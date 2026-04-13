import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)


def corn_loss(
    logits:      torch.Tensor,  # (batch, K-1)
    labels:      torch.Tensor,  # (batch,)  integer class indices in [0, K-1]
    num_classes: int,
) -> torch.Tensor:
    num_thresholds = num_classes - 1
    total_loss  = torch.zeros(1, device=logits.device, dtype=logits.dtype)
    total_count = 0

    for k in range(num_thresholds):
        # Only samples whose true class >= k are eligible for threshold k
        mask = labels >= k                      # (batch,) bool
        n_eligible = mask.sum().item()
        if n_eligible == 0:
            continue

        logits_k  = logits[mask, k]             # (n_eligible,)
        targets_k = (labels[mask] > k).float()  # 1 if class > k, else 0

        total_loss  = total_loss + F.binary_cross_entropy_with_logits(
            logits_k, targets_k, reduction='sum'
        )
        total_count += n_eligible

    return total_loss / total_count


def corn_label_from_logits(logits: torch.Tensor) -> torch.Tensor:

    fired      = (torch.sigmoid(logits) > 0.5).long()  # (batch, K-1)
    consistent = torch.cumprod(fired, dim=1)            # stops at first 0
    return consistent.sum(dim=1)                        # (batch,)


GRANULARITIES = {
    "coarse": {
        "bin_size":   10,
        "num_classes": 6,               # 5 distance bins + "no car"
        "bin_edges":  [10, 20, 30, 40, 50],
        "labels":     ["(0,10]", "(10,20]", "(20,30]", "(30,40]", "(40,50]", "100"],
    },
    "medium": {
        "bin_size":   5,
        "num_classes": 11,              # 10 distance bins + "no car"
        "bin_edges":  [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "labels": [
            "(0,5]", "(5,10]", "(10,15]", "(15,20]", "(20,25]",
            "(25,30]", "(30,35]", "(35,40]", "(40,45]", "(45,50]", "100",
        ],
    },
    "fine": {
        "bin_size":   2,
        "num_classes": 26,              # 25 distance bins + "no car"
        "bin_edges":  list(range(2, 52, 2)),
        "labels":     [f"({i},{i+2}]" for i in range(0, 50, 2)] + ["100"],
    },
}


def distance_to_label(distance: float, granularity: str) -> int:

    cfg = GRANULARITIES[granularity]
    if distance >= 100:
        return cfg["num_classes"] - 1
    for i, edge in enumerate(cfg["bin_edges"]):
        if distance <= edge:
            return i
    raise ValueError(
        f"distance_to_label received distance={distance:.2f}m which is > 50m "
        f"but < 100. This sample should have been filtered out in DrivingDataset."
    )


class DistanceCTECNN(nn.Module):

    def __init__(
        self,
        backbone_name: str,
        granularity:   str,
        pretrained:    bool  = True,
        dropout:       float = 0.3,
        shared_dim:    int   = 256,
    ):
        super().__init__()

        assert backbone_name in ("resnet18", "resnet50", "resnet101")
        assert granularity   in ("coarse", "medium", "fine")

        self.granularity    = granularity
        self.num_classes    = GRANULARITIES[granularity]["num_classes"]
        self.num_thresholds = self.num_classes - 1

        # ── Backbone ──────────────────────────────────────────────────────
        weights_map = {
            "resnet18":  ResNet18_Weights.DEFAULT  if pretrained else None,
            "resnet50":  ResNet50_Weights.DEFAULT  if pretrained else None,
            "resnet101": ResNet101_Weights.DEFAULT if pretrained else None,
        }
        builder_map = {
            "resnet18":  models.resnet18,
            "resnet50":  models.resnet50,
            "resnet101": models.resnet101,
        }
        backbone    = builder_map[backbone_name](weights=weights_map[backbone_name])
        in_features = backbone.fc.in_features  # 512 for R18, 2048 for R50/R101

        # Strip the original ImageNet FC head; keep everything up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output shape: (batch, in_features, 1, 1)

        # Both heads branch off here, so the backbone optimises one unified
        # representation rather than being pulled in two directions directly.
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, shared_dim),
            nn.ReLU(),
        )

        self.cte_head = nn.Linear(shared_dim, 1)

        self.distance_head = nn.Linear(shared_dim, self.num_thresholds)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 3, H, W)
        Returns:
            cte_pred:    (batch, 1)   — raw CTE in meters
            dist_logits: (batch, K-1) — CORN threshold logits
        """
        features    = self.backbone(x)           # (batch, in_features, 1, 1)
        shared      = self.shared(features)      # (batch, shared_dim)
        cte_pred    = self.cte_head(shared)      # (batch, 1)
        dist_logits = self.distance_head(shared) # (batch, K-1)
        return cte_pred, dist_logits

    def predict(self, x: torch.Tensor) -> dict:

        self.eval()
        with torch.no_grad():
            cte_pred, dist_logits = self.forward(x)

        dist_classes = corn_label_from_logits(dist_logits)
        dist_labels  = [
            GRANULARITIES[self.granularity]["labels"][i]
            for i in dist_classes.cpu().tolist()
        ]
        return {
            "cte":            cte_pred.squeeze(1),
            "distance_class": dist_classes,
            "distance_label": dist_labels,
        }



class MultiTaskLoss(nn.Module):
    """
    Combined loss: lambda_cte * Huber(cte) + lambda_dist * CORN(distance)

    Args:
        num_classes:  number of distance classes K
        lambda_cte:   weight for the CTE loss term
        lambda_dist:  weight for the distance loss term
        huber_delta:  Huber transition point; good rule = 1 bin width in meters
                        coarse → 10.0,  medium → 5.0,  fine → 2.0
    """

    def __init__(
        self,
        num_classes:  int,
        lambda_cte:   float = 1.0,
        lambda_dist:  float = 1.0,
        huber_delta:  float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cte  = lambda_cte
        self.lambda_dist = lambda_dist
        self.huber_delta = huber_delta

    def forward(
        self,
        cte_pred:    torch.Tensor,  # (batch, 1)
        cte_true:    torch.Tensor,  # (batch,)
        dist_logits: torch.Tensor,  # (batch, K-1)
        dist_labels: torch.Tensor,  # (batch,)  integer class indices
    ) -> tuple[torch.Tensor, dict]:
        loss_cte  = F.huber_loss(
            cte_pred.squeeze(1), cte_true, delta=self.huber_delta
        )
        loss_dist = corn_loss(
            dist_logits, dist_labels, num_classes=self.num_classes
        )
        total = self.lambda_cte * loss_cte + self.lambda_dist * loss_dist

        return total, {
            "loss_total": total.item(),
            "loss_cte":   loss_cte.item(),
            "loss_dist":  loss_dist.item(),
        }



def train_step(
    model:     DistanceCTECNN,
    images:    torch.Tensor,
    cte_true:  torch.Tensor,
    dist_true: torch.Tensor,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
) -> dict:
    model.train()
    optimizer.zero_grad()

    cte_pred, dist_logits = model(images)
    total_loss, breakdown = criterion(cte_pred, cte_true, dist_logits, dist_true)

    total_loss.backward()
    optimizer.step()

    return breakdown



def build_all_models(pretrained: bool = True) -> dict:
    backbone_names = ["resnet18", "resnet50", "resnet101"]
    granularities  = ["coarse", "medium", "fine"]
    huber_deltas   = {"coarse": 10.0, "medium": 5.0, "fine": 2.0}

    all_models = {}
    for backbone in backbone_names:
        for granularity in granularities:
            name = f"{backbone}_{granularity}"
            all_models[name] = {
                "model": DistanceCTECNN(
                    backbone_name=backbone,
                    granularity=granularity,
                    pretrained=pretrained,
                ),
                "criterion": MultiTaskLoss(
                    num_classes=GRANULARITIES[granularity]["num_classes"],
                    huber_delta=huber_deltas[granularity],
                ),
            }
            cfg = GRANULARITIES[granularity]
            print(
                f"Built {name:25s} | "
                f"classes={cfg['num_classes']:2d} | "
                f"thresholds={cfg['num_classes']-1:2d} | "
                f"huber_delta={huber_deltas[granularity]}"
            )
    return all_models

