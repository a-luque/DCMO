import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)
from model_utils import (
    build_midpoint_table,
    class_to_distance,
    get_test_transform,
    predict_single,
    load_model,
)

def classes_to_distance(
    class_idx:   torch.Tensor, # (batch, 1)
    granularity: str,
    nocar_value: float | None = None,
) -> float | None:

    table = build_midpoint_table(granularity)
    value = table[class_idx]
    if value is None:
        return nocar_value
    return value

class MoE(nn.Module):



    def __init__(
        self,
        controllers_dir: str,
        dropout:         float = 0.3,
        device:          str   = "cuda",
        hidden_dim:      int   = 50,
        shared_dim:      int   = 50,
        contexts_dim:    int   = 16,
        ):
        super().__init__()


        # ── Backbone ──────────────────────────────────────────────────────
        controllers_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(f"{controllers_dir}")) for f in fn]
        controllers_paths.sort()
        self.controllers = []
        self.dim_dists = 0
        for c in controllers_paths:
            controller, _ = load_model(c, device)
            for param in controller.parameters():
                param.requires_grad = False
            self.controllers += [controller]
            self.dim_dists += controller.num_thresholds
        self.dim_ctes = len(self.controllers)

       
        self.fc1 = torch.nn.Linear(contexts_dim+self.dim_ctes+self.dim_dists,hidden_dim)
        
        self.dropout1 = torch.nn.Dropout(dropout)
        
        self.fc2 = torch.nn.Linear(hidden_dim,shared_dim)
        
        self.cte_head = torch.nn.Linear(shared_dim, 1)
        self.distance_head = torch.nn.Linear(shared_dim, 1)


    def forward(
        self,
        img:      torch.Tensor,  # (batch, 3, H, W)
        maneuver: torch.Tensor,  # (batch,)  integer codes: 1=straight, 2=left, 3=right
        context:  torch.Tensor,  # (batch, 16)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:        (batch, 3, H, W)
            maneuver: (batch,) int64 with values 1, 2, or 3
            context:  (batch,16) 
        Returns:
            cte_pred:    (batch, 1)   — raw CTE in radians
            dist_pred:   (batch, 1)   — raw distance in meters
        """
        
        

        ctes = []
        dists = []
        for model in self.controllers:
            cte_pred, dist_logits = model(img, maneuver)
            # dist_classes = corn_label_from_logits(dist_logits)
            # dist_m     = class_to_distance(dist_class, model.granularity, nocar_value)
            ctes += [cte_pred]
            dists += [dist_logits]

        # Size: (batch, 1). After cat, we want (batch, dim_ctes)
        ctes_tensor = torch.cat(ctes, dim=1)
        # Size: (batch, K-1). After cat, we want (batch, dim_dists)
        dists_tensor = torch.cat(dists, dim=1)

        # Combining with contexts
        combined = torch.cat([ctes_tensor, dists_tensor, context], dim=1)

        x = self.fc1(combined)
        x = self.dropout1(x)
        x = self.fc2(x)

        cte_pred    = self.cte_head(x)       # (batch, 1)
        dist_pred = self.distance_head(x)  # (batch, K-1)
        
        return cte_pred, dist_pred

    def predict(
        self,
        x:        torch.Tensor,
        maneuver: torch.Tensor,
        context:  torch.Tensor,
    ) -> dict:
        self.eval()
        with torch.no_grad():
            cte_pred, dist_pred = self.forward(x, maneuver, context)

        return {
            "cte":            cte_pred.squeeze(1),
            "distance": dist_pred.squeeze(1),
        }

    @torch.no_grad()
    def predict_single(
        self,
        input_img:   Image.Image,
        maneuver:    int,               # 1=straight, 2=left, 3=right
        w:           np.array,
        d:           int,
        s:           int,  
        device:      torch.device,
        transform,
        nocar_value: float | None = None,
    ) -> dict:

        #image = Image.open(image_path).convert("RGB")
        x     = transform(input_img).unsqueeze(0).to(device)          # (1, 3, H, W)
        m     = torch.tensor([maneuver], dtype=torch.long).to(device)
        c     = torch.cat([torch.tensor(w).unsqueeze(0), torch.tensor([d]).unsqueeze(0), torch.tensor([s]).unsqueeze(0)], dim=1).to(device)

        cte_pred, dist_pred = self(x, m, c.to(torch.float32))


        return {
            "cte_pred":            cte_pred.squeeze().item(),
            "dist_pred":      dist_pred.squeeze().item(),
        }



class MoELoss(nn.Module):

    def __init__(
        self,
        lambda_cte:   float = 1.0,
        lambda_dist:  float = 1.0,
        huber_delta:  float = 2.0,
    ):
        super().__init__()
        self.lambda_cte  = lambda_cte
        self.lambda_dist = lambda_dist
        self.huber_delta = huber_delta

    def forward(
        self,
        cte_pred:    torch.Tensor,  # (batch, 1)
        cte_true:    torch.Tensor,  # (batch,)
        dist_pred: torch.Tensor,  # (batch, 1)
        dist_true: torch.Tensor,  # (batch,)  
    ) -> tuple[torch.Tensor, dict]:

        loss_cte  = F.huber_loss(
            cte_pred.squeeze(1), cte_true, delta=self.huber_delta
        )
        loss_dist = F.huber_loss(
            dist_pred.squeeze(1), dist_true, delta=self.huber_delta
        )
        total = self.lambda_cte * loss_cte + self.lambda_dist * loss_dist

        return total, {
            "loss_total": total.item(),
            "loss_cte":   loss_cte.item(),
            "loss_dist":  loss_dist.item(),
        }

def train_step(
    model:     MoE,
    images:    torch.Tensor,    # (batch, 3, H, W)
    maneuver:  torch.Tensor,    # (batch,)  int64: 1=straight, 2=left, 3=right
    cte_true:  torch.Tensor,    # (batch,)  real-valued CTE in radians
    dist_true: torch.Tensor,    # (batch,)  real-value distance
    context:   torch.Tensor,
    criterion: MoELoss,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.amp.GradScaler | None = None,
) -> dict:
    """Single training step with optional mixed-precision (AMP) support.

    Pass a GradScaler from torch.amp to enable FP16/BF16 training, which
    gives 1.5-4x speedup depending on GPU (larger on A100/H100 than T4).
    Pass scaler=None to use full FP32 (original behaviour).
    """
    model.train()
    optimizer.zero_grad()
    

    with torch.autocast(device_type="cuda", enabled=(scaler is not None)):
        cte_pred, dist_pred = model(images, maneuver, context)
        total_loss, breakdown = criterion(cte_pred, cte_true, dist_pred, dist_true)

    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        optimizer.step()

    return breakdown