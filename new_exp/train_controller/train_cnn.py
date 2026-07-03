"""
Train a ResNet18-based CNN for steering angle and distance prediction.

Architecture:
    Input : image (3, H, W)  +  maneuver (0/1/2 → learned embedding)
    Backbone : ResNet18 (pretrained), final FC replaced
    Fusion   : concat(image features, maneuver embedding) → shared FC head
    Outputs  : steering angle  (regression, [-1, 1])
               distance        (regression, metres, ~5 50 m)

Loss scale problem & solution
------------------------------
Steering is in [-1, 1]   → Huber loss is O(0.01)
Distance is in [5, 50] m → Huber loss is O(1 10)

We keep both targets in raw units and use --dist_weight to down-scale the
distance loss term so both heads contribute equally to training.
A good starting point is --dist_weight 0.005.

Usage:
    python train_cnn.py \
        --h5_file      /path/to/aggressive.h5 \
        --out_dir      ./checkpoints/aggressive_cnn \
        --epochs       30 \
        --batch_size   64 \
        --dist_weight  0.005
"""

import argparse
import io
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm


# ── Dataset ───────────────────────────────────────────────────────────────────

class DrivingDataset(Dataset):
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(self, h5_path: str, augment: bool = False, indices=None):
        self.h5_path = h5_path
        self.augment = augment

        with h5py.File(h5_path, "r") as f:
            self.cte      = f["cte"][:]       # (N,) float32
            self.dist     = f["dist"][:]      # (N,) float32  raw metres
            self.maneuver = f["maneuver"][:]  # (N,) int64

        self.indices = np.asarray(indices) if indices is not None else np.arange(len(self.cte))

        base_tf = [
            transforms.ToTensor(),
            transforms.Normalize(self.IMG_MEAN, self.IMG_STD),
        ]
        aug_tf = [
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        ] + base_tf
        # NOTE: RandomHorizontalFlip omitted — flipping breaks the steering sign.

        self.tf_train = transforms.Compose(aug_tf)
        self.tf_eval  = transforms.Compose(base_tf)
        self._handle  = None

    def _get_handle(self):
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r")
        return self._handle

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        f   = self._get_handle()

        raw_bytes  = bytes(f["images"][idx])
        img        = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img_tensor = (self.tf_train if self.augment else self.tf_eval)(img)

        return (
            img_tensor,
            torch.tensor(int(self.maneuver[idx]),   dtype=torch.long),
            torch.tensor(float(self.cte[idx]),      dtype=torch.float32),
            torch.tensor(float(self.dist[idx]),     dtype=torch.float32),  # raw metres
        )


# ── Model ─────────────────────────────────────────────────────────────────────

class CNNDrivingModel(nn.Module):
    """ResNet18 backbone + maneuver embedding → steering and distance heads."""

    def __init__(self, maneuver_embed_dim: int = 16, dropout: float = 0.3):
        super().__init__()

        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features      # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.maneuver_embed = nn.Embedding(3, maneuver_embed_dim)

        fused_dim = feat_dim + maneuver_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head_steering = nn.Linear(256, 1)
        self.head_distance = nn.Linear(256, 1)

    def forward(self, img: torch.Tensor, maneuver: torch.Tensor):
        feat     = self.backbone(img)
        m_emb    = self.maneuver_embed(maneuver)
        fused    = self.fusion(torch.cat([feat, m_emb], dim=1))
        steering = self.head_steering(fused).squeeze(1)
        distance = self.head_distance(fused).squeeze(1)
        return steering, distance


# ── Data loaders ──────────────────────────────────────────────────────────────

def make_loaders(h5_path: str, val_split: float, batch_size: int, num_workers: int):
    with h5py.File(h5_path, "r") as f:
        n_total = f["cte"].shape[0]

    rng     = np.random.default_rng(42)
    all_idx = rng.permutation(n_total)
    n_val   = int(n_total * val_split)
    val_idx   = all_idx[:n_val]
    train_idx = all_idx[n_val:]

    train_ds = DrivingDataset(h5_path, augment=True,  indices=train_idx)
    val_ds   = DrivingDataset(h5_path, augment=False, indices=val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    return train_loader, val_loader


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file",      required=True)
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--val_split",    type=float, default=0.1)
    parser.add_argument("--workers",      type=int,   default=4)
    parser.add_argument("--freeze_bn",    action="store_true",
                        help="Freeze BatchNorm layers in backbone")
    parser.add_argument("--loss",         default="huber", choices=["mse", "l1", "huber"])
    parser.add_argument("--steer_weight", type=float, default=1.0,
                        help="Weight on steering loss (raw scale [-1,1])")
    parser.add_argument("--dist_weight",  type=float, default=0.005,
                        help="Weight on distance loss (raw metres). "
                             "Distance loss is ~100-1000x larger than steering loss, "
                             "so this should be ~0.001 to 0.01. Default: 0.005")
    parser.add_argument("--patience",     type=int,   default=5,
                        help="Early stopping patience. Set 0 to disable.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"steer_weight : {args.steer_weight}   dist_weight : {args.dist_weight}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = make_loaders(
        args.h5_file, args.val_split, args.batch_size, args.workers
    )
    print(f"Train: {len(train_loader.dataset):,}  |  Val: {len(val_loader.dataset):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = CNNDrivingModel().to(device)

    if args.freeze_bn:
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    # ── Loss ──────────────────────────────────────────────────────────────
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "l1":
        criterion = nn.L1Loss()
    else:
        criterion = nn.HuberLoss(delta=1.0)

    sw, dw = args.steer_weight, args.dist_weight

    # ── Optimizer — differential LR ───────────────────────────────────────
    backbone_params = list(model.backbone.parameters())
    head_params     = (list(model.maneuver_embed.parameters()) +
                       list(model.fusion.parameters()) +
                       list(model.head_steering.parameters()) +
                       list(model.head_distance.parameters()))

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    def _run_epoch(train: bool):
        loader = train_loader if train else val_loader
        model.train(train)
        total = steer_total = dist_total = 0.0

        with torch.set_grad_enabled(train):
            pbar = tqdm(loader, desc="train" if train else "val  ", leave=False)
            for imgs, maneuvers, ctes, dists in pbar:
                imgs      = imgs.to(device, non_blocking=True)
                maneuvers = maneuvers.to(device, non_blocking=True)
                ctes      = ctes.to(device, non_blocking=True)
                dists     = dists.to(device, non_blocking=True)

                pred_steer, pred_dist = model(imgs, maneuvers)

                ls   = criterion(pred_steer, ctes)
                ld   = criterion(pred_dist,  dists)
                loss = sw * ls + dw * ld

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total       += loss.item()
                steer_total += ls.item()
                dist_total  += ld.item()

        n = len(loader)
        return total / n, steer_total / n, dist_total / n

    for epoch in range(1, args.epochs + 1):
        train_loss, train_s, train_d = _run_epoch(train=True)
        val_loss,   val_s,   val_d   = _run_epoch(train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.4f} (steer {train_s:.4f} dist {train_d:.4f}) | "
            f"val   {val_loss:.4f}  (steer {val_s:.4f} dist {val_d:.4f}) | "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        # ── Single improved flag drives both checkpoint and early stop ────
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss":  val_loss,
                "args":      vars(args),
            }, out_dir / "best.pt")
            print(f"  ✓ Saved best checkpoint (val={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{args.patience})")

        if epoch % 5 == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss":  val_loss,
                "args":      vars(args),
            }, out_dir / f"epoch_{epoch:03d}.pt")

        # ── Early stopping ────────────────────────────────────────────────
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()