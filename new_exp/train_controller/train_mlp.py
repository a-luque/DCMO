"""
Train an MLP to predict acceleration using a frozen CNN for distance prediction.

Pipeline:
    Image + Maneuver → [Frozen CNN] → predicted distance
    (predicted_dist, ego_speed, leader_speed, maneuver) → [MLP] → acceleration

Inputs to MLP  : predicted_dist (1), ego_speed (1), leader_speed (1), maneuver embedding (16) = 19
Output         : acceleration in [-1, 1]  (negative=brake, positive=throttle)

Usage:
    python train_mlp.py \
        --h5_file     /path/to/aggressive.h5 \
        --cnn_ckpt    ./checkpoints/aggressive_cnn/best.pt \
        --out_dir     ./checkpoints/aggressive_mlp \
        --epochs      30 \
        --batch_size  256
"""

import argparse
import io
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


# ── CNN (frozen, identical to train_cnn.py) ───────────────────────────────────

class CNNDrivingModel(nn.Module):
    def __init__(self, maneuver_embed_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
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

    def forward(self, img, maneuver):
        feat     = self.backbone(img)
        m_emb    = self.maneuver_embed(maneuver)
        fused    = self.fusion(torch.cat([feat, m_emb], dim=1))
        steering = self.head_steering(fused).squeeze(1)
        distance = self.head_distance(fused).squeeze(1)
        return steering, distance


# ── MLP ───────────────────────────────────────────────────────────────────────

class AccelerationMLP(nn.Module):
    """
    Inputs  : predicted_dist (1) + ego_speed (1) + leader_speed (1)
              + maneuver embedding (16)  →  total 19-d
    Output  : acceleration scalar in [-1, 1]

    Maneuver is embedded separately (shared embedding weight with CNN is NOT
    used — the MLP learns its own embedding since the maneuver-acceleration
    relationship is different from maneuver-steering).
    """

    def __init__(self, maneuver_embed_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        self.maneuver_embed = nn.Embedding(3, maneuver_embed_dim)

        # 3 scalar inputs + maneuver embedding
        in_dim = 3 + maneuver_embed_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh(),              # output in [-1, 1]
        )

    def forward(self, dist, ego_speed, leader_speed, maneuver):
        """
        dist, ego_speed, leader_speed : (B,) float
        maneuver                      : (B,) long
        """
        m_emb   = self.maneuver_embed(maneuver)                        # (B, 16)
        scalars = torch.stack([dist, ego_speed, leader_speed], dim=1)  # (B, 3)
        x       = torch.cat([scalars, m_emb], dim=1)                   # (B, 19)
        return self.net(x).squeeze(1)                                  # (B,)


# ── Dataset ───────────────────────────────────────────────────────────────────

class MLPDataset(Dataset):
    """
    Returns everything needed to:
      1. Run the frozen CNN to get predicted distance  (image + maneuver)
      2. Feed the MLP                                  (pred_dist, ego_speed, leader_speed, maneuver)
      3. Supervise against ground-truth acceleration   (acc)
    """

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(self, h5_path: str, indices=None):
        self.h5_path = h5_path

        with h5py.File(h5_path, "r") as f:
            self.acc          = f["acc"][:]
            self.ego_speed    = f["ego_speed"][:]
            self.leader_speed = f["leader_speed"][:]
            self.maneuver     = f["maneuver"][:]


        self.indices = np.asarray(indices) if indices is not None else np.arange(len(self.acc))

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.IMG_MEAN, self.IMG_STD),
        ])
        self._handle = None

    def _get_handle(self):
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r")
        return self._handle

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        f   = self._get_handle()

        raw_bytes = bytes(f["images"][idx])
        img       = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img_t     = self.tf(img)

        return (
            img_t,
            torch.tensor(int(self.maneuver[idx]),       dtype=torch.long),
            torch.tensor(float(self.ego_speed[idx]),    dtype=torch.float32),
            torch.tensor(float(self.leader_speed[idx]), dtype=torch.float32),
            torch.tensor(float(self.acc[idx]),          dtype=torch.float32),
        )


# ── Data loaders ──────────────────────────────────────────────────────────────

def make_loaders(h5_path: str, val_split: float, batch_size: int, num_workers: int):
    with h5py.File(h5_path, "r") as f:
        n_total = f["acc"].shape[0]

    rng     = np.random.default_rng(42)
    all_idx = rng.permutation(n_total)
    n_val   = int(n_total * val_split)

    val_idx   = all_idx[:n_val]
    train_idx = all_idx[n_val:]

    train_ds = MLPDataset(h5_path, indices=train_idx)
    val_ds   = MLPDataset(h5_path, indices=val_idx)

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
    parser.add_argument("--h5_file",     required=True)
    parser.add_argument("--cnn_ckpt",    required=True,
                        help="Path to trained CNN checkpoint (best.pt)")
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=256,
                        help="MLP is lightweight so larger batches are fine")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--val_split",   type=float, default=0.1)
    parser.add_argument("--workers",     type=int,   default=4)
    parser.add_argument("--loss",        default="huber", choices=["mse", "l1", "huber"])
    parser.add_argument("--patience",    type=int,   default=10,
                        help="Early stopping patience. Set 0 to disable.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load and freeze CNN ───────────────────────────────────────────────
    cnn_ckpt = torch.load(args.cnn_ckpt, map_location=device)
    cnn = CNNDrivingModel().to(device)
    cnn.load_state_dict(cnn_ckpt["model"])
    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad = False
    print(f"CNN loaded from {args.cnn_ckpt} (epoch {cnn_ckpt['epoch']}) — frozen")

    # ── MLP ───────────────────────────────────────────────────────────────
    mlp = AccelerationMLP().to(device)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = make_loaders(
        args.h5_file, args.val_split, args.batch_size, args.workers
    )
    print(f"Train: {len(train_loader.dataset):,}  |  Val: {len(val_loader.dataset):,}")

    # ── Loss ──────────────────────────────────────────────────────────────
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "l1":
        criterion = nn.L1Loss()
    else:
        criterion = nn.HuberLoss(delta=1.0)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    def _run_epoch(train: bool):
        loader = train_loader if train else val_loader
        mlp.train(train)
        total = 0.0

        with torch.set_grad_enabled(train):
            pbar = tqdm(loader, desc="train" if train else "val  ", leave=False)
            for imgs, maneuvers, ego_speed, leader_speed, gt_acc in pbar:
                imgs         = imgs.to(device, non_blocking=True)
                maneuvers    = maneuvers.to(device, non_blocking=True)
                ego_speed    = ego_speed.to(device, non_blocking=True)
                leader_speed = leader_speed.to(device, non_blocking=True)
                gt_acc       = gt_acc.to(device, non_blocking=True)

                # CNN inference (no grad, frozen)
                with torch.no_grad():
                    _, pred_dist = cnn(imgs, maneuvers)

                pred_acc = mlp(pred_dist, ego_speed, leader_speed, maneuvers)
                loss     = criterion(pred_acc, gt_acc)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total += loss.item()

        return total / len(loader)

    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(train=True)
        val_loss   = _run_epoch(train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch":    epoch,
                "mlp":      mlp.state_dict(),
                "optimizer":optimizer.state_dict(),
                "val_loss": val_loss,
                "args":     vars(args),
                "cnn_ckpt": args.cnn_ckpt,
            }, out_dir / "best.pt")
            print(f"  ✓ Saved best checkpoint (val={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{args.patience})")

        if epoch % 5 == 0:
            torch.save({
                "epoch":    epoch,
                "mlp":      mlp.state_dict(),
                "optimizer":optimizer.state_dict(),
                "val_loss": val_loss,
                "args":     vars(args),
                "cnn_ckpt": args.cnn_ckpt,
            }, out_dir / f"epoch_{epoch:03d}.pt")

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()