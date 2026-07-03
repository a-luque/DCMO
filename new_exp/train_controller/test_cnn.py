"""
Evaluate a trained CNN checkpoint on a test HDF5 dataset.

Metrics reported (all in original units):
    Steering : MAE, RMSE
    Distance : MAE, RMSE  (metres)
    Per-maneuver breakdown for steering MAE (straight / left / right)

Usage:
    python test_cnn.py \
        --checkpoint  ./checkpoints/aggressive_cnn/best.pt \
        --h5_file     /path/to/test.h5 \
        --batch_size  64
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


# ── Model (must match train_cnn.py exactly) ───────────────────────────────────

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


# ── Dataset ───────────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(self, h5_path: str):
        self.h5_path = h5_path

        with h5py.File(h5_path, "r") as f:
            self.cte      = f["cte"][:]
            self.dist     = f["dist"][:]
            self.maneuver = f["maneuver"][:]


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
        return len(self.cte)

    def __getitem__(self, idx):
        f         = self._get_handle()
        raw_bytes = bytes(f["images"][idx])
        img       = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img_t     = self.tf(img)

        return (
            img_t,
            torch.tensor(int(self.maneuver[idx]),  dtype=torch.long),
            torch.tensor(float(self.cte[idx]),     dtype=torch.float32),
            torch.tensor(float(self.dist[idx]),    dtype=torch.float32),  # raw metres
        )


# ── Metrics ───────────────────────────────────────────────────────────────────

def mae(pred, gt):
    return float(np.abs(pred - gt).mean())

def rmse(pred, gt):
    return float(np.sqrt(((pred - gt) ** 2).mean()))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--h5_file",     required=True, help="Path to test .h5 file")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--workers",     type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"Checkpoint : epoch {ckpt['epoch']},  val_loss={ckpt['val_loss']:.4f}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = CNNDrivingModel().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Data ──────────────────────────────────────────────────────────────
    ds     = TestDataset(args.h5_file)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)
    print(f"Test samples : {len(ds):,}")

    # ── Inference ─────────────────────────────────────────────────────────
    all_pred_steer = []
    all_pred_dist  = []
    all_gt_steer   = []
    all_gt_dist    = []
    all_maneuver   = []

    with torch.no_grad():
        for imgs, maneuvers, gt_cte, gt_dist in tqdm(loader, desc="Evaluating"):
            imgs      = imgs.to(device, non_blocking=True)
            maneuvers = maneuvers.to(device, non_blocking=True)

            pred_steer, pred_dist = model(imgs, maneuvers)

            all_pred_steer.append(pred_steer.cpu().numpy())
            all_pred_dist .append(pred_dist.cpu().numpy())
            all_gt_steer  .append(gt_cte.numpy())
            all_gt_dist   .append(gt_dist.numpy())
            all_maneuver  .append(maneuvers.cpu().numpy())

    pred_steer = np.concatenate(all_pred_steer)
    pred_dist  = np.concatenate(all_pred_dist)
    gt_steer   = np.concatenate(all_gt_steer)
    gt_dist    = np.concatenate(all_gt_dist)
    maneuvers  = np.concatenate(all_maneuver)

    # ── Overall metrics ───────────────────────────────────────────────────
    print("\n─── Overall ─────────────────────────────────────────────────")
    print(f"  Steering  MAE  : {mae(pred_steer, gt_steer):.4f}")
    print(f"  Steering  RMSE : {rmse(pred_steer, gt_steer):.4f}")
    print(f"  Distance  MAE  : {mae(pred_dist, gt_dist):.4f} m")
    print(f"  Distance  RMSE : {rmse(pred_dist, gt_dist):.4f} m")

    # ── Per-maneuver steering breakdown ───────────────────────────────────
    maneuver_names = {0: "Straight", 1: "Turn left", 2: "Turn right"}
    print("\n─── Steering MAE per maneuver ────────────────────────────────")
    for code, name in maneuver_names.items():
        mask = maneuvers == code
        if mask.sum() == 0:
            print(f"  {name:<12}: no samples")
            continue
        print(f"  {name:<12}: MAE={mae(pred_steer[mask], gt_steer[mask]):.4f}  "
              f"RMSE={rmse(pred_steer[mask], gt_steer[mask]):.4f}  (n={mask.sum():,})")

    # ── Distance error distribution ───────────────────────────────────────
    dist_err = np.abs(pred_dist - gt_dist)
    print("\n─── Distance absolute error distribution (metres) ────────────")
    for pct in [50, 75, 90, 95]:
        print(f"  p{pct}  : {np.percentile(dist_err, pct):.3f} m")

    # ── Save predictions ──────────────────────────────────────────────────
    out_path = Path(args.checkpoint).parent / "test_predictions.npz"
    np.savez(out_path,
             pred_steering=pred_steer,
             pred_distance=pred_dist,
             gt_steering=gt_steer,
             gt_distance=gt_dist,
             maneuver=maneuvers)
    print(f"\nPredictions saved → {out_path}")


if __name__ == "__main__":
    main()