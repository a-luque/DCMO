"""
evaluate_checkpoint.py  —  evaluate a saved checkpoint on val/test splits.

Usage:
    python3 evaluate_checkpoint.py \
        --checkpoint  checkpoints/resnet50_fine_best.pt \
        --h5_file     dataset_packed.h5 \
        --batch_size  256 \
        --num_workers 4

The backbone and granularity are read directly from the checkpoint so you
don't need to pass them manually.
"""

import os
import io
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py

from distance_cte_cnns_turn import (
    DistanceCTECNN,
    MultiTaskLoss,
    GRANULARITIES,
    distance_to_label,
    corn_label_from_logits,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class DrivingDatasetHDF5(Dataset):
    def __init__(self, h5_path, indices, granularity):
        self.h5_path  = h5_path
        self.indices  = np.sort(indices)   # sorted = sequential reads
        self._file    = None               # opened lazily per worker

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.transform = normalize

        # Pre-load scalar arrays into RAM (tiny compared to images)
        with h5py.File(h5_path, "r") as f:
            raw_dist  = f["dist"][self.indices]
            raw_cte   = f["cte"][self.indices]
            raw_man   = f["maneuver"][self.indices]

        self.dist_labels = np.array(
            [distance_to_label(float(d), granularity) for d in raw_dist],
            dtype=np.int64,
        )
        self.ctes      = raw_cte.astype(np.float32)
        self.maneuvers = raw_man.astype(np.int64)

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open()
        jpeg_bytes = self._file["images"][int(self.indices[idx])].tobytes()
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        img = transforms.functional.to_tensor(img)
        img = self.transform(img)
        return (
            img,
            torch.tensor(self.ctes[idx],       dtype=torch.float32),
            torch.tensor(self.dist_labels[idx], dtype=torch.long),
            torch.tensor(self.maneuvers[idx],   dtype=torch.long),
        )


def split_indices(h5_path, val_ratio=0.15, test_ratio=0.10, seed=42):
    with h5py.File(h5_path, "r") as f:
        n = f["images"].shape[0]
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test  = int(n * test_ratio)
    n_val   = int(n * val_ratio)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    print(f"Split: {n - n_test - n_val:,} train / {n_val:,} val / {n_test:,} test samples")
    return val_idx, test_idx


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name="test"):
    model.eval()

    total_loss = total_cte_loss = total_dist_loss = 0.0
    n_batches  = 0
    all_cte_pred,  all_cte_true  = [], []
    all_dist_pred, all_dist_true = [], []

    for images, cte_true, dist_true, maneuver in loader:
        images    = images.to(device, non_blocking=True)
        cte_true  = cte_true.to(device, non_blocking=True)
        dist_true = dist_true.to(device, non_blocking=True)
        maneuver  = maneuver.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            cte_pred, dist_logits = model(images, maneuver)
            _, breakdown = criterion(cte_pred, cte_true, dist_logits, dist_true)

        total_loss      += breakdown["loss_total"]
        total_cte_loss  += breakdown["loss_cte"]
        total_dist_loss += breakdown["loss_dist"]
        n_batches       += 1

        all_cte_pred.append(cte_pred.squeeze(1).cpu())
        all_cte_true.append(cte_true.cpu())
        all_dist_pred.append(corn_label_from_logits(dist_logits).cpu())
        all_dist_true.append(dist_true.cpu())

        if n_batches % 50 == 0:
            print(f"  [{split_name}] {n_batches} batches...", flush=True)

    cte_pred_t  = torch.cat(all_cte_pred)
    cte_true_t  = torch.cat(all_cte_true)
    dist_pred_t = torch.cat(all_dist_pred)
    dist_true_t = torch.cat(all_dist_true)

    cte_mae      = (cte_pred_t - cte_true_t).abs().mean().item()
    dist_acc     = (dist_pred_t == dist_true_t).float().mean().item()
    dist_acc_1   = (dist_pred_t - dist_true_t).abs().le(1).float().mean().item()

    labels    = GRANULARITIES[model.granularity]["labels"]
    per_class = {}
    for i, label in enumerate(labels):
        mask = dist_true_t == i
        if mask.sum() > 0:
            per_class[label] = {
                "acc":   (dist_pred_t[mask] == i).float().mean().item(),
                "count": int(mask.sum().item()),
            }

    return {
        "loss_total":        total_loss      / n_batches,
        "loss_cte":          total_cte_loss  / n_batches,
        "loss_dist":         total_dist_loss / n_batches,
        "cte_mae":           cte_mae,
        "dist_acc":          dist_acc,
        "dist_acc_off_by_1": dist_acc_1,
        "dist_per_class":    per_class,
        "n_samples":         len(cte_pred_t),
    }


def print_metrics(metrics, split_name):
    print(f"\n{'='*52}")
    print(f"  {split_name.upper()}  ({metrics['n_samples']:,} samples)")
    print(f"{'='*52}")
    print(f"  loss_total        : {metrics['loss_total']:.4f}")
    print(f"  loss_cte          : {metrics['loss_cte']:.4f}")
    print(f"  loss_dist         : {metrics['loss_dist']:.4f}")
    print(f"  cte_mae           : {metrics['cte_mae']:.4f}")
    print(f"  dist_acc          : {metrics['dist_acc']*100:.2f}%")
    print(f"  dist_acc_off_by_1 : {metrics['dist_acc_off_by_1']*100:.2f}%")
    print(f"\n  Per-class accuracy:")
    print(f"  {'Label':>12}  {'Acc':>7}  {'Count':>8}")
    print(f"  {'-'*32}")
    for label, info in metrics["dist_per_class"].items():
        print(f"  {label:>12}  {info['acc']*100:>6.1f}%  {info['count']:>8,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--h5_file",     required=True)
    parser.add_argument("--backbone",    default=None,
                        choices=["resnet18", "resnet50", "resnet101"],
                        help="Inferred from checkpoint filename if not given.")
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--split",       default="both",
                        choices=["val", "test", "both"])
    parser.add_argument("--lambda_cte",  type=float, default=100.0)
    parser.add_argument("--lambda_dist", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"Loading: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Strip _orig_mod. prefix from checkpoints saved under torch.compile()
    state = {
        k.replace("_orig_mod.", "", 1): v
        for k, v in checkpoint["model_state"].items()
    }

    granularity = checkpoint["granularity"]
    print(f"  granularity : {granularity}")
    print(f"  saved epoch : {checkpoint['epoch']}")
    print(f"  val loss    : {checkpoint['val_metrics']['loss_total']:.4f}")
    print(f"  val cte_mae : {checkpoint['val_metrics']['cte_mae']:.4f}")
    print(f"  val dist_acc: {checkpoint['val_metrics']['dist_acc']*100:.2f}%")

    # Infer backbone from filename if not provided
    backbone = args.backbone
    if backbone is None:
        fname = os.path.basename(args.checkpoint)
        for b in ["resnet101", "resnet50", "resnet18"]:  # longest first
            if b in fname:
                backbone = b
                break
    assert backbone is not None, \
        "Could not infer backbone from filename — pass --backbone explicitly."
    print(f"  backbone    : {backbone}")

    # ── Build model ───────────────────────────────────────────────────────
    model = DistanceCTECNN(
        backbone_name=backbone,
        granularity=granularity,
        pretrained=False,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    criterion = MultiTaskLoss(
        num_classes=GRANULARITIES[granularity]["num_classes"],
        lambda_cte=args.lambda_cte,
        lambda_dist=args.lambda_dist,
        huber_delta=0.1,
    )

    # ── Build data loaders (same seed as training → same splits) ─────────
    val_idx, test_idx = split_indices(args.h5_file)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    if args.split in ("val", "both"):
        print(f"\nEvaluating on val ({len(val_idx):,} samples)...")
        val_loader = DataLoader(
            DrivingDatasetHDF5(args.h5_file, val_idx, granularity),
            shuffle=False, **loader_kwargs,
        )
        print_metrics(evaluate(model, val_loader, criterion, device, "val"), "Validation")

    if args.split in ("test", "both"):
        print(f"\nEvaluating on test ({len(test_idx):,} samples)...")
        test_loader = DataLoader(
            DrivingDatasetHDF5(args.h5_file, test_idx, granularity),
            shuffle=False, **loader_kwargs,
        )
        print_metrics(evaluate(model, test_loader, criterion, device, "test"), "Test")


if __name__ == "__main__":
    main()