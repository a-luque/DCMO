import os
import io
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py

from distance_cte_cnns_turn import (
    DistanceCTECNN,
    MultiTaskLoss,
    GRANULARITIES,
    distance_to_label,
    train_step,
    corn_label_from_logits,
)


class DrivingDatasetHDF5(Dataset):
    """
    Reads pre-packed (image, cte, dist, maneuver) samples from an HDF5 file
    produced by pack_dataset.py.

    All image data is stored as uint8 (C, H, W) — normalization is applied
    at __getitem__ time.  No filesystem seek per sample: HDF5 reads are
    sequential-chunk-friendly and work well with multiple DataLoader workers.

    Args:
        h5_path:     path to the packed .h5 file
        indices:     subset of row indices (e.g. train split)
        granularity: "coarse" | "medium" | "fine"
        transform:   torchvision transforms (applied after converting uint8→float)
    """

    def __init__(
        self,
        h5_path:     str,
        indices:     np.ndarray,
        granularity: str,
        transform=None,
    ):
        self.h5_path     = h5_path
        self.indices     = np.sort(indices)   # sorted = sequential HDF5 reads
        self.granularity = granularity
        self.transform   = transform
        self._file       = None   # opened lazily per worker (see _open)

        # Pre-compute dist labels from raw dist values
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

        print(f"Dataset: {len(self.indices)} samples ({granularity} granularity) "
              f"from {h5_path}")

    def _open(self):
        """Lazily open the HDF5 file. Each DataLoader worker gets its own handle."""
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open()
        h5_idx = int(self.indices[idx])

        # Decode JPEG bytes → PIL → float tensor
        jpeg_bytes = self._file["images"][h5_idx].tobytes()
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        img = transforms.functional.to_tensor(img)  # (3, H, W) float, 0-1

        if self.transform:
            # transform expects (3, H, W) float or PIL; we pass tensor directly
            img = self.transform(img)

        return (
            img,
            torch.tensor(self.ctes[idx],        dtype=torch.float32),
            torch.tensor(self.dist_labels[idx],  dtype=torch.long),
            torch.tensor(self.maneuvers[idx],    dtype=torch.long),
        )


def get_transforms(train: bool):
    """
    Normalization-only (resize was done at pack time).
    Training adds ColorJitter.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            normalize,
        ])
    else:
        return normalize


def split_indices(
    h5_path:    str,
    val_ratio:  float = 0.15,
    test_ratio: float = 0.10,
    seed:       int   = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        n = f["images"].shape[0]

    rng     = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    print(f"Split: {len(train_idx):,} train / {len(val_idx):,} val / "
          f"{len(test_idx):,} test samples")
    return train_idx, val_idx, test_idx


@torch.no_grad()
def evaluate(
    model:     DistanceCTECNN,
    loader:    DataLoader,
    criterion: MultiTaskLoss,
    device:    torch.device,
) -> dict:
    model.eval()

    total_loss = total_cte_loss = total_dist_loss = 0.0
    all_cte_pred, all_cte_true   = [], []
    all_dist_pred, all_dist_true = [], []

    for images, cte_true, dist_true, maneuver in loader:
        images    = images.to(device, non_blocking=True)
        cte_true  = cte_true.to(device, non_blocking=True)
        dist_true = dist_true.to(device, non_blocking=True)
        maneuver  = maneuver.to(device, non_blocking=True)

        cte_pred, dist_logits = model(images, maneuver)
        _, breakdown = criterion(cte_pred, cte_true, dist_logits, dist_true)

        total_loss      += breakdown["loss_total"]
        total_cte_loss  += breakdown["loss_cte"]
        total_dist_loss += breakdown["loss_dist"]

        all_cte_pred.append(cte_pred.squeeze(1).cpu())
        all_cte_true.append(cte_true.cpu())
        all_dist_pred.append(corn_label_from_logits(dist_logits).cpu())
        all_dist_true.append(dist_true.cpu())

    n = len(loader)
    cte_pred_t  = torch.cat(all_cte_pred)
    cte_true_t  = torch.cat(all_cte_true)
    dist_pred_t = torch.cat(all_dist_pred)
    dist_true_t = torch.cat(all_dist_true)

    cte_mae  = (cte_pred_t - cte_true_t).abs().mean().item()
    dist_acc = (dist_pred_t == dist_true_t).float().mean().item()

    labels    = GRANULARITIES[model.granularity]["labels"]
    per_class = {}
    for i, label in enumerate(labels):
        mask = dist_true_t == i
        if mask.sum() > 0:
            per_class[label] = (dist_pred_t[mask] == i).float().mean().item()

    return {
        "loss_total":         total_loss      / n,
        "loss_cte":           total_cte_loss  / n,
        "loss_dist":          total_dist_loss / n,
        "cte_mae":            cte_mae,
        "dist_acc":           dist_acc,
        "dist_per_class_acc": per_class,
    }


def train(
    model:        DistanceCTECNN,
    criterion:    MultiTaskLoss,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    optimizer:    torch.optim.Optimizer,
    scheduler:    torch.optim.lr_scheduler._LRScheduler,
    device:       torch.device,
    epochs:       int,
    save_path:    str,
    early_stop_patience: int | None = 5,
):
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_totals = {"loss_total": 0., "loss_cte": 0., "loss_dist": 0.}

        for images, cte_true, dist_true, maneuver in train_loader:
            images    = images.to(device, non_blocking=True)
            cte_true  = cte_true.to(device, non_blocking=True)
            dist_true = dist_true.to(device, non_blocking=True)
            maneuver  = maneuver.to(device, non_blocking=True)

            breakdown = train_step(
                model, images, maneuver, cte_true, dist_true, criterion, optimizer
            )
            for k in train_totals:
                train_totals[k] += breakdown[k]

        n_train = len(train_loader)
        train_metrics = {k: v / n_train for k, v in train_totals.items()}

        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss_total"])

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train loss={train_metrics['loss_total']:.4f} "
            f"(cte={train_metrics['loss_cte']:.4f}, dist={train_metrics['loss_dist']:.4f}) | "
            f"val loss={val_metrics['loss_total']:.4f} "
            f"cte_mae={val_metrics['cte_mae']:.4f} "
            f"dist_acc={val_metrics['dist_acc']:.3f}"
        )

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            epochs_without_improvement = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "granularity": model.granularity,
            }, save_path)
            print(f"  ✓ Saved best model → {save_path}")
        else:
            epochs_without_improvement += 1

        if (
            early_stop_patience is not None
            and epochs_without_improvement >= early_stop_patience
        ):
            print(f"  Early stopping after {early_stop_patience} epoch(s) without improvement.")
            break

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file",     required=True,
                        help="Path to packed HDF5 file from pack_dataset.py")
    parser.add_argument("--granularity", default="fine",
                        choices=["coarse", "medium", "fine"])
    parser.add_argument("--backbone",    default="resnet18",
                        choices=["resnet18", "resnet50", "resnet101"])
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=4e-4)
    parser.add_argument("--lambda_cte",  type=float, default=100.0)
    parser.add_argument("--lambda_dist", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--output_dir",  default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    train_idx, val_idx, test_idx = split_indices(args.h5_file)

    train_dataset = DrivingDatasetHDF5(args.h5_file, train_idx, args.granularity,
                                       transform=get_transforms(train=True))
    val_dataset   = DrivingDatasetHDF5(args.h5_file, val_idx,   args.granularity,
                                       transform=get_transforms(train=False))
    test_dataset  = DrivingDatasetHDF5(args.h5_file, test_idx,  args.granularity,
                                       transform=get_transforms(train=False))

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    model = DistanceCTECNN(
        backbone_name=args.backbone,
        granularity=args.granularity,
        pretrained=True,
    ).to(device)

    if hasattr(torch, "compile"):
        print("torch.compile() enabled")
        model = torch.compile(model)

    criterion = MultiTaskLoss(
        num_classes=GRANULARITIES[args.granularity]["num_classes"],
        lambda_cte=args.lambda_cte,
        lambda_dist=args.lambda_dist,
        huber_delta=0.1,
    )

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.shared.parameters(),   "lr": args.lr},
        {"params": model.cte_head.parameters(), "lr": args.lr},
        {"params": model.distance_head.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    save_path = os.path.join(args.output_dir, f"{args.backbone}_{args.granularity}_best.pt")
    print(f"\nTraining {args.backbone} + {args.granularity} granularity\n")

    history = train(
        model, criterion,
        train_loader, val_loader,
        optimizer, scheduler,
        device, args.epochs, save_path,
    )

    print("\n=== Test set evaluation ===")
    checkpoint = torch.load(save_path, map_location=device)
    model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_to_load.load_state_dict(checkpoint["model_state"])

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  loss_total : {test_metrics['loss_total']:.4f}")
    print(f"  loss_cte   : {test_metrics['loss_cte']:.4f}")
    print(f"  loss_dist  : {test_metrics['loss_dist']:.4f}")
    print(f"  cte_mae    : {test_metrics['cte_mae']:.4f}")
    print(f"  dist_acc   : {test_metrics['dist_acc']:.4f}")
    print("  per-class accuracy:")
    for label, acc in test_metrics["dist_per_class_acc"].items():
        print(f"    {label:10s}: {acc:.3f}")


if __name__ == "__main__":
    main()