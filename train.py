import os
import glob
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Import our model definitions
from distance_cte_cnns import (
    DistanceCTECNN,
    MultiTaskLoss,
    GRANULARITIES,
    distance_to_label,
    train_step,
    corn_label_from_logits,
)


class DrivingDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        granularity: str,
        transform=None,
        runs: list[int] | None = None,
    ):
        self.granularity = granularity
        self.transform   = transform
        self.samples     = []   # list of (img_path, cte, dist_label)

        root = Path(root_dir)
        run_dirs = sorted(root.glob("[0-9]*"), key=lambda p: int(p.name))

        if runs is not None:
            run_dirs = [root / str(r) for r in runs]

        for run_dir in run_dirs:
            dist_path = run_dir / "dist.npz"
            cte_path  = run_dir / "cte.npz"
            img_dir   = run_dir / "img"

            if not dist_path.exists() or not cte_path.exists():
                print(f"  [skip] {run_dir} — missing dist.npz or cte.npz")
                continue

            # Recorded distance is center-to-center. Subtract 4.6m to get
            # bumper-to-bumper (tail of leading car to head of ego car).
            # Clip to 0 in case of simulation glitches where distance < 4.6m.
            raw = np.load(dist_path)["values"].astype(np.float32)
            distances = np.clip(raw - 4.6, a_min=0.0, a_max=None)

            # Preserve the no-car sentinel: original value >= 100 stays 100.
            # Discard samples where a leading car exists but is beyond 50m —
            # those images are visually ambiguous (car barely visible) and
            # would confuse the model if labelled as "no car".
            no_car_mask  = raw >= 100.0
            far_car_mask = (~no_car_mask) & (distances > 50.0)
            distances[no_car_mask] = 100.0      # restore no-car sentinel

            ctes = np.load(cte_path)["values"].astype(np.float32)

            # Collect sorted image paths for this run
            img_paths = sorted(
                img_dir.glob("front_rgb_*.jpg"),
                key=lambda p: float(p.stem.replace("front_rgb_", ""))
            )

            # Align: use the minimum length across images and labels
            n = min(len(img_paths), len(distances), len(ctes))
            if n == 0:
                print(f"  [skip] {run_dir} — no aligned samples found")
                continue

            for i in range(n):
                if far_car_mask[i]:
                    continue        # discard: car present but beyond 50m
                dist_label = distance_to_label(float(distances[i]), granularity)
                self.samples.append((
                    str(img_paths[i]),
                    float(ctes[i]),
                    dist_label,
                ))

        print(f"Dataset: {len(self.samples)} samples from {len(run_dirs)} runs "
              f"({granularity} granularity)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cte, dist_label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor(cte,        dtype=torch.float32),
            torch.tensor(dist_label, dtype=torch.long),
        )


def get_transforms(train: bool, img_height: int = 112, img_width: int = 224):
    """
    Training: resize + color jitter + normalize.
    Validation: resize + normalize only.

    Default size 224x112 preserves the 2:1 aspect ratio of the 640x320
    camera images (image_size_x=640, image_size_y=320).

    Note: RandomHorizontalFlip is intentionally omitted — flipping a driving
    image would require negating the CTE sign, which is error-prone.
    ColorJitter simulates lighting variation across simulation conditions.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats — good starting point
        std=[0.229, 0.224, 0.225],    # for pretrained ResNet backbones
    )
    if train:
        return transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            normalize,
        ])


def split_runs(
    total_runs: int = 500,
    val_ratio:  float = 0.15,
    test_ratio: float = 0.10,
    seed:       int   = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split run indices into train / val / test at the RUN level (not sample
    level). This avoids data leakage: frames from the same simulation run
    will never appear in both train and val.

    Returns (train_runs, val_runs, test_runs).
    """
    rng  = np.random.default_rng(seed)
    runs = rng.permutation(total_runs).tolist()

    n_test = int(total_runs * test_ratio)
    n_val  = int(total_runs * val_ratio)

    test_runs  = runs[:n_test]
    val_runs   = runs[n_test : n_test + n_val]
    train_runs = runs[n_test + n_val:]

    print(f"Split: {len(train_runs)} train / {len(val_runs)} val / "
          f"{len(test_runs)} test runs")
    return train_runs, val_runs, test_runs


@torch.no_grad()
def evaluate(
    model:     DistanceCTECNN,
    loader:    DataLoader,
    criterion: MultiTaskLoss,
    device:    torch.device,
) -> dict:
    """
    Returns a dict of evaluation metrics:
        loss_total, loss_cte, loss_dist  — average losses
        cte_mae                          — mean absolute error on CTE (meters)
        dist_acc                         — distance class accuracy
        dist_per_class_acc               — per-class accuracy dict
    """
    model.eval()

    total_loss = total_cte_loss = total_dist_loss = 0.0
    all_cte_pred, all_cte_true         = [], []
    all_dist_pred, all_dist_true       = [], []

    for images, cte_true, dist_true in loader:
        images   = images.to(device)
        cte_true = cte_true.to(device)
        dist_true = dist_true.to(device)

        cte_pred, dist_logits = model(images)
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

    # Per-class accuracy
    labels    = GRANULARITIES[model.granularity]["labels"]
    per_class = {}
    for i, label in enumerate(labels):
        mask = dist_true_t == i
        if mask.sum() > 0:
            per_class[label] = (dist_pred_t[mask] == i).float().mean().item()

    return {
        "loss_total":        total_loss      / n,
        "loss_cte":          total_cte_loss  / n,
        "loss_dist":         total_dist_loss / n,
        "cte_mae":           cte_mae,
        "dist_acc":          dist_acc,
        "dist_per_class_acc": per_class,
    }


def train(
    model:       DistanceCTECNN,
    criterion:   MultiTaskLoss,
    train_loader: DataLoader,
    val_loader:  DataLoader,
    optimizer:   torch.optim.Optimizer,
    scheduler:   torch.optim.lr_scheduler._LRScheduler,
    device:      torch.device,
    epochs:      int,
    save_path:   str,
    early_stop_patience: int | None = 5,
):
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_totals = {"loss_total": 0., "loss_cte": 0., "loss_dist": 0.}

        for images, cte_true, dist_true in train_loader:
            images    = images.to(device)
            cte_true  = cte_true.to(device)
            dist_true = dist_true.to(device)

            breakdown = train_step(
                model, images, cte_true, dist_true, criterion, optimizer
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
            print(
                f"  Early stopping: no validation improvement for "
                f"{early_stop_patience} epoch(s)."
            )
            break

    return history



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",    default="simulation_results")
    parser.add_argument("--granularity", default="fine",
                        choices=["coarse", "medium", "fine"])
    parser.add_argument("--backbone",    default="resnet50",
                        choices=["resnet18", "resnet50", "resnet101"])
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--lambda_cte",  type=float, default=1.0)
    parser.add_argument("--lambda_dist", type=float, default=1.0)
    parser.add_argument("--total_runs",  type=int,   default=3000)
    parser.add_argument("--img_height",  type=int,   default=112,
                        help="Resize height — default 112 preserves 2:1 ratio of 640x320 camera")
    parser.add_argument("--img_width",   type=int,   default=224,
                        help="Resize width  — default 224 preserves 2:1 ratio of 640x320 camera")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--output_dir",  default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    train_runs, val_runs, test_runs = split_runs(total_runs=args.total_runs)

    train_dataset = DrivingDataset(
        args.root_dir, args.granularity,
        transform=get_transforms(train=True,  img_height=args.img_height, img_width=args.img_width),
        runs=train_runs,
    )
    val_dataset = DrivingDataset(
        args.root_dir, args.granularity,
        transform=get_transforms(train=False, img_height=args.img_height, img_width=args.img_width),
        runs=val_runs,
    )
    test_dataset = DrivingDataset(
        args.root_dir, args.granularity,
        transform=get_transforms(train=False, img_height=args.img_height, img_width=args.img_width),
        runs=test_runs,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,  batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    #huber_deltas = {"coarse": 10.0, "medium": 5.0, "fine": 2.0}

    cte_huber_delta = 0.1

    model = DistanceCTECNN(
        backbone_name=args.backbone,
        granularity=args.granularity,
        pretrained=True,
    ).to(device)

    criterion = MultiTaskLoss(
        num_classes=GRANULARITIES[args.granularity]["num_classes"],
        lambda_cte=args.lambda_cte,
        lambda_dist=args.lambda_dist,
        huber_delta=cte_huber_delta,
    )

    # Use different learning rates for backbone vs heads (standard practice
    # with pretrained backbones: fine-tune slowly, train head faster)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.shared.parameters(),   "lr": args.lr},
        {"params": model.cte_head.parameters(), "lr": args.lr},
        {"params": model.distance_head.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    save_path = os.path.join(
        args.output_dir, f"{args.backbone}_{args.granularity}_best.pt"
    )
    print(f"\nTraining {args.backbone} + {args.granularity} granularity\n")
    history = train(
        model, criterion,
        train_loader, val_loader,
        optimizer, scheduler,
        device, args.epochs, save_path,
    )

    print("\n=== Test set evaluation ===")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  loss_total : {test_metrics['loss_total']:.4f}")
    print(f"  loss_cte   : {test_metrics['loss_cte']:.4f}")
    print(f"  loss_dist  : {test_metrics['loss_dist']:.4f}")
    print(f"  cte_mae    : {test_metrics['cte_mae']:.4f}")
    print(f"  dist_acc   : {test_metrics['dist_acc']:.4f}")
    print(f"  per-class accuracy:")
    for label, acc in test_metrics["dist_per_class_acc"].items():
        print(f"    {label:10s}: {acc:.3f}")


if __name__ == "__main__":
    main()