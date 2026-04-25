import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from distance_cte_cnns_turn import (
    DistanceCTECNN,
    GRANULARITIES,
    corn_label_from_logits,
)

# change predicted bin to a value (midpoint)
# For the "no car" class (last index), returns None
def build_midpoint_table(granularity: str) -> list[float | None]:

    cfg    = GRANULARITIES[granularity]
    edges  = cfg["bin_edges"]          # e.g. [10, 20, 30, 40, 50] for coarse
    table  = []

    prev_edge = 0.0
    for edge in edges:
        midpoint = (prev_edge + edge) / 2.0
        table.append(midpoint)
        prev_edge = edge

    table.append(None)                 # last class = "no car"
    return table

#Convert a predicted class index to a real-valued distance in meters.
def class_to_distance(
    class_idx:   int,
    granularity: str,
    nocar_value: float | None = None,
) -> float | None:

    table = build_midpoint_table(granularity)
    value = table[class_idx]
    if value is None:
        return nocar_value
    return value


def get_test_transform(img_height: int = 112, img_width: int = 224):
    return transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])



def load_model(checkpoint_path: str, device: torch.device) -> tuple[DistanceCTECNN, dict]:

    checkpoint = torch.load(checkpoint_path, map_location=device)

    state = {
        k.replace("_orig_mod.", "", 1): v
        for k, v in checkpoint["model_state"].items()
    }

    # Infer backbone and granularity from the checkpoint filename if not stored
    # e.g. "resnet18_coarse_best.pt" → backbone=resnet18, granularity=coarse
    granularity  = checkpoint.get("granularity")
    backbone     = checkpoint.get("backbone")

    if granularity is None or backbone is None:
        name = Path(checkpoint_path).stem          # e.g. "resnet50_medium_best"
        parts = name.split("_")
        backbone    = backbone    or parts[0]      # "resnet50"
        granularity = granularity or parts[1]      # "medium"
        #print(f"  [inferred from filename] backbone={backbone}, granularity={granularity}")

    model = DistanceCTECNN(
        backbone_name=backbone,
        granularity=granularity,
        pretrained=False,          # weights come from checkpoint, not ImageNet
    ).to(device)

    #model.load_state_dict(checkpoint["model_state"])
    model.load_state_dict(state)
    model.eval()

    #print(f"Loaded checkpoint: {checkpoint_path}")
    #print(f"  backbone={backbone}, granularity={granularity}")
    if "val_metrics" in checkpoint:
        m = checkpoint["val_metrics"]
        """
        print(f"  best val — loss={m['loss_total']:.4f}, "
              f"cte_mae={m['cte_mae']:.4f} rad, dist_acc={m['dist_acc']:.4f}")
        """

    return model, checkpoint



@torch.no_grad()
def predict_single(
    model:       DistanceCTECNN,
    input_img:  Image.Image,
    maneuver:    int,               # 1=straight, 2=left, 3=right
    device:      torch.device,
    transform,
    nocar_value: float | None = None,
) -> dict:

    #image = Image.open(image_path).convert("RGB")
    x     = transform(input_img).unsqueeze(0).to(device)          # (1, 3, H, W)
    m     = torch.tensor([maneuver], dtype=torch.long).to(device)

    cte_pred, dist_logits = model(x, m)

    dist_class = corn_label_from_logits(dist_logits).item()
    dist_label = GRANULARITIES[model.granularity]["labels"][dist_class]
    dist_m     = class_to_distance(dist_class, model.granularity, nocar_value)

    return {
        "cte":            cte_pred.squeeze().item(),
        "distance_class": dist_class,
        "distance_label": dist_label,
        "distance_m":     dist_m,
    }