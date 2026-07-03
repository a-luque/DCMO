import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

snapshot_dir_se = "../bandit_checkpoint_s_e_snapshots"
snapshot_dir_es = "../bandit_checkpoint_e_s_snapshots"

save_dir = "plots_se/2"
os.makedirs(save_dir, exist_ok=True)

cell_idx = 0

controller_names = {
    0: "resnet101_coarse",
    1: "resnet101_fine",
    2: "resnet101_medium",
    3: "resnet18_coarse",
    4: "resnet18_fine",
    5: "resnet18_medium",
    6: "resnet50_coarse",
    7: "resnet50_fine",
    8: "resnet50_medium",
}


def pareto_front(points):
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominated = (
            np.all(points >= points[i], axis=1) &
            np.any(points > points[i], axis=1)
        )
        dominated[i] = False
        if np.any(dominated):
            is_pareto[i] = False
    return is_pareto


def extract_t(filename):
    return int(re.search(r"snapshot_t(\d+)", os.path.basename(filename)).group(1))


files_se = {
    extract_t(f): f
    for f in glob.glob(os.path.join(snapshot_dir_se, "snapshot_t*.npz"))
}

files_es = {
    extract_t(f): f
    for f in glob.glob(os.path.join(snapshot_dir_es, "snapshot_t*.npz"))
}

common_ts = sorted(set(files_se.keys()) & set(files_es.keys()))

for t in common_ts:
    data_se = np.load(files_se[t], allow_pickle=True)
    data_es = np.load(files_es[t], allow_pickle=True)

    pts_se = data_se["mu_hat"][:, cell_idx, :]
    pts_es = data_es["mu_hat"][:, cell_idx, :]

    x_se, y_se = pts_se[:, 1], pts_se[:, 0]
    x_es, y_es = pts_es[:, 0], pts_es[:, 1]

    mask_se = pareto_front(pts_se)
    mask_es = pareto_front(pts_es)

    plt.figure(figsize=(7, 6))

    # --- s_e (GREEN) ---
    plt.scatter(x_se[~mask_se], y_se[~mask_se],
                color="lightgreen", s=70, alpha=0.6, label="$S \succ E$ Dominated")

    plt.scatter(x_se[mask_se], y_se[mask_se],
                color="green", s=120, edgecolors="darkgreen",
                linewidths=1.2, label="$S \succ E$ Pareto")

    order = np.argsort(x_se[mask_se])
    plt.plot(x_se[mask_se][order], y_se[mask_se][order],
             color="green", linewidth=1.5)

    # --- e_s (BLUE) ---
    plt.scatter(x_es[~mask_es], y_es[~mask_es],
                color="lightblue", s=70, alpha=0.6, label="$E \succ S$ Dominated")

    plt.scatter(x_es[mask_es], y_es[mask_es],
                color="steelblue", s=120, edgecolors="navy",
                linewidths=1.2, label="$E \succ S$ Pareto")

    order = np.argsort(x_es[mask_es])
    plt.plot(x_es[mask_es][order], y_es[mask_es][order],
             color="steelblue", linewidth=1.5)

    # ===============================
    # Annotations (ONLY Pareto points)
    # ===============================

    # s→e (green) — offset up-right
    for c in range(len(x_se)):
        if not mask_se[c]:
            continue
        label = controller_names.get(c, str(c))
        plt.annotate(
            label,
            (x_se[c], y_se[c]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            color="darkgreen",
            ha="left",
            va="bottom",
            zorder=10,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1)
        )

    # e→s (blue) — offset down-left
    for c in range(len(x_es)):
        if not mask_es[c]:
            continue
        label = controller_names.get(c, str(c))
        plt.annotate(
            label,
            (x_es[c], y_es[c]),
            textcoords="offset points",
            xytext=(-6, -6),
            fontsize=7,
            color="navy",
            ha="right",
            va="top",
            zorder=10,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1)
        )

    plt.xlabel("Reward Efficiency")
    plt.ylabel("Reward Stability")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outfile = os.path.join(save_dir, f"compare_t{t}_cell{cell_idx}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", outfile)