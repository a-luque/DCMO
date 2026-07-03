import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

# ===============================
# Paths
# ===============================
snapshot_dir = "../bandit_checkpoint_s_e_snapshots"
save_dir = "plots_se"
os.makedirs(save_dir, exist_ok=True)

cell_idx = 22

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


files = sorted(
    glob.glob(os.path.join(snapshot_dir, "snapshot_t*.npz")),
    key=lambda x: int(re.search(r"snapshot_t(\d+)", os.path.basename(x)).group(1))
)

for file in files:
    data = np.load(file, allow_pickle=True)
    mu_hat = data["mu_hat"]   # shape: (n_controllers, n_cells, 2)

    pts = mu_hat[:, cell_idx, :]
    if save_dir == "plots_es":
        x = pts[:, 0]
        y = pts[:, 1]
    else:
        x = pts[:, 1]
        y = pts[:, 0]

    mask = pareto_front(pts)

    fname = os.path.basename(file).replace(".npz", "")

    plt.figure(figsize=(7, 6))

    # Non-Pareto points
    plt.scatter(x[~mask], y[~mask], s=80, color="lightblue", alpha=0.7, label="Dominated")

    # Pareto points
    plt.scatter(x[mask], y[mask], s=150, color="steelblue", zorder=5,
                edgecolors="navy", linewidths=1.2, label="Pareto Front")

    # Connect Pareto front with a line
    order = np.argsort(x[mask])
    px, py = x[mask][order], y[mask][order]
    plt.plot(px, py, color="steelblue", linewidth=1.5, zorder=4)

    # Annotate all points with controller index
    """
    for c in range(len(x)):
        color = "navy" if mask[c] else "steelblue"
        plt.text(x[c], y[c], str(c), fontsize=9, color=color,
                 ha="left", va="bottom", zorder=10)
    """
    for c in range(len(x)):
        color = "navy" if mask[c] else "steelblue"
        label = controller_names.get(c, str(c))  # fallback to index if missing

        plt.annotate(
            label,
            (x[c], y[c]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,   # slightly smaller since names are longer
            color=color,
            ha="left",
            va="bottom",
            zorder=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1)
        )
    plt.xlabel("Reward Efficiency")
    plt.ylabel("Reward Stability")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outfile = os.path.join(save_dir, f"{fname}_cell{cell_idx}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", outfile)