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

cell_idx = 0   # choose context row

# ===============================
# Load files (sorted by timestep)
# ===============================
files = sorted(
    glob.glob(os.path.join(snapshot_dir, "snapshot_t*.npz")),
    key=lambda x: int(re.search(r"snapshot_t(\d+)", os.path.basename(x)).group(1))
)

# ===============================
# One figure per file
# ===============================

for file in files:
    data = np.load(file, allow_pickle=True)
    mu_hat = data["mu_hat"]   # shape: (n_controllers, n_cells, 2)

    pts = mu_hat[:, cell_idx, :]

    x = pts[:, 0]   # reward0 efficiency
    y = pts[:, 1]   # reward1 stability

    fname = os.path.basename(file).replace(".npz", "")

    plt.figure(figsize=(7,6))
    plt.scatter(x, y, s=100)

    # annotate controller index
    for c in range(len(x)):
        plt.text(x[c], y[c], str(c), fontsize=9)

    plt.xlabel("Reward 0 (Efficiency)")
    plt.ylabel("Reward 1 (Stability)")
    plt.title(f"{fname} | Cell {cell_idx}")
    plt.grid(True)
    plt.tight_layout()

    outfile = os.path.join(save_dir, f"{fname}_cell{cell_idx}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", outfile)