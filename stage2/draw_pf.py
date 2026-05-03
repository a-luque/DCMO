import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

# ===============================
# Paths
# ===============================
snapshot_dir = "../bandit_checkpoint_e_s_snapshots"
save_dir = "plots_es"
os.makedirs(save_dir, exist_ok=True)

cell_idx = 8   # choose context row


files = sorted(
    glob.glob(os.path.join(snapshot_dir, "snapshot_t*.npz")),
    key=lambda x: int(re.search(r"snapshot_t(\d+)", os.path.basename(x)).group(1))
)



for file in files:
    data = np.load(file, allow_pickle=True)
    mu_hat = data["mu_hat"]   # shape: (n_controllers, n_cells, 2)

    pts = mu_hat[:, cell_idx, :]

    x = pts[:, 0]   # x_axix efficiency, 0 for es, 1 for se
    y = pts[:, 1]   # y_axis stability, 1 for es, 0 for se

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