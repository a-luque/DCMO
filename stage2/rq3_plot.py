import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from scipy.spatial import ConvexHull


save_dir = "rq3_plots"
os.makedirs(save_dir, exist_ok=True)

cell_idx = 12

EXPERIMENTS = [
    {
        "snapshot_dir": "../bandit_checkpoint_s_e_snapshots",
        "label": "s_e",
        "base_color": np.array([0.17, 0.63, 0.17]),  # green
    },
    {
        "snapshot_dir": "../bandit_checkpoint_e_s_snapshots",
        "label": "e_s",
        "base_color": np.array([0.13, 0.47, 0.71]),  # blue
    },
]

# ===============================
# Pareto Front Helper
# ===============================
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

def pareto_area(pts):
    """
    Compute the area under the Pareto front using the trapezoidal rule,
    w.r.t. the origin as reference point.
    """
    mask = pareto_front(pts)
    pareto_pts = pts[mask]
    order = np.argsort(pareto_pts[:, 0])
    px = pareto_pts[order, 0]
    py = pareto_pts[order, 1]

    # Trapezoidal area between consecutive Pareto points
    area = np.trapezoid(py, px)

    # Add the rectangle from x=0 to the leftmost point (baseline slab)
    area += px[0] * py[0]

    return area

def load_experiment(exp):
    snapshot_dir = exp["snapshot_dir"]
    files = sorted(
        glob.glob(os.path.join(snapshot_dir, "snapshot_t*.npz")),
        key=lambda x: int(re.search(r"snapshot_t(\d+)", os.path.basename(x)).group(1))
    )
    timesteps = [
        int(re.search(r"snapshot_t(\d+)", os.path.basename(f)).group(1))
        for f in files
    ]
    return files, timesteps


fig, ax = plt.subplots(figsize=(9, 7))

for exp in EXPERIMENTS:
    files, timesteps = load_experiment(exp)
    base_color = exp["base_color"]
    exp_label  = exp["label"]
    alphas = np.linspace(0.25, 1.0, len(files))
    colors = [(*base_color, a) for a in alphas]

    for idx, (file, t, color) in enumerate(zip(files, timesteps, colors)):
        data   = np.load(file, allow_pickle=True)
        mu_hat = data["mu_hat"]
        pts    = mu_hat[:, cell_idx, :]
        if exp_label == "s_e":
            pts = pts[:, ::-1]

        mask       = pareto_front(pts)
        pareto_pts = pts[mask]
        order      = np.argsort(pareto_pts[:, 0])
        px, py     = pareto_pts[order, 0], pareto_pts[order, 1]

        is_last      = (idx == len(files) - 1)
        lw           = 2.5 if is_last else 1.5
        ms           = 100 if is_last else 70
        zorder       = 5   if is_last else 3
        legend_label = (f"{exp_label} t={t}" + (" ★" if is_last else "")) \
                       if (idx == 0 or is_last) else "_nolegend_"

        ax.plot(px, py, color=color, linewidth=lw, zorder=zorder)
        ax.scatter(px, py, s=ms, color=color, zorder=zorder + 1, label=legend_label)

        if is_last:
            controller_indices = np.where(mask)[0][order]
            for ci, xi, yi in zip(controller_indices, px, py):
                ax.annotate(str(ci), (xi, yi),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=8, color=color)

ax.set_xlabel("Reward 0 (Efficiency)", fontsize=12)
ax.set_ylabel("Reward 1 (Stability)", fontsize=12)
ax.set_title(f"Pareto Front Evolution Comparison | Cell {cell_idx}", fontsize=13)
ax.legend(title="Experiment / Timestep", fontsize=9, title_fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"pareto_comparison_cell{cell_idx}.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: pareto_comparison")



# Collect areas per experiment per timestep, averaged over all cells
exp_areas = {}  # label -> {t -> [area per cell]}

for exp in EXPERIMENTS:
    files, timesteps = load_experiment(exp)
    exp_label = exp["label"]
    areas_by_t = {}

    for file, t in zip(files, timesteps):
        data   = np.load(file, allow_pickle=True)
        mu_hat = data["mu_hat"]   # (n_controllers, n_cells, 2)
        n_cells = mu_hat.shape[1]
        cell_areas = []

        for c in range(n_cells):
            pts = mu_hat[:, c, :]
            if exp_label == "s_e":
                pts = pts[:, ::-1]
            cell_areas.append(pareto_area(pts))

        areas_by_t[t] = np.array(cell_areas)

    exp_areas[exp_label] = areas_by_t

# Compute mean difference (e_s - s_e) per timestep
files_ref, timesteps_ref = load_experiment(EXPERIMENTS[0])

# ===============================
# Plot 2: Area Difference (Box-Whisker) + Absolute HV lines
# ===============================
timesteps_all = sorted(exp_areas[EXPERIMENTS[0]["label"]].keys())

diffs_per_t = []
for t in timesteps_all:
    diff = np.abs(exp_areas["e_s"][t] - exp_areas["s_e"][t])
    diffs_per_t.append(diff)

fig, ax1 = plt.subplots(figsize=(9, 6))
ax2 = ax1.twinx()

# --- Box-whisker on ax1 ---
ax1.boxplot(
    diffs_per_t,
    positions=timesteps_all,
    widths=150,
    patch_artist=True,
    boxprops=dict(facecolor="gray", alpha=0.4),
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker="o", markersize=4, linestyle="none", color="gray", alpha=0.5),
)

# --- Absolute HV lines on ax2 ---
for exp in EXPERIMENTS:
    exp_label  = exp["label"]
    base_color = exp["base_color"]
    color      = (*base_color, 1.0)

    means = np.array([exp_areas[exp_label][t].mean() for t in timesteps_all])
    stds  = np.array([exp_areas[exp_label][t].std()  for t in timesteps_all])

    ax2.plot(timesteps_all, means, marker="o", linewidth=2,
             color=color, label=exp_label, zorder=5)
    ax2.fill_between(timesteps_all, means - stds, means + stds,
                     alpha=0.15, color=color)

# --- Formatting ---
ax1.set_xlabel("Number of Simulations", fontsize=12)
ax1.set_ylabel("|e_s - s_e| Pareto Area Difference", fontsize=12)
ax2.set_ylabel("Mean Absolute Hypervolume", fontsize=12)

ax1.set_xticks(timesteps_all)
ax1.grid(True, alpha=0.3, axis="y")

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

plt.title("Pareto Area Difference & Absolute HV Over Time", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "pareto_area_diff.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: pareto_area_diff")


# ===============================
# Print outlier contexts per timestep
# ===============================
print("\n=== Outlier Contexts (Box-Whisker Plot 2) ===")
for t, diffs in zip(timesteps_all, diffs_per_t):
    q1, q3 = np.percentile(diffs, [25, 75])
    iqr     = q3 - q1
    upper   = q3 + 1.5 * iqr
    lower   = q1 - 1.5 * iqr

    outlier_cells = np.where((diffs > upper) | (diffs < lower))[0]

    if len(outlier_cells) == 0:
        print(f"  t={t}: no outliers")
    else:
        print(f"  t={t}: {len(outlier_cells)} outlier(s)")
        for c in outlier_cells:
            area_es = exp_areas["e_s"][t][c]
            area_se = exp_areas["s_e"][t][c]
            print(f"    cell {c:3d} | |diff|={diffs[c]:.4f} | e_s={area_es:.4f} | s_e={area_se:.4f}")