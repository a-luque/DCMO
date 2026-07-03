"""
Compare aggressive vs smooth driving simulation data from ClearNoon folder.
Directory structure:
  ClearNoon/aggressive/0/, 1/, ..., 300/  -> ego.npz, acc.npz, dist.npz
  ClearNoon/smooth/0/,    1/, ..., 300/  -> ego.npz, acc.npz, dist.npz
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────────────────
AGGRESSIVE_DIR = "/cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/experiments/new_exp/data_gen/data/smooth/ClearNoon/515"
SMOOTH_DIR     = "/cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/experiments/new_exp/data_gen/data/moderate/ClearNoon/515"
NUM_SIMS       = 100          # folders 0 … 300
DT             = 0.1          # seconds per timestep

COLORS = {
    "aggressive": "#E63946",   # vivid red
    "smooth":     "#457B9D",   # steel blue
}

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_simulation(sim_dir, sim_id) -> dict | None:
    """Load ego.npz, acc.npz, dist.npz for one simulation run."""
    folder = os.path.join(sim_dir, str(sim_id))
    try:
        ego_path = os.path.join(folder, "ego.npz")
        
        ego  = np.load(ego_path)
        acc  = np.load(os.path.join(folder, "acc.npz"))
        dist = np.load(os.path.join(folder, "dist.npz"))


        # Try common array-key names; fall back to first key
        def extract(npz):
            return npz['values'].flatten()
        return {
            "speed":    extract(ego),
            "accel":    extract(acc),
            "distance": extract(dist),
        }
    except Exception:
        return None


def load_all(sim_dir, n = 100):
    """Load all simulation runs and return padded arrays."""
    all_speed, all_accel, all_dist = [], [], []

    for i in range(n):
        run = load_simulation(sim_dir, i)
        if run is not None:
            all_speed.append(run["speed"])
            all_accel.append(run["accel"])
            all_dist.append(run["distance"])

    if not all_speed:
        print("all_speed", all_speed)
        raise FileNotFoundError(f"No data found in {sim_dir!r}")

    # Pad/truncate to the median length so we can stack
    lengths   = [len(s) for s in all_speed]
    target_len = int(np.median(lengths))

    def pad(arrays, length):
        out = []
        for a in arrays:
            if len(a) >= length:
                out.append(a[:length])
            else:
                out.append(np.pad(a, (0, length - len(a)), constant_values=np.nan))
        return np.array(out)   # shape (n_runs, length)

    return {
        "speed":    pad(all_speed, target_len),
        "accel":    pad(all_accel, target_len),
        "distance": pad(all_dist,  target_len),
        "n_runs":   len(all_speed),
        "timesteps": target_len,
    }


# ── Statistics ─────────────────────────────────────────────────────────────────

def summary_stats(data_2d: np.ndarray) -> dict:
    """Per-timestep mean ± std over all runs (ignoring NaN)."""
    return {
        "mean": np.nanmean(data_2d, axis=0),
        "std":  np.nanstd(data_2d,  axis=0),
        "p5":   np.nanpercentile(data_2d, 5,  axis=0),
        "p95":  np.nanpercentile(data_2d, 95, axis=0),
    }


def flat_values(data_2d: np.ndarray) -> np.ndarray:
    """All non-NaN values as a 1-D array (for distributions)."""
    return data_2d.flatten()[~np.isnan(data_2d.flatten())]


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_timeseries(ax, time, stats_a, stats_s, ylabel, title):
    """Mean ± shaded band over time."""
    for label, st, c in [("Aggressive", stats_a, COLORS["aggressive"]),
                          ("Smooth",     stats_s, COLORS["smooth"])]:
        ax.plot(time, st["mean"], color=c, lw=2, label=label)
        ax.fill_between(time, st["p5"], st["p95"],
                         color=c, alpha=0.18, label=f"{label} 5–95 pct")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_distribution(ax, vals_a, vals_s, xlabel, title, bins=60):
    """Overlapping histograms + KDE."""
    for label, vals, c in [("Aggressive", vals_a, COLORS["aggressive"]),
                             ("Smooth",     vals_s, COLORS["smooth"])]:
        ax.hist(vals, bins=bins, density=True, alpha=0.35, color=c)
        kde_x = np.linspace(vals.min(), vals.max(), 500)
        kde   = stats.gaussian_kde(vals)
        ax.plot(kde_x, kde(kde_x), color=c, lw=2, label=label)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_boxplot(ax, vals_a, vals_s, ylabel, title):
    """Side-by-side box plots."""
    bp = ax.boxplot(
        [vals_a, vals_s],
        labels=["Aggressive", "Smooth"],
        patch_artist=True,
        medianprops=dict(color="white", lw=2),
        whiskerprops=dict(lw=1.5),
        capprops=dict(lw=1.5),
        flierprops=dict(marker=".", ms=3, alpha=0.3),
    )
    for patch, c in zip(bp["boxes"], [COLORS["aggressive"], COLORS["smooth"]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)


def plot_scatter_density(ax, speed_vals, accel_vals, color, label, alpha=0.15):
    ax.scatter(speed_vals, accel_vals, s=2, alpha=alpha, color=color, label=label)
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.grid(True, alpha=0.3)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading aggressive data …")
    agg  = load_all(AGGRESSIVE_DIR)
    print(f"  Loaded {agg['n_runs']} runs, {agg['timesteps']} timesteps each")

    print("Loading smooth data …")
    smo  = load_all(SMOOTH_DIR)
    print(f"  Loaded {smo['n_runs']} runs, {smo['timesteps']} timesteps each")

    # Shared time axis (use the shorter of the two)
    T    = min(agg["timesteps"], smo["timesteps"])
    time = np.arange(T) * DT

    # Trim to shared length
    for key in ("speed", "accel", "distance"):
        agg[key] = agg[key][:, :T]
        smo[key] = smo[key][:, :T]

    # Compute stats
    stats_agg = {k: summary_stats(agg[k]) for k in ("speed", "accel", "distance")}
    stats_smo = {k: summary_stats(smo[k]) for k in ("speed", "accel", "distance")}

    # Flat distributions
    flat_a = {k: flat_values(agg[k]) for k in ("speed", "accel", "distance")}
    flat_s = {k: flat_values(smo[k]) for k in ("speed", "accel", "distance")}

    # ── Figure 1: Time-series comparison ──────────────────────────────────────
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig1.suptitle("Aggressive vs Smooth Driving — Time-Series Comparison\n"
                  "(shaded band = 5th–95th percentile across all runs)",
                  fontsize=14, fontweight="bold")

    plot_timeseries(axes1[0], time, stats_agg["speed"],    stats_smo["speed"],
                    "Speed (m/s)",      "Ego Speed over Time")
    plot_timeseries(axes1[1], time, stats_agg["accel"],    stats_smo["accel"],
                    "Acceleration (m/s²)", "Ego Acceleration over Time")
    plot_timeseries(axes1[2], time, stats_agg["distance"], stats_smo["distance"],
                    "Distance (m)",     "Distance to Leading Car over Time")

    plt.tight_layout()
    fig1.savefig("fig1_timeseries.png", dpi=150, bbox_inches="tight")
    print("Saved fig1_timeseries.png")

    # ── Figure 2: Distributions ────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle("Aggressive vs Smooth Driving — Value Distributions",
                  fontsize=14, fontweight="bold")

    plot_distribution(axes2[0], flat_a["speed"],    flat_s["speed"],
                      "Speed (m/s)",      "Speed Distribution")
    plot_distribution(axes2[1], flat_a["accel"],    flat_s["accel"],
                      "Acceleration (m/s²)", "Acceleration Distribution")
    plot_distribution(axes2[2], flat_a["distance"], flat_s["distance"],
                      "Distance (m)",     "Headway Distance Distribution")

    plt.tight_layout()
    fig2.savefig("fig2_distributions.png", dpi=150, bbox_inches="tight")
    print("Saved fig2_distributions.png")

    # ── Figure 3: Box plots ────────────────────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 6))
    fig3.suptitle("Aggressive vs Smooth Driving — Box Plots",
                  fontsize=14, fontweight="bold")

    plot_boxplot(axes3[0], flat_a["speed"],    flat_s["speed"],
                 "Speed (m/s)",      "Speed")
    plot_boxplot(axes3[1], flat_a["accel"],    flat_s["accel"],
                 "Acceleration (m/s²)", "Acceleration")
    plot_boxplot(axes3[2], flat_a["distance"], flat_s["distance"],
                 "Distance (m)",     "Headway Distance")

    plt.tight_layout()
    fig3.savefig("fig3_boxplots.png", dpi=150, bbox_inches="tight")
    print("Saved fig3_boxplots.png")

    # ── Figure 4: Speed vs Acceleration scatter ────────────────────────────────
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
    fig4.suptitle("Aggressive vs Smooth — Speed–Acceleration Phase Space",
                  fontsize=14, fontweight="bold")

    # Sample up to 50k points per mode to keep plot readable
    def sample(arr, n=50_000):
        return arr[np.random.choice(len(arr), min(n, len(arr)), replace=False)]

    sa_sp = sample(flat_a["speed"])
    sa_ac = sample(flat_a["accel"])
    ss_sp = sample(flat_s["speed"])
    ss_ac = sample(flat_s["accel"])

    plot_scatter_density(axes4[0], sa_sp, sa_ac, COLORS["aggressive"], "Aggressive")
    axes4[0].set_title("Aggressive", fontweight="bold")
    plot_scatter_density(axes4[1], ss_sp, ss_ac, COLORS["smooth"],     "Smooth")
    axes4[1].set_title("Smooth", fontweight="bold")

    plt.tight_layout()
    fig4.savefig("fig4_phase_space.png", dpi=150, bbox_inches="tight")
    print("Saved fig4_phase_space.png")

    # ── Figure 5: Aggregate summary statistics bar chart ──────────────────────
    metrics = {
        "Mean Speed\n(m/s)":          (np.nanmean(flat_a["speed"]),    np.nanmean(flat_s["speed"])),
        "Std Speed\n(m/s)":           (np.nanstd(flat_a["speed"]),     np.nanstd(flat_s["speed"])),
        "Mean |Accel|\n(m/s²)":       (np.nanmean(np.abs(flat_a["accel"])), np.nanmean(np.abs(flat_s["accel"]))),
        "Std Accel\n(m/s²)":          (np.nanstd(flat_a["accel"]),     np.nanstd(flat_s["accel"])),
        "Mean Distance\n(m)":         (np.nanmean(flat_a["distance"]), np.nanmean(flat_s["distance"])),
        "Std Distance\n(m)":          (np.nanstd(flat_a["distance"]),  np.nanstd(flat_s["distance"])),
    }

    labels  = list(metrics.keys())
    vals_ag = [v[0] for v in metrics.values()]
    vals_sm = [v[1] for v in metrics.values()]
    x       = np.arange(len(labels))
    w       = 0.35

    fig5, ax5 = plt.subplots(figsize=(14, 6))
    ax5.bar(x - w/2, vals_ag, w, label="Aggressive", color=COLORS["aggressive"], alpha=0.85)
    ax5.bar(x + w/2, vals_sm, w, label="Smooth",     color=COLORS["smooth"],     alpha=0.85)
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels, fontsize=10)
    ax5.set_title("Summary Statistics — Aggressive vs Smooth", fontweight="bold", fontsize=14)
    ax5.set_ylabel("Value")
    ax5.legend()
    ax5.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig5.savefig("fig5_summary_stats.png", dpi=150, bbox_inches="tight")
    print("Saved fig5_summary_stats.png")

    # ── Statistical significance tests ────────────────────────────────────────
    print("\n── Mann-Whitney U tests (aggressive vs smooth) ──")
    for key, label in [("speed", "Speed"), ("accel", "Acceleration"), ("distance", "Distance")]:
        # subsample for speed (Mann-Whitney is O(n²))
        n      = min(10_000, len(flat_a[key]), len(flat_s[key]))
        idx_a  = np.random.choice(len(flat_a[key]), n, replace=False)
        idx_s  = np.random.choice(len(flat_s[key]), n, replace=False)
        u, p   = stats.mannwhitneyu(flat_a[key][idx_a], flat_s[key][idx_s],
                                     alternative="two-sided")
        sig    = "✓ significant" if p < 0.05 else "✗ not significant"
        print(f"  {label:12s}: U={u:.0f},  p={p:.2e}  →  {sig}")

    plt.show()
    print("\nDone. All figures saved.")


if __name__ == "__main__":
    main()