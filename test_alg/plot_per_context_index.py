"""
Plot efficiency (x-axis) vs comfort/stability (y-axis) for each controller,
for a single context cell, read from a bandit checkpoint .npz file.

Usage
-----
    python plot_per_context_index.py --checkpoint bandit_checkpoint_e_s_45.npz --context 0
    python plot_per_context_index.py --checkpoint bandit_checkpoint_e_s_45.npz --list-contexts
    python plot_per_context_index.py --checkpoint bandit_checkpoint_e_s_45.npz --context 0 --out my_plot.png

Notes
-----
- The checkpoint stores mu_hat with shape (n_controllers, n_cells, K), where
  K=2 and column 0 = efficiency reward, column 1 = comfort/stability
  (smoothness) reward -- this matches the objective order ("e", "s") used in
  training (see get_reward(): `return [reward_e, reward_s, safety_info]`).
- Context indices come from the SAME ContextSpace construction used during
  training (weather x distance x speed grid, deduplicated). This script
  rebuilds that exact space so index <-> (weather, distance, speed) tuple
  lookups match what was used when the checkpoint was written. If you ever
  change ContextSpace in the training script, update the copy below to match.
"""

# python3 plot_per_context_index.py --checkpoint bandit_checkpoint_e_s_45.npz --context 21

import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from alg_es import ContextSpace


# ---------------------------------------------------------------------------
# Same ContextSpace construction as in the training script, so that context
# indices map to the same (weather, distance, speed) cells.
# ---------------------------------------------------------------------------
"""
class ContextSpace:
    DIM_NAMES = ("weather", "distance", "speed")

    def __init__(self):
        weather = ['ClearNoon', 'HardRainNoon', 'CloudyNoon']
        distance_between = 5.0
        dists = [1, 10, 20, 40]
        dists = [x + distance_between for x in dists]
        speeds = [4, 8, 12]

        raw = (
            list(itertools.product(weather, dists, speeds)) +
            list(itertools.product(weather, [100], [0]))
        )

        seen = set()
        self.cells = []
        for c in raw:
            if c not in seen:
                seen.add(c)
                self.cells.append(c)

    def __len__(self):
        return len(self.cells)

    def cell(self, idx: int) -> tuple:
        return self.cells[idx]
"""

CONTROLLER_NAMES = ['aggressive', 'smooth', 'moderate']  # matches controllers_dir in training script
OBJECTIVE_NAMES = ['efficiency (reward_e)', 'comfort/stability (reward_s)']
COLORS = {'aggressive': '#d62728', 'smooth': '#2ca02c', 'moderate': '#1f77b4'}
MARKERS = {'aggressive': 'o', 'smooth': 's', 'moderate': '^'}


def load_checkpoint(path: str):
    data = np.load(path, allow_pickle=True)
    counts = data["counts"]   # (n_controllers, n_cells)
    mu_hat = data["mu_hat"]   # (n_controllers, n_cells, K)
    t = int(data["t"]) if "t" in data else None
    return counts, mu_hat, t


def list_contexts(ctx: ContextSpace, counts: np.ndarray):
    print(f"{'idx':>4}  {'weather':<14} {'distance':>9} {'speed':>6}   total plays")
    print("-" * 55)
    for i, cell in enumerate(ctx.cells):
        weather, distance, speed = cell
        total_plays = int(counts[:, i].sum())
        print(f"{i:>4}  {weather:<14} {distance:>9} {speed:>6}   {total_plays}")


def plot_context(ctx: ContextSpace, counts: np.ndarray, mu_hat: np.ndarray,
                  context_idx: int, out_path: str):
    n_controllers, n_cells, K = mu_hat.shape
    if K != 2:
        raise ValueError(
            f"Expected K=2 objectives (efficiency, comfort/stability), got K={K}."
        )
    if not (0 <= context_idx < n_cells):
        raise ValueError(f"context index {context_idx} out of range [0, {n_cells - 1}]")

    cell = ctx.cell(context_idx)
    weather, distance, speed = cell

    fig, ax = plt.subplots(figsize=(6.5, 6))

    for c in range(n_controllers):
        name = CONTROLLER_NAMES[c] if c < len(CONTROLLER_NAMES) else f"controller_{c}"
        eff = mu_hat[c, context_idx, 0]
        comf = mu_hat[c, context_idx, 1]
        n_plays = int(counts[c, context_idx])

        ax.scatter(
            eff, comf,
            s=160,
            color=COLORS.get(name, None),
            marker=MARKERS.get(name, 'o'),
            edgecolors='black',
            linewidths=1,
            label=f"{name} (n={n_plays})",
            zorder=3,
        )
        ax.annotate(
            name,
            (eff, comf),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    ax.set_xlabel("Efficiency (reward_e)")
    ax.set_ylabel("Comfort / Stability (reward_s)")
    ax.set_title(
        f"Context #{context_idx}: weather={weather}, distance={distance}, speed={speed}"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to .npz checkpoint/snapshot file")
    parser.add_argument("--context", type=int, default=None, help="Context index to plot")
    parser.add_argument("--list-contexts", action="store_true", help="List all context indices and exit")
    parser.add_argument("--out", default=None, help="Output PNG path (default: context_<idx>.png)")
    args = parser.parse_args()

    ctx = ContextSpace()
    counts, mu_hat, t = load_checkpoint(args.checkpoint)

    if t is not None:
        print(f"Checkpoint round: {t}")
    print(f"n_controllers={mu_hat.shape[0]}, n_cells={mu_hat.shape[1]}, K={mu_hat.shape[2]}")

    if args.list_contexts:
        list_contexts(ctx, counts)
        return

    if args.context is None:
        parser.error("--context is required unless --list-contexts is used")

    out_path = args.out or f"context_{args.context}.png"
    plot_context(ctx, counts, mu_hat, args.context, out_path)


if __name__ == "__main__":
    main()