"""
Plot efficiency (x-axis) vs comfort/stability (y-axis) for each controller,
sweeping over ONE context dimension while holding the other TWO fixed.

The context is (weather, distance, speed). Give any two of the three via
flags; the script finds every cell matching those two and draws one subplot
per value of the remaining ("free") dimension.

Usage
-----
    # Fix weather + distance, sweep over speed:
    python3 plot_per_two_contexts.py --checkpoint bandit_checkpoint_e_s_45.npz \
        --weather ClearNoon --distance 15.0

    # Fix weather + speed, sweep over distance:
    python3 plot_per_two_contexts.py --checkpoint bandit_checkpoint_e_s_45.npz \
        --weather ClearNoon --speed 8

    # Fix distance + speed, sweep over weather:
    python3 plot_per_two_contexts.py --checkpoint bandit_checkpoint_e_s_45.npz \
        --distance 15.0 --speed 8

    # See valid weather / distance / speed values first:
    python3 plot_per_two_contexts.py --checkpoint bandit_checkpoint_e_s_45.npz --list-values

Notes
-----
- mu_hat column 0 = efficiency (reward_e), column 1 = comfort/stability
  (reward_s), matching training's `return [reward_e, reward_s, safety_info]`.
- ContextSpace here is rebuilt identically to the training script so indices
  line up with what's stored in the checkpoint.
"""

import argparse
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Same ContextSpace construction as in the training script.
# ---------------------------------------------------------------------------
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

    def unique_values(self, dim: str):
        i = self.DIM_NAMES.index(dim)
        vals = sorted(set(c[i] for c in self.cells), key=lambda v: (isinstance(v, str), v))
        return vals


CONTROLLER_NAMES = ['sport', 'aggressive', 'dynamic', 'balanced', 'comfort', 'conservative', 'defensive']
#['aggressive', 'smooth', 'moderate']  # matches controllers_dir in training script
#COLORS = {'aggressive': '#d62728', 'smooth': '#2ca02c', 'moderate': '#1f77b4'}
COLORS = {
    'sport': '#d62728',         # Red
    'aggressive': '#ff7f0e',    # Orange
    'dynamic': '#2ca02c',       # Green
    'balanced': '#1f77b4',      # Blue
    'comfort': '#9467bd',       # Purple
    'conservative': '#8c564b',  # Brown
    'defensive': '#7f7f7f'      # Gray
}

MARKERS = {
    'sport': 'X',
    'aggressive': 'o',
    'dynamic': '^',
    'balanced': 's',
    'comfort': 'D',
    'conservative': 'v',
    'defensive': 'P'
}
#MARKERS = {'aggressive': 'o', 'smooth': 's', 'moderate': '^'}


def load_checkpoint(path: str):
    data = np.load(path, allow_pickle=True)
    counts = data["counts"]   # (n_controllers, n_cells)
    mu_hat = data["mu_hat"]   # (n_controllers, n_cells, K)
    t = int(data["t"]) if "t" in data else None
    return counts, mu_hat, t


def parse_value(dim: str, raw: str):
    """Cast a CLI string to the type used in ContextSpace cells for that dim."""
    if dim == "weather":
        return raw
    if dim == "distance":
        return float(raw)
    if dim == "speed":
        # speeds are ints in ContextSpace, except the stopped-case speed=0
        try:
            return int(raw)
        except ValueError:
            return float(raw)
    raise ValueError(f"Unknown dim {dim!r}")


def list_values(ctx: ContextSpace):
    for dim in ContextSpace.DIM_NAMES:
        print(f"{dim}: {ctx.unique_values(dim)}")


def plot_sweep(ctx: ContextSpace, counts: np.ndarray, mu_hat: np.ndarray,
               fixed: dict, out_path: str):
    """
    fixed: dict with exactly two of {"weather","distance","speed"} set to a value.
    The remaining dim is swept over all its values.
    """
    all_dims = list(ContextSpace.DIM_NAMES)
    free_dims = [d for d in all_dims if d not in fixed]
    if len(free_dims) != 1:
        raise ValueError(f"Exactly two of {all_dims} must be fixed; got fixed={fixed}")
    free_dim = free_dims[0]
    free_values = ctx.unique_values(free_dim)

    # Find, for each free value, the matching cell index (if it exists).
    matches = []  # list of (free_value, context_idx, cell)
    for i, cell in enumerate(ctx.cells):
        cell_d = dict(zip(all_dims, cell))
        if all(cell_d[k] == v for k, v in fixed.items()):
            matches.append((cell_d[free_dim], i, cell))
    matches.sort(key=lambda x: (isinstance(x[0], str), x[0]))

    if not matches:
        raise ValueError(
            f"No context cells match fixed={fixed}. "
            f"Valid values: { {d: ctx.unique_values(d) for d in all_dims} }"
        )

    n = len(matches)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows), squeeze=False)

    fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items())
    fig.suptitle(f"Sweeping {free_dim}  (fixed: {fixed_str})", fontsize=14)

    n_controllers = mu_hat.shape[0]

    for panel_idx, (free_val, ctx_idx, cell) in enumerate(matches):
        r, c = divmod(panel_idx, ncols)
        ax = axes[r][c]

        for ctrl in range(n_controllers):
            name = CONTROLLER_NAMES[ctrl] if ctrl < len(CONTROLLER_NAMES) else f"controller_{ctrl}"
            eff = mu_hat[ctrl, ctx_idx, 0]
            comf = mu_hat[ctrl, ctx_idx, 1]
            n_plays = int(counts[ctrl, ctx_idx])

            ax.scatter(
                eff, comf,
                s=140,
                color=COLORS.get(name, None),
                marker=MARKERS.get(name, 'o'),
                edgecolors='black',
                linewidths=1,
                label=f"{name} (n={n_plays})",
                zorder=3,
            )
            ax.annotate(name, (eff, comf), textcoords="offset points",
                        xytext=(6, 5), fontsize=8)

        ax.set_title(f"{free_dim} = {free_val}")
        ax.set_xlabel("Efficiency (reward_e)")
        ax.set_ylabel("Comfort / Stability (reward_s)")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        if panel_idx == 0:
            ax.legend(loc="best", fontsize=8)

    # Hide unused panels
    for panel_idx in range(n, nrows * ncols):
        r, c = divmod(panel_idx, ncols)
        axes[r][c].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to .npz checkpoint/snapshot file")
    parser.add_argument("--weather", default=None, help="Fix weather, e.g. ClearNoon")
    parser.add_argument("--distance", default=None, help="Fix distance, e.g. 15.0")
    parser.add_argument("--speed", default=None, help="Fix speed, e.g. 8")
    parser.add_argument("--list-values", action="store_true", help="List valid values for each dim and exit")
    parser.add_argument("--out", default=None, help="Output PNG path (default: sweep_<freedim>.png)")
    args = parser.parse_args()

    ctx = ContextSpace()
    counts, mu_hat, t = load_checkpoint(args.checkpoint)

    if t is not None:
        print(f"Checkpoint round: {t}")
    print(f"n_controllers={mu_hat.shape[0]}, n_cells={mu_hat.shape[1]}, K={mu_hat.shape[2]}")

    if args.list_values:
        list_values(ctx)
        return

    raw_fixed = {"weather": args.weather, "distance": args.distance, "speed": args.speed}
    given = {k: v for k, v in raw_fixed.items() if v is not None}
    if len(given) != 2:
        parser.error(
            "Provide exactly TWO of --weather / --distance / --speed "
            f"(got {list(given.keys())}). Use --list-values to see valid values."
        )

    fixed = {k: parse_value(k, v) for k, v in given.items()}
    free_dim = [d for d in ContextSpace.DIM_NAMES if d not in fixed][0]
    out_path = args.out or f"sweep_{free_dim}.png"

    plot_sweep(ctx, counts, mu_hat, fixed, out_path)


if __name__ == "__main__":
    main()