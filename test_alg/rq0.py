import itertools
import os
import time
import numpy as np
from enum import Enum
from scipy.special import expit


class Weather(Enum):
    ClearNoon = [5.0, 0.0, 0.0, 10.0, -1.0, 45.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    CloudyNoon = [60.0, 0.0, 0.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetNoon = [5.0, 0.0, 50.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetCloudyNoon = [60.0, 0.0, 50.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    MidRainyNoon = [60.0, 60.0, 60.0, 60.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    HardRainNoon = [100.0, 100.0, 90.0, 100.0, -1.0, 45.0, 7.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    SoftRainNoon = [20.0, 30.0, 50.0, 30.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    ClearSunset = [5.0, 0.0, 0.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    CloudySunset = [60.0, 0.0, 0.0, 10.0, -1.0, 15.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetSunset = [5.0, 0.0, 50.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetCloudySunset = [60.0, 0.0, 50.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    MidRainSunset = [60.0, 60.0, 60.0, 60.0, -1.0, 15.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    HardRainSunset = [100.0, 100.0, 90.0, 100.0, -1.0, 15.0, 7.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    SoftRainSunset = [20.0, 30.0, 50.0, 30.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]


class ContextSpace:
    """Sparse, irregular discretized context space (weather, distance, speed)."""

    DIM_NAMES = ("weather", "distance", "speed")

    def __init__(self):
        weather = ['ClearNoon', 'HardRainNoon', 'CloudyNoon']
        distance_between = 5.0
        dists = [1, 10, 20, 40]
        dists = [x + distance_between for x in dists]
        speeds = [4, 8, 12]
        self.sampled_ctx = None

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

        self._cell_to_idx = {c: i for i, c in enumerate(self.cells)}

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def total_cells(self) -> int:
        return len(self.cells)

    def index(self, cell: tuple) -> int:
        try:
            return self._cell_to_idx[cell]
        except KeyError:
            raise ValueError(
                f"Context {cell} is not a valid cell. "
                f"Expected (weather, distance, speed). "
                f"Valid cells: {self.cells}"
            )

    def cell(self, idx: int) -> tuple:
        return self.cells[idx]


def get_reward(results_path: str, controller_path: str, timestep: float = 0.1,
               tau: float = 1.0) -> list:
    speed_results = np.load(results_path + "speed.npz")
    acc_results = np.load(results_path + "acc.npz")
    lead_speed_results = np.load(results_path + "leader_speed.npz")

    avg_speed = np.mean(speed_results["values"])
    avg_lead_speed = np.mean(lead_speed_results["values"])
    acc = np.array(acc_results["values"])
    jerks = np.diff(acc) / timestep
    jerk_rms = np.sqrt(np.mean(jerks ** 2))
    reward_s = np.exp(-jerk_rms / tau)

    reward_e = avg_speed / avg_lead_speed
    reward_e = float(np.clip(reward_e, 0.0, 1.0))

    dist_vals = np.load(results_path + "dist.npz")
    cte_vals = np.load(results_path + "cte.npz")

    min_dist = float(dist_vals["values"].min()) - 4.6
    lane_invasion = int((np.absolute(cte_vals["values"]) > 0.7).sum())

    if min_dist < 1.0 or lane_invasion > 30:
        safety_violation = 1
    else:
        safety_violation = 0

    safety_info = {
        "min_dist": min_dist,
        "avg_speed": avg_speed,
        "avg_jerks": jerk_rms,
        "lane_invasion": lane_invasion,
        "safety_violation": safety_violation,
    }

    return [reward_e, reward_s, safety_info]


def simulate(cell, controller_path: str, run_id) -> tuple:
    """Run one simulation for the given (weather, distance, speed) cell and
    controller, returning (reward_vector, safety_info)."""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir = os.path.join(
        current_file_dir, f"fixed_ctx_results/{controller_path}/{run_id}/"
    )
    if os.path.exists(results_dir):
        import shutil
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    sampled_weather, sampled_distance, sampled_speed = cell

    scenic_file_path = os.path.join(current_file_dir, "gen_sim_7.scenic")

    os.system(
        f"scenic -S {scenic_file_path} --count 1 --time 300 --2d "
        f"--param result_path {results_dir} "
        f"--param ego_idm {controller_path} "
        f"--param weather {sampled_weather} "
        f"--param car_dist {sampled_distance} "
        f"--param leader_speed {sampled_speed}"
    )

    reward_e, reward_s, safety_info = get_reward(results_dir, controller_path)
    rewards = np.array([reward_e, reward_s])
    return rewards, safety_info


# ----------------------------------------------------------------------
# Fixed-context empirical evaluation, single controller
# ----------------------------------------------------------------------

def run_fixed_context_eval(
    cell: tuple,
    controller: str,
    n_runs: int = 1000,
    checkpoint_path: str = "fixed_context_eval.npz",
    snapshot_every: int = 200,
    snapshot_dir: str = None,
):
    """
    Run `simulate()` n_runs times for ONE controller at a single fixed
    context `cell`, collecting reward vectors and safety info so you can
    compute empirical (mean/std) rewards for that controller at that
    context.

    Resumable: if `checkpoint_path` already exists, already-completed runs
    are skipped and the run continues from where it left off.

    A permanent snapshot (never overwritten) is written every
    `snapshot_every` completed runs, in addition to the rolling
    `checkpoint_path` file (overwritten after every run so progress is
    never lost to a crash).

    Returns
    -------
    dict: {"rewards": (n_runs, 2), "safety_violation": (n_runs,),
           "min_dist": (n_runs,), "avg_speed": (n_runs,),
           "avg_jerks": (n_runs,), "lane_invasion": (n_runs,)}
    """
    if snapshot_dir is None:
        base = os.path.splitext(checkpoint_path)[0]
        snapshot_dir = base + "_snapshots"

    results = {
        "rewards": np.full((n_runs, 2), np.nan),
        "safety_violation": np.full(n_runs, np.nan),
        "min_dist": np.full(n_runs, np.nan),
        "avg_speed": np.full(n_runs, np.nan),
        "avg_jerks": np.full(n_runs, np.nan),
        "lane_invasion": np.full(n_runs, np.nan),
    }

    start = 0
    if os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path, allow_pickle=True)
        results = data["results"].item()
        done_mask = ~np.isnan(results["rewards"][:, 0])
        start = int(done_mask.sum())
        print(f"[checkpoint] Resuming controller={controller} from run {start}/{n_runs}")

    def _save(path):
        np.savez(path, results=results, cell=np.array(cell, dtype=object),
                  controller=controller)

    for i in range(start, n_runs):
        while True:
            try:
                r, safety_info = simulate(cell, controller, i)
                break
            except Exception as e:
                print(f"Simulation failed for controller={controller}, run={i}: {e}")
                time.sleep(5)

        results["rewards"][i] = r
        results["safety_violation"][i] = safety_info["safety_violation"]
        results["min_dist"][i] = safety_info["min_dist"]
        results["avg_speed"][i] = safety_info["avg_speed"]
        results["avg_jerks"][i] = safety_info["avg_jerks"]
        results["lane_invasion"][i] = safety_info["lane_invasion"]

        done = i + 1
        print(f"[{done}/{n_runs}] controller={controller} run={i} "
              f"reward={r} safety_violation={safety_info['safety_violation']}")

        # Rolling checkpoint: overwritten after every run so a crash never
        # loses more than the in-flight simulation.
        _save(checkpoint_path)

        # Permanent snapshot: new file every snapshot_every runs, never
        # overwritten.
        if done % snapshot_every == 0:
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_path = os.path.join(snapshot_dir, f"snapshot_{controller}_{done}.npz")
            _save(snapshot_path)
            print(f"[snapshot]   Saved permanent snapshot at run {done} -> {snapshot_path}")

    _save(checkpoint_path)
    return results


def summarize(controller: str, results: dict):
    rewards = results["rewards"]
    valid = ~np.isnan(rewards[:, 0])
    n = int(valid.sum())
    print("=" * 60)
    print(f"Controller: {controller}   (#runs: {n})")
    if n == 0:
        print("No completed runs.")
        print("=" * 60)
        return
    mean_r = np.nanmean(rewards[valid], axis=0)
    std_r = np.nanstd(rewards[valid], axis=0)
    viol_pct = 100.0 * np.nanmean(results["safety_violation"][valid])
    print(f"Avg reward_e (R^1): {mean_r[0]:.4f}  (std {std_r[0]:.4f})")
    print(f"Avg reward_s (R^2): {mean_r[1]:.4f}  (std {std_r[1]:.4f})")
    print(f"Safety violation rate: {viol_pct:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run 1000 fixed-context simulations for a single controller."
    )
    parser.add_argument(
        "--controller", type=str, required=True,
        choices=['sport', 'aggressive', 'dynamic', 'balanced',
                 'comfort', 'conservative', 'defensive'],
        help="Which controller to evaluate.",
    )
    parser.add_argument("--n-runs", type=int, default=1000)
    parser.add_argument("--snapshot-every", type=int, default=200)
    parser.add_argument(
        "--checkpoint-path", type=str, default=None,
        help="Defaults to fixed_context_eval_<controller>.npz",
    )
    args = parser.parse_args()

    np.random.seed(42)

    ctx = ContextSpace()

    # ---- fixed context: (weather, distance, speed) ----
    # NOTE: `distance` in a cell = raw distance + distance_between(5.0).
    # For "distance 6" with distance_between=5.0 that's raw distance 1,
    # i.e. cell distance value 6.0 -- matches an existing sampled cell.
    fixed_cell = ("ClearNoon", 6.0, 8)

    # Sanity check: raises ValueError with the full list of valid cells
    # if this tuple isn't one of them.
    ctx.index(fixed_cell)

    checkpoint_path = args.checkpoint_path or f"fixed_context_eval_{args.controller}.npz"

    results = run_fixed_context_eval(
        cell=fixed_cell,
        controller=args.controller,
        n_runs=args.n_runs,
        checkpoint_path=checkpoint_path,
        snapshot_every=args.snapshot_every,
    )

    summarize(args.controller, results)