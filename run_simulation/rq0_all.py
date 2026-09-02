import itertools
import os
import time
import numpy as np
from enum import Enum
from scipy.special import expit
from alg_es import Weather, ContextSpace, get_reward
import shutil

# ----------------------------------------------------------------------
# Fixed-context empirical evaluation, single controller
# ----------------------------------------------------------------------

def run_fixed_context_eval(
    cell: tuple,
    controller: str,
    save_path: str,
    n_runs: int = 500,
    checkpoint_path: str = "fixed_context_rq0.npz",
    snapshot_every: int = 100,
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
        "near_collision": np.full(n_runs, np.nan),
        "min_dist": np.full(n_runs, np.nan),
        "avg_speed": np.full(n_runs, np.nan),
        "avg_jerks": np.full(n_runs, np.nan),
        "lane_invasion": np.full(n_runs, np.nan),
        "lane_invasion_count": np.full(n_runs, np.nan),
        "collision_happened": np.full(n_runs, np.nan),
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
            #sim_seed = np.random.randint(1, 2**32 - 1) # we don't use sim_seed here.
            try:
                r, safety_info = simulate(cell, controller, save_path, order="es", t=i)
                break
            except Exception as e:
                print(f"Simulation failed for controller={controller}, run={i}: {e}")
                time.sleep(5)

        results["rewards"][i] = r
        results["safety_violation"][i] = safety_info["safety_violation"]
        results["lane_invasion"][i] = safety_info["lane_invasion"]
        results["lane_invasion_count"][i] = safety_info["lane_invasion_count"]
        results["collision_happened"][i] = safety_info["collision_happened"]
        results["near_collision"][i] = safety_info["near_collision"]
        results["min_dist"][i] = safety_info["min_dist"]
        results["avg_speed"][i] = safety_info["avg_speed"]
        results["avg_jerks"][i] = safety_info["avg_jerks"]

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


def run_all_controllers(
    cell: tuple,
    controllers: list,
    base_save_path: str,
    n_runs: int = 500,
    checkpoint_dir: str = ".",
    snapshot_every: int = 100,
    cell_str: str = None,
):
    """
    Run `run_fixed_context_eval` sequentially for every controller in
    `controllers`, at the same fixed context `cell`.

    Each controller gets:
      - its own save_path:       {base_save_path}/{controller}
      - its own checkpoint file: {checkpoint_dir}/rq0_{cell_str}_{controller}.npz

    so runs for different controllers never clobber each other and each
    controller is independently resumable.

    Returns
    -------
    dict: {controller_name: results_dict} for every controller run.
    """
    if cell_str is None:
        cell_str = f"{int(cell[1])}_{int(cell[2])}"

    all_results = {}
    for controller in controllers:
        save_path = os.path.join(base_save_path, controller)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"rq0_{cell_str}_{controller}.npz"
        )

        # Cheap pre-check: if this controller's checkpoint already shows
        # n_runs completed runs, skip it entirely instead of calling
        # run_fixed_context_eval (which would still load/resave the npz).
        if os.path.exists(checkpoint_path):
            data = np.load(checkpoint_path, allow_pickle=True)
            existing_results = data["results"].item()
            done_mask = ~np.isnan(existing_results["rewards"][:, 0])
            done_count = int(done_mask.sum())
            if done_count >= n_runs:
                print(f"\n=== Controller already complete, skipping: "
                      f"{controller} ({done_count}/{n_runs}) ===")
                all_results[controller] = existing_results
                continue
            else:
                print(f"\n=== Running controller: {controller} "
                      f"(resuming from {done_count}/{n_runs}) ===")
        else:
            print(f"\n=== Running controller: {controller} ===")

        results = run_fixed_context_eval(
            cell=cell,
            controller=controller,
            save_path=save_path,
            n_runs=n_runs,
            checkpoint_path=checkpoint_path,
            snapshot_every=snapshot_every,
        )
        all_results[controller] = results

    return all_results


def simulate(cell, controller_path: str, save_path: str, order: str, t: int) -> np.ndarray:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    #results_dir = f"sim_results/{t}/"
    results_dir = os.path.join(current_file_dir, f"{save_path}/{t}/")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    # sampled_weather, sampled_intersect, sampled_distance, sampled_speed = cell
    sampled_weather, sampled_distance, sampled_speed = cell

    scenic_file_path = os.path.join(current_file_dir, "new_sim.scenic")
    """
    cmd = [
        "scenic",
        "-S", scenic_file_path,
        "--count", "1",
        "--time", "200",
        "--2d",
        "--param", "results_path", results_dir,
        "--param", "controller_path", controller_path,
        "--param", "weather", str(sampled_weather),
        "--param", "dist_car", str(sampled_distance),
        "--param", "speed_car", str(sampled_speed),
    ]
    subprocess.run(cmd, check=True)
    """

    #print(f"Simulated round {t} with controller {controller_path} at context {sampled_weather} {-1 * sampled_distance} {sampled_speed}. Results in {results_dir}")

    os.system(
        f"scenic -S {scenic_file_path} --count 1 --time 300 --2d "
        f"--param result_path {results_dir} "
        #f"--param controller_path {controller_path} "
        f"--param ego_idm {controller_path} "
        f"--param weather {sampled_weather} "
        #f"--param intersect {sampled_intersect} "
        f"--param car_dist {sampled_distance} "
        f"--param leader_speed {sampled_speed}"
    )

    #rewards = get_reward(results_dir, controller_path)
    [reward_e, reward_s, safety_info] = get_reward(results_dir, order)
    rewards = [reward_e, reward_s]
    return np.array(rewards), safety_info


if __name__ == "__main__":
    import argparse

    CONTROLLERS = ['sport', 'aggressive', 'dynamic', 'balanced',
                   'comfort', 'conservative', 'defensive']

    parser = argparse.ArgumentParser(
        description="Run 500 fixed-context simulations for each controller, "
                     "in sequence.",
    )
    parser.add_argument("--n-runs", type=int, default=500)
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument(
        "--checkpoint-dir", type=str, default=".",
        help="Directory to store per-controller checkpoint files "
             "(rq0_<cell>_<controller>.npz).",
    )
    args = parser.parse_args()

    np.random.seed(42)

    ctx = ContextSpace()

    # ---- fixed context: (weather, distance, speed) ----

    #fixed_cell = ("ClearNoon", 6.0, 8)

    fixed_cell = ("ClearNoon", 15.0, 12)
    #fixed_cell = ("ClearNoon", 15.0, 4)

    # Sanity check: raises ValueError with the full list of valid cells
    # if this tuple isn't one of them.
    ctx.index(fixed_cell)

    cell_str = f"{int(fixed_cell[1])}_{int(fixed_cell[2])}"
    base_save_path = f"rq0_results/{cell_str}"

    all_results = run_all_controllers(
        cell=fixed_cell,
        controllers=CONTROLLERS,
        base_save_path=base_save_path,
        n_runs=args.n_runs,
        checkpoint_dir=args.checkpoint_dir,
        snapshot_every=args.snapshot_every,
        cell_str=cell_str,
    )