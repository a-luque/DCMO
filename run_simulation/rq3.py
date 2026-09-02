import argparse
import glob
import os
import sys
import numpy as np
import time
from alg_es import Weather, ContextSpace, get_reward


# All (distance, speed) context cells we have per-controller ensemble
# results for (weather is held fixed, e.g. "ClearNoon").
CONTEXT_CELLS = [
    "6_4", "6_8", "6_12",
    "15_4", "15_8", "15_12",
    "25_4", "25_8", "25_12",
    "35_4", "35_8", "35_12",
]

# code for our appraoch simulation
CONTROLLER_NAMES = ['sport', 'aggressive', 'dynamic', 'balanced', 'comfort', 'conservative', 'defensive']


def load_controller_results(path: str):
    """
    Load one controller's fixed_context_eval .npz file and return
    (avg_efficiency, avg_comfort).
    """
    data = np.load(path, allow_pickle=True)
    results = data["results"].item()
    rewards = results["rewards"]  # (n_runs, 2) -> [reward_e, reward_s]
    #valid = ~np.isnan(rewards[:, 0])
    valid = ~np.isnan(rewards).any(axis=1)
    if not valid.all():
        raise ValueError(f"Uncompleted runs found in {path}")

    avg_eff = float(np.mean(rewards[valid, 0]))
    avg_comf = float(np.mean(rewards[valid, 1]))

    return avg_eff, avg_comf


def get_controller_weights(
    result_dir: str,
    alphas,
    prefix: str = "rq0",
    contexts=None,
    controller_names=None,
):
    """
    For each controller, load its results for every context cell (files
    named "{prefix}_{context}_{controller}.npz", e.g. "rq0_6_4_aggressive.npz"),
    and average (avg_efficiency, avg_comfort) across all of those contexts.

    Scalarizing:
        score = alphas[0] * mean_efficiency
              + alphas[1] * mean_comfort

    Scores are normalized so they sum to 1 and returned as
    {controller_name: normalized_weight}.
    """
    if contexts is None:
        contexts = CONTEXT_CELLS
    if controller_names is None:
        controller_names = CONTROLLER_NAMES

    names, scalar_values = [], []
    for controller in controller_names:
        eff_vals, comf_vals = [], []
        for context in contexts:
            path = os.path.join(result_dir, f"{prefix}_{context}_{controller}.npz")
            if not os.path.exists(path):
                # Not every controller has a results file for every context
                # (e.g. some controllers were only evaluated on a subset of
                # cells) -- just skip it and average over whatever exists.
                print(f"[weights] skipping missing file for controller "
                      f"'{controller}' in context '{context}': {path}")
                continue
            avg_eff, avg_comf = load_controller_results(path)
            eff_vals.append(avg_eff)
            comf_vals.append(avg_comf)

        if (not eff_vals) or (not comf_vals):
            print(f"[weights] no results found for controller '{controller}' "
                  f"in any context, skipping it entirely")
            continue

        mean_eff = float(np.mean(eff_vals))
        mean_comf = float(np.mean(comf_vals))

        names.append(controller)
        scalar_values.append(
            alphas[0] * mean_eff  + alphas[1] * mean_comf
        )

    total = np.sum(scalar_values)
    if np.isclose(total, 0.0):
        raise ValueError("Sum of scalar values is ~0; cannot normalize to sum-to-one.")

    normalized = np.asarray(scalar_values, dtype=float) / total
    return {name: round(float(w), 3) for name, w in zip(names, normalized)}


def names_from_history(history, n_controllers):
    """
    Reconstruct {index -> controller name} from the checkpoint's own history
    log, where each played round recorded both c_t (index) and
    controller_path (name). Skipped rounds (no c_t) are ignored.
 
    Returns a list of length n_controllers, or None if any index never got
    played (so its name can't be recovered this way).
    """
    controller_map = {}
    for entry in history:
        c_idx = entry.get("c_t")
        c_name = entry.get("controller_path")
        if c_idx is None or c_name is None:
            continue
        controller_map.setdefault(c_idx, c_name)
 
    if len(controller_map) < n_controllers:
        return None  # some controller index was never played, can't recover its name
 
    return [controller_map[i] for i in range(n_controllers)]


def load_checkpoint(path: str):
    data = np.load(path, allow_pickle=True)
    mu_hat = data["mu_hat"]  # (n_controllers, n_cells, K)
    n_controllers = mu_hat.shape[0]
 
    # 2) Otherwise, reconstruct from the played-rounds history log.
    if "history" in data:
        names = names_from_history(data["history"], n_controllers)
    else:
        names = CONTROLLER_NAMES
 
    return mu_hat, names


def pareto_front(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Indices maximizing both x (efficiency) and y (stability)."""
    n = len(xs)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if (xs[j] >= xs[i] and ys[j] >= ys[i]) and (xs[j] > xs[i] or ys[j] > ys[i]):
                is_pareto[i] = False
                break
    return is_pareto


def get_pareto_controllers(mu_hat: np.ndarray, ctx_idx: int, names=None):
    """Returns list of (name, (efficiency, stability)) for controllers on the Pareto front."""
    n_controllers = mu_hat.shape[0]
    xs = mu_hat[:, ctx_idx, 0]  # efficiency
    ys = mu_hat[:, ctx_idx, 1]  # stability
    is_pareto = pareto_front(xs, ys)
 
    if names is None:
        names = [CONTROLLER_NAMES[i] if i < len(CONTROLLER_NAMES) else f"controller_{i}"
                 for i in range(n_controllers)]
 
    return [(names[i], (float(xs[i]), float(ys[i])))
            for i in range(n_controllers) if is_pareto[i]]


def select_best(pareto_points, weights):
    """Pick the controller maximizing weights[0]*efficiency + weights[1]*stability."""
    best_name, best_score, best_vec = None, -np.inf, None
    for name, (eff, stab) in pareto_points:
        score = weights[0] * eff + weights[1] * stab
        if score > best_score:
            best_name, best_score, best_vec = name, score, (eff, stab)
    return best_name, best_score, best_vec

"""
### run simulations
def get_reward(results_path: str, timestep: float = 0.1,
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
"""

def simulate(cell, weights, save_path, run_id, seed) -> tuple:
    """Run one simulation for the given (weather, distance, speed) cell and
    controller, returning (reward_vector, safety_info)."""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir = os.path.join(
        current_file_dir, f"{save_path}/{run_id}/"
    )
    if os.path.exists(results_dir):
        import shutil
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    sampled_weather, sampled_distance, sampled_speed = cell

    scenic_file_path = os.path.join(current_file_dir, "rq3_sim.scenic")

    os.system(
        f"scenic -S {scenic_file_path} --seed {seed} --count 1 --time 300 --2d "
        f"--param result_path {results_dir} "
        f"--param aggressive {weights['aggressive']} "
        f"--param balanced {weights['balanced']} "
        f"--param comfort {weights['comfort']} "
        f"--param conservative {weights['conservative']} "
        f"--param defensive {weights['defensive']} "
        f"--param dynamic {weights['dynamic']} "
        f"--param sport {weights['sport']} "
        f"--param weather {sampled_weather} "
        f"--param car_dist {sampled_distance} "
        f"--param leader_speed {sampled_speed}"
    )

    reward_e, reward_s, safety_info = get_reward(results_dir, order="es")
    rewards = np.array([reward_e, reward_s])
    return rewards, safety_info


def simulate_alg(cell, controller_path: str, save_path, run_id, seed) -> tuple:
    """Run one simulation for the given (weather, distance, speed) cell and
    controller, returning (reward_vector, safety_info)."""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir = os.path.join(
        current_file_dir, f"{save_path}/{run_id}/"
    )

    if os.path.exists(results_dir):
        import shutil
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    sampled_weather, sampled_distance, sampled_speed = cell

    scenic_file_path = os.path.join(current_file_dir, "new_sim.scenic")

    os.system(
        f"scenic -S {scenic_file_path} --seed {seed} --count 1 --time 300 --2d "
        f"--param result_path {results_dir} "
        f"--param ego_idm {controller_path} "
        f"--param weather {sampled_weather} "
        f"--param car_dist {sampled_distance} "
        f"--param leader_speed {sampled_speed}"
    )

    reward_e, reward_s, safety_info = get_reward(results_dir, order="es")
    rewards = np.array([reward_e, reward_s])
    return rewards, safety_info


def run_random_context_eval(
    ctx: ContextSpace,
    weights: dict,
    alphas,
    mu_hat: np.ndarray,
    names,
    n_runs: int = 2000,
    checkpoint_path: str = "random_context_eval.npz",
    alg_checkpoint_path: str = "random_context_eval.npz",
    file_address: str = "random_ctx",
    file_address_alg: str = "random_ctx_alg",
    snapshot_every: int = 200,
    snapshot_dir: str = None,
):
    """
    Same as the old fixed-context evaluation, except the context cell is no
    longer fixed: on every run we draw a fresh cell from `ctx` via
    ctx.sample_one_context(), and the alg-side "best" controller is
    re-selected (Pareto front + select_best) for that specific cell before
    simulating it. The scalarization/weighting logic itself
    (get_controller_weights, select_best) is unchanged.

    Returns
    -------
    dict: {"rewards": (n_runs, 2), "safety_violation": (n_runs,),
           "min_dist": (n_runs,), "avg_speed": (n_runs,),
           "avg_jerks": (n_runs,), "lane_invasion": (n_runs,),
           "cells": (n_runs,) object array of the sampled (weather, distance, speed)
           tuples, "best_controller": (n_runs,) object array of the alg-selected
           controller name for that run's cell}
    """
    if snapshot_dir is None:
        base = os.path.splitext(checkpoint_path)[0]
        snapshot_dir = base + "_snapshots"
        
        alg_base = os.path.splitext(alg_checkpoint_path)[0]
        alg_snapshot_dir = alg_base + "_snapshots"

    def _fresh_results():
        return {
            "rewards": np.full((n_runs, 2), np.nan),
            "safety_violation": np.full(n_runs, np.nan),
            "near_collision": np.full(n_runs, np.nan),
            "lane_invasion": np.full(n_runs, np.nan),
            "lane_invasion_count": np.full(n_runs, np.nan),
            "collision_happened": np.full(n_runs, np.nan),
            "min_dist": np.full(n_runs, np.nan),
            "avg_speed": np.full(n_runs, np.nan),
            "avg_jerks": np.full(n_runs, np.nan),
            "cells": np.full(n_runs, None, dtype=object),
        }

    results = _fresh_results()
    results_alg = _fresh_results()
    results_alg["best_controller"] = np.full(n_runs, None, dtype=object)

    def _save(path):
        np.savez(path, results=results, info=checkpoint_path)

    def _save_alg(path):
        np.savez(path, results=results_alg, info=alg_checkpoint_path)

    start = 0
    if os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path, allow_pickle=True)
        results = data["results"].item()
        done_mask = ~np.isnan(results["rewards"][:, 0])
        start = int(done_mask.sum())
        print(f"[checkpoint] Resuming baseline from run {start}/{n_runs}")

    alg_start = 0
    if os.path.exists(alg_checkpoint_path):
        alg_data = np.load(alg_checkpoint_path, allow_pickle=True)
        results_alg = alg_data["results"].item()
        alg_done_mask = ~np.isnan(results_alg["rewards"][:, 0])
        alg_start = int(alg_done_mask.sum())
        print(f"[checkpoint] Resuming alg run from run {alg_start}/{n_runs}")

    if start != alg_start:
        print(f"[checkpoint] WARNING: baseline ({start}) and alg ({alg_start}) "
              f"checkpoints are out of sync; resuming both from {min(start, alg_start)}")
        start = min(start, alg_start)

    sim_seed = 0

    for i in range(start, n_runs):
        # Randomly sample a fresh context cell for this run instead of
        # reusing a fixed cell.
        cell = ctx.sample_one_context()
        ctx_idx = ctx.index(cell)

        # Re-derive the alg's "best" controller for THIS cell (weight
        # scalarization itself is unchanged -- only the cell it's applied to
        # varies now).
        pareto_points = get_pareto_controllers(mu_hat, ctx_idx, names)
        best_name, _, _ = select_best(pareto_points, alphas)

        while True:
            sim_seed = np.random.randint(1, 2**32)
            try:
                r, safety_info = simulate(cell, weights, file_address, i, sim_seed)
                break
            except Exception as e:
                print(f"Simulation failed for {checkpoint_path}, run={i}, cell={cell}: {e}")
                time.sleep(5)

        while True:
            try:
                r_alg, safety_info_alg = simulate_alg(cell, best_name, file_address_alg, i, sim_seed)
                break
            except Exception as e:
                print(f"Simulation failed for {alg_checkpoint_path}, run={i}, cell={cell}: {e}")
                time.sleep(5)

        results["rewards"][i] = r
        results["safety_violation"][i] = safety_info["safety_violation"]
        results["min_dist"][i] = safety_info["min_dist"]
        results["avg_speed"][i] = safety_info["avg_speed"]
        results["avg_jerks"][i] = safety_info["avg_jerks"]
        results["lane_invasion"][i] = safety_info["lane_invasion"]
        results["lane_invasion_count"][i] = safety_info["lane_invasion_count"]
        results["collision_happened"][i] = safety_info["collision_happened"]
        results["near_collision"][i] = safety_info["near_collision"]
        results["cells"][i] = cell

        results_alg["rewards"][i] = r_alg
        results_alg["safety_violation"][i] = safety_info_alg["safety_violation"]
        results_alg["min_dist"][i] = safety_info_alg["min_dist"]
        results_alg["avg_speed"][i] = safety_info_alg["avg_speed"]
        results_alg["avg_jerks"][i] = safety_info_alg["avg_jerks"]
        results_alg["lane_invasion"][i] = safety_info_alg["lane_invasion"]
        results_alg["lane_invasion_count"][i] = safety_info_alg["lane_invasion_count"]
        results_alg["collision_happened"][i] = safety_info_alg["collision_happened"]
        results_alg["near_collision"][i] = safety_info_alg["near_collision"]
        results_alg["cells"][i] = cell
        results_alg["best_controller"][i] = best_name

        done = i + 1
        print(f"[{done}/{n_runs}] info={checkpoint_path} run={i} cell={cell} "
              f"reward={r} safety_violation={safety_info['safety_violation']}")
        print(f"[{done}/{n_runs}] info={alg_checkpoint_path} run={i} cell={cell} "
              f"controller={best_name} reward={r_alg} "
              f"safety_violation={safety_info_alg['safety_violation']}")

        # Rolling checkpoint: overwritten after every run so a crash never
        # loses more than the in-flight simulation.
        _save(checkpoint_path)
        _save_alg(alg_checkpoint_path)

        if done % snapshot_every == 0:
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_path = os.path.join(snapshot_dir, f"snapshot_{done}.npz")
            _save(snapshot_path)
            print(f"[snapshot]   Saved permanent baseline snapshot at run {done} -> {snapshot_path}")

            os.makedirs(alg_snapshot_dir, exist_ok=True)
            alg_snapshot_path = os.path.join(alg_snapshot_dir, f"snapshot_{done}.npz")
            _save_alg(alg_snapshot_path)
            print(f"[snapshot]   Saved permanent alg snapshot at run {done} -> {alg_snapshot_path}")

    _save(checkpoint_path)
    _save_alg(alg_checkpoint_path)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run randomly-sampled-context simulations for ensemble controllers."
    )
    parser.add_argument("--n-runs", type=int, default=2000)
    parser.add_argument("--snapshot-every", type=int, default=100)

    # Scalarization weights (alphas) used to combine efficiency/comfort
    # into a single score when computing ensemble controller weights.
    parser.add_argument("--alpha-eff", type=float, default=0.5,
                         help="Weight on (bias-adjusted) efficiency in the scalarized score.")
    parser.add_argument("--alpha-comf", type=float, default=0.5,
                         help="Weight on (bias-adjusted) comfort in the scalarized score.")

    args = parser.parse_args()

    np.random.seed(42)

    alphas = [args.alpha_eff, args.alpha_comf]
    alpha_str = "_".join(f"{int(round(w * 10))}" for w in alphas)

    # Directory holding the per-controller, per-context ensemble reward
    # files, e.g. rq0_6_4_aggressive.npz, rq0_6_4_comfort.npz, ...
    base_dir = "/proj/berzelius-2026-227/users/x_menwa/CMO_new/new_exp/run_simulations"
    npz_dir = os.path.join(base_dir, "rq3_ensemble_rewards")

    weights = get_controller_weights(npz_dir, alphas)

    alg_npz_path = "/proj/berzelius-2026-227/users/x_menwa/CMO_new/new_exp/main_alg_sim_es.npz"
    ctx = ContextSpace()
    mu_hat, names = load_checkpoint(alg_npz_path)
    if names is None:
        print("(could not recover controller names from checkpoint, using hardcoded fallback list)")


    file_address = f"rq3_results/rq3_w_{alpha_str}"
    file_address_alg = f"rq3_results/rq3_w_{alpha_str}_alg"

    checkpoint_path = os.path.join(".", f"rq3_{alpha_str}_ensemble.npz")
    
    
    alg_checkpoint_path = os.path.join(".", f"rq3_{alpha_str}_alg.npz")

    results = run_random_context_eval(
        ctx=ctx,
        weights=weights,
        alphas=alphas,
        mu_hat=mu_hat,
        names=names,
        n_runs=args.n_runs,
        checkpoint_path=checkpoint_path,
        alg_checkpoint_path=alg_checkpoint_path,
        file_address=file_address,
        file_address_alg=file_address_alg,
        snapshot_every=args.snapshot_every,
    )