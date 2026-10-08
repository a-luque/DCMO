import argparse
import glob
import os
import sys
import numpy as np
import time
from alg_es import get_reward, CONTROLLER_NAMES

global SEEDS

seed_file_path = "./sim_seed_context.npz"
SEEDS = np.load(seed_file_path, allow_pickle=True)

def simulate(cell, weights, save_path, run_id, seed) -> tuple:
    """Run one simulation for the given (weather, distance, speed) cell and
    controller, returning (reward_vector, safety_info)."""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir = os.path.join(
        current_file_dir, f"{save_path}/{run_id}/"
    )
    print(f"Results will be saved to: {results_dir}")

    if os.path.exists(results_dir):
        import shutil
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    #print(f"cell: {cell}")

    sampled_weather, sampled_distance, sampled_speed = cell

    sampled_distance = float(sampled_distance)
    sampled_speed = float(sampled_speed)

    weight_eff = float(weights[0])
    weight_stab = float(weights[1])

    scenic_file_path = os.path.join(current_file_dir, "rq4_sim_test.scenic")

    os.system(
        f"scenic -S {scenic_file_path} --seed {seed} --count 1 --time 300 --2d "
        f"--param result_path {results_dir} "
        f"--param weight_eff {weight_eff} "
        f"--param weight_stab {weight_stab} "
        f"--param weather {sampled_weather} "
        f"--param car_dist {sampled_distance} "
        f"--param leader_speed {sampled_speed}"
    )

    reward_e, reward_s, safety_info = get_reward(results_dir, order="es")
    rewards = np.array([reward_e, reward_s])
    return rewards, safety_info



def run_random_context_eval(
    weights,
    n_runs,
    checkpoint_path: str = "random_context_eval.npz",
    file_address: str = "random_ctx",
    snapshot_every: int = 100,
    snapshot_dir: str = None,
):
    if snapshot_dir is None:
        base = os.path.splitext(checkpoint_path)[0]
        snapshot_dir = base + "_snapshots"

    def _fresh_results():
        return {
            "rewards": np.full((n_runs, 2), np.nan),
            "sim_seed": np.full(n_runs, np.nan),
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

    def _save(path):
        np.savez(path, results=results, info=checkpoint_path)

    start = 0
    if os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path, allow_pickle=True)
        results = data["results"].item()
        done_mask = ~np.isnan(results["rewards"][:, 0])
        start = int(done_mask.sum())
        print(f"[checkpoint] Resuming baseline from run {start}/{n_runs}")


    for i in range(start, n_runs):
        # Randomly sample a fresh context cell for this run instead of
        # reusing a fixed cell.
        sim_seed = SEEDS["sim_seed"][i]
        cell = SEEDS["cells"][i]

        #cell = ctx.sample_one_context()

        while True:
            try:
                r, safety_info = simulate(cell, weights, file_address, i, int(sim_seed))
                break
            except Exception as e:
                print(f"Simulation failed for {checkpoint_path}, run={i}, cell={cell}: {e}")
                time.sleep(5)

        results["rewards"][i] = r
        results["sim_seed"][i] = sim_seed
        results["safety_violation"][i] = safety_info["safety_violation"]
        results["min_dist"][i] = safety_info["min_dist"]
        results["avg_speed"][i] = safety_info["avg_speed"]
        results["avg_jerks"][i] = safety_info["avg_jerks"]
        results["lane_invasion"][i] = safety_info["lane_invasion"]
        results["lane_invasion_count"][i] = safety_info["lane_invasion_count"]
        results["collision_happened"][i] = safety_info["collision_happened"]
        results["near_collision"][i] = safety_info["near_collision"]
        results["cells"][i] = cell

        done = i + 1
        print(f"[{done}/{n_runs}] info={checkpoint_path} run={i} cell={cell} "
              f"reward={r} safety_violation={safety_info['safety_violation']}")
        
        # Rolling checkpoint: overwritten after every run so a crash never
        # loses more than the in-flight simulation.
        _save(checkpoint_path)

        if done % snapshot_every == 0:
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_path = os.path.join(snapshot_dir, f"snapshot_{done}.npz")
            _save(snapshot_path)
            print(f"[snapshot]   Saved permanent baseline snapshot at run {done} -> {snapshot_path}")

    _save(checkpoint_path)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run randomly-sampled-context simulations for ensemble controllers."
    )
    parser.add_argument("--n-runs", type=int, default=1000)
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


    file_address = f"rq4_results_n2n/w_{alpha_str}"
    checkpoint_path = os.path.join(".", f"n2n_rq4_{alpha_str}.npz")

    results = run_random_context_eval(
        weights=alphas,
        n_runs=args.n_runs,
        checkpoint_path=checkpoint_path,
        file_address=file_address,
        snapshot_every=args.snapshot_every,
    )