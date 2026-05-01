import sys
import os
import argparse
import numpy as np
from enum import Enum
import pickle 
import random
import shutil
import pandas as pd
import re
import time
import itertools


from system import System
from trainer import Trainer, Logger
from bandits import LinearExplorer, LogisticExplorer
import scenic 
from scenic.simulators.newtonian import NewtonianSimulator 
from scenic.simulators.carla.simulator import CarlaSimulator 


# python3 /cephyr/users/luque/Alvis/DCMO/src/evaluation_moe.py --scenic /cephyr/users/luque/Alvis/DCMO/src/evaluation_moe.scenic --num_steps 300 --num_sim 500 --results_dir /mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/testing_moe  --moe_path /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/checkpoints/train_faster 


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)



def get_reward(results_path: str, timestep: float = 0.1,
               tau: float = 1.0) -> list:
    # --- efficiency reward: avg_speed / model_size ---
    # Extract model size (18, 50, or 101) from controller filename
    # e.g. "../controllers/resnet101_coarse_best.pt" -> 101
    # controller_basename = os.path.basename(controller_path)
    # model_size = None
    # for candidate in [18, 50, 101]:
    #     if f"resnet{candidate}" in controller_basename:
    #         model_size = candidate
    #         break
    # if model_size is None:
    #     raise ValueError(
    #         f"Cannot determine model size from controller path: {controller_path!r}. "
    #         f"Expected filename to contain 'resnet18', 'resnet50', or 'resnet101'."
    #     )

    speed_results = np.load(results_path + "speed.npz")
    acc_results   = np.load(results_path + "acc.npz")

    avg_speed = np.mean(speed_results["values"])
    acc       = np.array(acc_results["values"])
    jerks     = np.diff(acc) / timestep
    jerks_mean = np.mean(np.abs(jerks))

    #print(f"Jerk Mean: {np.mean(np.abs(jerks))}, Max: {np.max(np.abs(jerks))}, Min: {np.min(np.abs(jerks))}")

    # Min-max normalization for reward_e
    # Max possible value: fastest speed (8) / smallest model (18)  -> 8/18
    # Min possible value: 0 (absolute minimum)
    #RAW_MAX = 8.0 / 18.0
    RAW_MAX = 8.0 / np.log(19.0) 
    RAW_MIN = 0.0
    # raw_e   = avg_speed / model_size
    raw_e = avg_speed / np.log(3*18 + 3*50 + 3*101 + 1) 
    reward_e = (raw_e - RAW_MIN) / (RAW_MAX - RAW_MIN)     # efficiency (normalized)
    reward_e = float(np.clip(reward_e, 0.0, 1.0))          # guard against edge cases

    # reward_c = np.mean(np.exp(-np.abs(jerks) / tau))       # comfort
    #reward_c = np.mean(np.exp(-(jerks / tau)**2))
    reward_s = np.exp(- ( jerks_mean / tau)) 

    #### logging safety violation info
    dist_vals = np.load(results_path + "dist.npz")
    cte_vals = np.load(results_path + "true_cte.npz")
    collision_vals = np.load(results_path + "collision.npz")

    min_dist = float(dist_vals["values"].min())-4.6 
    lane_invasion = int((np.abs(cte_vals["values"])>0.7).sum())
    collision_happened = int(collision_vals["values"].max()) 

    # TODO: Change safe distance value to something proportional to the speed?
    if min_dist < 1 or lane_invasion > 30 or collision_happened:
        safety_violation = 1
    else: 
        safety_violation = 0
    
    safety_info = {
        "min_dist": min_dist,
        "avg_speed": avg_speed,
        "avg_jerks": jerks_mean,
        "lane_invasion": lane_invasion,
        "collision_happened": collision_happened,
        "safety_violation": safety_violation
    }

    # to controll the priorities of objectives, we only need to change the order of reward.
    #return [reward_e, reward_s, safety_info] # # objective order: e, s
    return [reward_s, reward_e, 1-safety_violation, safety_info] # objective order: s, e



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='modd',usage='later', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## arguments 
    parser.add_argument('--num_sim', help='number of simulations',type=int,default=500)
    parser.add_argument('--num_steps', help='number of steps per simulation',type=int,default=300)
    parser.add_argument("--results_dir", type=str, default="")
    parser.add_argument("--moe_path", type=str, default="")
    parser.add_argument("--controllers_folder", type=str, default="")
    parser.add_argument('--scenic', help='scenic program',type=str,default="")

    args = parser.parse_args()
    
    CONTROLLERS_FOLDER = args.controllers_folder
    NUM_STEPS= args.num_steps
    NUM_SIM = args.num_sim
    RESULTS_DIR = args.results_dir
    SCENIC = args.scenic
    MOE_PATH = args.moe_path

    # Contexts and controllers
    weathers = ['ClearNoon', 'HardRainNoon', 'ClearSunset']
    dists = [5,10,15] # [0,7]; [8,12]; [13,20]; [21+]
    speeds = [4,6,8]
    contexts = list(itertools.product(weathers, dists, speeds)) + \
                    list(itertools.product(weathers, [100], [0]))

       
      
    df = pd.DataFrame(columns=["weather", "dist", "speed", "rew_sta", "rew_eff", "rew_safe"])
      
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR)

    for i in range(NUM_SIM):
        
        # j = i % len_space
        random.seed(i)
        seed = int("".join(["{}".format(random.randint(0,9)) for num in range(5)]))
        [weather, dist, speed] = random.choice(contexts)
        
        results_dir = RESULTS_DIR
        if results_dir[-1] != "/":
            results_dir += "/"
        # Initialize logger folder structure
        os.makedirs(results_dir, exist_ok=True)
        # print(f"Iteration {i}", flush=True)

        while True:
            try: 
                current_modd = -1
                for folder in os.scandir(results_dir):    
                    str_num = re.sub("[^0-9]","",os.path.basename(folder))
                    if len(str_num) > 0:
                        id = int(str_num)
                        if id>current_modd:
                            current_modd=id
                modd_dir = f"{results_dir}modd_{current_modd+1}/"

                # print("Loading scenic", flush=True)
                os.system(f"scenic -S {SCENIC} --count 1 --time {NUM_STEPS} --2d -s {seed} --param controllers_dir {CONTROLLERS_FOLDER} --param moe_path {MOE_PATH} --param weather {weather} --param results_path {modd_dir} --param dist_car {dist} --param speed_car {speed} --param render 0")  
                # print("Simulation worked", flush=True)

                r_sta, r_eff, r_safe, _ = get_reward(results_path=modd_dir)
                # print("Computed rewards", flush=True)
                df.loc[i] = [weather, dist, speed, r_sta, r_eff, r_safe]
                df.to_csv(f"{results_dir}results.csv")
                print(f"Saved data for iteration {i}", flush=True)
                break

            except Exception as e: 
                print(f"Simulation failed: {e}", flush=True)
                time.sleep(5)