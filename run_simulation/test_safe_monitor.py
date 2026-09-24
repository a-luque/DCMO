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

sys.path.append("../src")
sys.path.append('..')
sys.path.append('./')

from src.monitor import Monitor
from src.system import System
from src.trainer import Trainer, Logger
from src.bandits import LinearExplorer, LogisticExplorer
import scenic
from scenic.simulators.newtonian import NewtonianSimulator
from scenic.simulators.carla.simulator import CarlaSimulator

from alg_es import Weather, ContextSpace, get_reward

CONTROLLERS = ['sport', 'aggressive', 'dynamic', 'balanced', 'comfort', 'conservative', 'defensive']
SAVE_PATH =  f"safety_monitor_testing"




def simulate(cell, controller_path: str, save_path: str, order: str, t: int, seed: int) -> np.ndarray:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    #results_dir = f"sim_results/{t}/"
    results_dir = os.path.join(current_file_dir, f"{save_path}/{t}/")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    # sampled_weather, sampled_intersect, sampled_distance, sampled_speed = cell
    sampled_weather, sampled_distance, sampled_speed = cell

    scenic_file_path = os.path.join(current_file_dir, "new_sim.scenic")

    #print(f"Simulated round {t} with controller {controller_path} at context {sampled_weather} {-1 * sampled_distance} {sampled_speed}. Results in {results_dir}")

    os.system(
        f"scenic -S {scenic_file_path} --count 1 --time 300 --2d --seed {int(seed)} "
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
    return reward_e, reward_s, 1-safety_info["safety_violation"]




class BasicSystem(System):


    def step(self, t_step):
        """
        Execute the controller (given by index) in the system on context "context".


        Parameters:
        - index: The index to be executed.
        - context: The context for the system. Example: ("ClearNoon", 15.0, 12)

        Returns:
        - The safety reward for the executed controller.
        """
        seed = SEEDS["sim_seed"][t_step]
        context = SEEDS["cells"][t_step]

        controller_index = MONITOR.optimal_controller(context, LOSS_WEIGHTS)

        if controller_index is None:
            print(f"No safe controller found for context {context} at step {t_step}. Skipping this step.", flush=True)
            return 0, 0, 1

        (weather, dist_car, speed) = context
        controller = CONTROLLERS[controller_index]

        print(controller, weather, dist_car, speed, flush=True)

        save_path = os.path.join(SAVE_PATH, controller)
        
        while True:
            try:
                re, rs, safety = simulate(context, CONTROLLERS[controller_index], save_path, order="es", t=t_step, seed=seed)
                break
            except Exception as e:
                print(f"Simulation failed for controller={controller}, run={t_step}: {e}", flush=True)
                time.sleep(5)

        return re,rs,safety
        



    def sample_context(self):
        """
        Sample a context for the system.

        Sample a vector from scenic and return it. (Initial configuration of the system)

        Returns:
        - The context for the system.
        """
        return self.contexts.sample_one_context()

    
class BasicLogger(Logger):
    def log_data(self, bandit_alg, system, i_init=None):
        """
        Log the performance of the bandit algorithm.

        Parameters:
        - t: The current time step.
        - bandit_alg: The bandit algorithm.
        - system: The system.
        """
        if i_init is None:
            i_init = 0
        res_reward_e = 0
        res_reward_s = 0
        res_safety = 0
        self.log["t"].append(0)
        
        for i in range(i_init, i_init + self.log_samples):
            
            if f"rew_e_{i}" not in self.log.keys():
                self.log[f"rew_e_{i}"] = []
            if f"rew_s_{i}" not in self.log.keys():
                self.log[f"rew_s_{i}"] = []
            if f"safety_{i}" not in self.log.keys():
                self.log[f"safety_{i}"] = []
                
            reward_e, reward_s, safety = system.step(t_step=i)
            self.log[f"rew_e_{i}"].append(reward_e)
            self.log[f"rew_s_{i}"].append(reward_s)
            self.log[f"safety_{i}"].append(safety)
            
            # print(self.log, flush=True)
            if i == 0 or i % 10 == 0 or i == (self.log_samples - 1):
                pd.DataFrame.from_dict(self.log).to_csv(f"{LOG_PATH[:-4]}_{i}.csv")
            res_reward_e += reward_e
            res_reward_s += reward_s
            res_safety += safety

        res_reward_e /= self.log_samples
        res_reward_s /= self.log_samples
        res_safety /= self.log_samples

        self.log["t"].append(t)
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
        pd.DataFrame.from_dict(self.log).to_csv(LOG_PATH)
        # self.log["expected_reward"].append(res_reward)


if __name__ == "__main__":
    np.random.seed(42)
    
    parser = argparse.ArgumentParser(description='modd',usage='later', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## arguments 
    parser.add_argument('--num_steps', help='number of steps per simulation',type=int,default=300)
    parser.add_argument('--threshold_invasions', help='number of steps per simulation',type=float,default=0.1)
    parser.add_argument("--results_dir", type=str, default="/home/luque/Documents/safety_monitor_testing")
    parser.add_argument("--log_path", type=str, default="../../log_loss_05_05.csv")
    parser.add_argument('--n_steps', help='number of rounds per simulation',type=int,default=1001)
    parser.add_argument('--i_init', help='log data initial simulation for seed',type=int,default=0)
    parser.add_argument('--log_samples', help='number of steps per simulation',type=int,default=1000)
    parser.add_argument('--log_at', help='number of steps per simulation',type=int,default=25)
    parser.add_argument('--recompute_every', help='number of steps per simulation',type=int,default=25)
    parser.add_argument('--initial_step', help='index of initial simulation',type=int,default=0)
    parser.add_argument('--safety_monitor', help='path to safety monitor weights',type=str,default="/home/luque/Documents/safety_monitor_training/weights_1000.npy")
    parser.add_argument('--performance_monitor', help='path to performance monitor weights',type=str,default="/home/luque/Documents/DCMO/main_alg_sim_es.npz")
    parser.add_argument('--safety_threshold', help='number of steps per simulation',type=float,default=0.8)
    parser.add_argument('--seed_file', help='path to seed file',type=str,default="/home/luque/Downloads/sim_seed_context.npz")
    
    args = parser.parse_args()


    global THRESHOLD_INVASIONS
    global NUM_STEPS 
    global RESULTS_DIR 
    global LOG_PATH 
    global CONTROLLERS_FOLDER
    global MONITOR
    global SEEDS
    global LOSS_WEIGHTS

    THRESHOLD_INVASIONS = args.threshold_invasions
    NUM_STEPS= args.num_steps
    RESULTS_DIR = args.results_dir
    LOG_PATH = args.log_path
    n_steps = args.n_steps
    log_at = args.log_at
    log_samples = args.log_samples
    recompute_every = args.recompute_every
    i_init = args.i_init
    initial_step = args.initial_step


    LOSS_WEIGHTS = np.array([0.5,0.5])
    


    SEEDS = np.load(args.seed_file,allow_pickle=True)

    contexts = ContextSpace()
    
    MONITOR = Monitor(args.safety_monitor, args.performance_monitor, args.safety_threshold, contexts)
    
    system = BasicSystem(controllers=CONTROLLERS, scenic=['follow_lane.scenic', 'follow_lane_car.scenic'], contexts=contexts)
    
    logger = BasicLogger(log_samples=log_samples)
    explorer = LogisticExplorer(n_arms=len(CONTROLLERS), feature_dim=17, recompute_every=recompute_every)
    
    
    system.results_dir = RESULTS_DIR
    explorer.contexts = contexts
    system.weights_file = ""

    
    # _, log = trainer.train(logger=logger, initial=initial_step, n_steps=n_steps)

    logger.log_data(explorer, system, i_init=i_init)
    
    # if log is not None:
    #     pd.DataFrame.from_dict(log).to_csv(LOG_PATH)
