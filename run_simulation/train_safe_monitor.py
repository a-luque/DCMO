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

from src.system import System
from src.trainer import Trainer, Logger
from src.bandits import LinearExplorer, LogisticExplorer
import scenic
from scenic.simulators.newtonian import NewtonianSimulator
from scenic.simulators.carla.simulator import CarlaSimulator

from alg_es import Weather, ContextSpace, get_reward
from rq0_all import simulate

CONTROLLERS = ['sport', 'aggressive', 'dynamic', 'balanced', 'comfort', 'conservative', 'defensive']
SAVE_PATH =  f"safety_monitor"

class BasicSystem(System):


    def step(self, index, context, t_step):
        """
        Execute the controller (given by index) in the system on context "context".


        Parameters:
        - index: The index to be executed.
        - context: The context for the system. Example: ("ClearNoon", 15.0, 12)

        Returns:
        - The safety reward for the executed controller.
        """
        (weather, dist_car, speed) = context
        controller = CONTROLLERS[index]
        print(controller, weather, dist_car, speed, flush=True)

        save_path = os.path.join(SAVE_PATH, controller)
        
        while True:
            try:
                r, safety_info = simulate(context, CONTROLLERS[index], save_path, order="es", t=t_step)
                break
            except Exception as e:
                print(f"Simulation failed for controller={controller}, run={t_step}: {e}", flush=True)
                time.sleep(5)

        return 1-safety_info["safety_violation"]
            
        



    def sample_context(self):
        """
        Sample a context for the system.

        Sample a vector from scenic and return it. (Initial configuration of the system)

        Returns:
        - The context for the system.
        """
        return self.contexts.sample_one_context()
    

class BasicTrainer(Trainer):

    def train(self, initial, n_steps, logger=None):
        """
        Train the system for n_steps.

        Parameters:
        - n_steps: The number of steps to train the system.
        """
        
        
        for t_step in range(initial, n_steps):
            (weather, dist_car, speed) = self.system.sample_context()
            # intersection = random.randint(0,1)

            x = np.concatenate([np.array(Weather[weather].value), np.array([dist_car]), np.array([speed]), np.array([1.])])
            index = self.bandit_alg.act(x)
            
            uncertainties = []
            product_contexts = []

            if not all(self.bandit_alg.logistic_models.values()):
                (w, dc, s) = self.system.sample_context()
            else: 
                for ctx_i in range(len(self.system.contexts)):
                    (w, dc, s) = self.system.contexts.cell(ctx_i)
                    product_contexts += [(w, dc, s)]
                    X = np.concatenate([np.array(Weather[w].value), np.array([dc]), np.array([s]), np.array([1.])])

                    uncertainties += [
                        np.sqrt(np.dot(np.dot(X, self.bandit_alg.arm_hessians_inv[index]), X.T))
                    ]
                # (c, i, dc, dp) = random.choice(contexts_product)
                index_context = random.choice([i for i in range(len(uncertainties)) if uncertainties[i] == max(uncertainties)])
                # index_context = np.argmax(uncertainties)
                (w, dc, s) = product_contexts[index_context]
            x = np.concatenate([np.array(Weather[w].value), np.array([dc]), np.array([s]), np.array([1.])])
            index = self.bandit_alg.act(x)
            # index = random.randrange(15)
            reward = self.system.step(index, [w, dc, s], t_step)
            self.bandit_alg.update(index, x, reward)
            if t_step % self.bandit_alg.recompute_every == 0 and t_step > 0: 
                with open(f"{self.results_dir}/weights_{t_step}.npy", "wb") as f:
                    np.save(f, self.bandit_alg.weights)
                with open(f"{self.results_dir}/arm_hessians_inv_{t_step}.pkl", "wb") as f:
                    pickle.dump(self.bandit_alg.arm_hessians_inv, f)
                with open(f"{self.results_dir}/arm_data_{t_step}.pkl", "wb") as f:
                    pickle.dump(self.bandit_alg.arm_data, f)
                with open(f"{self.results_dir}/logistic_models_{t_step}.pkl", "wb") as f:
                    pickle.dump(self.bandit_alg.logistic_models, f)
            if logger is not None and (t_step % self.log_at == 0 or t_step == last_folder+n_steps - 1) and t_step >0:
                logger.log_data(t_step, self.bandit_alg, self.system)
                log = logger.get_log()
                if os.path.exists(LOG_PATH):
                    os.remove(LOG_PATH)
                pd.DataFrame.from_dict(log).to_csv(LOG_PATH)
                
        if logger is not None:
            return self.bandit_alg, logger.get_log()
        else:
            return self.bandit_alg, None
    
class BasicLogger(Logger):
    def log_data(self, t_step, bandit_alg, system, i_init=None):
        """
        Log the performance of the bandit algorithm.

        Parameters:
        - t: The current time step.
        - bandit_alg: The bandit algorithm.
        - system: The system.
        """
        if i_init is None:
            i_init = 0
        res_reward = 0
        self.log["t"].append(0)
        
        for i in range(i_init, i_init + self.log_samples):
            
            if f"rew_{i}" not in self.log.keys():
                self.log[f"rew_{i}"] = []
                
            reward = system.step(None, None, t_step=i)
            self.log[f"rew_{i}"].append(reward)
            
            print(self.log, flush=True)
            if i == 0 or i % 10 == 0 or i == (self.log_samples - 1):
                pd.DataFrame.from_dict(self.log).to_csv(f"{LOG_PATH[:-4]}_{i}.csv")
            res_reward += reward

        res_reward /= self.log_samples

        self.log["t"].append(t)
        # self.log["expected_reward"].append(res_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='modd',usage='later', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## arguments 
    parser.add_argument('--num_steps', help='number of steps per simulation',type=int,default=300)
    parser.add_argument('--threshold_invasions', help='number of steps per simulation',type=float,default=0.1)
    parser.add_argument("--results_dir", type=str, default="/proj/berzelius-2026-227/users/x_alluq/CMO/safety_monitor_training")
    parser.add_argument("--log_path", type=str, default="../../log.csv")
    parser.add_argument('--n_steps', help='number of steps per simulation',type=int,default=1001)
    parser.add_argument('--i_init', help='log data initial simulation for seed',type=int,default=0)
    parser.add_argument('--log_samples', help='number of steps per simulation',type=int,default=25)
    parser.add_argument('--log_at', help='number of steps per simulation',type=int,default=25)
    parser.add_argument('--recompute_every', help='number of steps per simulation',type=int,default=25)
    parser.add_argument('--initial_step', help='index of initial simulation',type=int,default=0)
    
    args = parser.parse_args()

    global THRESHOLD_INVASIONS
    global NUM_STEPS 
    global RESULTS_DIR 
    global LOG_PATH 
    global CONTROLLERS_FOLDER


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
    


    contexts = ContextSpace()
    
    system = BasicSystem(controllers=CONTROLLERS, scenic=['follow_lane.scenic', 'follow_lane_car.scenic'], contexts=contexts)
    
    logger = BasicLogger(log_samples=log_samples)
    explorer = LogisticExplorer(n_arms=len(CONTROLLERS), feature_dim=17, recompute_every=recompute_every)
    trainer = BasicTrainer(system=system, bandit_alg=explorer, log_at=log_at)
    
    
    system.results_dir = RESULTS_DIR
    trainer.results_dir = RESULTS_DIR
    explorer.contexts = contexts
    trainer.contexts = contexts
    system.weights_file = ""

    
    _, log = trainer.train(logger=None, initial=initial_step, n_steps=n_steps)
    
    if log is not None:
        pd.DataFrame.from_dict(log).to_csv(LOG_PATH)
