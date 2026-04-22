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


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

class Weather(Enum):
    ClearNoon = [5.0, 0.0, 0.0, 10.0, -1.0, 45.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0] #
    CloudyNoon = [60.0, 0.0, 0.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetNoon = [5.0, 0.0, 50.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetCloudyNoon = [60.0, 0.0, 50.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    MidRainyNoon = [60.0, 60.0, 60.0, 60.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    HardRainNoon = [100.0, 100.0, 90.0, 100.0, -1.0, 45.0, 7.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0] #
    SoftRainNoon = [20.0, 30.0, 50.0, 30.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    ClearSunset = [5.0, 0.0, 0.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0] #
    CloudySunset = [60.0, 0.0, 0.0, 10.0, -1.0, 15.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetSunset = [5.0, 0.0, 50.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetCloudySunset = [60.0, 0.0, 50.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    MidRainSunset = [60.0, 60.0, 60.0, 60.0, -1.0, 15.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0] 
    HardRainSunset = [100.0, 100.0, 90.0, 100.0, -1.0, 15.0, 7.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0] #
    SoftRainSunset = [20.0, 30.0, 50.0, 30.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]


class BasicSystem(System):


    def step(self, index, context, t_step=None):
        """
        Execute the controller corresponding index in the system with context x.


        Parameters:
        - index: The index to be executed.
        - context: The context for the system.
        - t_step: ID that helps using the same seed. Used only for testing 

        Returns:
        - The reward for the executed index.
        """
        if t_step is None:
            [weather, dist, speed] = context
            print(self.controllers[index], weather, dist, speed)
        
        results_dir = self.results_dir
        if results_dir[-1] != "/":
            results_dir += "/"
        print(results_dir)
        # Initialize logger folder structure
        os.makedirs(results_dir, exist_ok=True)

        try: 
            current_modd = -1
            for folder in os.scandir(results_dir):
                id = int(re.sub("[^0-9]","",os.path.basename(folder)))
                if id>current_modd:
                    current_modd=id
            modd_dir = f"{results_dir}modd_{current_modd+1}"

            if t_step is not None:
                random.seed(t_step)
                seed = int("".join(["{}".format(random.randint(0,9)) for num in range(5)]))
                print(f"Seed: {seed}")
                # TODO: Generate parameters for a given seed so that they can be fed to the scenic program
                weathers = ['ClearNoon','CloudyNoon','WetNoon','WetCloudyNoon','MidRainyNoon', 'HardRainNoon', 'SoftRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'MidRainSunset', 'HardRainSunset', 'SoftRainSunset']
                dists = [5,10,20,30,50]
                # TODO: Speeds?
                speeds = [4,6,8]
                contexts_product = list(itertools.product(weathers, [0], dists, speeds)) + \
                        list(itertools.product(weathers, [1], [5], speeds)) + \
                        list(itertools.product(weathers, [0], [100], [0])) + \
                        list(itertools.product(weathers, [1], [100], [0])) 
                [weather, intersection, dist, speed] = random.choice(contexts_product)

                # print(self.sample_context())
                # [weather, intersection, dist, speed] = self.sample_context()
                print(weather, intersection, dist, speed, MONITOR_PATH)
                print(f"scenic -S {self.scenic['testing']} --count 1 --time {NUM_STEPS} --2d -s {seed} --param weather {weather} --param results_path {modd_dir} --param dist_car {dist} --param speed_car {speed} --param monitor {MONITOR_PATH} --param render 0")
                # os.system(f"scenic -S test_cnn_controller.scenic --count 1 --time {NUM_STEPS} --2d --param controller_dir {self.controllers} --param weather {weather} --param results_path {modd_dir} --param dist_car {dist} --param intersec {intersection} --param render 0 -s {seed}")
                # os.system(f"scenic -S follow_lane_car_testing_ensemble.scenic --count 1 --time {NUM_STEPS} --2d --param global_folder {modd_dir} --param controller_path {self.controllers} -s {seed} --param render 0")
                os.system(f"scenic -S {self.scenic['testing']} --count 1 --time {NUM_STEPS} --2d --param controllers_dir {CONTROLLERS_FOLDER} -s {seed} --param weather {weather} --param results_path {modd_dir} --param intersec {intersection} --param dist_car {dist} --param speed_car {speed} --param monitor {MONITOR_PATH} --param render 0")  

            else:
                # os.system(f"scenic -S follow_lane_car_generic.scenic --count 1 --time {NUM_STEPS} --2d --param global_folder {modd_dir} --param weather {weather} --param controller_path {self.controllers[index]} --param dist_car {dist} --param intersec {intersection} --param render 0")
                os.system(f"scenic -S {self.scenic['training']} --count 1 --time {NUM_STEPS} --2d --param controller_path {self.controllers[index]} --param weather {weather} --param results_path {modd_dir} --param dist_car {dist} --param speed_car {speed} --param render 0")  


            dist_vals = np.load(f"{modd_dir}/dist.npz")
            cte_vals = np.load(f"{modd_dir}/true_cte.npz")
            collision_vals = np.load(f"{modd_dir}/collision.npz")

            min_dist = float(dist_vals["values"].min())-4.6
            lane_invasion = int((np.absolute(cte_vals["values"])>0.7).sum())
            collision_happened = int(collision_vals["values"].max())

            print(f"-- Min. dist: {min_dist}")
            print(f"-- Lane invasion: {lane_invasion}")
            print(f"-- Collision happened?: {collision_happened}")

            # TODO: Change safe distance value to something proportional to the speed?
            if min_dist-4.6 < 1 or lane_invasion > 30 or collision_happened:
                return 0
            else: 
                return 1

        except: 
            print(f"Simulation failed.")
            time.sleep(10)
            if os.path.isdir(modd_dir):
                if len(os.listdir(modd_dir)) > 0:
                    dist_vals = np.load(f"{modd_dir}/dist.npz")
                    cte_vals = np.load(f"{modd_dir}/true_cte.npz")
                    collision_vals = np.load(f"{modd_dir}/collision.npz")

                    min_dist = float(dist_vals["values"].min())
                    lane_invasion = int((np.absolute(cte_vals["values"])>0.7).sum())
                    collision_happened = int(collision_vals["values"].max())
                    
                    print(f"-- Min. dist: {min_dist}")
                    print(f"-- Lane invasion: {lane_invasion}")
                    print(collision_vals["values"])
                    print(f"-- Collision happened?: {collision_happened}")

                    # TODO: Change safe distance value to something proportional to the speed?
                    if min_dist-4.6 < 1 or lane_invasion > 30 or collision_happened:
                        return 0
                    else: 
                        return 1
                else:
                    return self.step(index, context, t_step=t_step)
            else:
                return self.step(index, context, t_step=t_step)
        




    def sample_context(self):
        """
        Sample a context for the system.

        Sample a vector from scenic and return it. (Initial configuration of the system)

        Returns:
        - The context for the system.
        """
        # dists = [5,10,15,100] # [0,7]; [8,12]; [13,20]; [21+]
        # speeds = [4,6,8]
        # contexts_product = list(itertools.product(system.contexts, dists, speeds)) + \
        #                 list(itertools.product(system.contexts, [5], speeds)) + \
        #                 list(itertools.product(system.contexts, [100], [0])) + \
        #                 list(itertools.product(system.contexts, [100], [0])) 

        [weather, dist, speed] = random.choice(self.contexts)
        return [weather, dist, speed]
    

class BasicTrainer(Trainer):

    def train(self, n_steps, last_folder, logger=None):
        """
        Train the system for n_steps.

        Parameters:
        - n_steps: The number of steps to train the system.
        """
        # dists = [5,10,15,100] # [0,7]; [8,12]; [13,20]; [21+]
        # speeds = [4,6,8]
        # contexts_product = list(itertools.product(system.contexts, dists, speeds)) + \
        #                 list(itertools.product(system.contexts, [5], speeds)) + \
        #                 list(itertools.product(system.contexts, [100], [0])) + \
        #                 list(itertools.product(system.contexts, [100], [0])) 
        results_dir = self.results_dir
        
        for t_step in range(last_folder,last_folder+n_steps):
            # Initialize logger folder structure
            if t_step==0:
                if os.path.exists(results_dir):
                    shutil.rmtree(results_dir)
                os.makedirs(results_dir)
            [weather, dist, speed] = self.system.sample_context()

            x = np.concatenate([np.array(Weather[weather].value), np.array([dist]), np.array([speed]), np.array([1.])])
            index = self.bandit_alg.act(x)
            uncertainties = []
            product_contexts = []
            if not all(self.bandit_alg.logistic_models.values()):
                (w,d,s) = random.choice(self.system.contexts)
            else: 
                for [w,  d, s] in self.system.contexts:
                    product_contexts += [(w,d,s)]
                    X = np.concatenate([np.array(Weather[w].value), np.array([d]), np.array([s]), np.array([1.])])

                    uncertainties += [
                        np.sqrt(np.dot(np.dot(X, self.bandit_alg.arm_hessians_inv[index]), X.T))
                    ]
                index_context = random.choice([i for i in range(len(uncertainties)) if uncertainties[i] == max(uncertainties)])
                # index_context = np.argmax(uncertainties)
                (w,d,s) = product_contexts[index_context]
            x = np.concatenate([np.array(Weather[w].value), np.array([d]), np.array([s]), np.array([1.])])

            reward = self.system.step(index, [w, d, s])
            print(f"Reward! {reward}")
            self.bandit_alg.update(index, x, reward)
            if t_step % self.bandit_alg.recompute_every == 0 and t_step > 0: 
                with open(f"{results_dir}/weights_{t_step}.npy", "wb") as f:
                    np.save(f, self.bandit_alg.weights)
                with open(f"{results_dir}/arm_hessians_inv_{t_step}.pkl", "wb") as f:
                    pickle.dump(self.bandit_alg.arm_hessians_inv, f)
                with open(f"{results_dir}/arm_data_{t_step}.pkl", "wb") as f:
                    pickle.dump(self.bandit_alg.arm_data, f)
                with open(f"{results_dir}/logistic_models_{t_step}.pkl", "wb") as f:
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
            print(self.log)
            if i == 0 or i % 50 == 0 or i == (self.log_samples - 1):
                pd.DataFrame.from_dict(self.log).to_csv(f"{LOG_PATH[:-4]}_{i}.csv")
            res_reward += reward

        res_reward /= self.log_samples

        self.log["t"].append(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='modd',usage='later', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## arguments 
    parser.add_argument('--num_steps', help='number of steps per simulation',type=int,default=300)
    parser.add_argument('--threshold_invasions', help='number of steps per simulation',type=float,default=0.1)
    parser.add_argument("--results_dir", type=str, default="/mimer/NOBACKUP/groups/naiss2024-22-1336/simulation_data_system_main")
    parser.add_argument("--log_path", type=str, default="../../log.csv")
    parser.add_argument("--controllers_folder", type=str, default="../../controllers_models/")
    parser.add_argument('--n_steps', help='number of monitor algorithm steps',type=int,default=1001)
    parser.add_argument('--i_init', help='log data initial simulation for seed',type=int,default=0)
    parser.add_argument('--log_samples', help='number of simulations to evaluate',type=int,default=500)
    parser.add_argument('--log_at', help='number of steps per simulation',type=int,default=25)
    parser.add_argument('--recompute_every', help='number of steps per simulation',type=int,default=25)
    parser.add_argument('--monitor', help='path to monitor', type=str, default="")
    parser.add_argument('--training', help='training?', type=int, default=1)

    # python3 safety_monitor_training.py --results_dir /mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/test_monitor_training --log_path /mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/test_monitor_training/log.csv --controllers_folder /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/checkpoints/train_faster --num_steps 300 --recompute_every 25

    args = parser.parse_args()

    global THRESHOLD_INVASIONS
    global NUM_STEPS 
    global RESULTS_DIR 
    global LOG_PATH 
    global CONTROLLERS_FOLDER
    global MONITOR_PATH


    THRESHOLD_INVASIONS = args.threshold_invasions
    NUM_STEPS= args.num_steps
    RESULTS_DIR = args.results_dir
    LOG_PATH = args.log_path
    CONTROLLERS_FOLDER = args.controllers_folder
    n_steps = args.n_steps
    log_at = args.log_at
    log_samples = args.log_samples
    recompute_every = args.recompute_every
    i_init = args.i_init
    MONITOR_PATH = args.monitor
    training = int(args.training)
    print(f"Training {training}")


    # controllers = [CONTROLLERS_FOLDER + e for e in os.listdir(CONTROLLERS_FOLDER)]
    controllers = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(f"{CONTROLLERS_FOLDER}")) for f in fn]
    controllers.sort()
    print(f"Controllers list: {controllers}")
    # controllers = [CONTROLLERS_FOLDER + f"/adamodel_v2_{i}.pth" for i in range(15)]
    weathers = ['ClearNoon', 'HardRainNoon', 'ClearSunset']
    dists = [5,10,15,100] # [0,7]; [8,12]; [13,20]; [21+]
    speeds = [4,6,8]
    contexts = list(itertools.product(weathers, dists, speeds)) + \
                    list(itertools.product(weathers, [5], speeds)) + \
                    list(itertools.product(weathers, [100], [0])) + \
                    list(itertools.product(weathers, [100], [0])) 

    scenic_progs = {}
    scenic_progs["training"] = "safety_program.scenic"
    scenic_progs["testing"] = "safety_program_testing.scenic"

    # contexts = ['ClearNoon','CloudyNoon','WetNoon','WetCloudyNoon','MidRainyNoon', 'HardRainNoon', 'SoftRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'MidRainSunset', 'HardRainSunset', 'SoftRainSunset']
    # controllers = ["./controllers/monitor0.sav", "./controllers/monitor1.sav", "./controllers/monitor4.sav", "./controllers/monitor6.sav"]
    system = BasicSystem(controllers=controllers, scenic=scenic_progs, contexts=contexts)
    # system = BasicSystem(controllers=controllers, scenic=['crossing_collision.scenic','crossing_collision.scenic'], contexts=contexts)
    logger = BasicLogger(log_samples=log_samples)
    explorer = LogisticExplorer(n_arms=len(controllers), feature_dim=17, recompute_every=recompute_every)

    if i_init > 0:
        remove_empty_folders(RESULTS_DIR)
        with open(f"{RESULTS_DIR}/weights_{i_init}.npy", "rb") as f:
            explorer.weights = np.load(f)
        with open(f"{RESULTS_DIR}/arm_hessians_inv_{i_init}.pkl", "rb") as f:
            explorer.arm_hessians_inv = pickle.load(f)
        with open(f"{RESULTS_DIR}/arm_data_{i_init}.pkl", "rb") as f:
            explorer.arm_data = pickle.load(f)
        with open(f"{RESULTS_DIR}/logistic_models_{i_init}.pkl", "rb") as f:
            explorer.logistic_models = pickle.load(f)


    trainer = BasicTrainer(system=system, bandit_alg=explorer, log_at=log_at)
    system.results_dir = RESULTS_DIR
    trainer.results_dir = RESULTS_DIR
    explorer.contexts = contexts
    trainer.contexts = contexts
    system.controllers_folder_path = CONTROLLERS_FOLDER
    system.weights_file = ""
    
    if training:
        _, log = trainer.train(logger=None, last_folder=i_init, n_steps=n_steps)
        if log is not None:
            pd.DataFrame.from_dict(log).to_csv(LOG_PATH)
    else:
        print("-- Testing")
        if os.path.exists(RESULTS_DIR):
            shutil.rmtree(RESULTS_DIR)
        os.makedirs(RESULTS_DIR)
        for t in range(log_at,log_at+1, 100):
            system.weights_file = f"{MONITOR_PATH}"
            logger.log_data(t, explorer, system, i_init=i_init)

