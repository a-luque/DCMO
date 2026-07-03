import os
import shutil
import time
import numpy as np

seed = 42
np.random.seed(seed)

num_of_simulations = 100
idm = "moderate"
weather = "HardRainNoon"


data_dir = f"data/{idm}/{weather}/515"
car_distance = -15
#car_distance = -54 # pass lower bound

last_folder = 0

# Reset directory only if starting fresh
if last_folder == 0:
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

for i in range(last_folder, num_of_simulations):
    #weather = np.random.choice(['ClearNoon', 'HardRainNoon', 'ClearSunset'])

    leader_speed = np.random.choice([2, 6, 10])

    sim_dir = os.path.join(data_dir, str(i))
    os.makedirs(sim_dir, exist_ok=True)

    print(f"Executing sim {i}")

    cmd = f"scenic -S gen_data_turn.scenic --2d --count 1 --time 800 --param ego_idm {idm} --param result_path {sim_dir} --param car_dist {car_distance} --param weather {weather} --param leader_speed {leader_speed}"

    ret = os.system(cmd)

    if ret != 0:
        print(f"Simulation {i} failed. Retrying...")
        time.sleep(5)
        os.system(cmd)