import os
import shutil
import time
import numpy as np

seed = 42
np.random.seed(seed)

num_of_simulations = 455

data_dir = "training_data/2325"
car_distance = -25
#car_distance = -54 # pass lower bound

last_folder = 0

# Reset directory only if starting fresh
if last_folder == 0:
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

for i in range(last_folder, num_of_simulations):
    weather = np.random.choice(['ClearNoon', 'HardRainNoon', 'ClearSunset'])
    leader_speed = np.random.choice([4, 6, 8])

    sim_dir = os.path.join(data_dir, str(i))
    os.makedirs(sim_dir, exist_ok=True)

    print(f"Executing sim {i}")

    cmd = f"scenic -S datagen_far.scenic --2d --count 1 --time 300 --param result_path {sim_dir} --param car_dist {car_distance} --param weather {weather} --param leader_speed {leader_speed}"

    ret = os.system(cmd)

    if ret != 0:
        print(f"Simulation {i} failed. Retrying...")
        time.sleep(5)
        os.system(cmd)