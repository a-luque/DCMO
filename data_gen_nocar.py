import os
import shutil
import time
import numpy as np

seed = 42
np.random.seed(seed)

num_of_simulations = 455
data_dir = "training_data/no_car"
last_folder = 0

# Reset directory only if starting fresh
if last_folder == 0:
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

for i in range(last_folder, num_of_simulations):
    weather = np.random.choice(['ClearNoon', 'HardRainNoon', 'ClearSunset'])

    sim_dir = os.path.join(data_dir, str(i))
    os.makedirs(sim_dir, exist_ok=True)

    print(f"Executing sim {i}")

    #cmd = f"scenic -S datagen_nocar.scenic --2d --count 1 --time 300 --param result_path {sim_dir}"
    cmd = f"scenic -S datagen_nocar.scenic --2d --count 1 --time 300 --param result_path {sim_dir} --param weather {weather}"

    ret = os.system(cmd)

    if ret != 0:
        print(f"Simulation {i} failed. Retrying...")
        time.sleep(5)
        os.system(cmd)