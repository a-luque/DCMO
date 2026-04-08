import os
import shutil
import time

num_of_simulations = 500
data_dir = "simulation_results/515_noise"
car_distance = -15 # pass lower bound

last_folder = 0

# Reset directory only if starting fresh
if last_folder == 0:
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

for i in range(last_folder, num_of_simulations):

    sim_dir = os.path.join(data_dir, str(i))
    os.makedirs(sim_dir, exist_ok=True)

    print(f"Executing sim {i}")

    cmd = f"scenic -S datagen.scenic --2d --count 1 --time 200 --param result_path {sim_dir} --param car_dist {car_distance}"

    ret = os.system(cmd)

    if ret != 0:
        print(f"Simulation {i} failed. Retrying...")
        time.sleep(5)
        os.system(cmd)