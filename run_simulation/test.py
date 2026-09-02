import numpy as np


path = "/cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/new_exp/run_simulations/rq0_results/35_12/balanced/2/leader_speed.npz"
data = np.load(path, allow_pickle=True)
print(data["values"])
