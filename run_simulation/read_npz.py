import numpy as np
"""
path = "/cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/new_exp/rq0_6_4_sport.npz"
data = np.load(path, allow_pickle=True)
results = data['results'].item()


lane_invasion = results['lane_invasion']
safety_violation = results['safety_violation']
#near_collision = results['near_collision']
#lane_invasion_count = results['lane_invasion_count']

lane_invasion_pct = np.mean(lane_invasion == 1) * 100
safety_violation_pct = np.mean(safety_violation == 1) * 100
#near_collision_pct = np.mean(near_collision == 1) * 100

# both_ones = (lane_invasion == 1) & (near_collision == 1)
# both_ones_sum = np.sum(both_ones)

print("Lane Invasion:", np.sum(lane_invasion))
print("Safety Violation:", np.sum(safety_violation))
# print("Near Collision:", np.sum(near_collision))
# print("Both Ones:", both_ones_sum)
print("Lane Invasion (%):", lane_invasion_pct)
print("Safety Violation (%):", safety_violation_pct)
# print("Near Collision (%):", near_collision_pct)
print(lane_invasion)
# print(lane_invasion_count)
"""
"""
path = "/cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/new_exp/run_simulations/redo_sim/cte.npz"
data = np.load(path, allow_pickle=True)
cte_vals = data['values']

indices = np.where(np.absolute(cte_vals) > 0.7)
matching_vals = cte_vals[indices]
violation = np.sum(np.absolute(cte_vals) > 0.7) > 30
print(matching_vals)
print(violation)
"""

"""
file_path = "/cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/new_exp/run_simulations/test_results_all/6_4/testresults_rq0_6_4_sport.npz"
file_data = np.load(file_path, allow_pickle=True)
results = file_data['results'].item()
print(f"Lane Invasion: {results['lane_invasion'][0]}")
print(f"sim_seed: {results['sim_seed'][0]}")
"""

"""
for i in range(15):
    speed = 8
    path = f"/cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/new_exp/run_simulations/rq0_results/25_{speed}/sport/{i}/leader_speed.npz"
    data = np.load(path, allow_pickle=True)
    mean = np.mean(data["values"])
    if np.mean(data["values"]) != speed:
        print("!!!")
        print(mean)
        break
"""


simu_num = 8
path = f"/cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/new_exp/run_simulations/rq0_results/6_4/sport/{simu_num}"
maneuver_file = path + "/maneuver.npz"
cte_file = path + "/cte.npz"
maneuver_data = np.load(maneuver_file, allow_pickle=True)
cte_data = np.load(cte_file, allow_pickle=True)
maneuver_vals = maneuver_data["values"]
cte_vals = cte_data["values"]

indices = np.where(np.absolute(cte_vals) > 0.7)
print(cte_vals[indices])
print(maneuver_vals[indices])
print(cte_vals)
print(maneuver_vals)



npz_file = "/cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/new_exp/rq0_6_4_sport.npz"
npz_data = np.load(npz_file, allow_pickle=True)
npz_vals = npz_data["results"].item()
lane_invasion = npz_vals["lane_invasion"]
lane_invasion_count = npz_vals["lane_invasion_count"]
print(lane_invasion[simu_num])
print(lane_invasion_count[simu_num])





