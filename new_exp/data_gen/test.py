import numpy as np


# path1 = "data/dist.npz"
# path2 = "data/acc.npz"
path3 = "test_data/maneuver.npz"
# dist = np.load(path1)
# acc = np.load(path2)
maneuver = np.load(path3)

print(maneuver['values'])

