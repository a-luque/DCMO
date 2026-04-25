"""
import os

controllers = sorted([
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(os.path.expanduser("../controllers"))
            for f in fn
            if f.endswith(".pt")
        ])


print(f"{controllers[0]}")
"""



"""
import numpy as np


path = "../bandit_checkpoint.npz"
# path = "sim_results/1/speed.npz"
data = np.load(path, allow_pickle=True)

for key in data.files:
    print(key, data[key])
    #print(len(data[key]))
    #print(min(data[key]))
"""


"""
import os

if not os.path.exists("../bandit_checkpoint.npz"):
    print("[checkpoint] No checkpoint found — starting from scratch.")
"""

import numpy as np
a = np.zeros((3, 2), dtype=np.float64)
print(a[0])
print(a.shape)
