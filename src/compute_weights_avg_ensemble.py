import os
import numpy as np

cont_path = "/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/checkpoints/train_faster"
controllers = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(f"{cont_path}")) for f in fn]
controllers.sort()

weights_ensemble = [0.0538, 0.0372, 0.0483, 0.0417, 0.0557, 0.0512, 0.0543, 0.0377, 0.0496]
weights_ensemble = np.array(weights_ensemble) 
weights_ensemble = 1/weights_ensemble
weights_ensemble = weights_ensemble/weights_ensemble.sum()

with open("/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/weights_ensemble/weights_avg_ensemble.npy", "wb") as f: 
	np.save(f, weights_ensemble) 
print(weights_ensemble)