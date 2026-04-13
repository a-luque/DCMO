import numpy as np

"""
path = "training_data/no_car_noise/2/dist.npz"
#path = "data/extra_data/turn_left/6/maneuver.npz"
data = np.load(path)

for key in data.files:
    print(key, data[key])
    print(len(data[key]))
    print(min(data[key]))


"""

"""
#copy dist.npz to all no_car folders

import os
import shutil

base_dir = "simulation_results/inf"
source_file = "dist.npz"

for i in range(100):
    folder_path = os.path.join(base_dir, str(i))
    img_folder = os.path.join(folder_path, "img")

    if not os.path.isdir(img_folder):
        print(f"{i}, img_folder does not exist")

    images = [f for f in os.listdir(img_folder) if f.endswith(".jpg")]
    has_471_images = len(images) == 471

    has_specific_file = os.path.exists(
        os.path.join(img_folder, "front_rgb_50.0.jpg")
    )

    if has_471_images and has_specific_file:
        dest_file = os.path.join(folder_path, "dist.npz")

        if not os.path.exists(dest_file):
            shutil.copy2(source_file, dest_file)
        else:
            print(f"{i}, file already existed")

        #print(f"Copied to {folder_path}")
    else:
        print(f"{i}, Skipped {folder_path}")

"""

"""
# change folder index
import os

base_dir = "training_data/no_car_noise"
#s_index = 10

for i in range(0, 500):
    old_path = os.path.join(base_dir, str(i))
    #new_path = os.path.join(base_dir, str(i + 500*s_index))
    new_index = 7600 + i
    new_path = os.path.join(base_dir, str(new_index))

    if not os.path.exists(old_path):
        print(f"Skip {old_path} (not found)")
        continue

    if os.path.exists(new_path):
        print(f"Skip {new_path} (already exists)")
        continue

    os.rename(old_path, new_path)
    print(f"Renamed {i} -> {new_index}")
"""


# move folders
import os
import shutil

src_base = "training_data/no_car_noise"
dst_base = "important_training_dataset"

#num = 10

#for i in range(500*(num-1), 500*num):
for i in range(7600, 8100):
    src_path = os.path.join(src_base, str(i))
    dst_path = os.path.join(dst_base, str(i))

    if not os.path.exists(src_path):
        print(f"Skip {src_path} (not found)")
        continue

    if os.path.exists(dst_path):
        print(f"Skip {dst_path} (already exists)")
        continue

    shutil.move(src_path, dst_path)
    #print(f"Moved {src_path} -> {dst_path}")
print("done")


"""
import os
path = 'data'

existing = {d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))}


missing = [i for i in range(7448) if str(i) not in existing]


if missing:
    print(f"missing file{missing}")
else:
    print("all good")

"""
"""
import shutil
import os

path = "/mimer/NOBACKUP/groups/naiss2025-22-1298/containers"

if os.path.exists(path):
    shutil.rmtree(path)
"""