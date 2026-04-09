import numpy as np


path = "test_controller_sim_results/0/maneuver.npz"

data = np.load(path)

for key in data.files:
    print(key, data[key])
    print(len(data[key]))
    print(min(data[key]))



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

base_dir = "simulation_results/3545"
s_index = 1500

for i in range(78, 500):
    old_path = os.path.join(base_dir, str(i))
    new_path = os.path.join(base_dir, str(i + s_index))

    if not os.path.exists(old_path):
        print(f"Skip {old_path} (not found)")
        continue

    if os.path.exists(new_path):
        print(f"Skip {new_path} (already exists)")
        continue

    os.rename(old_path, new_path)
    print(f"Renamed {i} -> {i+s_index}")

"""
"""
# move folders
import os
import shutil

src_base = "simulation_results/no_car"
dst_base = "data"

num = 6

for i in range(500*(num-1), 500*num):
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