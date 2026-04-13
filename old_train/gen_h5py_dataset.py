#python3 gen_h5py_dataset.py --root_dir important_training_dataset --output /mimer/NOBACKUP/groups/naiss2024-22-1336/CMO/datasets/dataset_fine.h5 --granularity fine
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import h5py
import concurrent.futures
import os

OFFSET_M = 4.6
NO_CAR_SENTINEL = 100.0

GRANULARITIES = {
    "coarse": {
        "num_classes": 6,
        "bin_edges":  [10, 20, 30, 40, 50],
    },
    "medium": {
        "num_classes": 11,
        "bin_edges":  [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    },
    "fine": {
        "num_classes": 26,
        "bin_edges":  list(range(2, 52, 2)),
    },
}

def write_split_streaming(h5file, split_name, runs, args):
    grp = h5file.create_group(split_name)
    
    dset_img = grp.create_dataset("images", shape=(0, args.img_height, args.img_width, 3), 
                                  maxshape=(None, args.img_height, args.img_width, 3), 
                                  chunks=(64, args.img_height, args.img_width, 3), 
                                  dtype='uint8', compression="lzf")
    dset_cte = grp.create_dataset("cte", shape=(0,), maxshape=(None,), dtype='float32')
    dset_label = grp.create_dataset("dist_label", shape=(0,), maxshape=(None,), dtype='int16')
    dset_man = grp.create_dataset("maneuver", shape=(0,), maxshape=(None,), dtype='int8')

    current_idx = 0
    run_infos = [(r, args.root_dir, args.granularity, args.img_height, args.img_width) for r in runs]
    
    max_workers = min(os.cpu_count(), 24)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:

        future_to_run = {executor.submit(process_single_run, info): info for info in run_infos}
        
        pbar = tqdm(total=len(runs), desc=f"Packing {split_name}")
        
        for future in concurrent.futures.as_completed(future_to_run):
            res = future.result()
            pbar.update(1)
            
            if res is None:
                continue
            
            imgs, ctes, labels, mans = res
            num_new = len(imgs)
            new_size = current_idx + num_new
            
            dset_img.resize(new_size, axis=0)
            dset_cte.resize(new_size, axis=0)
            dset_label.resize(new_size, axis=0)
            dset_man.resize(new_size, axis=0)
            
            dset_img[current_idx:new_size] = imgs
            dset_cte[current_idx:new_size] = ctes
            dset_label[current_idx:new_size] = labels
            dset_man[current_idx:new_size] = mans
            
            current_idx = new_size
            
            del res, imgs, ctes, labels, mans
            
        pbar.close()

def distance_to_label(distance: float, granularity: str) -> int:
    cfg = GRANULARITIES[granularity]
    if distance >= 100:
        return cfg["num_classes"] - 1
    for i, edge in enumerate(cfg["bin_edges"]):
        if distance <= edge:
            return i
    return cfg["num_classes"] - 1

def process_distances(raw):
    distances = np.clip(raw - OFFSET_M, a_min=0.0, a_max=None)
    distances[distances > 50.0] = NO_CAR_SENTINEL
    return distances

def split_runs(total_runs, val_ratio=0.15, test_ratio=0.10, seed=42):
    rng = np.random.default_rng(seed)
    runs = rng.permutation(total_runs).tolist()
    n_test = int(total_runs * test_ratio)
    n_val = int(total_runs * val_ratio)
    test_runs = runs[:n_test]
    val_runs = runs[n_test : n_test + n_val]
    train_runs = runs[n_test + n_val:]
    return train_runs, val_runs, test_runs


def process_single_run(run_info):
    run_idx, root_dir, granularity, img_height, img_width = run_info
    root = Path(root_dir)
    run_dir = root / str(run_idx)
    
    try:
        dist_path = run_dir / "dist.npz"
        cte_path = run_dir / "cte.npz"
        maneuver_path = run_dir / "maneuver.npz"
        img_dir = run_dir / "img"

        if not (dist_path.exists() and cte_path.exists() and maneuver_path.exists()):
            return None

        raw = np.load(dist_path)["values"].astype(np.float32)
        distances = process_distances(raw)
        ctes = np.load(cte_path)["values"].astype(np.float32)
        maneuvers = np.load(maneuver_path)["values"].astype(np.int8)

        img_paths = sorted(
            img_dir.glob("front_rgb_*.jpg"),
            key=lambda p: float(p.stem.replace("front_rgb_", ""))
        )
        
        n = min(len(img_paths), len(distances), len(ctes), len(maneuvers))
        
        run_images, run_ctes, run_labels, run_maneuvers = [], [], [], []

        for i in range(n):
            img = Image.open(img_paths[i]).convert("RGB")
            img = img.resize((img_width, img_height), Image.BILINEAR)
            run_images.append(np.array(img, dtype=np.uint8))
            run_ctes.append(float(ctes[i]))
            run_labels.append(distance_to_label(float(distances[i]), granularity))
            run_maneuvers.append(int(maneuvers[i]))

        if not run_images:
            return None

        return (np.stack(run_images), np.array(run_ctes, dtype=np.float32), 
                np.array(run_labels, dtype=np.int16), np.array(run_maneuvers, dtype=np.int8))
    except Exception:
        return None

def write_split_streaming(h5file, split_name, runs, args):
    grp = h5file.create_group(split_name)
    
    # Initialize datasets
    dset_img = grp.create_dataset("images", shape=(0, args.img_height, args.img_width, 3), 
                                  maxshape=(None, args.img_height, args.img_width, 3), 
                                  chunks=(64, args.img_height, args.img_width, 3), 
                                  dtype='uint8', compression="lzf")
    dset_cte = grp.create_dataset("cte", shape=(0,), maxshape=(None,), dtype='float32')
    dset_label = grp.create_dataset("dist_label", shape=(0,), maxshape=(None,), dtype='int16')
    dset_man = grp.create_dataset("maneuver", shape=(0,), maxshape=(None,), dtype='int8')

    current_idx = 0
    run_infos = [(r, args.root_dir, args.granularity, args.img_height, args.img_width) for r in runs]
    
    # Use a conservative number of workers to keep RAM usage stable
    max_workers = min(os.cpu_count(), 8)
    # Define batch size: how many runs to keep in memory at once
    batch_size = 50 

    pbar = tqdm(total=len(runs), desc=f"Packing {split_name}")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Loop through run_infos in steps of batch_size
        for i in range(0, len(run_infos), batch_size):
            batch = run_infos[i : i + batch_size]
            
            # Submit only the current batch to the executor
            future_to_run = {executor.submit(process_single_run, info): info for info in batch}
            
            for future in concurrent.futures.as_completed(future_to_run):
                res = future.result()
                pbar.update(1)
                
                if res is None:
                    continue
                
                imgs, ctes, labels, mans = res
                num_new = len(imgs)
                new_size = current_idx + num_new
                
                # Resize and write to the same single H5 file
                dset_img.resize(new_size, axis=0)
                dset_cte.resize(new_size, axis=0)
                dset_label.resize(new_size, axis=0)
                dset_man.resize(new_size, axis=0)
                
                dset_img[current_idx:new_size] = imgs
                dset_cte[current_idx:new_size] = ctes
                dset_label[current_idx:new_size] = labels
                dset_man[current_idx:new_size] = mans
                
                current_idx = new_size
                
                # Explicitly delete the local reference to large arrays to help GC
                del res, imgs, ctes, labels, mans
                
    pbar.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",    default="data")
    parser.add_argument("--output",      default="dataset_fine.h5")
    parser.add_argument("--granularity", default="fine")
    parser.add_argument("--total_runs",  type=int, default=5000)
    parser.add_argument("--img_height",  type=int, default=112)
    parser.add_argument("--img_width",   type=int, default=224)
    args = parser.parse_args()

    train_runs, val_runs, test_runs = split_runs(args.total_runs)
    print(f"Split: {len(train_runs)} train / {len(val_runs)} val / {len(test_runs)} test runs")

    with h5py.File(args.output, "w") as f:
        f.attrs["granularity"] = args.granularity
        f.attrs["img_height"]  = args.img_height
        f.attrs["img_width"]   = args.img_width
        
        for name, runs in [("train", train_runs), ("val", val_runs), ("test", test_runs)]:
            if len(runs) > 0:
                write_split_streaming(f, name, runs, args)

    size_gb = os.path.getsize(args.output) / 1e9
    print(f"\nSuccessfully packed into {args.output} ({size_gb:.2f} GB)")

if __name__ == "__main__":
    main()