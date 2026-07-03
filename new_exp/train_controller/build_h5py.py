"""
# python3 build_h5py.py --root_dir "/cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/experiments/new_exp/data_gen/data/aggressive" --out_file "/cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/experiments/new_exp/train_controller/data/aggressive.h5"
"""

import argparse
import io
import numpy as np
from pathlib import Path
from PIL import Image
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def load_npz(path: Path) -> np.ndarray:
    return np.load(path)["values"]


def process_single_run(run_info):
    run_dir, n, img_height, img_width, jpeg_quality = run_info
    pil_size = (img_width, img_height)
    run_dir = Path(run_dir)

    raw = load_npz(run_dir / "dist.npz").astype(np.float32)
    distances = np.clip(raw - 4.6, a_min=0.0, a_max=None)
    #distances[distances > 50.0] = 100.0

    ctes         = load_npz(run_dir / "cte.npz").astype(np.float32)
    maneuvers = load_npz(run_dir / "maneuver.npz").astype(np.int64) - 1 # 1,2,3 to 0,1,2
    leader_speed = load_npz(run_dir / "leader_speed.npz").astype(np.float32)
    acc          = load_npz(run_dir / "acc.npz").astype(np.float32)
    ego_speed    = load_npz(run_dir / "ego.npz").astype(np.float32)

    img_paths = sorted(
        (run_dir / "img").glob("front_rgb_*.jpg"),
        key=lambda p: float(p.stem.replace("front_rgb_", ""))
    )[:n]

    encoded_images = []
    for img_path in img_paths:
        with Image.open(img_path).convert("RGB") as img:
            img_resized = img.resize(pil_size, resample=Image.BILINEAR)
            buf = io.BytesIO()
            img_resized.save(buf, format="JPEG", quality=jpeg_quality)
            encoded_images.append(np.frombuffer(buf.getvalue(), dtype=np.uint8).copy())

    return {
        "images":       encoded_images,
        "cte":          ctes[:n],
        "dist":         distances[:n],
        "maneuver":     maneuvers[:n],
        "leader_speed": leader_speed[:n],
        "acc":          acc[:n],
        "ego_speed":    ego_speed[:n],
        "count":        n,
    }


def discover_run_dirs(root: Path) -> list[Path]:
    """
    Walk the nested structure rooted at a single driving-style directory:
        root / <weather> / <scenario> / <run_id (0-99)>

    e.g. root = .../data/aggressive
              aggressive / ClearNoon / 515 / 0
              aggressive / ClearNoon / 515 / 1
              aggressive / HardRainNoon / 3545 / 99
              ...

    Returns a sorted list of leaf run directories.
    """
    run_dirs = []
    for weather_dir in sorted(root.iterdir()):
        if not weather_dir.is_dir():
            continue
        for scenario_dir in sorted(weather_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            for run_dir in sorted(
                scenario_dir.iterdir(),
                key=lambda p: int(p.name) if p.name.isdigit() else -1,
            ):
                if not run_dir.is_dir() or not run_dir.name.isdigit():
                    continue
                run_dirs.append(run_dir)

    print(f"  Discovered {len(run_dirs)} candidate run directories under {root}")
    return run_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",     required=True)
    parser.add_argument("--out_file",     required=True)
    parser.add_argument("--img_height",   type=int, default=112)
    parser.add_argument("--img_width",    type=int, default=224)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--workers",      type=int, default=min(8, os.cpu_count()),
                        help="Number of worker processes. Default: min(8, cpu_count). "
                             "Don't set this to os.cpu_count() — leave headroom to avoid OOM.")
    args = parser.parse_args()

    root = Path(args.root_dir)
    run_dirs = discover_run_dirs(root)

    # ── Pre-scan: only include runs with all required files ───────────────
    print("Pre-scanning directories...")
    valid_tasks = []
    total_samples = 0

    required_npz = ["dist.npz", "cte.npz", "maneuver.npz",
                    "leader_speed.npz", "acc.npz", "ego.npz"]

    for run_dir in run_dirs:
        img_dir = run_dir / "img"

        if not img_dir.exists() or not all((run_dir / f).exists() for f in required_npz):
            continue

        n_imgs = len(list(img_dir.glob("front_rgb_*.jpg")))
        lengths = [n_imgs] + [len(load_npz(run_dir / f)) for f in required_npz]
        n = min(lengths)

        if n == 0:
            continue

        valid_tasks.append((str(run_dir), n, args.img_height, args.img_width, args.jpeg_quality))
        total_samples += n

    print(f"Total: {total_samples:,} samples across {len(valid_tasks)} runs. "
          f"Using {args.workers} workers.")

    if total_samples == 0:
        print("ERROR: No valid samples found. Check that --root_dir points to the "
              "driving-style folder (e.g. .../data/aggressive) and that each leaf "
              "run directory contains img/ and all required .npz files.")
        return

    # ── Write HDF5 ────────────────────────────────────────────────────────
    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(args.out_file, "w") as f:
        ds_images       = f.create_dataset("images",       shape=(total_samples,), dtype=vlen_uint8, chunks=(16,))
        ds_cte          = f.create_dataset("cte",          shape=(total_samples,), dtype=np.float32)
        ds_dist         = f.create_dataset("dist",         shape=(total_samples,), dtype=np.float32)
        ds_maneuver     = f.create_dataset("maneuver",     shape=(total_samples,), dtype=np.int64)
        ds_leader_speed = f.create_dataset("leader_speed", shape=(total_samples,), dtype=np.float32)
        ds_acc          = f.create_dataset("acc",          shape=(total_samples,), dtype=np.float32)
        ds_ego_speed    = f.create_dataset("ego_speed",    shape=(total_samples,), dtype=np.float32)

        f.attrs["img_height"] = args.img_height
        f.attrs["img_width"]  = args.img_width

        idx = 0
        completed = 0
        results_store = {}
        next_to_write = 0

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task_idx = {
                executor.submit(process_single_run, task): i
                for i, task in enumerate(valid_tasks)
            }

            for future in as_completed(future_to_task_idx):
                task_idx = future_to_task_idx[future]
                results_store[task_idx] = future.result()

                while next_to_write in results_store:
                    result = results_store.pop(next_to_write)
                    n = result["count"]

                    for i in range(n):
                        ds_images[idx + i] = result["images"][i]

                    ds_cte         [idx:idx + n] = result["cte"]
                    ds_dist        [idx:idx + n] = result["dist"]
                    ds_maneuver    [idx:idx + n] = result["maneuver"]
                    ds_leader_speed[idx:idx + n] = result["leader_speed"]
                    ds_acc         [idx:idx + n] = result["acc"]
                    ds_ego_speed   [idx:idx + n] = result["ego_speed"]

                    idx += n
                    next_to_write += 1
                    completed += 1
                    print(f"  [{completed}/{len(valid_tasks)}] runs | "
                          f"{idx:,}/{total_samples:,} samples", end="\r")

    size_gb = os.path.getsize(args.out_file) / 1e9
    print(f"\nDone. {idx:,} samples -> {args.out_file}  ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()