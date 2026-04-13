import argparse
import io
import numpy as np
from pathlib import Path
from PIL import Image
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def process_single_run(run_info):
    run_dir, n, img_height, img_width, jpeg_quality = run_info
    # NOTE: PIL's resize() takes (width, height) — opposite of numpy/torchvision.
    # We keep (img_width, img_height) here, which is correct.
    pil_size = (img_width, img_height)

    run_dir = Path(run_dir)  # re-wrap after pickling

    raw = np.load(run_dir / "dist.npz")["values"].astype(np.float32)
    distances = np.clip(raw - 4.6, a_min=0.0, a_max=None)
    distances[distances > 50.0] = 100.0
    ctes      = np.load(run_dir / "cte.npz")["values"].astype(np.float32)
    maneuvers = np.load(run_dir / "maneuver.npz")["values"].astype(np.int64)

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
            # .copy() is required — frombuffer gives a read-only view of the
            # BytesIO buffer which becomes invalid once buf goes out of scope
            encoded_images.append(np.frombuffer(buf.getvalue(), dtype=np.uint8).copy())

    return {
        "images":   encoded_images,
        "cte":      ctes[:n],
        "dist":     distances[:n],
        "maneuver": maneuvers[:n],
        "count":    n,
    }


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
    run_dirs = sorted(root.glob("[0-9]*"), key=lambda p: int(p.name))

    # ── Pre-scan: only include runs with all required files ───────────────
    print("Pre-scanning directories...")
    valid_tasks = []
    total_samples = 0
    for run_dir in run_dirs:
        dist_path     = run_dir / "dist.npz"
        cte_path      = run_dir / "cte.npz"
        maneuver_path = run_dir / "maneuver.npz"
        img_dir       = run_dir / "img"

        if not (dist_path.exists() and cte_path.exists()
                and maneuver_path.exists() and img_dir.exists()):
            continue

        n_imgs = len(list(img_dir.glob("front_rgb_*.jpg")))
        n_dist = len(np.load(dist_path)["values"])
        n_cte  = len(np.load(cte_path)["values"])
        n_man  = len(np.load(maneuver_path)["values"])
        n = min(n_imgs, n_dist, n_cte, n_man)

        if n == 0:
            continue

        # Pass run_dir as string — Path objects don't always pickle cleanly
        valid_tasks.append((str(run_dir), n, args.img_height, args.img_width, args.jpeg_quality))
        total_samples += n

    print(f"Total: {total_samples:,} samples across {len(valid_tasks)} runs. "
          f"Using {args.workers} workers.")

    # ── Write HDF5 ────────────────────────────────────────────────────────
    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(args.out_file, "w") as f:
        ds_images   = f.create_dataset("images",   shape=(total_samples,), dtype=vlen_uint8, chunks=(16,))
        ds_cte      = f.create_dataset("cte",      shape=(total_samples,), dtype=np.float32)
        ds_dist     = f.create_dataset("dist",     shape=(total_samples,), dtype=np.float32)
        ds_maneuver = f.create_dataset("maneuver", shape=(total_samples,), dtype=np.int64)

        f.attrs["img_height"] = args.img_height
        f.attrs["img_width"]  = args.img_width

        # ── Submit all jobs; write results as they complete ───────────────
        # Using as_completed() instead of executor.map() means we write each
        # run the moment it finishes rather than buffering ALL results in RAM
        # simultaneously (5300 runs × 220 images would easily OOM).
        #
        # Results arrive out-of-order, so we buffer them in results_store and
        # flush in task-index order to keep HDF5 writes sequential.
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

                # Flush any contiguous block that is ready to write
                while next_to_write in results_store:
                    result = results_store.pop(next_to_write)
                    n = result["count"]

                    for i in range(n):
                        ds_images[idx + i] = result["images"][i]

                    ds_cte[idx:idx + n]      = result["cte"]
                    ds_dist[idx:idx + n]     = result["dist"]
                    ds_maneuver[idx:idx + n] = result["maneuver"]

                    idx += n
                    next_to_write += 1
                    completed += 1
                    print(f"  [{completed}/{len(valid_tasks)}] runs | "
                          f"{idx:,}/{total_samples:,} samples", end="\r")

    size_gb = os.path.getsize(args.out_file) / 1e9
    print(f"\nDone. {idx:,} samples -> {args.out_file}  ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()