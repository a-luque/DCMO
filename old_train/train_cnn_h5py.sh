#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1298 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -n 1
#SBATCH -t 1-00:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/cnn_h5py/train_resnet18_f.out
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/cnn_h5py/train_resnet18_f.err

# Path to the pre-packed HDF5 file (created once with pack_dataset.py).
# Storing it on GPFS is fine — HDF5 is a single large file and reads
# sequentially, so GPFS handles it well. No rsync needed.
H5_FILE=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/h5py_dataset/dataset_packed.h5

cd /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/

srun -n1 --overlap apptainer exec --nv \
    /mimer/NOBACKUP/groups/naiss2024-22-404/my_container_verifai_h5.sif \
    python3 train_cnn_h5py.py \
        --h5_file       "$H5_FILE" \
        --backbone      resnet18 \
        --granularity   fine \
        --epochs        30 \
        --batch_size    256 \
        --lr            8e-4 \
        --lambda_cte    100.0 \
        --lambda_dist   1.0 \
        --num_workers   4 \
        --output_dir    /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/checkpoints/cnn_h5py/resnet18_f

wait