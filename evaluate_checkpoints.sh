#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1572 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -n 1
#SBATCH -t 1-00:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/evaluation/resnet101_c.out
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/evaluation/resnet101_c.err


H5_FILE=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/h5py_dataset/dataset_packed.h5

cd /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/

srun -n1 --overlap apptainer exec --nv \
    /mimer/NOBACKUP/groups/naiss2024-22-404/my_container_verifai_h5.sif \
    python3 evaluate_checkpoints.py \
        --checkpoint checkpoints/train_faster/resnet101_c/resnet101_coarse_best.pt \
        --h5_file    h5py_dataset/dataset_packed.h5 \
        --batch_size 256 \
        --num_workers 4
wait