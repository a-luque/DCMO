#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1298 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -n 1
#SBATCH -t 1-00:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/train_turn_resnet18_c_20.out
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/train_turn_resnet18_c_20.err

cd /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/

srun -n1 --overlap apptainer exec --nv \
    /mimer/NOBACKUP/groups/naiss2024-22-404/my_container_verifai3.sif \
    python3 train_turn.py \
        --backbone      resnet18 \
        --granularity   coarse \
        --root_dir      data \
        --epochs        10 \
        --batch_size    32 \
        --lr            1e-4 \
        --lambda_cte    20.0 \
        --lambda_dist   1.0 \
        --img_height    112 \
        --img_width     224 \
        --total_runs    3000 \
        --num_workers   2 \
        --output_dir    /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/checkpoints/turn_resnet18_c_20

wait