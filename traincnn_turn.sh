#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1298 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -n 1
#SBATCH -t 1-00:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/cnn_turn/train_resnet18_c.out
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/cnn_turn/train_resnet18_c.err

cd /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/

srun -n1 --overlap apptainer exec --nv \
    /mimer/NOBACKUP/groups/naiss2024-22-404/my_container_verifai3.sif \
    python3 train_turn.py \
        --backbone      resnet18 \
        --granularity   coarse \
        --root_dir      important_training_dataset \
        --epochs        30 \
        --batch_size    256 \
        --lr            8e-4 \
        --lambda_cte    100.0 \
        --lambda_dist   1.0 \
        --img_height    112 \
        --img_width     224 \
        --total_runs    8100 \
        --num_workers   8 \
        --output_dir    /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/checkpoints/cnn_turn/resnet18_c

wait