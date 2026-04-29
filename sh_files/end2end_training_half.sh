#!/usr/bin/env bash

#SBATCH -A NAISS2025-22-1572 -p alvis

#SBATCH -N 1 --gpus-per-node=T4:1  # We're launching 1 node with 1 Nvidia T4 GPU
#SBATCH -n 2 # NEW
#SBATCH -t 1-00:00:00

#SBATCH --output=/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/end2end_training_half.out
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/end2end_training_half.err

cd /cephyr/users/luque/Alvis/DCMO/src
srun -n1 --overlap apptainer exec --nv /mimer/NOBACKUP/groups/naiss2024-22-404/my_container_verifai3.sif python3 end2end_half.py