#!/usr/bin/env bash

#SBATCH -A NAISS2025-22-1572 -p alvis

#SBATCH -N 1 --gpus-per-node=T4:1  # We're launching 1 node with 1 Nvidia T4 GPU
#SBATCH -n 2 # NEW
#SBATCH -t 1-00:00:00

#SBATCH --output=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/new_exp/debug/data_gen/moderate/hardrain_noon/515.out

#SBATCH --error=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/new_exp/debug/data_gen/moderate/hardrain_noon/515.err

cd /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/new_exp/data_gen/

srun -n1 --overlap apptainer exec --nv /mimer/NOBACKUP/groups/naiss2024-22-404/MODD/carla_container.sif ./CarlaUE4_restart.sh &
sleep 20
srun -n1 --overlap apptainer exec --nv /mimer/NOBACKUP/groups/naiss2024-22-404/my_container_verifai_h5.sif python3 gen_data.py
wait