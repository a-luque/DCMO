#!/usr/bin/env bash

#SBATCH -A NAISS2025-22-1298 -p alvis

#SBATCH -N 1 --gpus-per-node=T4:1  # We're launching 1 node with 1 Nvidia T4 GPU
#SBATCH -n 2 # NEW
#SBATCH -t 1-00:00:00

#SBATCH --output=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/data_gen/no_car.out

#SBATCH --error=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/data_gen/no_car.err

cd /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/

srun -n1 --overlap apptainer exec --nv /mimer/NOBACKUP/groups/naiss2024-22-404/MODD/carla_container.sif ./CarlaUE4_restart.sh &
sleep 20
srun -n1 --overlap apptainer exec --nv /mimer/NOBACKUP/groups/naiss2024-22-404/my_container_verifai3.sif python3 data_gen_nocar.py
wait


