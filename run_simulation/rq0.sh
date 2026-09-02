#!/usr/bin/env bash

#SBATCH -A NAISS2025-22-1298 -p alvis

#SBATCH -N 1 --gpus-per-node=A40:1  # We're launching 1 node with 1 Nvidia T4 GPU
#SBATCH -n 2 # NEW
#SBATCH -t 10:00:00

#SBATCH --output=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/new_exp/run_simulations/debug/rq0_100_0_defensive.out
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/new_exp/run_simulations/debug/rq0_100_0_defensive.err

cd /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/new_exp/

srun -n1 --overlap apptainer exec --nv /mimer/NOBACKUP/groups/naiss2025-22-1298/MODD_alj/carla_container.sif ./CarlaUE4_restart.sh &
sleep 5
srun -n1 --overlap apptainer exec --nv /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/containers/my_container.sif python3 run_simulations/rq0.py --controller defensive
wait