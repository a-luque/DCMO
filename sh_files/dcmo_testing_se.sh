#!/usr/bin/env bash

#SBATCH -A NAISS2025-22-1572 -p alvis

#SBATCH -N 1 --gpus-per-node=T4:1  # We're launching 1 node with 1 Nvidia T4 GPU
#SBATCH -n 2 # NEW
#SBATCH -t 1-00:00:00

#SBATCH --output=/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/dcmo_testing_se.out
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/dcmo_testing_se.err

cd /mimer/NOBACKUP/groups/naiss2024-22-404/MODD/experiments/contextual_MODD/carla/lane_keeping/

srun -n1 --overlap apptainer exec --nv ../../../../carla_container.sif ./CarlaUE4_restart.sh &
sleep 20 
cd /cephyr/users/luque/Alvis/DCMO/src
srun -n1 --overlap apptainer exec --nv /mimer/NOBACKUP/groups/naiss2024-22-404/my_container_verifai3.sif python3 /cephyr/users/luque/Alvis/DCMO/src/evaluation_dcmo.py --scenic /cephyr/users/luque/Alvis/DCMO/src/evaluation_dcmo.scenic --num_steps 300 --num_sim 500 --results_dir /mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/testing_dcmo_se --controllers_folder /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/checkpoints/train_faster --dcmo_path /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/bandit_checkpoint_s_e_snapshots/snapshot_t5000.npz --pi_safe_path /mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/safety_monitor_reduced_1/weights_1000.npy --rulebook_path /mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/rulebook_preferences.pkl --contexts_order /cephyr/users/luque/Alvis/DCMO/stage2/context_cells.csv --eff_first 0 &
wait