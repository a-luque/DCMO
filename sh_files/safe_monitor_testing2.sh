#!/usr/bin/env bash

#SBATCH -A NAISS2025-22-1572 -p alvis

#SBATCH -N 1 --gpus-per-node=T4:1  # We're launching 1 node with 1 Nvidia T4 GPU
#SBATCH -n 2 # NEW
#SBATCH -t 1-00:00:00

#SBATCH --output=/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/safety_monitor_testing_reduced_2.out
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/safety_monitor_testing_reduced_2.err

cd /mimer/NOBACKUP/groups/naiss2024-22-404/MODD/experiments/contextual_MODD/carla/lane_keeping/

srun -n1 --overlap apptainer exec --nv ../../../../carla_container.sif ./CarlaUE4_restart.sh &
sleep 20 
cd /cephyr/users/luque/Alvis/DCMO/src
srun -n1 --overlap apptainer exec --nv /mimer/NOBACKUP/groups/naiss2024-22-404/my_container_verifai3.sif python3 /cephyr/users/luque/Alvis/DCMO/src/safety_monitor_training_testing.py --num_steps 300 --results_dir /mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/safety_monitor_testing_reduced_2 --log_path /mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/safety_monitor_testing_reduced_2/log.csv --controllers_folder /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/checkpoints/train_faster --i_init 0 --monitor /mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/safety_monitor_reduced_2/weights_1000.npy --training 0 &
wait