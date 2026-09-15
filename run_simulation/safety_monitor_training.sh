#!/usr/bin/env bash

#SBATCH -A Berzelius-2026-227

#SBATCH --gpus 1
#SBATCH -n 3 # NEW
#SBATCH -t 20:00:00

#SBATCH --output=/proj/berzelius-2026-227/users/x_alluq/CMO/safety_monitor_training/debug/safety_monitor_train.out
#SBATCH --error=/proj/berzelius-2026-227/users/x_alluq/CMO/safety_monitor_training/debug/safety_monitor_train.err

cd /proj/berzelius-2026-227/users/x_alluq/DCMO/

srun -n1 --overlap apptainer exec --nv /proj/berzelius-2026-227/users/x_alluq/containers/carla_container0_9_16.sif ./CarlaUE4_restart.sh &
sleep 5
srun -n1 --overlap apptainer exec --nv /proj/berzelius-2026-227/users/x_alluq/containers/my_container.sif python3 run_simulation/train_safe_monitor.py  &
sleep 1
srun -n1 --overlap apptainer exec --nv /proj/berzelius-2026-227/users/x_alluq/containers/my_container.sif python3 run_simulation/train_nn.py
wait