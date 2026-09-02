#!/usr/bin/env bash

#SBATCH -A Berzelius-2026-227

#SBATCH --gpus 1
#SBATCH -n 3 # NEW
#SBATCH -t 10:00:00

#SBATCH --output=/proj/berzelius-2026-227/users/x_menwa/CMO_new/new_exp/run_simulations/debug/rq3_1_9.out
#SBATCH --error=/proj/berzelius-2026-227/users/x_menwa/CMO_new/new_exp/run_simulations/debug/rq3_1_9.err

cd /proj/berzelius-2026-227/users/x_menwa/CMO_new/new_exp/

srun -n1 --overlap apptainer exec --nv /proj/berzelius-2026-227/users/x_menwa/containers/carla_container.sif ./CarlaUE4_restart.sh &
sleep 5
srun -n1 --overlap apptainer exec --nv /proj/berzelius-2026-227/users/x_menwa/containers/my_container.sif python3 run_simulations/rq3.py --alpha-eff 0.1 --alpha-comf 0.9 &
sleep 1
srun -n1 --overlap apptainer exec --nv /proj/berzelius-2026-227/users/x_menwa/containers/my_container.sif python3 run_simulations/train_nn.py
wait