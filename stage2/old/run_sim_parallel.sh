#!/usr/bin/env bash

#SBATCH -A NAISS2025-22-1572 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:4      # 4 T4 GPUs on one node (one Carla per GPU)
#SBATCH -n 5                            # 4 Carla processes + 1 Python process
#SBATCH -t 2-00:00:00                   # 2 days (4300 sims left / 4 workers ~ 17hrs, with margin)

#SBATCH --output=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/stage2/run_sim_parallel_es.out
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/stage2/run_sim_parallel_es.err

cd /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/

# ---------------------------------------------------------------------------
# Start one Carla instance per GPU, each on its own RPC port.
# CUDA_VISIBLE_DEVICES pins each instance to its own GPU so they don't
# share VRAM (each T4 has ~15 GB and one Carla uses ~14 GB).
# Carla also needs a matching streaming port = RPC port + 1, and a
# secondary port = RPC port + 2, so we space ports 4 apart to be safe.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 srun -n1 --overlap apptainer exec --nv \
    /mimer/NOBACKUP/groups/naiss2024-22-404/MODD/carla_container.sif \
    ./CarlaUE4_restart.sh -carla-port=2000 &

CUDA_VISIBLE_DEVICES=1 srun -n1 --overlap apptainer exec --nv \
    /mimer/NOBACKUP/groups/naiss2024-22-404/MODD/carla_container.sif \
    ./CarlaUE4_restart.sh -carla-port=2004 &

CUDA_VISIBLE_DEVICES=2 srun -n1 --overlap apptainer exec --nv \
    /mimer/NOBACKUP/groups/naiss2024-22-404/MODD/carla_container.sif \
    ./CarlaUE4_restart.sh -carla-port=2008 &

CUDA_VISIBLE_DEVICES=3 srun -n1 --overlap apptainer exec --nv \
    /mimer/NOBACKUP/groups/naiss2024-22-404/MODD/carla_container.sif \
    ./CarlaUE4_restart.sh -carla-port=2012 &

# ---------------------------------------------------------------------------
# Wait for all 4 Carla instances to finish starting up.
# 4 instances take longer than 1, so we give 120s instead of the old 30s.
# Increase this if you see "Connection refused" errors at the start of the run.
# ---------------------------------------------------------------------------
echo "Waiting for Carla instances to start..."
sleep 120

# ---------------------------------------------------------------------------
# Run the bandit. The Python script reads the checkpoint and resumes
# automatically if a previous run was interrupted.
# The port list here must match the ports launched above.
# ---------------------------------------------------------------------------
echo "Starting bandit..."
srun -n1 --overlap apptainer exec --nv \
    /mimer/NOBACKUP/groups/naiss2024-22-404/my_container_verifai3.sif \
    python3 stage2/alg_parallel.py
wait