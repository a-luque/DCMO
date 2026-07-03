#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1298 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -n 1
#SBATCH -t 1-00:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/new_exp/debug/train/cnn/moderate.out
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/new_exp/debug/train/cnn/moderate.err

H5_FILENAME="moderate"

REMOTE_H5="/cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/experiments/new_exp/train_controller/data/${H5_FILENAME}.h5"
LOCAL_H5="$TMPDIR/${H5_FILENAME}.h5"

if [ ! -f "$LOCAL_H5" ]; then
    echo "Copying dataset to local scratch: $TMPDIR"
    cp "$REMOTE_H5" "$LOCAL_H5"
else
    echo "Dataset already exists, skip copy"
fi

cd /cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/experiments/new_exp/train_controller

srun -n1 --overlap apptainer exec --nv \
    /mimer/NOBACKUP/groups/naiss2024-22-404/my_container_verifai_h5.sif \
    python3 train_cnn.py \
        --h5_file "$LOCAL_H5" \
        --epochs 15 \
        --batch_size 64 \
        --lr 4e-4 \
        --workers 4 \
        --dist_weight  0.005 \
        --out_dir "/cephyr/users/mengyuan/Alvis/Desktop/mimer_naiss2025-22-1298/CMO/experiments/new_exp/train_controller/checkpoints/cnn/${H5_FILENAME}"

wait