#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1298 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1        # T4 -> A100 for ~6x speedup
#SBATCH -n 1
#SBATCH -t 1-00:00:00                    # 8h is plenty for 30 epochs on A100
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/train_faster/train_resnet101_c.out
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/debug_outputs/train_faster/train_resnet101_c.err

H5_FILENAME="dataset_packed.h5"
REMOTE_H5="/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/h5py_dataset/$H5_FILENAME"
LOCAL_H5="$TMPDIR/$H5_FILENAME"

if [ ! -f "$LOCAL_H5" ]; then
    echo "Copying dataset to local scratch: $TMPDIR"
    cp $REMOTE_H5 $LOCAL_H5
else
    echo "Dataset already exists, skip copy"
fi

cd /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/

srun -n1 --overlap apptainer exec --nv \
    /mimer/NOBACKUP/groups/naiss2024-22-404/my_container_verifai_h5.sif \
    python3 -u train_faster.py \
        --h5_file       $LOCAL_H5 \
        --backbone      resnet101 \
        --granularity   coarse \
        --epochs        30 \
        --batch_size    256 \
        --lr            8e-4 \
        --lambda_cte    100.0 \
        --lambda_dist   1.0 \
        --num_workers   4 \
        --amp \
        --output_dir    /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/checkpoints/train_faster/resnet101_c

wait