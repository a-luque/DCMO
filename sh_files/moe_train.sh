#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1298 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1        # T4 -> A100 for ~6x speedup
#SBATCH -n 1
#SBATCH -t 1-00:00:00                    # 8h is plenty for 30 epochs on A100
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/moe_train.out
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/moe_train.err

H5_FILENAME="moe.h5"
REMOTE_H5="/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/$H5_FILENAME"
LOCAL_H5="$TMPDIR/$H5_FILENAME"

if [ ! -f "$LOCAL_H5" ]; then
    echo "Copying dataset to local scratch: $TMPDIR"
    cp $REMOTE_H5 $LOCAL_H5
else
    echo "Dataset already exists, skip copy"
fi

cd /cephyr/users/luque/Alvis/DCMO/src

srun -n1 --overlap apptainer exec --nv \
    /mimer/NOBACKUP/groups/naiss2024-22-404/my_container_verifai_h5.sif \
    python3 -u train_MOE.py \
        --h5_file       $LOCAL_H5 \
		--controllers_dir /mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/checkpoints/train_faster \
        --epochs        30 \
        --batch_size    256 \
        --lr            8e-4 \
        --lambda_cte    100.0 \
        --lambda_dist   1.0 \
        --num_workers   4 \
        --amp \
        --output_dir    /mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/train_MOE
