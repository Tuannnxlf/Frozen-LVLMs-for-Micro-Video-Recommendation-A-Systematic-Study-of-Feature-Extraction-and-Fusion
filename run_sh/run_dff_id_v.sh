#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1

seeds=(2025)
lr=1e-5
batch_size=512

for seed in "${seeds[@]}"
do
    run_id="dff_id_v_s${seed}"

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port=$((12648 + RANDOM % 10000)) \
        main.py \
        --run_id="$run_id" \
        --seed="$seed" \
        --model="sasrec" \
        --epoch=100 \
        --lr="$lr" \
        --batch_size="$batch_size" \
        --eval_num=1
done
