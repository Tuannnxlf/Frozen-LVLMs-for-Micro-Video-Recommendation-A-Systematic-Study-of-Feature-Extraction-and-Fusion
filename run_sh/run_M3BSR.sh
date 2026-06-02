#!/bin/bash
cd /opt/data/private/work/vllmembs4rec/sasrec || exit 1
seeds=(2025)
lrs=(1e-5)
drop_rates=(0.1)
weight_decays=(0.1)
embedding_dims=(2048)

for seed in "${seeds[@]}"
do
    run_id="M3BSR_v1_seed_${seed}"

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=$((12648 + ${RANDOM} % 10000)) \
        main.py \
        --run_id="$run_id" \
        --seed="$seed" \
        --model="sasrec" \
        --epoch=100 \
        --lr=1e-5 \
        --batch_size=128 \
        --diffusion_loss_weight=0.1 \
        --eval_num=1 \
        --method='M3BSR'
done