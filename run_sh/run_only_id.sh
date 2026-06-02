#!/bin/bash
# dff(id+a)+condition diffusion(v)
cd /opt/data/private/work/vllmembs4rec/sasrec || exit 1
seeds=(2025)
lrs=(1e-5)
drop_rates=(0.1)
weight_decays=(0.1)
embedding_dims=(2048)

# for seed in "${seeds[@]}"
# do
#     run_id="onlyid_tuan_seed_${seed}"

#     CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=$((12648 + ${RANDOM} % 10000)) \
#         main.py \
#         --run_id="$run_id" \
#         --seed="$seed" \
#         --model="sasrec" \
#         --epoch=100 \
#         --lr=1e-5 \
#         --batch_size=512 \
#         --eval_num=1 \
#         --method='only_id'
# done

for seed in "${seeds[@]}"
do
    run_id="diffusion_id_v_con_a_tuan_seed${seed}"

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=$((12648 + ${RANDOM} % 10000)) \
        main.py \
        --run_id="$run_id" \
        --seed="$seed" \
        --model="sasrec" \
        --epoch=100 \
        --lr=1e-5 \
        --batch_size=512 \
        --eval_num=1 \
        --method='only_id' \
        --mode='print' \
        --infer_ckpt='/opt/data/private/work/vllmembs4rec/sasrec/checkpoint/checkpoint_dff_id_v_tuan_seed_2025/cpt__bs_512_ed_2048_lr_1e-05_l2_0.1_maxLen_10/epoch-46.pt'
done