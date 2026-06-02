#!/bin/bash
cd /opt/data/private/work/vllmembs4rec/sasrec || exit 1
seeds=(2025)
lrs=(1e-5)
drop_rates=(0.1)
weight_decays=(0.1)
embedding_dims=(2048)

# # h1
# for seed in "${seeds[@]}"
# do
#     run_id="diffusion_id_v_con_a_tuan_seed${seed}"

#     CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=$((12648 + ${RANDOM} % 10000)) \
#         main.py \
#         --run_id="$run_id" \
#         --seed="$seed" \
#         --model="sasrec" \
#         --epoch=100 \
#         --lr=1e-5 \
#         --batch_size=512 \
#         --diffusion_loss_weight=0.1 \
#         --eval_num=1 \
#         --method='dff_diffusion_id_v_condition_a' \
#         --timesteps=4 \
#         --beta_start=0.001 \
#         --beta_end=0.05 \
#         --w=0.9
# done

# h2
# for seed in "${seeds[@]}"
# do
#     run_id="diffusion_id_v_con_a_h2_seed${seed}"

#     CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=$((12648 + ${RANDOM} % 10000)) \
#         main.py \
#         --run_id="$run_id" \
#         --seed="$seed" \
#         --model="sasrec" \
#         --epoch=100 \
#         --lr=1e-5 \
#         --batch_size=512 \
#         --diffusion_loss_weight=0.1 \
#         --eval_num=1 \
#         --method='dff_diffusion_id_v_condition_a' \
#         --timesteps=4 \
#         --beta_start=0.001 \
#         --beta_end=0.05 \
#         --w=0.7
# done

# h3
# for seed in "${seeds[@]}"
# do
#     run_id="diffusion_id_v_con_a_h3_seed${seed}"

#     CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=$((12648 + ${RANDOM} % 10000)) \
#         main.py \
#         --run_id="$run_id" \
#         --seed="$seed" \
#         --model="sasrec" \
#         --epoch=100 \
#         --lr=1e-5 \
#         --batch_size=512 \
#         --diffusion_loss_weight=0.1 \
#         --eval_num=1 \
#         --method='dff_diffusion_id_v_condition_a' \
#         --timesteps=4 \
#         --beta_start=0.001 \
#         --beta_end=0.05 \
#         --w=0.8
# done

# h4
for seed in "${seeds[@]}"
do
    run_id="diffusion_id_v_con_a_h4_seed${seed}"

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=$((12648 + ${RANDOM} % 10000)) \
        main.py \
        --run_id="$run_id" \
        --seed="$seed" \
        --model="sasrec" \
        --epoch=100 \
        --lr=1e-5 \
        --batch_size=512 \
        --diffusion_loss_weight=0.3 \
        --eval_num=1 \
        --method='dff_diffusion_id_v_condition_a' \
        --timesteps=4 \
        --beta_start=0.001 \
        --beta_end=0.05 \
        --w=0.9
done

# print
# for seed in "${seeds[@]}"
# do
#     run_id="diffusion_id_v_con_a_tuan_seed${seed}"

#     CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=$((12648 + ${RANDOM} % 10000)) \
#         main.py \
#         --run_id="$run_id" \
#         --seed="$seed" \
#         --model="sasrec" \
#         --epoch=100 \
#         --lr=1e-5 \
#         --batch_size=512 \
#         --diffusion_loss_weight=0.1 \
#         --eval_num=1 \
#         --method='dff_diffusion_id_v_condition_a' \
#         --mode='print' \
#         --infer_ckpt='/opt/data/private/work/vllmembs4rec/sasrec/checkpoint/checkpoint_diffusion_id_v_con_a_tuan_seed2025/cpt__bs_512_ed_2048_lr_1e-05_l2_0.1_maxLen_10/epoch-54.pt'
# done