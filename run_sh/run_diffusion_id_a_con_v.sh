#!/bin/bash
cd /opt/data/private/work/vllmembs4rec/sasrec || exit 1
seeds=(2025)
lrs=(1e-5)
drop_rates=(0.1)
weight_decays=(0.1)
embedding_dims=(2048)

for seed in "${seeds[@]}"
do
    run_id="diffusion_id_a_con_v_tuan_new_seed${seed}"
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=$((12648 + ${RANDOM} % 10000)) \
        main.py \
        --run_id="$run_id" \
        --seed="$seed" \
        --model="sasrec" \
        --epoch=100 \
        --lr=1e-5 \
        --batch_size=512 \
        --diffusion_loss_weight=0.1 \
        --eval_num=1 \
        --method='dff_diffusion_id_a_condition_v'
done

# parser.add_argument('--timesteps', type=int, default=4)
# parser.add_argument('--beta_start', default=0.001, type=float)
# parser.add_argument('--beta_end', default=0.05, type=float)
# parser.add_argument('--w', default=0.9, type=float)
# parser.add_argument('--diffusion_loss_weight', default=0.1, type=float)