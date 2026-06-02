cd /opt/data/private/vllmembs4rec/sasrec || exit 1
seed=2023

# run_id="videolavitlastlayer_modelsize"
# save_test_path="/opt/data/private/vllmembs4rec/sasrec/performance/${run_id}.csv"
# pretrained_embs_file="/opt/data/private/vllmembs4rec/sasrec/data/pretrained_embs/all_layers_tensor.pt"
# file_base=$(basename "$pretrained_embs_file" .pt)
# echo "Running with seed $seed (run_id: $run_id)"

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=$((12648 + ${RANDOM} % 10000)) \
#     model_size.py \
#     --run_id="$run_id" \
#     --save_test="$save_test_path" \
#     --pretrained_embs="$pretrained_embs_file" \
#     --filtered_embs_path="/opt/data/private/Tuan/VLLMEmbs4Rec/SASRec/tmp/filtered_embs/${file_base}.pt" \
#     --seed="$seed" \
#     --embs_mode="id+layers" \
#     --eval_num=5 \
#     --model="sasrec" \
#     --epoch=100 \
#     --lr=1e-5 \
#     --batch_size=512

# run_id="videolavitlastlayer_modelsize"
# save_test_path="/opt/data/private/vllmembs4rec/sasrec/performance/${run_id}.csv"
# pretrained_embs_file="/opt/data/private/vllmembs4rec/sasrec/data/pretrained_embs/last_layer_tensor.pt"
# file_base=$(basename "$pretrained_embs_file" .pt)
# echo "Running with seed $seed (run_id: $run_id)"

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=$((12648 + ${RANDOM} % 10000)) \
#     model_size.py \
#     --run_id="$run_id" \
#     --save_test="$save_test_path" \
#     --pretrained_embs="$pretrained_embs_file" \
#     --filtered_embs_path="/opt/data/private/Tuan/VLLMEmbs4Rec/SASRec/tmp/filtered_embs/${file_base}.pt" \
#     --seed="$seed" \
#     --embs_mode="id+linear_mapping" \
#     --eval_num=5 \
#     --model="sasrec" \
#     --epoch=100 \
#     --lr=1e-5 \
#     --batch_size=512

run_id="videolavitlastlayer_modelsize"
save_test_path="/opt/data/private/vllmembs4rec/sasrec/performance/${run_id}.csv"
echo "Running with seed $seed (run_id: $run_id)"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=$((12648 + ${RANDOM} % 10000)) \
    model_size.py \
    --run_id="$run_id" \
    --save_test="$save_test_path" \
    --seed="$seed" \
    --embs_mode="id" \
    --eval_num=5 \
    --model="sasrec" \
    --epoch=100 \
    --lr=1e-5 \
    --batch_size=512