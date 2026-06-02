import os
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
_DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "MicroLens-100k")
_DEFAULT_FEATURES = os.path.join(_DEFAULT_DATA, "features")

def parse_args():
    parser = argparse.ArgumentParser()
    # ============== new parameter =========
    parser.add_argument('--run_id', type=str, default='id')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--model', type=str, default='sasrec', choices=['sasrec', 'gru4rec'])

    # VLVM embs
    parser.add_argument(
        '--pretrained_embs',
        type=str,
        default=os.path.join(_DEFAULT_FEATURES, 'all_layers_tensor.pt'),
    )
    # # plt
    # parser.add_argument('--pltpath', type=str, default='')
    # parser.add_argument('--ckpt_path', type=str, default='')
    # parser.add_argument('--tmp_filtered_embs_path', type=str, default='')
    # parser.add_argument('--checkpoint', type=str, default='')


    # ============== data_dir ==============
    parser.add_argument('--root_data_dir', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    parser.add_argument('--dataset', type=str, default='MicroLens-100k')
    parser.add_argument('--behaviors', type=str, default='MicroLens-100k_pairs.tsv')
    parser.add_argument('--min_video_no', type=int, default=1)
    parser.add_argument('--max_video_no', type=int, default=19738)

    # ============== train parameters==============
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--scheduler', type=str, default='step_schedule_with_warmup')
    parser.add_argument('--scheduler_gap', type=int, default=1)
    parser.add_argument('--scheduler_alpha', type=float, default=1)
    # parser.add_argument('--neg_num', type=int, default=100)
    parser.add_argument('--eval_num', type=int, default=1)

    # ============== model parameters ==============
    parser.add_argument('--embedding_dim', type=int, default=2048)
    parser.add_argument('--max_seq_len', type=int, default=10)
    parser.add_argument('--min_seq_len', type=int, default=5)

    # ============== SASRec parameters ==============
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--transformer_block', type=int, default=2)

    # ============== GRU4Rec parameters ==============
    parser.add_argument('--block_num', type=int, default=2)

    # ============== switch and logging setting ==============
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--load_ckpt_name', type=str, default='')
    parser.add_argument('--label_screen', type=str, default='tuan')
    parser.add_argument('--logging_num', type=int, default=10)
    parser.add_argument('--testing_num', type=int, default=10)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--node_rank', default=0, type=int)

    # ============== fusion methods ==============
    # parser.add_argument('--fusion_method', type=str, default='concat', choices=['none', 'sum', 'concat', 'film', 'gated'])
    # ============== only test ==============
    parser.add_argument('--if_only_test', action='store_true', default=False)
    parser.add_argument('--testckpt_path', type=str, default='tuan')
    parser.add_argument('--testckpt_epoch', type=int, default=50)
    # ============== tensorboard ==============
    parser.add_argument('--tb_log_dir', type=str, default=os.path.join(PROJECT_ROOT, 'tblogs'))
    # ============== diffusion ==============
    parser.add_argument('--timesteps', type=int, default=4)
    parser.add_argument('--beta_start', default=0.001, type=float)
    parser.add_argument('--beta_end', default=0.05, type=float)
    parser.add_argument('--w', default=0.9, type=float)
    parser.add_argument('--diffusion_loss_weight', default=0.1, type=float)
    # ============== second ==============
    parser.add_argument('--first_ckpt_path', type=str, default='')
    # ============== audio ==============
    parser.add_argument(
        '--audio_feature_path',
        type=str,
        default=os.path.join(_DEFAULT_FEATURES, 'laionclap_fusion_audio_feature.npy'),
    )
    # ============== method ==============
    parser.add_argument('--method', type=str, default='dff', choices=['only_id', 'dff_id_v', 'dff_id_a', 'dff_diffusion_id_a_condition_v', 'dff_diffusion_id_v_condition_a'])
    # 对比学习
    parser.add_argument('--cl_weight', default=0.1, type=float)
    parser.add_argument('--cl_temp', default=0.1, type=float)
    # print
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'print'])
    parser.add_argument('--infer_ckpt', type=str, default=None)

    args = parser.parse_args()

    return args

