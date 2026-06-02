import os
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
_DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "MicroLens-100k")
_DEFAULT_FEATURES = os.path.join(_DEFAULT_DATA, "features")


def parse_args():
    parser = argparse.ArgumentParser(description="DFF id+v on MicroLens-100k")
    parser.add_argument('--run_id', type=str, default='dff_id_v')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--model', type=str, default='sasrec', choices=['sasrec'])

    parser.add_argument(
        '--pretrained_embs',
        type=str,
        default=os.path.join(_DEFAULT_FEATURES, 'all_layers_tensor.pt'),
    )
    parser.add_argument('--root_data_dir', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    parser.add_argument('--dataset', type=str, default='MicroLens-100k')
    parser.add_argument('--behaviors', type=str, default='MicroLens-100k_pairs.tsv')
    parser.add_argument('--min_video_no', type=int, default=1)
    parser.add_argument('--max_video_no', type=int, default=19738)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--scheduler', type=str, default='step_schedule_with_warmup')
    parser.add_argument('--scheduler_gap', type=int, default=1)
    parser.add_argument('--scheduler_alpha', type=float, default=1)
    parser.add_argument('--eval_num', type=int, default=1)

    parser.add_argument('--embedding_dim', type=int, default=2048)
    parser.add_argument('--max_seq_len', type=int, default=10)
    parser.add_argument('--min_seq_len', type=int, default=5)
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--transformer_block', type=int, default=2)

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--label_screen', type=str, default='dff_id_v')
    parser.add_argument('--logging_num', type=int, default=10)
    parser.add_argument('--testing_num', type=int, default=10)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--node_rank', default=0, type=int)
    parser.add_argument('--tb_log_dir', type=str, default=os.path.join(PROJECT_ROOT, 'tblogs'))

    return parser.parse_args()
