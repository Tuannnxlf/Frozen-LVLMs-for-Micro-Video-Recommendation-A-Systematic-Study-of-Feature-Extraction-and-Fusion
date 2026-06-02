import os
import time
import random

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast as autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from model.model_dff import Model_dff_id_v
from utils.lr_decay import *
from utils.parameters import parse_args
from utils.load_data import read_items, read_behaviors
from utils.logging_utils import para_and_log, report_time_train, report_time_eval, save_model, setuplogger, get_time
from utils.dataset import IdDataset
from utils.metrics import get_item_embedding, eval_model_dff_id_v

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
scaler = torch.cuda.amp.GradScaler()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def train(args, model_dir, Log_file, Log_screen, start_time, local_rank):
    writer = SummaryWriter(log_dir=args.tb_log_dir_runid)
    hparams_dict = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}
    writer.add_hparams(
        hparam_dict=hparams_dict,
        metric_dict={
            "Loss/train_epoch": 0.0,
            "Metrics/Valid_Hit10": 0.0,
            "Metrics/Valid_nDCG10": 0.0,
            "Metrics/Test_Hit10": 0.0,
            "Metrics/Test_nDCG10": 0.0,
        },
    )

    item_content = None
    item_id_to_keys = None
    pretrained_embs = torch.load(args.pretrained_embs)
    pretrained_embs = pretrained_embs.to(f"cuda:{local_rank}")

    before_item_id_to_keys, before_item_name_to_id = read_items(args)
    item_num, item_id_to_keys, users_train, users_valid, users_history_for_valid, users_test, users_history_for_test = \
        read_behaviors(before_item_id_to_keys, before_item_name_to_id, Log_file, args, local_rank)
    print("item_num:", item_num)

    Log_file.info('build model dff_id_v...')
    model = Model_dff_id_v(args, item_num, item_content, pretrained_embs).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    train_dataset = IdDataset(u2seq=users_train, item_num=item_num, max_seq_len=args.max_seq_len, args=args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    def worker_init_reset_seed(worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        multiprocessing_context='fork',
        worker_init_fn=worker_init_reset_seed,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    optimizer = optim.AdamW(
        [{'params': model.module.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay, 'initial_lr': args.lr}]
    )

    next_set_start_time = time.time()
    max_epoch, early_stop_epoch = 0, args.epoch
    max_eval_value, early_stop_count = 0, 0
    steps_for_log, steps_for_eval = para_and_log(model, len(users_train), args.batch_size, Log_file,
                                                 logging_num=args.logging_num, testing_num=args.testing_num)
    del steps_for_log, steps_for_eval
    Log_screen.info('{} train start'.format(args.label_screen))

    if args.scheduler == "cosine_schedule_with_warmup":
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.epoch, start_epoch=-1)
    elif args.scheduler == "linear_schedule_with_warmup":
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.epoch, start_epoch=-1)
    elif args.scheduler == "step_schedule_with_warmup":
        lr_scheduler = get_step_schedule_with_warmup(
            optimizer, num_warmup_steps=0, gap_steps=args.scheduler_gap,
            scheduler_alpha=args.scheduler_alpha, start_epoch=-1,
        )
    else:
        raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

    for ep in range(args.epoch):
        now_epoch = ep + 1
        train_dl.sampler.set_epoch(now_epoch)
        loss = 0.0
        batch_index = 1
        need_break = False
        need_test = False
        model.train()

        for sample_items, log_mask in tqdm(train_dl, desc=f"Epoch {now_epoch}"):
            sample_items = sample_items.to(local_rank)
            log_mask = log_mask.to(local_rank)
            sample_items_id = sample_items.view(-1)
            optimizer.zero_grad()
            with autocast(enabled=True):
                bz_loss = model(sample_items_id, log_mask, local_rank, args)
                loss += bz_loss.item()
            scaler.scale(bz_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=5, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
            if torch.isnan(bz_loss):
                need_break = True
                break
            batch_index += 1

        Log_file.info('')
        next_set_start_time = report_time_train(batch_index, now_epoch, loss, next_set_start_time, start_time, Log_file)
        epoch_batch_num = max(batch_index - 1, 1)
        mean_bz_loss = loss / epoch_batch_num
        writer.add_scalar("Loss/train_epoch", mean_bz_loss, now_epoch)
        Log_file.info(f'epoch {now_epoch} summary: mean_bz_loss={mean_bz_loss:.4f}')

        if not need_break and now_epoch % args.eval_num == 0:
            valid_Hit10 = eval_step(
                now_epoch, model, users_history_for_valid, users_valid, item_num,
                local_rank, args, Log_file, item_id_to_keys, writer, mode='valid',
            )
            if valid_Hit10 > max_eval_value:
                max_eval_value = valid_Hit10
                max_epoch = now_epoch
                early_stop_count = 0
                if dist.get_rank() == 0:
                    save_model(now_epoch, model, model_dir, optimizer, torch.get_rng_state(),
                               torch.cuda.get_rng_state(), Log_file)
            else:
                early_stop_count += 1
                if early_stop_count > 5:
                    need_break = True
                    need_test = True
                    early_stop_epoch = now_epoch

        if need_test:
            best_ckpt = os.path.abspath(os.path.join(model_dir, f'epoch-{max_epoch}.pt'))
            checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            eval_step(max_epoch, model, users_history_for_test, users_test, item_num,
                      local_rank, args, Log_file, item_id_to_keys, writer, mode='test')

        if need_break:
            break
        if lr_scheduler is not None:
            lr_scheduler.step()

    Log_file.info('%' * 90)
    Log_file.info('max eval Hit10 {:0.5f}  in epoch {}'.format(max_eval_value * 100, max_epoch))
    Log_file.info('early stop in epoch {}'.format(early_stop_epoch))
    Log_file.info('the End')
    Log_screen.info('{} train end in epoch {}'.format(args.label_screen, early_stop_epoch))


def eval_step(now_epoch, model, user_history, users_eval, item_num, local_rank, args,
              Log_file, item_id_to_keys, writer, mode='valid'):
    del item_id_to_keys
    score_embs, score_embs_id, score_embs_vlvm = get_item_embedding(model, item_num, 64, args, local_rank)
    Hit10, nDCG10, Hit20, nDCG20 = eval_model_dff_id_v(
        model, user_history, users_eval, score_embs, score_embs_id, score_embs_vlvm,
        64, args, item_num, Log_file, mode, local_rank, now_epoch,
    )
    if writer is not None:
        prefix = "Valid" if mode == 'valid' else "Test"
        writer.add_scalar(f"Metrics/{prefix}_Hit10", Hit10, now_epoch)
        writer.add_scalar(f"Metrics/{prefix}_Hit20", Hit20, now_epoch)
        writer.add_scalar(f"Metrics/{prefix}_nDCG10", nDCG10, now_epoch)
        writer.add_scalar(f"Metrics/{prefix}_nDCG20", nDCG20, now_epoch)
    report_time_eval(time.time(), Log_file)
    Log_file.info('')
    return Hit10


def main():
    args = parse_args()
    local_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    setup_seed(args.seed)

    dir_label = str(args.behaviors).strip().split('.')[0]
    log_paras = (
        f'_bs_{args.batch_size}_ed_{args.embedding_dim}_lr_{args.lr}'
        f'_l2_{args.weight_decay}_maxLen_{args.max_seq_len}'
    )
    model_dir = os.path.join('./checkpoint/checkpoint_' + args.run_id, 'cpt_' + log_paras)
    Log_file, Log_screen = setuplogger(dir_label, log_paras, args.run_id, dist.get_rank())
    Log_file.info(args)

    tb_log_dir_runid = os.path.join(args.tb_log_dir, args.run_id)
    os.makedirs(tb_log_dir_runid, exist_ok=True)
    args.tb_log_dir_runid = tb_log_dir_runid
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    train(args, model_dir, Log_file, Log_screen, start_time, local_rank)
    hour, minute, seconds = get_time(start_time, time.time())
    Log_file.info('#### (time) all: hours {} minutes {} seconds {} ####'.format(hour, minute, seconds))


if __name__ == '__main__':
    main()
