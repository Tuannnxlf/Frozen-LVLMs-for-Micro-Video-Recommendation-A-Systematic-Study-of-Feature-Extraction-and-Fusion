# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9505))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

from warnings import simplefilter
from transformers import logging
logging.set_verbosity_warning()
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

import os
import re
import time
import torch
import random
import subprocess
import wandb
import csv
from tqdm import tqdm

import numpy as np
import torch.optim as optim
import torch.distributed as dist
import torchvision.models as models

from torch import nn
from pathlib import Path
from statistics import mode
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast as autocast
# from transformers import CLIPVisionModel, SwinForImageClassification, ViTMAEModel
# from transformers import BertModel, BertTokenizer, BertConfig, RobertaTokenizer, RobertaModel, RobertaConfig
# from transformers import VideoMAEFeatureExtractor, VideoMAEModel, VideoMAEConfig

from model.model import Model

# from model.model_dnn import Model
from utils.lr_decay import *
from utils.parameters import parse_args
from utils.load_data import read_items, read_behaviors
from utils.logging_utils import para_and_log, report_time_train, report_time_eval, save_model, setuplogger, get_time
from utils.dataset import IdDataset
from utils.metrics import get_item_id_score, eval_model
from utils.newutils import get_file_name

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
scaler = torch.cuda.amp.GradScaler()

def setup_seed(seed):
    '''
    global seed config
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def zca_whitening_torch(X, eps=1e-5):
    # X: [n_samples, n_features], torch tensor
    X_mean = X.mean(dim=0, keepdim=True)
    Xc = X - X_mean
    # 协方差
    cov = (Xc.t() @ Xc) / (Xc.shape[0] - 1)  # torch不带rowvar参数，需自己写
    # SVD分解
    # U, S, V = torch.svd(cov)  # 适用于低版本pytorch
    # 或 torch.linalg.svd in new version (≥1.9)
    U, S, Vh = torch.linalg.svd(cov, full_matrices=False)
    # ZCA白化矩阵
    W = U @ torch.diag(1.0 / torch.sqrt(S + eps)) @ U.t()
    X_zca = Xc @ W
    return X_zca

def train(args, model_dir, Log_file, Log_screen, start_time, local_rank):

    # ========================================== Loading Data ===========================================
    item_content = None
    item_id_to_keys = None
    
    if args.pretrained_embs:
        pretrained_embs = torch.load(args.pretrained_embs)#[19739,4096]/[19739,33,4096]
    else:
        pretrained_embs = None

    before_item_id_to_keys, before_item_name_to_id = read_items(args)

    Log_file.info('read behaviors...')
    item_num, item_id_to_keys, users_train, users_valid, users_history_for_valid, users_test, users_history_for_test, pretrained_embs= \
        read_behaviors(before_item_id_to_keys, before_item_name_to_id, Log_file, args, pretrained_embs, local_rank)
    if args.embs_mode == 'ZCA_whitening':
        pretrained_embs = zca_whitening_torch(pretrained_embs)
    if args.embs_mode == 'ZCA_whitening_all_layers':
        X = pretrained_embs.reshape(-1, pretrained_embs.size(-1))
        X_zca = zca_whitening_torch(X)
        pretrained_embs = X_zca.reshape(*pretrained_embs.shape)  # [B, 33, 4096]
    # ========================================== Building Model ===========================================
    Log_file.info('build model...')
    model = Model(args, item_num, item_content, pretrained_embs).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # if args.load_ckpt_name and 'epoch' in args.load_ckpt_name:
    #     Log_file.info('load ckpt if not None...')

    #     ckpt_path = os.path.abspath(os.path.join(model_dir, args.load_ckpt_name))
        
    #     start_epoch = int(re.split(r'[._-]', args.load_ckpt_name)[1])
        
    #     checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    #     Log_file.info('load checkpoint...')
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     Log_file.info(f'Model loaded from {args.load_ckpt_name}')
    #     torch.set_rng_state(checkpoint['rng_state'])  # random seed status in loading torch
    #     torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])  # random seed status in loading torch.cuda
    #     is_early_stop = False
    # else:
    #     Log_file.info(' ckpt is None...')
    #     checkpoint = None  # new
    #     ckpt_path = None  # new
    #     start_epoch = 0
    #     is_early_stop = False

    # # ============================ Dataset and Dataloader ============================

    # train_dataset = IdDataset(u2seq=users_train, 
    #                             item_num=item_num, 
    #                             max_seq_len=args.max_seq_len,
    #                             args=args)

    # Log_file.info('build DDP sampler...')
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # def worker_init_reset_seed(worker_id):
    #     initial_seed = torch.initial_seed() % 2 ** 31
    #     worker_seed = initial_seed + worker_id
    #     random.seed(worker_seed)
    #     np.random.seed(worker_seed)

    # Log_file.info('build dataloader...')
    # train_dl = DataLoader(train_dataset, 
    #                     batch_size=args.batch_size, 
    #                     num_workers=args.num_workers,
    #                     multiprocessing_context='fork', 
    #                     worker_init_fn=worker_init_reset_seed, 
    #                     pin_memory=True, 
    #                     sampler=train_sampler,
    #                     drop_last=True)

    # # ============================ Optimizer ============================
    # recsys_params = [param for name, param in model.module.named_parameters() if param.requires_grad]

    # optimizer = optim.AdamW([
    #     {
    #         'params': recsys_params,
    #         'lr': args.lr,
    #         'weight_decay': args.weight_decay,
    #         'initial_lr': args.lr,
    #     }
    # ])
    # # optimizer = optim.Adam([{'params':model.module.parameters(), 'lr':args.lr, 'initial_lr': args.lr}])

    # if args.load_ckpt_name:   # load optimizer status
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     Log_file.info(f'optimizer loaded from {ckpt_path}')
    
    # # ============================  training  ============================

    # Log_file.info('\n')
    # Log_file.info('Training...')
    # next_set_start_time = time.time()
    # max_epoch, early_stop_epoch = 0, args.epoch
    # max_eval_value, early_stop_count = 0, 0

    # steps_for_log, steps_for_eval = para_and_log(model, len(users_train), args.batch_size, Log_file,
    #                                             logging_num=args.logging_num, testing_num=args.testing_num)
    # Log_screen.info('{} train start'.format(args.label_screen))

    # warmup_steps = 0
    # if args.scheduler == "cosine_schedule_with_warmup":
    #     lr_scheduler = get_cosine_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=warmup_steps,
    #         num_training_steps=args.epoch,
    #         start_epoch=start_epoch-1)
        
    # elif args.scheduler == "linear_schedule_with_warmup":
    #     lr_scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=warmup_steps,
    #         num_training_steps=args.epoch,
    #         start_epoch=start_epoch-1)
        
    # elif args.scheduler == "step_schedule_with_warmup":
    #     lr_scheduler = get_step_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=warmup_steps,
    #         gap_steps = args.scheduler_gap,
    #         scheduler_alpha = args.scheduler_alpha,
    #         start_epoch=start_epoch-1)
    # else:
    #     raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

    # epoch_left = args.epoch - start_epoch

    # for ep in range(epoch_left):
    #     now_epoch = start_epoch + ep + 1
    #     train_dl.sampler.set_epoch(now_epoch)
    #     loss = 0.0
    #     batch_index = 1
    #     need_break = False
        
    #     Log_file.info('\n')
    #     Log_file.info('epoch {} start'.format(now_epoch))
    #     Log_file.info('')
        
    #     model.train()
    #     if lr_scheduler is not None:
    #         Log_file.info('start of trainin epoch:  {} ,lr: {}'.format(now_epoch, lr_scheduler.get_lr()))

    #     for sample_items, log_mask in tqdm(train_dl, desc=f"Epoch {now_epoch}"):
    #         sample_items = sample_items.to(local_rank)
    #         log_mask = log_mask.to(local_rank)
    #         sample_items_id = sample_items.view(-1)

    #         optimizer.zero_grad()

    #         # Mixed accuracy (acceleration)
    #         with autocast(enabled=True):
    #             bz_loss = model(sample_items_id, log_mask, local_rank, args)
    #             loss += bz_loss.item()

    #         scaler.scale(bz_loss).backward()
    #         scaler.unscale_(optimizer)
    #         torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=5, norm_type=2)
    #         scaler.step(optimizer)
    #         scaler.update()


    #         if torch.isnan(bz_loss):
    #             need_break = True
    #             break

    #         #steps_for_log = 1
    #         if batch_index % steps_for_log == 0:
    #             Log_file.info('Ed: {}, batch_loss: {:.3f}, mean_batch_loss: {:.3f}'.format(
    #                 batch_index * args.batch_size, bz_loss.item(), loss / batch_index))
    #         batch_index += 1

    #     Log_file.info('')
    #     next_set_start_time = report_time_train(batch_index, now_epoch, loss, next_set_start_time, start_time, Log_file)
        
    #     epoch_batch_num = batch_index - 1  # 或者刚才batch_index-1
    #     if epoch_batch_num == 0:
    #         epoch_batch_num = 1  # 避免被零除

    #     mean_bz_loss = loss / epoch_batch_num

    #     Log_file.info(f'epoch {now_epoch} summary: mean_bz_loss={mean_bz_loss:.4f}')

    #     if args.wandb:
    #         wandb.log({
    #             "Loss/train_loss": mean_bz_loss
    #         }, step=now_epoch)

    #     Log_screen.info('{} training: epoch {}/{}'.format(args.label_screen, now_epoch, args.epoch))

    #     # eval
    #     if not need_break and now_epoch % args.eval_num == 0 and now_epoch > 0:
    #         valid_Hit10 = \
    #             eval(now_epoch, model, users_history_for_valid, \
    #                 users_valid, 64, item_num, 'valid', local_rank, args, Log_file, item_content, item_id_to_keys)
    #         if valid_Hit10 > max_eval_value:
    #             max_eval_value = valid_Hit10
    #             max_epoch = now_epoch
    #             early_stop_count = 0
    #             if dist.get_rank() == 0 :
    #                 save_model(now_epoch, model, model_dir, optimizer, torch.get_rng_state(), torch.cuda.get_rng_state(), Log_file)   # new
    #         else:
    #             early_stop_count += 1
    #             if early_stop_count > 5:
    #                 if is_early_stop:
    #                     need_break = True
    #                 early_stop_epoch = now_epoch

    #     if need_break:
    #         break
        
    #     if lr_scheduler is not None:
    #         lr_scheduler.step()
    #         Log_file.info('end of trainin epoch:  {} ,lr: {}'.format(now_epoch, lr_scheduler.get_lr()))

    # Log_file.info('\n')
    # Log_file.info('%' * 90)
    # Log_file.info('max eval Hit10 {:0.5f}  in epoch {}'.format(max_eval_value * 100, max_epoch-1))
    # Log_file.info('early stop in epoch {}'.format(early_stop_epoch))
    # Log_file.info('the End')
    # Log_screen.info('{} train end in epoch {}'.format(args.label_screen, early_stop_epoch))
    
    # # test
    # best_ckpt = os.path.abspath(os.path.join(model_dir, f'epoch-{max_epoch}.pt'))
    # checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['model_state_dict'])
    # torch.set_rng_state(checkpoint['rng_state'])  # random seed status in loading torch
    # torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])  # random seed status in loading torch.cuda
    # eval(max_epoch, model, users_history_for_test, \
    #                 users_test, 64, item_num, 'test', local_rank, args, Log_file, item_content, item_id_to_keys)

def eval(now_epoch, model, user_history, users_eval, batch_size, item_num,\
         mode, local_rank, args, Log_file, item_content=None, item_id_to_keys=None):

    eval_start_time = time.time()

    score_embs = get_item_id_score(model, item_num, batch_size, args, local_rank)

    Hit10, nDCG10, Hit20, nDCG20 = eval_model(model, user_history, users_eval, score_embs, batch_size, \
        args, item_num, Log_file, mode, local_rank, now_epoch)

    if args.wandb and mode == 'valid':
        wandb.log({f'{mode}_Hit10': Hit10}, step=now_epoch)
        wandb.log({f'{mode}_nDCG10': nDCG10}, step=now_epoch)
        wandb.log({f'{mode}_Hit10': Hit10}, step=now_epoch)
        wandb.log({f'{mode}_nDCG10': nDCG10}, step=now_epoch)
    if args.wandb and mode == 'test':
        wandb.log({
            "test/Hit10":Hit10,
            "test/nDCG10":nDCG10,
            "test/Hit20":Hit20,
            "test/nDCG20":nDCG20
        })

    if mode == 'test' and args.save_test:
        csv_path = args.save_test
        csv_dir = os.path.dirname(csv_path)
        os.makedirs(csv_dir, exist_ok=True)
        row = [str(args.run_id), float(Hit10), float(nDCG10)]
        # 检查文件是否存在，决定是否写header
        write_header = not os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['run_id', 'Hit10', 'nDCG10'])
            writer.writerow(row)

    report_time_eval(eval_start_time, Log_file)
    Log_file.info('')
    return Hit10

def main():
    args = parse_args()

    # ============== Distributed Computation Config ==============
    local_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    # ============== Experiment and Logging Config ===============
    if args.seed:
        setup_seed(args.seed)
    else:
        setup_seed(42 + dist.get_rank())  # magic number

    dir_label =  str(args.behaviors).strip().split('.')[0]

    log_paras = f'_bs_{args.batch_size}' \
                f'_ed_{args.embedding_dim}_lr_{args.lr}' \
                f'_l2_{args.weight_decay}' \
                f'_maxLen_{args.max_seq_len}'

    model_dir = os.path.join('./checkpoint/checkpoint_' + args.run_id, f'cpt_' + log_paras)
    Log_file, Log_screen = setuplogger(dir_label, log_paras, args.run_id, dist.get_rank())
    Log_file.info(args)

    if args.wandb:
        args_dict = vars(args)
        wandb.init(
            project='vllmfuse',
            config=args_dict,
            name=get_file_name(args_dict),
            mode='online'
        )

    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    train(args, model_dir, Log_file, Log_screen, start_time, local_rank)

    end_time = time.time()
    hour, minute, seconds = get_time(start_time, end_time)
    Log_file.info('#### (time) all: hours {} minutes {} seconds {} ####'.format(hour, minute, seconds))


if __name__ == '__main__':
    main()
