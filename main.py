from warnings import simplefilter

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
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from pathlib import Path
from statistics import mode
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast as autocast

from model.model import Model
from model.model_dff import Model_dff_id_v
from model.baseline import M3BSR
from model.model_only_id import Model_onlyid
from model.model_diffusion import Model_diffusion_id_v_con_a, Model_diffusion_id_a_con_v

from utils.lr_decay import *
from utils.parameters import parse_args
from utils.load_data import read_items, read_behaviors
from utils.logging_utils import para_and_log, report_time_train, report_time_eval, save_model, setuplogger, get_time
from utils.dataset import IdDataset
from utils.metrics import print_test_result_diffusion, print_test_result, get_item_embedding, get_item_id_score_dff, get_item_id_score, get_item_id_score_onlyid, eval_model, eval_model_dff_id_v, eval_model_M3BSR, eval_model_onlyid
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

def train(args, model_dir, Log_file, Log_screen, start_time, local_rank):
    # tensorboard
    writer = SummaryWriter(log_dir = args.tb_log_dir_runid)
    raw_hparams = vars(args)
    hparams_dixt = {}
    for k, v in raw_hparams.items():
        if isinstance(v, (int, float, str, bool)):
            hparams_dixt[k] = v
        elif isinstance(v, (list, tuple)): # 特殊处理：list 或 tuple 转为字符串 (防止路径列表等报错)
            hparams_dixt[k] = str(v)
        else: # 其他不支持的类型直接跳过 (比如 None, 或者复杂的自定义对象)
            pass
    metrics_dict = {
        "Loss/train_epoch": 0.0,
        "Metrics/Valid_Hit10": 0.0,
        "Metrics/Valid_Hit20": 0.0,
        "Metrics/Valid_nDCG10": 0.0,
        "Metrics/Valid_nDCG20": 0.0,
        "Metrics/Test_Hit10": 0.0,
        "Metrics/Test_Hit20": 0.0,
        "Metrics/Test_nDCG10": 0.0,
        "Metrics/Test_nDCG20": 0.0
    }
    writer.add_hparams(hparam_dict=hparams_dixt, metric_dict=metrics_dict)
    # ========================================== Loading Data ===========================================

    item_content = None
    item_id_to_keys = None

    if args.pretrained_embs:
        pretrained_embs = torch.load(args.pretrained_embs)#[19739,4096]/[19739,33,4096]
    else:
        pretrained_embs = None

    if args.audio_feature_path:
        audio_embs = np.load(args.audio_feature_path)
        audio_embs = torch.from_numpy(audio_embs)
    else:
        audio_embs = None

    before_item_id_to_keys, before_item_name_to_id = read_items(args)
    item_num, item_id_to_keys, users_train, users_valid, users_history_for_valid, users_test, users_history_for_test= \
        read_behaviors(before_item_id_to_keys, before_item_name_to_id, Log_file, args, local_rank)
    print("item_num:", item_num)
    
    # ========================================== Building Model ===========================================

    Log_file.info('build model...')
    if pretrained_embs is not None:
        device = f"cuda:{local_rank}"
        pretrained_embs = pretrained_embs.to(device)
    if audio_embs is not None:
        device = f"cuda:{local_rank}"
        audio_embs = audio_embs.to(device)

    if args.method == 'only_id':
        model = Model_onlyid(args, item_num, item_content, pretrained_embs, audio_embs).to(local_rank)
    elif args.method == 'dff_id_v':
        model = Model_dff_id_v(args, item_num, item_content, pretrained_embs).to(local_rank)
    elif args.method == 'dff_id_a':
        pass
    elif args.method == 'dff_diffusion_id_v_condition_a':
        model = Model_diffusion_id_v_con_a(args, item_num, item_content, pretrained_embs, audio_embs).to(local_rank)
    elif args.method == 'dff_diffusion_id_a_condition_v':
        model = Model_diffusion_id_a_con_v(args, item_num, item_content, pretrained_embs, audio_embs).to(local_rank)
    else:
        raise ValueError(f"Invalid method: {args.method}")
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    start_epoch = 0
    is_early_stop = True

    # ============================ Dataset and Dataloader ============================

    train_dataset = IdDataset(u2seq=users_train, 
                                item_num=item_num, 
                                max_seq_len=args.max_seq_len,
                                args=args)

    # Log_file.info('build DDP sampler...')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    def worker_init_reset_seed(worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    # Log_file.info('build dataloader...')
    train_dl = DataLoader(train_dataset, 
                        batch_size=args.batch_size, 
                        num_workers=args.num_workers,
                        multiprocessing_context='fork', 
                        worker_init_fn=worker_init_reset_seed, 
                        pin_memory=True, 
                        sampler=train_sampler,
                        drop_last=True)

    # ============================ Optimizer ============================
    recsys_params = [param for name, param in model.module.named_parameters() if param.requires_grad]

    optimizer = optim.AdamW([
        {
            'params': recsys_params,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'initial_lr': args.lr,
        }
    ])
    # optimizer = optim.Adam([{'params':model.module.parameters(), 'lr':args.lr, 'initial_lr': args.lr}])
    
    # ============================  training  ============================
    if not args.if_only_test:
        next_set_start_time = time.time()
        max_epoch, early_stop_epoch = 0, args.epoch
        max_eval_value, early_stop_count = 0, 0

        steps_for_log, steps_for_eval = para_and_log(model, len(users_train), args.batch_size, Log_file,
                                                    logging_num=args.logging_num, testing_num=args.testing_num)
        Log_screen.info('{} train start'.format(args.label_screen))

        warmup_steps = 0
        if args.scheduler == "cosine_schedule_with_warmup":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=args.epoch,
                start_epoch=start_epoch-1)
            
        elif args.scheduler == "linear_schedule_with_warmup":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=args.epoch,
                start_epoch=start_epoch-1)
            
        elif args.scheduler == "step_schedule_with_warmup":
            lr_scheduler = get_step_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                gap_steps = args.scheduler_gap,
                scheduler_alpha = args.scheduler_alpha,
                start_epoch=start_epoch-1)
        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        epoch_left = args.epoch - start_epoch

        for ep in range(epoch_left):
            now_epoch = start_epoch + ep + 1
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

                #steps_for_log = 1
                # if batch_index % steps_for_log == 0:
                #     Log_file.info('Ed: {}, batch_loss: {:.3f}, mean_batch_loss: {:.3f}'.format(
                #         batch_index * args.batch_size, bz_loss.item(), loss / batch_index))
                batch_index += 1

            Log_file.info('')
            next_set_start_time = report_time_train(batch_index, now_epoch, loss, next_set_start_time, start_time, Log_file)
            
            epoch_batch_num = batch_index - 1  # 或者刚才batch_index-1
            if epoch_batch_num == 0:
                epoch_batch_num = 1  # 避免被零除

            mean_bz_loss = loss / epoch_batch_num
            if writer is not None:
                writer.add_scalar(f"Loss/train_epoch", mean_bz_loss, now_epoch)
            Log_file.info(f'epoch {now_epoch} summary: mean_bz_loss={mean_bz_loss:.4f}')

            # eval
            if not need_break and now_epoch % args.eval_num == 0 and now_epoch > 0:
                valid_Hit10 = \
                    eval(now_epoch, model, users_history_for_valid, \
                        users_valid, 64, item_num, 'valid', local_rank, args, Log_file, item_content, item_id_to_keys, writer=writer)
                # eval(now_epoch, model, users_history_for_test, \
                #         users_test, 64, item_num, 'test', local_rank, args, Log_file, item_content, item_id_to_keys, writer)
                if valid_Hit10 > max_eval_value:
                    max_eval_value = valid_Hit10
                    max_epoch = now_epoch
                    early_stop_count = 0
                    if dist.get_rank() == 0 :
                        save_model(now_epoch, model, model_dir, optimizer, torch.get_rng_state(), torch.cuda.get_rng_state(), Log_file)   # new
                else:
                    early_stop_count += 1
                    if early_stop_count > 5:
                        if is_early_stop:
                            need_break = True
                            need_test = True
                        early_stop_epoch = now_epoch
            if need_test:
                # test
                best_ckpt = os.path.abspath(os.path.join(model_dir, f'epoch-{max_epoch}.pt'))
                checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
                model.load_state_dict(checkpoint['model_state_dict'])
                torch.set_rng_state(checkpoint['rng_state'])  # random seed status in loading torch
                torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])  # random seed status in loading torch.cuda
                eval(max_epoch, model, users_history_for_test, \
                        users_test, 64, item_num, 'test', local_rank, args, Log_file, item_content, item_id_to_keys, writer)
            if need_break:
                break
            
            if lr_scheduler is not None:
                lr_scheduler.step()
                # Log_file.info('end of trainin epoch:  {} ,lr: {}'.format(now_epoch, lr_scheduler.get_lr()))

        Log_file.info('\n')
        Log_file.info('%' * 90)
        Log_file.info('max eval Hit10 {:0.5f}  in epoch {}'.format(max_eval_value * 100, max_epoch))
        Log_file.info('early stop in epoch {}'.format(early_stop_epoch))
        Log_file.info('the End')
        Log_screen.info('{} train end in epoch {}'.format(args.label_screen, early_stop_epoch))
    
    if args.if_only_test:
        best_ckpt = args.testckpt_path
        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])  # random seed status in loading torch
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])  # random seed status in loading torch.cuda
        eval(args.testckpt_epoch, model, users_history_for_test, \
                        users_test, 64, item_num, 'test', local_rank, args, Log_file, item_content, item_id_to_keys, writer)

def eval(now_epoch, model, user_history, users_eval, batch_size, item_num,\
         mode, local_rank, args, Log_file, item_content=None, item_id_to_keys=None, writer=None):

    eval_start_time = time.time()
    if args.method == 'dff_id_v':
        score_embs, score_embs_id, score_embs_vlvm = get_item_embedding(model, item_num, batch_size, args, local_rank)
        Hit10, nDCG10, Hit20, nDCG20 = eval_model_dff_id_v(model, user_history, users_eval, score_embs, score_embs_id, score_embs_vlvm, batch_size, \
        args, item_num, Log_file, mode, local_rank, now_epoch)
    if args.method in ['mmfusion', 'dff']:
        score_embs, score_embs_id, score_embs_vlvm, score_embs_a = get_item_id_score(model, item_num, batch_size, args, local_rank)
        Hit10, nDCG10, Hit20, nDCG20 = eval_model(model, user_history, users_eval, score_embs, score_embs_id, score_embs_vlvm, score_embs_a, batch_size, \
        args, item_num, Log_file, mode, local_rank, now_epoch)
    if args.method in ['dff_diffusion_id_a_condition_v', 'dff_diffusion_id_v_condition_a']:
        score_embs, score_embs_id, score_embs_vlvm, score_embs_a = get_item_id_score_dff(model, item_num, batch_size, args, local_rank)
        Hit10, nDCG10, Hit20, nDCG20 = eval_model(model, user_history, users_eval, score_embs, score_embs_id, score_embs_vlvm, score_embs_a, batch_size, \
        args, item_num, Log_file, mode, local_rank, now_epoch)
    if args.method == 'M3BSR':
        score_embs_id, score_embs_vlvm, score_embs_a = get_item_id_score(model, item_num, batch_size, args, local_rank)
        # print(score_embs_id.shape, score_embs_vlvm.shape, score_embs_a.shape)
        Hit10, nDCG10, Hit20, nDCG20 = eval_model_M3BSR(model, user_history, users_eval, score_embs_id, score_embs_vlvm, score_embs_a, batch_size, \
        args, item_num, Log_file, mode, local_rank, now_epoch)
    if args.method == 'only_id':
        score_embs_id = get_item_id_score_onlyid(model, item_num, batch_size, args, local_rank)
        Hit10, nDCG10, Hit20, nDCG20 = eval_model_onlyid(model, user_history, users_eval, score_embs_id, batch_size, \
        args, item_num, Log_file, mode, local_rank, now_epoch)

    if writer is not None:
        prefix = "Valid" if mode == 'valid' else 'Test'
        writer.add_scalar(f"Metrics/{prefix}_Hit10", Hit10, now_epoch)
        writer.add_scalar(f"Metrics/{prefix}_Hit20", Hit20, now_epoch)
        writer.add_scalar(f"Metrics/{prefix}_nDCG10", nDCG10, now_epoch)
        writer.add_scalar(f"Metrics/{prefix}_nDCG20", nDCG20, now_epoch)

    report_time_eval(eval_start_time, Log_file)
    Log_file.info('')
    return Hit10

def print_infered_embs(args, model_dir, Log_file, local_rank):
    # ========================================== Loading Data ===========================================
    item_content = None
    item_id_to_keys = None

    if args.pretrained_embs:
        pretrained_embs = torch.load(args.pretrained_embs)#[19739,4096]/[19739,33,4096]
    else:
        assert False

    if args.audio_feature_path:
        audio_embs = np.load(args.audio_feature_path)
        audio_embs = torch.from_numpy(audio_embs)
    else:
        assert False

    before_item_id_to_keys, before_item_name_to_id = read_items(args)
    item_num, item_id_to_keys, users_train, users_valid, users_history_for_valid, users_test, users_history_for_test= \
        read_behaviors(before_item_id_to_keys, before_item_name_to_id, Log_file, args, local_rank)
    # buld model
    Log_file.info('build model...')

    pretrained_embs = pretrained_embs.to(args.device)
    audio_embs = audio_embs.to(args.device)

    if args.method == 'mmfusion':
        model = Model(args, item_num, item_content, pretrained_embs, audio_embs).to(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    if args.method == 'dff_id_v':
        model = Model_dff_id_v(args, item_num, item_content, pretrained_embs).to(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    if args.method == 'dff_diffusion_id_a_condition_v':
        model = Model_diffusion_id_a_con_v(args, item_num, item_content, pretrained_embs, audio_embs).to(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    if args.method == 'dff_diffusion_id_v_condition_a':
        model = Model_diffusion_id_v_con_a(args, item_num, item_content, pretrained_embs, audio_embs).to(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    if args.method == 'M3BSR':
        model = M3BSR(args, item_num, item_content, pretrained_embs, audio_embs).to(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    if args.method == 'only_id':
        model = Model_onlyid(args, item_num, item_content, pretrained_embs, audio_embs).to(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # load ckpt
    infer_ckpt = args.infer_ckpt
    checkpoint = torch.load(infer_ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # dataset
    if args.method == 'dff_id_v':
        score_embs, score_embs_id, score_embs_vlvm = get_item_embedding(model, item_num, 64, args, local_rank)
        print_test_result(model, users_history_for_test, users_test, score_embs, score_embs_id, score_embs_vlvm, 64, \
        args, item_num, Log_file, local_rank)
    if args.method in ['dff_diffusion_id_a_condition_v', 'dff_diffusion_id_v_condition_a']:
        score_embs, score_embs_id, score_embs_vlvm, score_embs_a = get_item_id_score_dff(model, item_num, 64, args, local_rank)
        print_test_result_diffusion(model, users_history_for_test, users_test, score_embs, score_embs_id, score_embs_vlvm, score_embs_a, 64, \
        args, item_num, Log_file, local_rank)


def main():
    args = parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ============== Distributed Computation Config ==============
    local_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    # ============== Experiment and Logging Config ===============
    setup_seed(args.seed)

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
    
    # tensorboard
    tb_log_dir_runid = os.path.join(args.tb_log_dir, args.run_id)
    os.makedirs(tb_log_dir_runid, exist_ok=True)
    args.tb_log_dir_runid = tb_log_dir_runid

    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == 'train':
        start_time = time.time()
        train(args, model_dir, Log_file, Log_screen, start_time, local_rank)
        end_time = time.time()
        hour, minute, seconds = get_time(start_time, end_time)
        Log_file.info('#### (time) all: hours {} minutes {} seconds {} ####'.format(hour, minute, seconds))
    if args.mode == 'print':
        print_infered_embs(args, model_dir, Log_file, local_rank)

if __name__ == '__main__':
    main()
