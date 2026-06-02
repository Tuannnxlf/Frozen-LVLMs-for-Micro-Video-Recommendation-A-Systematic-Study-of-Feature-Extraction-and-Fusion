import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import os
import math
import wandb
import csv
from .dataset import  EvalDataset, EvalDataset_dff_id_v, EvalDataset_M3BSR, EvalDataset_onlyid, SequentialDistributedSampler, IdEvalDataset, ItemsDataset



def item_collate_fn(arr):
    arr = torch.LongTensor(np.array(arr))
    return arr

def id_collate_fn(arr):
    arr = torch.LongTensor(arr)
    return arr

def print_metrics(x, Log_file, v_or_t):
    Log_file.info(v_or_t+'_results   {}'.format('\t'.join(['{:0.5f}'.format(i * 100) for i in x])))

def get_mean(arr):
    return [i.mean() for i in arr]

def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]

def eval_concat(eval_list, test_sampler):
    eval_result = []
    for eval_m in eval_list:
        eval_m_cpu = distributed_concat(eval_m, len(test_sampler.dataset))\
            .to(torch.device('cpu')).numpy()
        eval_result.append(eval_m_cpu.mean())
    return eval_result

def metrics_topK(y_score, y_true, item_rank, topK, local_rank):
    # print("metrics_topK_start")
    # print(y_true.shape)
    # print(y_score.shape)
    order = torch.argsort(y_score, descending=True)
    # print("order")
    y_true = torch.take(y_true, order)
    # print("y_true")
    # print("y_true.shape:", y_true.shape)
    # print("item_rank.shape:", item_rank.shape)
    rank = torch.sum(y_true * item_rank)
    # print("rank")
    eval_ra = torch.zeros(2).to(local_rank)
    # print("eval_ra")
    if rank <= topK:
        eval_ra[0] = 1
        eval_ra[1] = 1 / math.log2(rank + 1)
    return rank, eval_ra

def get_item_embedding(model, item_num, test_batch_size, args, local_rank):
    model.eval()
    item_dataset = IdEvalDataset(data=np.arange(item_num + 1))
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=item_collate_fn)
    item_scoring = []
    id_scoring = []
    VLVM_scoring = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            # item_emb = model.module.id_pretrained_fusion(input_ids).to(torch.device('cpu')).detach()
            item_emb, id_embs, VLLM_layers_embs = model.module.id_pretrained_fusion(input_ids)
            item_emb = item_emb.to(torch.device('cpu')).detach()
            id_embs = id_embs.to(torch.device('cpu')).detach()
            VLLM_layers_embs = VLLM_layers_embs.to(torch.device('cpu')).detach()
            item_scoring.extend(item_emb)
            id_scoring.extend(id_embs)
            VLVM_scoring.extend(VLLM_layers_embs)
    return torch.stack(tensors=item_scoring, dim=0), torch.stack(tensors=id_scoring, dim=0), torch.stack(tensors=VLVM_scoring, dim=0)

def get_item_id_score(model, item_num, test_batch_size, args, local_rank):
    if args.method in ['mmfusion', 'dff']:
        model.eval()
        item_dataset = IdEvalDataset(data=np.arange(item_num + 1))
        item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size,
                                    num_workers=args.num_workers, pin_memory=True, collate_fn=item_collate_fn)
        item_scoring = []
        id_scoring = []
        VLVM_scoring = []
        audio_scoring = []
        with torch.no_grad():
            for input_ids in item_dataloader:
                input_ids = input_ids.to(local_rank)
                # item_emb = model.module.id_pretrained_fusion(input_ids).to(torch.device('cpu')).detach()
                item_emb, id_embs, VLLM_layers_embs, audio_embs = model.module.id_pretrained_fusion(input_ids)
                item_emb = item_emb.to(torch.device('cpu')).detach()
                id_embs = id_embs.to(torch.device('cpu')).detach()
                VLLM_layers_embs = VLLM_layers_embs.to(torch.device('cpu')).detach()
                audio_embs = audio_embs.to(torch.device('cpu')).detach()
                item_scoring.extend(item_emb)
                id_scoring.extend(id_embs)
                VLVM_scoring.extend(VLLM_layers_embs)
                audio_scoring.extend(audio_embs)
        return torch.stack(tensors=item_scoring, dim=0), torch.stack(tensors=id_scoring, dim=0), torch.stack(tensors=VLVM_scoring, dim=0), torch.stack(tensors=audio_scoring, dim=0)
    if args.method == 'M3BSR':
        model.eval()
        item_dataset = IdEvalDataset(data=np.arange(item_num + 1))
        item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size,
                                    num_workers=args.num_workers, pin_memory=True, collate_fn=item_collate_fn)
        id_scoring = []
        VLVM_scoring = []
        audio_scoring = []
        with torch.no_grad():
            for input_ids in item_dataloader:
                input_ids = input_ids.to(local_rank)
                # item_emb = model.module.id_pretrained_fusion(input_ids).to(torch.device('cpu')).detach()
                id_embs, VLLM_layers_embs, audio_embs = model.module.id_pretrained_fusion(input_ids)
                id_embs = id_embs.to(torch.device('cpu')).detach()
                VLLM_layers_embs = VLLM_layers_embs.to(torch.device('cpu')).detach()
                audio_embs = audio_embs.to(torch.device('cpu')).detach()
                id_scoring.extend(id_embs)
                VLVM_scoring.extend(VLLM_layers_embs)
                audio_scoring.extend(audio_embs)
        return torch.stack(tensors=id_scoring, dim=0), torch.stack(tensors=VLVM_scoring, dim=0), torch.stack(tensors=audio_scoring, dim=0)

def get_item_id_score_dff(model, item_num, test_batch_size, args, local_rank):
    model.eval()
    item_dataset = IdEvalDataset(data=np.arange(item_num + 1))
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=item_collate_fn)
    item_scoring = []
    id_scoring = []
    VLVM_scoring = []
    audio_scoring = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            # item_emb = model.module.id_pretrained_fusion(input_ids).to(torch.device('cpu')).detach()
            item_emb, id_embs, VLLM_layers_embs, audio_embs = model.module.id_pretrained_fusion(input_ids)
            item_emb = item_emb.to(torch.device('cpu')).detach()
            id_embs = id_embs.to(torch.device('cpu')).detach()
            VLLM_layers_embs = VLLM_layers_embs.to(torch.device('cpu')).detach()
            audio_embs = audio_embs.to(torch.device('cpu')).detach()
            item_scoring.extend(item_emb)
            id_scoring.extend(id_embs)
            VLVM_scoring.extend(VLLM_layers_embs)
            audio_scoring.extend(audio_embs)
    return torch.stack(tensors=item_scoring, dim=0), torch.stack(tensors=id_scoring, dim=0), torch.stack(tensors=VLVM_scoring, dim=0), torch.stack(tensors=audio_scoring, dim=0)

def get_item_id_score_onlyid(model, item_num, test_batch_size, args, local_rank):
    model.eval()
    item_dataset = IdEvalDataset(data=np.arange(item_num + 1))
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=item_collate_fn)
    id_scoring = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            # item_emb = model.module.id_pretrained_fusion(input_ids).to(torch.device('cpu')).detach()
            id_embs = model.module.id_pretrained_fusion(input_ids)
            id_embs = id_embs.to(torch.device('cpu')).detach()
            id_scoring.extend(id_embs)
    return torch.stack(tensors=id_scoring, dim=0)

def eval_model_dff_id_v(model, user_history, eval_seq, item_scoring, id_scoring, lvlm_scoring, test_batch_size, args, item_num, Log_file, v_or_t, local_rank, epoch):
    import torch
    from tqdm import tqdm
    from numpy import savetxt
    import numpy as np
    
    eval_dataset = EvalDataset_dff_id_v(u2seq=eval_seq, item_content=item_scoring, id_content=id_scoring, lvlm_content=lvlm_scoring,
                               max_seq_len=args.max_seq_len+1, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK_list = [10, 20]   # 支持多个topK
    Log_file.info(v_or_t + '_methods   {}'.format('\t'.join(['Hit{}'.format(k) for k in topK_list] +
                                                            ['nDCG{}'.format(k) for k in topK_list])))
    item_scoring = item_scoring.to(local_rank)
    with torch.no_grad():
        eval_all_user = {k: [] for k in topK_list}
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, id_embs, lvlm_embs, log_mask, labels = data
            user_ids, input_embs, id_embs, lvlm_embs, log_mask, labels = \
                user_ids.to(args.device), input_embs.to(args.device), id_embs.to(args.device), lvlm_embs.to(args.device), \
                log_mask.to(args.device), labels.to(args.device).detach()
            
            prec_emb = model.module.predict(input_embs, id_embs, lvlm_embs, log_mask, args)
            prec_emb = prec_emb.view(-1, args.max_seq_len, args.embedding_dim)
            prec_emb = prec_emb[:, -1].detach()  # [bz, dim]
            scores = torch.matmul(prec_emb, item_scoring.t()).squeeze(dim=-1).detach()
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                for k in topK_list:
                    rank, res = metrics_topK(score, label, item_rank, k, local_rank)
                    eval_all_user[k].append(res)
        mean_eval = []
        for k in topK_list:
            eval_all_user_k = torch.stack(tensors=eval_all_user[k], dim=0).t().contiguous()
            Hit, nDCG = eval_all_user_k
            mean_eval.extend(eval_concat([Hit, nDCG], test_sampler))
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0], mean_eval[1], mean_eval[2], mean_eval[3]  # [Hit10, nDCG10, Hit20, nDCG20]


def eval_model(model, user_history, eval_seq, item_scoring, id_scoring, lvlm_scoring, a_scoring, test_batch_size, args, item_num, Log_file, v_or_t, local_rank, epoch):
    import torch
    from tqdm import tqdm
    from numpy import savetxt
    import numpy as np
    
    eval_dataset = EvalDataset(u2seq=eval_seq, item_content=item_scoring, id_content=id_scoring, lvlm_content=lvlm_scoring, a_content=a_scoring,
                               max_seq_len=args.max_seq_len+1, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK_list = [10, 20]   # 支持多个topK
    Log_file.info(v_or_t + '_methods   {}'.format('\t'.join(['Hit{}'.format(k) for k in topK_list] +
                                                            ['nDCG{}'.format(k) for k in topK_list])))
    item_scoring = item_scoring.to(local_rank)
    with torch.no_grad():
        eval_all_user = {k: [] for k in topK_list}
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, id_embs, lvlm_embs, a_embs, log_mask, labels = data
            user_ids, input_embs, id_embs, lvlm_embs, a_embs, log_mask, labels = \
                user_ids.to(args.device), input_embs.to(args.device), id_embs.to(args.device), lvlm_embs.to(args.device), a_embs.to(args.device),\
                log_mask.to(args.device), labels.to(args.device).detach()
            # prec_emb = model.module.user_encoder(input_embs, log_mask, local_rank)[:, -1].detach()
            prec_emb = model.module.predict(input_embs, id_embs, lvlm_embs, a_embs, log_mask, args)
            prec_emb = prec_emb.view(-1, args.max_seq_len, args.embedding_dim)
            prec_emb = prec_emb[:, -1].detach()  # [bz, dim]
            scores = torch.matmul(prec_emb, item_scoring.t()).squeeze(dim=-1).detach()
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                for k in topK_list:
                    rank, res = metrics_topK(score, label, item_rank, k, local_rank)
                    eval_all_user[k].append(res)
        mean_eval = []
        for k in topK_list:
            eval_all_user_k = torch.stack(tensors=eval_all_user[k], dim=0).t().contiguous()
            Hit, nDCG = eval_all_user_k
            mean_eval.extend(eval_concat([Hit, nDCG], test_sampler))
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0], mean_eval[1], mean_eval[2], mean_eval[3]  # [Hit10, nDCG10, Hit20, nDCG20]

def eval_model_onlyid(model, user_history, eval_seq, id_scoring, test_batch_size, args, item_num, Log_file, v_or_t, local_rank, epoch):
    import torch
    from tqdm import tqdm
    from numpy import savetxt
    import numpy as np
    
    eval_dataset = EvalDataset_onlyid(u2seq=eval_seq, id_content=id_scoring,
                               max_seq_len=args.max_seq_len+1, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK_list = [10, 20]   # 支持多个topK
    Log_file.info(v_or_t + '_methods   {}'.format('\t'.join(['Hit{}'.format(k) for k in topK_list] +
                                                            ['nDCG{}'.format(k) for k in topK_list])))
    id_scoring = id_scoring.to(local_rank)
    with torch.no_grad():
        eval_all_user = {k: [] for k in topK_list}
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, id_embs, log_mask, labels = data
            user_ids, id_embs, log_mask, labels = \
                user_ids.to(args.device), id_embs.to(args.device),\
                log_mask.to(args.device), labels.to(args.device).detach()

            prec_emb = model.module.predict(id_embs, log_mask, args)
            prec_emb = prec_emb.view(-1, args.max_seq_len, args.embedding_dim)
            prec_emb = prec_emb[:, -1].detach()  # [bz, dim]
            scores = torch.matmul(prec_emb, id_scoring.t()).squeeze(dim=-1).detach()
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                for k in topK_list:
                    rank, res = metrics_topK(score, label, item_rank, k, local_rank)
                    eval_all_user[k].append(res)
        mean_eval = []
        for k in topK_list:
            eval_all_user_k = torch.stack(tensors=eval_all_user[k], dim=0).t().contiguous()
            Hit, nDCG = eval_all_user_k
            mean_eval.extend(eval_concat([Hit, nDCG], test_sampler))
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0], mean_eval[1], mean_eval[2], mean_eval[3]  # [Hit10, nDCG10, Hit20, nDCG20]

def eval_model_M3BSR(model, user_history, eval_seq, id_scoring, lvlm_scoring, a_scoring, test_batch_size, args, item_num, Log_file, v_or_t, local_rank, epoch):
    import torch
    from tqdm import tqdm
    from numpy import savetxt
    import numpy as np
    
    eval_dataset = EvalDataset_M3BSR(u2seq=eval_seq, id_content=id_scoring, lvlm_content=lvlm_scoring, a_content=a_scoring,
                               max_seq_len=args.max_seq_len+1, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK_list = [10, 20]   # 支持多个topK
    Log_file.info(v_or_t + '_methods   {}'.format('\t'.join(['Hit{}'.format(k) for k in topK_list] +
                                                            ['nDCG{}'.format(k) for k in topK_list])))
    with torch.no_grad():
        eval_all_user = {k: [] for k in topK_list}
        # print(eval_all_user)
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, id_embs, lvlm_embs, a_embs, log_mask, labels = data
            user_ids, id_embs, lvlm_embs, a_embs, log_mask, labels = \
                user_ids.to(args.device), id_embs.to(args.device), lvlm_embs.to(args.device), a_embs.to(args.device),\
                log_mask.to(args.device), labels.to(args.device).detach()
            # print("data ok")
            prec_emb = model.module.predict(id_embs, lvlm_embs, a_embs, log_mask, args)
            # print("predict ok")
            # print(prec_emb.shape)
            prec_emb = prec_emb.view(-1, args.max_seq_len, args.embedding_dim)
            prec_emb = prec_emb[:, -1].detach()  # [bz, dim]
            id_embs = id_embs.reshape(-1, args.embedding_dim)
            # print(prec_emb.shape, id_embs.shape)
            # print(id_scoring.shape)
            # print(prec_emb.device)
            # print(id_scoring.device)
            id_scoring = id_scoring.to(args.device)
            scores = torch.matmul(prec_emb, id_scoring.t()).squeeze(dim=-1).detach()
            # print(scores.shape)
            # print("user_ids", user_ids)
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                # print("user_id")
                history = user_history[user_id].to(local_rank)
                # print("history")
                score[history] = -np.inf
                score = score[1:]
                # print("score")
                for k in topK_list:
                    # print("topK_list")
                    rank, res = metrics_topK(score, label, item_rank, k, local_rank)
                    # print("rank")
                    eval_all_user[k].append(res)
        mean_eval = []
        # print("mean_eval")
        for k in topK_list:
            eval_all_user_k = torch.stack(tensors=eval_all_user[k], dim=0).t().contiguous()
            Hit, nDCG = eval_all_user_k
            mean_eval.extend(eval_concat([Hit, nDCG], test_sampler))
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0], mean_eval[1], mean_eval[2], mean_eval[3]  # [Hit10, nDCG10, Hit20, nDCG20]

def print_test_result(model, user_history, eval_seq, item_scoring, id_scoring, lvlm_scoring, test_batch_size, args, item_num, Log_file, local_rank):
    import torch
    from tqdm import tqdm
    from numpy import savetxt
    import numpy as np
    
    eval_dataset = EvalDataset_dff_id_v(u2seq=eval_seq, item_content=item_scoring, id_content=id_scoring, lvlm_content=lvlm_scoring,
                               max_seq_len=args.max_seq_len+1, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK_list = [10, 20]   # 支持多个topK
    #
    rank_result = []
    #

    item_scoring = item_scoring.to(local_rank)
    with torch.no_grad():
        eval_all_user = {k: [] for k in topK_list}
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, id_embs, lvlm_embs, log_mask, labels = data
            user_ids, input_embs, id_embs, lvlm_embs, log_mask, labels = \
                user_ids.to(args.device), input_embs.to(args.device), id_embs.to(args.device), lvlm_embs.to(args.device), \
                log_mask.to(args.device), labels.to(args.device).detach()
            
            prec_emb = model.module.predict(input_embs, id_embs, lvlm_embs, log_mask, args)
            prec_emb = prec_emb.view(-1, args.max_seq_len, args.embedding_dim)
            prec_emb = prec_emb[:, -1].detach()  # [bz, dim]

            scores = torch.matmul(prec_emb, item_scoring.t()).squeeze(dim=-1).detach()
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]

                order = torch.argsort(score, descending=True)
                y_true = torch.take(label, order)
                rank = torch.sum(y_true * item_rank)
                #print
                # user_emb_np = prec_emb[i].cpu().numpy()
                # save_path = os.path.join('/opt/data/private/work/vllmembs4rec/sasrec/result/userembs', f"{user_id}.npy")
                # np.save(save_path, user_emb_np)
                rank_result.append([user_id, int(rank.item())])
                # print([user_id, int(rank.item)])
                
        with open('/opt/data/private/work/vllmembs4rec/sasrec/result/userid_rank_dff_id_v.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['UserID', 'Rank'])
            writer.writerows(rank_result)

def print_test_result_onlyid(model, user_history, eval_seq, id_scoring, test_batch_size, args, item_num, Log_file, local_rank):
    import torch
    from tqdm import tqdm
    from numpy import savetxt
    import numpy as np
    
    eval_dataset = EvalDataset_onlyid(u2seq=eval_seq, id_content=id_scoring,
                               max_seq_len=args.max_seq_len+1, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK_list = [10, 20]   # 支持多个topK
    id_scoring = id_scoring.to(local_rank)
    #
    rank_result = []
    #
    with torch.no_grad():
        eval_all_user = {k: [] for k in topK_list}
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, id_embs, log_mask, labels = data
            user_ids, id_embs, log_mask, labels = \
                user_ids.to(args.device), id_embs.to(args.device),\
                log_mask.to(args.device), labels.to(args.device).detach()

            prec_emb = model.module.predict(id_embs, log_mask, args)
            prec_emb = prec_emb.view(-1, args.max_seq_len, args.embedding_dim)
            prec_emb = prec_emb[:, -1].detach()  # [bz, dim]
            scores = torch.matmul(prec_emb, id_scoring.t()).squeeze(dim=-1).detach()
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]

                #
                order = torch.argsort(score, descending=True)
                y_true = torch.take(label, order)
                rank = torch.sum(y_true * item_rank)
                rank_result.append([user_id, int(rank.item())])
        with open('/opt/data/private/work/vllmembs4rec/sasrec/result/userid_rank_onlyid.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['UserID', 'Rank'])
            writer.writerows(rank_result)

def print_test_result_diffusion(model, user_history, eval_seq, item_scoring, id_scoring, lvlm_scoring, a_scoring, test_batch_size, args, item_num, Log_file, local_rank):
    import torch
    from tqdm import tqdm
    from numpy import savetxt
    import numpy as np
    
    eval_dataset = EvalDataset(u2seq=eval_seq, item_content=item_scoring, id_content=id_scoring, lvlm_content=lvlm_scoring, a_content=a_scoring,
                               max_seq_len=args.max_seq_len+1, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK_list = [10, 20]   # 支持多个topK
    rank_result = []
    item_scoring = item_scoring.to(local_rank)
    with torch.no_grad():
        eval_all_user = {k: [] for k in topK_list}
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, id_embs, lvlm_embs, a_embs, log_mask, labels = data
            user_ids, input_embs, id_embs, lvlm_embs, a_embs, log_mask, labels = \
                user_ids.to(args.device), input_embs.to(args.device), id_embs.to(args.device), lvlm_embs.to(args.device), a_embs.to(args.device),\
                log_mask.to(args.device), labels.to(args.device).detach()
            # prec_emb = model.module.user_encoder(input_embs, log_mask, local_rank)[:, -1].detach()
            prec_emb = model.module.predict(input_embs, id_embs, lvlm_embs, a_embs, log_mask, args)
            prec_emb = prec_emb.view(-1, args.max_seq_len, args.embedding_dim)
            prec_emb = prec_emb[:, -1].detach()  # [bz, dim]
            scores = torch.matmul(prec_emb, item_scoring.t()).squeeze(dim=-1).detach()
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]

                order = torch.argsort(score, descending=True)
                y_true = torch.take(label, order)
                rank = torch.sum(y_true * item_rank)
                rank_result.append([user_id, int(rank.item())])
        with open('/opt/data/private/work/vllmembs4rec/sasrec/result/userid_rank_dff_diffusion_id_v_condition_a.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['UserID', 'Rank'])
            writer.writerows(rank_result)