import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .dataset import EvalDataset_dff_id_v, IdEvalDataset, SequentialDistributedSampler


def item_collate_fn(arr):
    return torch.LongTensor(np.array(arr))


def print_metrics(x, Log_file, v_or_t):
    Log_file.info(v_or_t + '_results   {}'.format('\t'.join(['{:0.5f}'.format(i * 100) for i in x])))


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]


def eval_concat(eval_list, test_sampler):
    eval_result = []
    for eval_m in eval_list:
        eval_m_cpu = distributed_concat(eval_m, len(test_sampler.dataset)).to(torch.device('cpu')).numpy()
        eval_result.append(eval_m_cpu.mean())
    return eval_result


def metrics_topK(y_score, y_true, item_rank, topK, local_rank):
    order = torch.argsort(y_score, descending=True)
    y_true = torch.take(y_true, order)
    rank = torch.sum(y_true * item_rank)
    eval_ra = torch.zeros(2).to(local_rank)
    if rank <= topK:
        eval_ra[0] = 1
        eval_ra[1] = 1 / math.log2(rank + 1)
    return rank, eval_ra


def get_item_embedding(model, item_num, test_batch_size, args, local_rank):
    model.eval()
    item_dataloader = DataLoader(
        IdEvalDataset(data=np.arange(item_num + 1)),
        batch_size=test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=item_collate_fn,
    )
    item_scoring, id_scoring, vllm_scoring = [], [], []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            item_emb, id_embs, vllm_layers_embs = model.module.id_pretrained_fusion(input_ids)
            item_scoring.extend(item_emb.to(torch.device('cpu')).detach())
            id_scoring.extend(id_embs.to(torch.device('cpu')).detach())
            vllm_scoring.extend(vllm_layers_embs.to(torch.device('cpu')).detach())
    return (
        torch.stack(tensors=item_scoring, dim=0),
        torch.stack(tensors=id_scoring, dim=0),
        torch.stack(tensors=vllm_scoring, dim=0),
    )


def eval_model_dff_id_v(model, user_history, eval_seq, item_scoring, id_scoring, lvlm_scoring,
                        test_batch_size, args, item_num, Log_file, v_or_t, local_rank, epoch):
    del epoch
    eval_dataset = EvalDataset_dff_id_v(
        u2seq=eval_seq,
        item_content=item_scoring,
        id_content=id_scoring,
        lvlm_content=lvlm_scoring,
        max_seq_len=args.max_seq_len + 1,
        item_num=item_num,
    )
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(
        eval_dataset,
        batch_size=test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=test_sampler,
    )
    model.eval()
    topK_list = [10, 20]
    Log_file.info(
        v_or_t + '_methods   {}'.format(
            '\t'.join(['Hit{}'.format(k) for k in topK_list] + ['nDCG{}'.format(k) for k in topK_list])
        )
    )
    item_scoring = item_scoring.to(local_rank)
    with torch.no_grad():
        eval_all_user = {k: [] for k in topK_list}
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for user_ids, input_embs, id_embs, lvlm_embs, log_mask, labels in eval_dl:
            user_ids = user_ids.to(local_rank)
            input_embs = input_embs.to(local_rank)
            id_embs = id_embs.to(local_rank)
            lvlm_embs = lvlm_embs.to(local_rank)
            log_mask = log_mask.to(local_rank)
            labels = labels.to(local_rank).detach()

            prec_emb = model.module.predict(input_embs, id_embs, lvlm_embs, log_mask, local_rank)
            prec_emb = prec_emb.view(-1, args.max_seq_len, args.embedding_dim)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_scoring.t()).squeeze(dim=-1).detach()

            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                for k in topK_list:
                    _, res = metrics_topK(score, label, item_rank, k, local_rank)
                    eval_all_user[k].append(res)

        mean_eval = []
        for k in topK_list:
            eval_all_user_k = torch.stack(tensors=eval_all_user[k], dim=0).t().contiguous()
            Hit, nDCG = eval_all_user_k
            mean_eval.extend(eval_concat([Hit, nDCG], test_sampler))
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0], mean_eval[1], mean_eval[2], mean_eval[3]
