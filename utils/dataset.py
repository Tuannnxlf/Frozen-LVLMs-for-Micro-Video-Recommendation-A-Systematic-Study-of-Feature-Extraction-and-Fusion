import math
import numpy as np
import torch
from torch.utils.data import Dataset


class IdDataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, args):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.args = args

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_len = len(seq)
        tokens_len = seq_len - 1
        mask_len_head = self.max_seq_len - seq_len
        log_mask = [0] * mask_len_head + [1] * tokens_len
        sample_items = [0] * mask_len_head + seq
        return torch.LongTensor(np.array(sample_items)), torch.FloatTensor(log_mask)


class IdEvalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class EvalDataset_dff_id_v(Dataset):
    def __init__(self, u2seq, item_content, id_content, lvlm_content, max_seq_len, item_num):
        self.u2seq = u2seq
        self.item_content = item_content
        self.id_content = id_content
        self.lvlm_content = lvlm_content
        self.max_seq_len = max_seq_len
        self.item_num = item_num

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        tokens = seq[:-1]
        target = seq[-1]
        mask_len = self.max_seq_len - len(seq)
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        input_embs = self.item_content[pad_tokens]
        id_embs = self.id_content[pad_tokens]
        lvlm_embs = self.lvlm_content[pad_tokens]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return (
            torch.LongTensor([user_id]),
            input_embs,
            id_embs,
            lvlm_embs,
            torch.FloatTensor(log_mask),
            labels,
        )


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices += [indices[-1]] * (self.total_size - len(indices))
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
