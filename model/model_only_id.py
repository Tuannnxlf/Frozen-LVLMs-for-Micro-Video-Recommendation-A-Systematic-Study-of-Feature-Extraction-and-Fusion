import torch
import math
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_
from collections import Counter
import torch.nn.functional as F
from tqdm import tqdm

from .user_encoders import User_Encoder_SASRec, User_Encoder_GRU4Rec

class Model_onlyid(torch.nn.Module):
    def __init__(self, args, item_num, text_content=None, pretrained_embs=None, audio_embs=None):
        super(Model_onlyid, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.item_num = item_num

        self.user_encoder = User_Encoder_SASRec(args)

        self.id_encoder = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=args.embedding_dim,
            padding_idx=0
        )
        xavier_normal_(self.id_encoder.weight.data)

        self.criterion = nn.CrossEntropyLoss()

    def id_pretrained_fusion(self, sample_items_id):
        id_embs = self.id_encoder(sample_items_id)
        return id_embs

    def forward(self, sample_items_id, log_mask, local_rank, args):
        id_embs = self.id_pretrained_fusion(sample_items_id)
        input_embs = id_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)
        prec_vec = self.user_encoder(input_embs[:, :-1, :], log_mask, local_rank)
        prec_vec = prec_vec.reshape(-1, self.args.embedding_dim)

        bs, seq_len = log_mask.size(0), log_mask.size(1)
        logits = torch.matmul(prec_vec, id_embs.t())

        label = torch.arange(bs * (seq_len + 1)).reshape(bs, seq_len + 1)
        label = label[:, 1:].to(local_rank).view(-1)

        flatten_item_seq = sample_items_id
        user_history = torch.zeros(bs, seq_len + 2).type_as(sample_items_id)
        user_history[:, :-1] = sample_items_id.view(bs, -1)
        user_history = user_history.unsqueeze(-1).expand(-1, -1, len(flatten_item_seq))
        history_item_mask = (user_history == flatten_item_seq).any(dim=1)
        history_item_mask = history_item_mask.repeat_interleave(seq_len, dim=0)
        unused_item_mask = torch.scatter(history_item_mask, 1, label.view(-1, 1), False)
        
        logits[unused_item_mask] = -1e4
        indices = torch.where(log_mask.view(-1) != 0)
        logits = logits.view(bs * seq_len, -1)

        loss = self.criterion(logits[indices], label[indices])
        return loss
    
    def predict(self, input_embs, log_mask, args):
        prec_vec = self.user_encoder(input_embs, log_mask, args.device) # [bz, seqlen, dim]
        return prec_vec