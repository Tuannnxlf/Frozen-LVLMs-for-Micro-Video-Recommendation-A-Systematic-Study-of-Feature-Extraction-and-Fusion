# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import uniform_, xavier_normal_, constant_, xavier_uniform_

from .modules import TransformerEncoder

class User_Encoder_SASRec(torch.nn.Module):
    def __init__(self, args):
        super(User_Encoder_SASRec, self).__init__()

        self.transformer_encoder = TransformerEncoder(n_vocab=None, n_position=args.max_seq_len,
                                                      d_model=args.embedding_dim, n_heads=args.num_attention_heads,
                                                      dropout=args.drop_rate, n_layers=args.transformer_block)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input_embs, log_mask, local_rank):
        att_mask = (log_mask != 0)
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank)
        att_mask = torch.where(att_mask, 0., -1e9)
        return self.transformer_encoder(input_embs, log_mask, att_mask)

class User_Encoder_GRU4Rec(nn.Module):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:
        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, args):
        super().__init__()

        self.embedding_size = args.embedding_dim
        self.n_layers = args.block_num
        self.hidden_size = args.embedding_dim
        self.dropout = args.drop_rate

        # define layers
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bias=False,
            batch_first=True,
        )
        self.emb_dropout = nn.Dropout(self.dropout)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq_emb):
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)

        return gru_output