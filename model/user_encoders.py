import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_

from .modules import TransformerEncoder


class User_Encoder_SASRec(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.transformer_encoder = TransformerEncoder(
            n_vocab=None,
            n_position=args.max_seq_len,
            d_model=args.embedding_dim,
            n_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block,
        )
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
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank)
        att_mask = torch.where(att_mask, 0.0, -1e9)
        return self.transformer_encoder(input_embs, log_mask, att_mask)
