import torch
from torch import nn
from torch.nn.init import xavier_normal_

from .user_encoders import User_Encoder_SASRec


class Model_dff_id_v(torch.nn.Module):
    """DFF id+v: gated fusion of ID embedding and video (VLLM) features."""

    def __init__(self, args, item_num, text_content=None, pretrained_embs=None):
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.item_num = item_num
        self.pretrained_embs = pretrained_embs

        self.user_encoder = User_Encoder_SASRec(args)
        self.criterion = nn.CrossEntropyLoss()

        self.id_encoder = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=args.embedding_dim,
            padding_idx=0,
        )
        xavier_normal_(self.id_encoder.weight.data)

        self.layer_num = pretrained_embs.shape[1]
        self.pretrained_emb_dim = pretrained_embs.shape[-1]
        self.proj = nn.Linear(self.pretrained_emb_dim, args.embedding_dim)
        self.gating = nn.Sequential(
            nn.Linear(args.embedding_dim * 2, args.embedding_dim),
            nn.Sigmoid(),
        )
        self.weights = nn.Parameter(torch.ones(self.layer_num, dtype=torch.float32))

    def id_pretrained_fusion(self, sample_items_id):
        id_embs = self.id_encoder(sample_items_id)
        vllm_embs = self.pretrained_embs[sample_items_id].to(sample_items_id.device)

        norm_weights = torch.softmax(self.weights, dim=0)
        vllm_layers_embs = torch.einsum("...ld,l->...d", vllm_embs, norm_weights)
        vllm_layers_embs = torch.relu(self.proj(vllm_layers_embs))

        cat_embs = torch.cat((id_embs, vllm_layers_embs), dim=-1)
        gate = self.gating(cat_embs)
        score_embs = gate * id_embs + (1 - gate) * vllm_layers_embs
        return score_embs, id_embs, vllm_layers_embs

    def forward(self, sample_items_id, log_mask, local_rank, args):
        score_embs, _, _ = self.id_pretrained_fusion(sample_items_id)
        input_embs = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)
        prec_vec = self.user_encoder(input_embs[:, :-1, :], log_mask, local_rank)
        prec_vec = prec_vec.reshape(-1, self.args.embedding_dim)

        bs, seq_len = log_mask.size(0), log_mask.size(1)
        logits = torch.matmul(prec_vec, score_embs.t())
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
        return self.criterion(logits[indices], label[indices])

    def predict(self, input_embs, id_embs, lvlm_embs, log_mask, local_rank):
        del id_embs, lvlm_embs
        prec_vec = self.user_encoder(input_embs, log_mask, local_rank)
        return prec_vec.reshape(-1, self.args.embedding_dim)
