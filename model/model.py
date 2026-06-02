import torch
import math
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_
from collections import Counter
import torch.nn.functional as F
from tqdm import tqdm

from .user_encoders import User_Encoder_SASRec, User_Encoder_GRU4Rec

def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

class Model(torch.nn.Module):
    def __init__(self, args, item_num, text_content=None, pretrained_embs=None, audio_embs=None):
        super(Model, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.w = args.w
        self.item_num = item_num
        self.pretrained_embs = pretrained_embs
        self.audio_embs = audio_embs

        self.user_encoder = User_Encoder_SASRec(args)
        # self.user_encoder_id = User_Encoder_SASRec(args)
        self.user_encoder_v = User_Encoder_SASRec(args)
        # self.user_encoder_a = User_Encoder_SASRec(args)
        self.criterion = nn.CrossEntropyLoss()

        self.id_encoder = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=args.embedding_dim,
            padding_idx=0
        )
        xavier_normal_(self.id_encoder.weight.data)
        
        self.layer_num = pretrained_embs.shape[1]
        self.pretrained_emb_dim = pretrained_embs.shape[-1]
        self.audio_embs_dim = audio_embs.shape[-1]

        self.proj = nn.Linear(self.pretrained_emb_dim, args.embedding_dim)
        self.proj_a = nn.Linear(self.audio_embs_dim, args.embedding_dim)

        self.gating = nn.Sequential(
            nn.Linear(args.embedding_dim * 2, args.embedding_dim),
            nn.Sigmoid()
        )
        self.gating_a = nn.Sequential(
            nn.Linear(args.embedding_dim * 2, args.embedding_dim),
            nn.Sigmoid()
        )

        self.weights = nn.Parameter(torch.ones(self.layer_num, dtype=torch.float32))

        # diffusion
        self.dev = args.device
        self.timesteps = args.timesteps
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        # selfattention for denoise
        self.ln=torch.nn.LayerNorm(self.args.embedding_dim,elementwise_affine=False)
        self.w_q=torch.nn.Linear(self.args.embedding_dim,self.args.embedding_dim,bias=False)
        self.init(self.w_q)
        self.w_k=torch.nn.Linear(self.args.embedding_dim,self.args.embedding_dim)
        self.init(self.w_k)
        self.w_v=torch.nn.Linear(self.args.embedding_dim,self.args.embedding_dim)
        self.init(self.w_v)
        self.ln=torch.nn.LayerNorm(self.args.embedding_dim,elementwise_affine=False)
        # 预计算核心参数
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        # 前向加噪
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def init(self,m):
        if isinstance(m,torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias,0)
        elif isinstance(m,torch.nn.Parameter):
            torch.nn.init.xavier_normal_(m)

    def id_pretrained_fusion(self, sample_items_id):
        id_embs = self.id_encoder(sample_items_id)
        VLLM_embs = self.pretrained_embs[sample_items_id].to(sample_items_id.device)
        audio_embs = self.audio_embs[sample_items_id].to(sample_items_id.device)

        norm_weights = torch.softmax(self.weights, dim=0)
        VLLM_layers_embs = torch.einsum("...ld,l->...d", VLLM_embs, norm_weights)
        VLLM_layers_embs = torch.relu(self.proj(VLLM_layers_embs))
        audio_embs = torch.relu(self.proj_a(audio_embs))
        # id+a
        cat_embs = torch.cat((id_embs, audio_embs), dim=-1)
        gate = self.gating(cat_embs)
        score_embs = gate * id_embs + (1 - gate) * audio_embs
        # id+v + a
        # cat_a_embs = torch.cat((score_embs, audio_embs), dim=-1)
        # gate_a = self.gating_a(cat_a_embs)
        # score_a_embs = gate_a * score_embs + (1 - gate_a) * audio_embs
        return score_embs, id_embs, VLLM_layers_embs, audio_embs

    def forward(self, sample_items_id, log_mask, local_rank, args):
        if args.method == 'mmfusion':
            score_embs, id_embs, VLLM_layers_embs, audio_embs = self.id_pretrained_fusion(sample_items_id)
            input_embs = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)     
            prec_vec = self.user_encoder(input_embs[:, :-1, :], log_mask, local_rank)
            prec_vec = prec_vec.reshape(-1, self.args.embedding_dim)
            # # modify diffusion loss
            target_embs = input_embs[:, 1:, :].reshape(-1, self.args.embedding_dim)
            # # onlyid_user
            # input_embs_id = id_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)
            # prec_vec_id = self.user_encoder_id(input_embs_id[:, :-1, :], log_mask, args.device)
            # prec_vec_id = prec_vec_id.reshape(-1, self.args.embedding_dim)
            # v_user
            input_embs_v = VLLM_layers_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)
            prec_vec_v = self.user_encoder_v(input_embs_v[:, :-1, :], log_mask, args.device) # [bz, seqlen, dim]
            prec_vec_v = prec_vec_v.reshape(-1, self.args.embedding_dim)
            # audio_user
            # input_embs_a = VLLM_layers_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)
            # prec_vec_a = self.user_encoder_a(input_embs_a[:, :-1, :], log_mask, args.device) # [bz, seqlen, dim]
            # prec_vec_a = prec_vec_a.reshape(-1, self.args.embedding_dim)
            # diffusion
            bs, seq_len = log_mask.size(0), log_mask.size(1)
            times_info = torch.randint(0, self.timesteps, (bs*seq_len,), device=self.dev).long()
            prec_vec_noise = self.q_sample(prec_vec, times_info)
            times_info_embedding = self.get_time_s(times_info)

            diffu_log_feats=torch.cat([prec_vec_noise.unsqueeze(1), prec_vec_v.unsqueeze(1), times_info_embedding.unsqueeze(1)],dim=1)
            prec_vec_final_feats=self.selfAttention(diffu_log_feats)

            u_p=(1-self.w)*prec_vec_final_feats + self.w*prec_vec

            logits = torch.matmul(u_p, score_embs.t())

            
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

            celoss = self.criterion(logits[indices], label[indices])
            diffuloss = F.mse_loss(prec_vec_final_feats, target_embs)
            loss = celoss + self.args.diffusion_loss_weight * diffuloss
            # loss = self.criterion(logits[indices], label[indices])
        else :
            score_embs, id_embs, VLLM_layers_embs, audio_embs = self.id_pretrained_fusion(sample_items_id)
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

            loss = self.criterion(logits[indices], label[indices])
        return loss
    
    def predict(self, input_embs, id_embs, lvlm_embs, a_embs, log_mask, args):
            prec_vec = self.user_encoder(input_embs, log_mask, args.device) # [bz, seqlen, dim]
            prec_vec = prec_vec.reshape(-1, self.args.embedding_dim)    # [bz*seqlen, dim]

            if args.method == 'mmfusion':
                noise = torch.randn_like(prec_vec)
                # # id_user
                # prec_vec_id = self.user_encoder_id(id_embs, log_mask, args.device) # [bz, seqlen, dim]
                # prec_vec_id = prec_vec_id.reshape(-1, self.args.embedding_dim)
                # v_user
                prec_vec_v = self.user_encoder_v(lvlm_embs, log_mask, args.device) # [bz, seqlen, dim]
                prec_vec_v = prec_vec_v.reshape(-1, self.args.embedding_dim)
                # # a_user
                # prec_vec_a = self.user_encoder_a(lvlm_embs, log_mask, args.device) # [bz, seqlen, dim]
                # prec_vec_a = prec_vec_a.reshape(-1, self.args.embedding_dim)

                for i in reversed(range(0, self.timesteps)):
                    t = torch.tensor([i] * prec_vec.shape[0], dtype=torch.long).to(self.dev)
                    times_info_embeddings = self.get_time_s(t)
                    diffu_log_feats = torch.cat([prec_vec.unsqueeze(1), prec_vec_v.unsqueeze(1), times_info_embeddings.unsqueeze(1)], dim=1)

                    x_start_prec_vec=self.selfAttention(diffu_log_feats)

                    model_mean_prec_vec = (
                            self.extract(self.posterior_mean_coef1, t, x_start_prec_vec.shape) * x_start_prec_vec +
                            self.extract(self.posterior_mean_coef2, t, x_start_prec_vec.shape) * noise
                    )

                    if i == 0:
                        x_prec_vec = model_mean_prec_vec
                    else:
                        # ---
                        posterior_variance_t = self.extract(self.posterior_variance, t, prec_vec.shape)
                        noise_prec_vec = torch.randn_like(prec_vec)
                        x_prec_vec = model_mean_prec_vec + torch.sqrt(posterior_variance_t) * noise_prec_vec

                output_prec_vec = (1-self.w) * x_prec_vec + self.w * prec_vec
            else:
                output_prec_vec = prec_vec
            return output_prec_vec
    
    def get_time_s(self, time):
        half_dim = self.args.embedding_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=self.dev) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def extract(self, a, t, x_shape):
        # res = a.to(device=t.device)[t].float()
        # while len(res.shape) < len(x_shape):
        #     res = res[..., None]
        # return res.expand(x_shape)
        # batch_size = t.shape[0]
        # out = a.gather(-1, t.cpu())
        # return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.dev)
        res = a.to(device=t.device)[t].float()
        while len(res.shape) < len(x_shape):
            res = res[..., None]
        return res.expand(x_shape)

    def q_sample(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        # mean=torch.mean(x_start,dim=0)
        # noise=mean+(x_start - mean).pow(2).mean().sqrt()*noise
        # return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise*torch.sign(x_start)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def selfAttention(self,features):
        features=self.ln(features)
        q=self.w_q(features)
        k=self.w_k(features)
        v=self.w_v(features)

        attn=q.mul(self.args.embedding_dim**-0.5) @ k.transpose(-1,-2)
        attn=attn.softmax(dim=-1)

        features=attn @ v
        # print("features.size:",features.size())
        # y=features
        # print("y.size:",y.size())
        return features[:,0,:]