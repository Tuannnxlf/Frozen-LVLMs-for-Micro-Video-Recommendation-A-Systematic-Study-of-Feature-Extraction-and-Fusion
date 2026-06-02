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

class M3BSR(torch.nn.Module):
    def __init__(self, args, item_num, text_content=None, pretrained_embs=None, audio_embs=None):
        super(M3BSR, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.w = args.w
        self.item_num = item_num
        self.pretrained_embs = pretrained_embs
        self.audio_embs = audio_embs
        # 对比学习
        self.cl_weight = args.cl_weight
        self.cl_temp = args.cl_temp

        self.user_encoder_id = User_Encoder_SASRec(args)
        self.user_encoder_v = User_Encoder_SASRec(args)
        self.user_encoder_a = User_Encoder_SASRec(args)
        self.user_encoder_common = User_Encoder_SASRec(args)
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
        # lvlm layers weight
        self.weights = nn.Parameter(torch.ones(self.layer_num, dtype=torch.float32))
        # 模态特征映射
        self.proj = nn.Linear(self.pretrained_emb_dim, args.embedding_dim)
        self.proj_a = nn.Linear(self.audio_embs_dim, args.embedding_dim)
        # hcommon降维
        self.fusion_proj = nn.Linear(self.args.embedding_dim * 3, self.args.embedding_dim)
        # prec各模态拼接降维
        self.prec_proj = nn.Linear(self.args.embedding_dim * 3, self.args.embedding_dim)
        # gate net for Interest Routing Fusion
        self.gate_net = nn.Linear(args.embedding_dim * 2, 1)
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
        # score_embs = torch.cat([VLLM_layers_embs, audio_embs, id_embs], dim=-1)
        return id_embs, VLLM_layers_embs, audio_embs

    def forward(self, sample_items_id, log_mask, local_rank, args):
        # 获取各模态item embedding
        id_embs, VLLM_embs, audio_embs = self.id_pretrained_fusion(sample_items_id)
        # 对v a两模态加噪
        # print(VLLM_embs.shape)
        VLLM_embs = VLLM_embs.reshape(-1, self.args.embedding_dim)
        # print(VLLM_embs.shape)
        audio_embs = audio_embs.reshape(-1, self.args.embedding_dim)
        # diffusion forward
        bs, seq_len = log_mask.size(0), log_mask.size(1)
        times_info = torch.randint(0, self.timesteps, (bs*(seq_len+1),), device=self.dev).long()
        VLLM_embs_noise = self.q_sample(VLLM_embs, times_info)
        audio_embs_noise = self.q_sample(audio_embs, times_info)
        times_info_embedding = self.get_time_s(times_info)
        #以id为条件 对v a两模态进行去噪
        diffu_log_feats_v = torch.cat([VLLM_embs_noise.unsqueeze(1), id_embs.unsqueeze(1), times_info_embedding.unsqueeze(1)],dim=1)
        v_embs_final_feats = self.selfAttention(diffu_log_feats_v)
        diffu_log_feats_a = torch.cat([audio_embs_noise.unsqueeze(1), id_embs.unsqueeze(1), times_info_embedding.unsqueeze(1)],dim=1)
        a_embs_final_feats = self.selfAttention(diffu_log_feats_a)
        # preferance
        input_embs_v = v_embs_final_feats.view(-1, self.max_seq_len + 1, self.args.embedding_dim)
        input_embs_a = a_embs_final_feats.view(-1, self.max_seq_len + 1, self.args.embedding_dim)
        input_embs_id = id_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim) 
        fusion_embs = torch.cat([input_embs_v, input_embs_a, input_embs_id], dim=-1)
        input_embs_common = self.fusion_proj(fusion_embs)
        prec_vec_v = self.user_encoder_v(input_embs_v[:, :-1, :], log_mask, args.device)
        prec_vec_a = self.user_encoder_a(input_embs_a[:, :-1, :], log_mask, args.device)
        prec_vec_id = self.user_encoder_id(input_embs_id[:, :-1, :], log_mask, args.device)
        prec_vec_common = self.user_encoder_common(input_embs_common[:, :-1, :], log_mask, args.device)
        prec_vec_v = prec_vec_v.reshape(-1, self.args.embedding_dim)
        prec_vec_a = prec_vec_a.reshape(-1, self.args.embedding_dim)
        prec_vec_id = prec_vec_id.reshape(-1, self.args.embedding_dim)
        prec_vec_common = prec_vec_common.reshape(-1, self.args.embedding_dim)
        # Interest Routing Fusion
        prec_vec = self.prec_proj(torch.cat([prec_vec_v, prec_vec_a, prec_vec_id], dim=-1))
        gate_input = torch.cat([prec_vec_common, prec_vec], dim=-1)
        g = torch.sigmoid(self.gate_net(gate_input))
        final_output = g * prec_vec_common + (1 - g) * prec_vec

        logits = torch.matmul(final_output, id_embs.t())
        
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
        diffuloss = F.mse_loss(v_embs_final_feats, VLLM_embs) + F.mse_loss(a_embs_final_feats, audio_embs)
        # 对比学习loss 共四种特征
        features = [prec_vec_v, prec_vec_a, prec_vec_id, prec_vec_common]
        loss_cl = self.cl_loss_matrix(features)

        loss = celoss + self.args.diffusion_loss_weight * diffuloss + self.cl_weight * loss_cl
        return loss
    
    def predict(self, id_embs, lvlm_embs, a_embs, log_mask, args):
        
        id_embs = id_embs.reshape(-1, self.args.embedding_dim)
        lvlm_embs = lvlm_embs.reshape(-1, self.args.embedding_dim)
        a_embs = a_embs.reshape(-1, self.args.embedding_dim)
        noise_v = torch.randn_like(lvlm_embs)
        noise_a = torch.randn_like(a_embs)

        # diffusion reverse
        for i in reversed(range(0, self.timesteps)):
            t = torch.tensor([i] * lvlm_embs.shape[0], dtype=torch.long).to(self.dev)
            times_info_embeddings = self.get_time_s(t)
            diffu_log_lvlm_embs = torch.cat([lvlm_embs.unsqueeze(1), id_embs.unsqueeze(1), times_info_embeddings.unsqueeze(1)], dim=1)

            x_start_lvlm_embs=self.selfAttention(diffu_log_lvlm_embs)

            model_mean_lvlm_embs = (
                    self.extract(self.posterior_mean_coef1, t, x_start_lvlm_embs.shape) * x_start_lvlm_embs +
                    self.extract(self.posterior_mean_coef2, t, x_start_lvlm_embs.shape) * noise_v
            )

            if i == 0:
                x_lvlm_embs = model_mean_lvlm_embs
            else:
                # ---
                posterior_variance_t = self.extract(self.posterior_variance, t, lvlm_embs.shape)
                noise_lvlm_embs = torch.randn_like(lvlm_embs)
                x_lvlm_embs = model_mean_lvlm_embs + torch.sqrt(posterior_variance_t) * noise_lvlm_embs
        output_lvlm_embs = (1-self.w) * x_lvlm_embs + self.w * lvlm_embs

        for i in reversed(range(0, self.timesteps)):
            t = torch.tensor([i] * a_embs.shape[0], dtype=torch.long).to(self.dev)
            times_info_embeddings = self.get_time_s(t)
            diffu_log_feats_a = torch.cat([a_embs.unsqueeze(1), id_embs.unsqueeze(1), times_info_embeddings.unsqueeze(1)], dim=1)

            x_start_a_embs=self.selfAttention(diffu_log_feats_a)

            model_mean_a_embs = (
                    self.extract(self.posterior_mean_coef1, t, x_start_a_embs.shape) * x_start_a_embs +
                    self.extract(self.posterior_mean_coef2, t, x_start_a_embs.shape) * noise_a
            )

            if i == 0:
                x_a_embs = model_mean_a_embs
            else:
                # ---
                posterior_variance_t = self.extract(self.posterior_variance, t, a_embs.shape)
                noise_a_embs = torch.randn_like(a_embs)
                x_a_embs = model_mean_a_embs + torch.sqrt(posterior_variance_t) * noise_a_embs

        output_a_embs = (1-self.w) * x_a_embs + self.w * a_embs
        
        # print(output_lvlm_embs.shape)
        input_embs_v = output_lvlm_embs.view(-1, self.max_seq_len, self.args.embedding_dim)
        input_embs_a = output_a_embs.view(-1, self.max_seq_len, self.args.embedding_dim)
        input_embs_id = id_embs.view(-1, self.max_seq_len, self.args.embedding_dim) 
        fusion_embs = torch.cat([input_embs_v, input_embs_a, input_embs_id], dim=-1)
        input_embs_common = self.fusion_proj(fusion_embs)
        prec_vec_v = self.user_encoder_v(input_embs_v, log_mask, args.device)
        prec_vec_a = self.user_encoder_a(input_embs_a, log_mask, args.device)
        prec_vec_id = self.user_encoder_id(input_embs_id, log_mask, args.device)
        prec_vec_common = self.user_encoder_common(input_embs_common, log_mask, args.device)
        prec_vec_v = prec_vec_v.reshape(-1, self.args.embedding_dim)
        prec_vec_a = prec_vec_a.reshape(-1, self.args.embedding_dim)
        prec_vec_id = prec_vec_id.reshape(-1, self.args.embedding_dim)
        prec_vec_common = prec_vec_common.reshape(-1, self.args.embedding_dim)
        # Interest Routing Fusion
        prec_vec = self.prec_proj(torch.cat([prec_vec_v, prec_vec_a, prec_vec_id], dim=-1))
        gate_input = torch.cat([prec_vec_common, prec_vec], dim=-1)
        g = torch.sigmoid(self.gate_net(gate_input))
        final_output = g * prec_vec_common + (1 - g) * prec_vec

        return final_output

    # 对比学习loss
    def cl_loss_matrix(self, features_list):
        """
        基于矩阵运算的对比学习 Loss
        features_list: 包含多个特征张量的列表，每个张量形状为 [Batch*Seq, Dim]
        """
        # 堆叠特征 [Num_Views, Batch*Seq, Dim]
        # 这里 Num_Views = 4 (v, a, id, common)
        z = torch.stack(features_list, dim=0)
        
        num_views, batch_size, dim = z.shape
        z = F.normalize(z, dim=2)
        z_flat = z.view(-1, dim)
        similarity_matrix = torch.matmul(z_flat, z_flat.T) / self.cl_temp

        batch_idx = torch.arange(batch_size, device=z.device).repeat_interleave(num_views)

        mask = torch.eq(batch_idx.unsqueeze(1), batch_idx.unsqueeze(0)).float()

        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        loss = -mean_log_prob_pos
        loss = loss.view(num_views, batch_size).mean()
        
        return loss

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