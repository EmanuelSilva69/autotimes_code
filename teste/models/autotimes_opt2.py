import os
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].float()
        pos = position_ids[:, None, :].float()
        freqs = pos * inv_freq
        return freqs.cos(), freqs.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.transpose(1, 2)
    sin = sin.transpose(1, 2)
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(2)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(2)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_key_value_heads * n_rep, slen, head_dim)


class MLAAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.n_head
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.kv_latent_dim = self.hidden_size // 2
        self.attention_dropout = config.attn_pdrop
        self.is_causal = True

        bias = getattr(config, 'bias', True)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.kv_proj = nn.Linear(
            self.hidden_size, self.kv_latent_dim, bias=bias
        )
        self.k_expand = nn.Linear(
            self.kv_latent_dim, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.v_expand = nn.Linear(
            self.kv_latent_dim, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=bias
        )
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[object] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[object]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        kv_latent = self.kv_proj(hidden_states)
        key_states = self.k_expand(kv_latent)
        value_states = self.v_expand(kv_latent)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)

        model_dir = configs.llm_ckp_dir
        if not os.path.isdir(os.path.abspath(model_dir)):
            model_dir = "gpt2"
        self.gpt2 = GPT2Model.from_pretrained(
            model_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.hidden_dim_of_gpt2 = self.gpt2.config.n_embd
        self.kv_latent_dim = self.hidden_dim_of_gpt2 // 2
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        gpt2_config = self.gpt2.config
        for i, layer in enumerate(self.gpt2.h):
            mla = MLAAttention(config=gpt2_config, layer_idx=i)
            with torch.no_grad():
                c_attn_weight = layer.attn.c_attn.weight
                c_attn_bias = layer.attn.c_attn.bias
                d = self.hidden_dim_of_gpt2
                kld = self.kv_latent_dim
                q_weight, k_weight, v_weight = c_attn_weight.T.split(d, dim=0)
                mla.q_proj.weight.copy_(q_weight)
                mla.kv_proj.weight.copy_(k_weight[:kld, :])
                mla.k_expand.weight.copy_(k_weight[:, :kld])
                mla.v_expand.weight.copy_(v_weight[:, :kld])
                if c_attn_bias is not None:
                    q_bias, k_bias, v_bias = c_attn_bias.split(d)
                    if mla.q_proj.bias is not None:
                        mla.q_proj.bias.copy_(q_bias)
                    if mla.kv_proj.bias is not None:
                        mla.kv_proj.bias.copy_(k_bias[:kld])
                    if mla.k_expand.bias is not None:
                        mla.k_expand.bias.copy_(k_bias)
                    if mla.v_expand.bias is not None:
                        mla.v_expand.bias.copy_(v_bias)
                mla.o_proj.weight.copy_(layer.attn.c_proj.weight.T)
                if mla.o_proj.bias is not None and layer.attn.c_proj.bias is not None:
                    mla.o_proj.bias.copy_(layer.attn.c_proj.bias)
            layer.attn = mla

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False
        for name, param in self.gpt2.named_parameters():
            if 'attn' in name:
                param.requires_grad = True

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.token_len)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len, self.hidden_dim_of_gpt2,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_gpt2, self.token_len,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, _, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]
        times_embeds = self.encoder(fold_out)
        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc

        outputs = self.gpt2(inputs_embeds=times_embeds.contiguous()).last_hidden_state

        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
