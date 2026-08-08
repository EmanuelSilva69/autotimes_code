import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP


def compute_statistical_features(x, token_len):
    B, S = x.shape
    num_tokens = S // token_len
    features = []
    for t in range(num_tokens):
        start = t * token_len
        end = start + token_len
        chunk = x[:, start:end]
        trend = (chunk[:, -1] - chunk[:, 0]).unsqueeze(1)
        local_var = chunk.var(dim=1, unbiased=False).unsqueeze(1)
        mean = chunk.mean(dim=1).unsqueeze(1)
        max_v = chunk.max(dim=1).values.unsqueeze(1)
        min_v = chunk.min(dim=1).values.unsqueeze(1)
        feat = torch.cat([mean, local_var, trend, max_v - min_v], dim=1)
        features.append(feat)
    return torch.stack(features, dim=1)


class AdaptiveFusion(nn.Module):
    def __init__(self, hidden_dim, n_stat_feats=4):
        super().__init__()
        self.stat_proj = nn.Linear(n_stat_feats, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, times_embeds, stat_feats):
        stat_embeds = self.stat_proj(stat_feats)
        combined = torch.cat([times_embeds, stat_embeds], dim=-1)
        gate = self.gate(combined)
        fused = gate * times_embeds + (1 - gate) * stat_embeds
        return self.norm(fused)


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
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

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

        self.adaptive_fusion = AdaptiveFusion(
            self.hidden_dim_of_gpt2,
            n_stat_feats=getattr(configs, 'n_stat_feats', 4),
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, _, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_flat = x_enc.reshape(bs * n_vars, -1)

        fold_out = x_flat.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]

        times_embeds = self.encoder(fold_out)

        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc

        stat_feats = compute_statistical_features(x_flat, self.token_len)
        times_embeds = self.adaptive_fusion(times_embeds, stat_feats)

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
