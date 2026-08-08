import os
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP


class FinGPTAdapter(nn.Module):
    def __init__(self, configs, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_len = configs.token_len
        self.use_cross_attn = getattr(configs, 'use_cross_attn', True)
        self._proj = None

        n_heads = getattr(configs, 'fingpt_heads', max(1, hidden_dim // 64))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=getattr(configs, 'dropout', 0.1),
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def _build_proj(self, in_features, device, dtype):
        proj = nn.Linear(in_features, self.hidden_dim).to(device=device, dtype=dtype)
        self._proj = proj
        return proj

    def forward(self, times_embeds, x_mark_enc):
        if not self.use_cross_attn:
            return times_embeds

        bs, seq_len, n_feats = x_mark_enc.shape
        bs_nvars, token_num, _ = times_embeds.shape
        nvars = bs_nvars // bs

        xm = x_mark_enc.unsqueeze(1).expand(-1, nvars, -1, -1)
        xm = xm.reshape(bs * nvars, seq_len, n_feats)

        n_tokens = seq_len // self.token_len
        keep = n_tokens * self.token_len
        if keep < seq_len:
            xm = xm[:, :keep, :]
        elif keep > seq_len:
            pad = xm.new_zeros(bs_nvars, keep - seq_len, n_feats)
            xm = torch.cat([xm, pad], dim=1)
            n_tokens = keep // self.token_len

        xm = xm.view(bs_nvars, n_tokens, self.token_len, n_feats).mean(dim=2)

        if n_tokens < token_num:
            pad = xm.new_zeros(bs_nvars, token_num - n_tokens, n_feats)
            xm = torch.cat([xm, pad], dim=1)
        elif n_tokens > token_num:
            xm = xm[:, :token_num, :]

        if self._proj is None or self._proj.in_features != xm.shape[-1]:
            sent_embeds = self._build_proj(xm.shape[-1], xm.device, xm.dtype)(xm)
        else:
            sent_embeds = self._proj(xm)

        enhanced, _ = self.cross_attn(
            query=times_embeds,
            key=sent_embeds,
            value=sent_embeds,
            need_weights=False,
        )
        enhanced = self.norm(enhanced + times_embeds)
        return enhanced


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

        self.sentiment_adapter = FinGPTAdapter(configs, hidden_dim=self.hidden_dim_of_gpt2)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, _, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc = x_enc.reshape(bs * n_vars, -1)

        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]

        times_embeds = self.encoder(fold_out)

        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc

        times_embeds = self.sentiment_adapter(times_embeds, x_mark_enc)

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
