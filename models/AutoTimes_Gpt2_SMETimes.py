import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP


def compute_statistical_features(x, token_len):
    B, S = x.shape
    num_tokens = S // token_len
    chunks = x.view(B, num_tokens, token_len)

    # 1) Estabilidade numérica: força FP32 (var/slope sofrem underflow em FP16 sob AMP)
    chunks_fp32 = chunks.float()

    mean = chunks_fp32.mean(dim=-1, keepdim=True)
    local_var = chunks_fp32.var(dim=-1, unbiased=False, keepdim=True)

    # Range (Max - Min)
    rng = (chunks_fp32.max(dim=-1, keepdim=True).values -
           chunks_fp32.min(dim=-1, keepdim=True).values)

    # 2) Tendência via OLS (mínimos quadrados), vetorizado sem loops:
    #    m = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
    #    x: eixo do tempo [0, 1, ..., token_len-1]; y: amostras do chunk
    x_time = torch.arange(token_len, dtype=torch.float32, device=chunks.device)
    x_centered = x_time - x_time.mean()
    y_centered = chunks_fp32 - mean                     # y_mean = mean (keepdim)
    numerator = (y_centered * x_centered).sum(dim=-1, keepdim=True)
    denominator = (x_centered ** 2).sum()               # escalar: só depende de token_len
    trend = numerator / denominator

    stat_feats = torch.cat([mean, local_var, trend, rng], dim=-1)

    # 3) Devolve no dtype original do input (ex: FP16 sob AMP)
    return stat_feats.to(chunks.dtype)


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

        torch.set_float32_matmul_precision('high' if configs.use_amp else 'highest')
        torch.backends.cudnn.benchmark = configs.use_amp
        self.use_amp = configs.use_amp

        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)

        self.gpt2 = GPT2Model.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.hidden_dim_of_gpt2 = self.gpt2.config.n_embd
        self.mix = configs.mix_embeds

        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
            self.time_proj = None

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
        with torch.cuda.amp.autocast(enabled=self.use_amp):
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
                if self.time_proj is None:
                    self.time_proj = nn.Linear(x_mark_enc.shape[-1], self.hidden_dim_of_gpt2).to(x_mark_enc.device, x_mark_enc.dtype)
                x_mark_enc = self.time_proj(x_mark_enc)
                x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
                times_embeds = times_embeds + self.add_scale * x_mark_enc

            stat_feats = compute_statistical_features(x_flat, self.token_len)
            times_embeds = self.adaptive_fusion(times_embeds, stat_feats)

            outputs = self.gpt2(
                inputs_embeds=times_embeds.contiguous()).last_hidden_state
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
