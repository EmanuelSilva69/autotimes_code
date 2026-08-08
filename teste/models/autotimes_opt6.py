import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = d_model * expand_factor

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        batch, seq_len, _ = x.shape
        x_and_res = self.in_proj(x)
        x_half, res = x_and_res.chunk(2, dim=-1)

        x_half = x_half.transpose(1, 2)
        x_half = self.conv1d(x_half)[:, :, :seq_len]
        x_half = F.silu(x_half)
        x_half = x_half.transpose(1, 2)

        proj = self.x_proj(x_half)
        delta_raw, B, C = proj.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta_raw))
        delta = torch.clamp(delta, max=10.0)

        A = -torch.exp(torch.clamp(self.A_log, max=4.0))
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y = []
        for t in range(seq_len):
            dt = delta[:, t]
            dA = torch.exp((dt.unsqueeze(-1) * A.unsqueeze(0)).clamp(min=-10.0, max=10.0))
            dB = dt.unsqueeze(-1) * B[:, t].unsqueeze(1)
            h = dA * h + dB * x_half[:, t].unsqueeze(-1)
            h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
            y_t = (C[:, t].unsqueeze(1) * h).sum(-1) + self.D * x_half[:, t]
            y.append(y_t)
        y = torch.stack(y, dim=1)

        y = y * F.silu(res)
        y = self.out_proj(y)
        return y


class MambaBlockWrapper(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand_factor)

    def forward(self, hidden_states, *args, **kwargs):
        hidden_states = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        return self.mamba(hidden_states)


class HybridGPT2Model(nn.Module):
    def __init__(self, gpt2_model, d_state=16, mamba_layers=12):
        super().__init__()
        self.model = gpt2_model
        original_layers = self.model.h
        num_layers = len(original_layers)

        keep_first = 2
        keep_last = 2
        replace_range = list(range(keep_first, num_layers - keep_last))
        if mamba_layers < len(replace_range):
            replace_range = replace_range[:mamba_layers]

        hidden_dim = self.model.config.n_embd

        new_layers = nn.ModuleList()
        for i, layer in enumerate(original_layers):
            if i in replace_range:
                new_layers.append(MambaBlockWrapper(hidden_dim, d_state=d_state))
            else:
                new_layers.append(layer)
                for param in layer.parameters():
                    param.requires_grad = False

        self.model.h = new_layers

    def forward(self, inputs_embeds):
        return self.model(inputs_embeds=inputs_embeds).last_hidden_state


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

        d_state = getattr(configs, 'd_state', 16)
        mamba_layers = getattr(configs, 'mamba_layers', 12)
        self.hybrid_model = HybridGPT2Model(self.gpt2, d_state=d_state, mamba_layers=mamba_layers)

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
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-7)
        x_enc = x_enc / (stdev + 1e-7)

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

        outputs = self.hybrid_model(inputs_embeds=times_embeds.contiguous())

        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))

        dec_out = torch.nan_to_num(dec_out, nan=0.0, posinf=1e4, neginf=-1e4)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
