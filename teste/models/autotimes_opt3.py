import os
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP


class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=8, top_k=2,
                 hidden_dim=256, hidden_layers=2, dropout=0.1,
                 activation='tanh', noise_std=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        self.shared_expert = MLP(input_dim, output_dim, hidden_dim, hidden_layers, dropout, activation)
        self.experts = nn.ModuleList([
            MLP(input_dim, output_dim, hidden_dim, hidden_layers, dropout, activation)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        B, S, D = x.shape
        router_logits = self.router(x)
        router_logits = torch.clamp(router_logits, min=-10.0, max=10.0)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        routing_weights = torch.softmax(router_logits.float(), dim=-1).to(x.dtype)
        routing_weights = torch.clamp(routing_weights, min=0.0, max=1.0)
        routing_weights = torch.nan_to_num(routing_weights, nan=0.0)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        weight_sum = top_k_weights.sum(dim=-1, keepdim=True) + 1e-8
        top_k_weights = top_k_weights / weight_sum
        shared_out = self.shared_expert(x)
        x_flat = x.reshape(-1, D)
        indices_flat = top_k_indices.reshape(-1, self.top_k)
        weights_flat = top_k_weights.reshape(-1, self.top_k)
        out_dim = shared_out.shape[-1]
        final_out = torch.zeros(B * S, out_dim, device=x.device, dtype=x.dtype)
        for expert_idx in range(self.num_experts):
            expert_mask = (indices_flat == expert_idx)
            token_mask = expert_mask.any(dim=-1)
            if not token_mask.any():
                continue
            expert_in = x_flat[token_mask]
            expert_out = self.experts[expert_idx](expert_in.unsqueeze(1)).squeeze(1)
            token_weights = weights_flat[token_mask] * expert_mask[token_mask].to(x.dtype)
            agg_weight = token_weights.sum(dim=-1, keepdim=True)
            final_out[token_mask] += expert_out * agg_weight
        final_out = final_out.reshape(B, S, -1)
        out = shared_out + final_out
        return torch.nan_to_num(out, nan=0.0)


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

        num_experts = getattr(configs, 'num_experts', 8)
        top_k = getattr(configs, 'top_k', 2)
        noise_std = getattr(configs, 'moe_noise_std', 0.0)

        self.encoder = MoELayer(
            self.token_len, self.hidden_dim_of_gpt2,
            num_experts=num_experts, top_k=top_k,
            hidden_dim=configs.mlp_hidden_dim,
            hidden_layers=configs.mlp_hidden_layers,
            dropout=configs.dropout,
            activation=configs.mlp_activation,
            noise_std=noise_std,
        ).to(self.gpt2.dtype)
        self.decoder = MoELayer(
            self.hidden_dim_of_gpt2, self.token_len,
            num_experts=num_experts, top_k=top_k,
            hidden_dim=configs.mlp_hidden_dim,
            hidden_layers=configs.mlp_hidden_layers,
            dropout=configs.dropout,
            activation=configs.mlp_activation,
            noise_std=noise_std,
        ).to(self.gpt2.dtype)

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
        times_embeds = self.encoder(fold_out.to(self.gpt2.dtype))
        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc
        times_embeds = times_embeds.to(self.gpt2.dtype)
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
