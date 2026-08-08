import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import math
import random
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')



torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

from benchmark_logger import BenchmarkLogger, Timer
from utils.metrics import metric
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class MLP(nn.Module):
    def __init__(self, f_in, f_out, hidden_dim=256, hidden_layers=2, dropout=0.1, activation='tanh'):
        super().__init__()
        act = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'gelu': nn.GELU()}[activation]
        layers = [nn.Linear(f_in, hidden_dim), act, nn.Dropout(dropout)]
        for _ in range(hidden_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), act, nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand_factor

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1, groups=self.d_inner, bias=True)
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
        ys = []
        for t in range(seq_len):
            dt = delta[:, t]
            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt.unsqueeze(-1) * B[:, t].unsqueeze(1)
            h = dA * h + dB * x_half[:, t].unsqueeze(-1)
            h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
            y_t = (C[:, t].unsqueeze(1) * h).sum(-1) + self.D * x_half[:, t]
            y_t = torch.clamp(y_t, min=-1e4, max=1e4)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)

        y = y * F.silu(res)
        y = self.out_proj(y)
        return y


class MambaBlockWrapper(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand_factor)

    def forward(self, hidden_states, *args, **kwargs):
        return (self.mamba(hidden_states),)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_position = max_position

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :x.shape[2], :x.shape[-1]]
    sin = sin[:, :x.shape[2], :x.shape[-1]]
    return (x * cos) + (rotate_half(x) * sin)


class MLAAttention(nn.Module):
    def __init__(self, d_model, latent_dim=None):
        super().__init__()
        if latent_dim is None:
            latent_dim = d_model // 2
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.num_heads = d_model // 64
        self.head_dim = 64

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, latent_dim * 2, bias=False)
        self.k_expand = nn.Linear(latent_dim, d_model, bias=False)
        self.v_expand = nn.Linear(latent_dim, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, hidden_states):
        B, T, D = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(hidden_states)
        k_latent, v_latent = kv.chunk(2, dim=-1)
        k = self.k_expand(k_latent).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_expand(v_latent).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos_sin = self.rotary(q)
        cos, sin = cos_sin.cos(), cos_sin.sin()
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        attn_weights = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, T, D)

        return self.o_proj(attn_output)


class MLABlock(nn.Module):
    def __init__(self, original_block, d_model, latent_dim=None):
        super().__init__()
        self.ln_1 = original_block.ln_1
        self.ln_2 = original_block.ln_2
        self.mlp = original_block.mlp
        self.attn = MLAAttention(d_model, latent_dim)

    def forward(self, hidden_states, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class HybridModel(nn.Module):
    def __init__(self, gpt2, d_state=16, d_conv=4, expand_factor=2, latent_dim=None):
        super().__init__()
        self.wte = gpt2.wte
        self.wpe = gpt2.wpe
        self.drop = gpt2.drop
        self.ln_f = gpt2.ln_f
        self.config = gpt2.config

        original_layers = gpt2.h
        num_layers = len(original_layers)
        hidden_dim = gpt2.config.n_embd

        keep_first = 2
        keep_last = 2
        num_replace = num_layers - keep_first - keep_last
        if num_replace < 2:
            replace_range = []
        else:
            replace_range = list(range(keep_first, num_layers - keep_last))

        if latent_dim is None:
            latent_dim = hidden_dim // 2

        mid = len(replace_range) // 2
        mamba_indices = set(replace_range[:mid])
        mla_indices = set(replace_range[mid:])

        new_layers = []
        for i, layer in enumerate(original_layers):
            if i in mamba_indices:
                new_layers.append(MambaBlockWrapper(hidden_dim, d_state, d_conv, expand_factor))
            elif i in mla_indices:
                new_layers.append(MLABlock(layer, hidden_dim, latent_dim))
            else:
                new_layers.append(layer)
                for p in layer.parameters():
                    p.requires_grad = False
        self.h = nn.ModuleList(new_layers)

    def forward(self, inputs_embeds):
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            out = block(hidden_states)
            hidden_states = out[0] if isinstance(out, tuple) else out

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"

        load_kwargs = dict(
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
            low_cpu_mem_usage=True,
        )
        model_dir = configs.llm_ckp_dir
        if not os.path.isdir(os.path.abspath(model_dir)):
            model_dir = "gpt2"
        self.gpt2 = GPT2Model.from_pretrained(model_dir, attn_implementation="eager", **load_kwargs)
        self.hidden_dim_of_gpt2 = self.gpt2.config.n_embd
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        for p in self.gpt2.parameters():
            p.requires_grad = False

        d_state = getattr(configs, 'd_state', 16)
        expand_factor = getattr(configs, 'expand_factor', 2)
        latent_dim = getattr(configs, 'latent_dim', None)
        self.hybrid_model = HybridModel(self.gpt2, d_state=d_state, expand_factor=expand_factor, latent_dim=latent_dim)

        if configs.mlp_hidden_layers == 0:
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.token_len)
        else:
            self.encoder = MLP(self.token_len, self.hidden_dim_of_gpt2,
                              configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                              configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_gpt2, self.token_len,
                              configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                              configs.dropout, configs.mlp_activation)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        bs, _, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc = x_enc.reshape(bs * n_vars, -1)

        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]

        times_embeds = self.encoder(fold_out)
        if self.mix:
            times_embeds = times_embeds / (times_embeds.norm(dim=2, keepdim=True) + 1e-8)
            x_mark_enc = x_mark_enc / (x_mark_enc.norm(dim=2, keepdim=True) + 1e-8)
            times_embeds = times_embeds + self.add_scale * x_mark_enc

        times_embeds = times_embeds.to(dtype=self.gpt2.dtype)
        outputs = self.hybrid_model(inputs_embeds=times_embeds)
        outputs = outputs.to(dtype=torch.float32)

        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1))
        dec_out = torch.nan_to_num(dec_out, nan=0.0, posinf=1e4, neginf=-1e4)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)


class Exp_ComboEficiencia(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _acquire_device(self):
        if self.args.use_multi_gpu:
            return torch.device(f'cuda:{self.args.local_rank}')
        return torch.device(f'cuda:{self.args.gpu}')

    def _build_model(self):
        model = Model(self.args).float().to(self.device, non_blocking=True)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = [p for n, p in self.model.named_parameters() if p.requires_grad]
        return torch.optim.Adam([{'params': p_list}],
                                lr=self.args.learning_rate,
                                weight_decay=self.args.weight_decay)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:, :].to(self.device, non_blocking=True)
                    loss = criterion(outputs, batch_y)

                total_loss.append(loss.detach().cpu())
                if self.args.smoke_test: break

        self.model.train()
        return np.average(total_loss) if len(total_loss) > 0 else 0.0

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            loss_val = 0.0
            count = 0
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    loss = criterion(outputs, batch_y)
                loss_val += loss.item()
                count += 1

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    print(f'\tspeed: {speed:.4f}s/iter; left: {left:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()
                if self.args.smoke_test: break

            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.4f}s")
            train_loss = loss_val / max(count, 1)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss_eval = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss_eval:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.cosine:
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    model_optim, T_max=self.args.tmax, eta_min=1e-8).step()
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            load_item = torch.load(best_model_path)
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                inference_steps = self.args.test_pred_len // self.args.token_len
                dis = self.args.test_pred_len - inference_steps * self.args.token_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j - 1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    pred_y.append(outputs[:, -self.args.token_len:, :])

                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-(self.args.token_len - dis), :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :]

                preds.append(pred_y.detach().cpu())
                trues.append(batch_y.detach().cpu())

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}, rmse:{rmse}')

        return {'mse': mse, 'mae': mae, 'rmse': rmse}


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='AutoTimes - Combo Eficiencia (AMD+MLA+Mamba)')

    parser.add_argument('--opt_id', type=str, default='opt10')
    parser.add_argument('--opt_name', type=str, default='Combo Eficiencia (AMD+MLA+Mamba)')

    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='test')
    parser.add_argument('--model', type=str, default='AutoTimes_Gpt2')

    parser.add_argument('--data', type=str, default='ETTm1')
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--test_data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--drop_last', action='store_true', default=False)
    parser.add_argument('--val_set_shuffle', action='store_false', default=True)
    parser.add_argument('--drop_short', action='store_true', default=False)

    parser.add_argument('--seq_len', type=int, default=672)
    parser.add_argument('--label_len', type=int, default=576)
    parser.add_argument('--token_len', type=int, default=96)
    parser.add_argument('--test_seq_len', type=int, default=672)
    parser.add_argument('--test_label_len', type=int, default=576)
    parser.add_argument('--test_pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--llm_ckp_dir', type=str, default='./gpt2')
    parser.add_argument('--mlp_hidden_dim', type=int, default=256)
    parser.add_argument('--mlp_hidden_layers', type=int, default=2)
    parser.add_argument('--mlp_activation', type=str, default='tanh')

    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--expand_factor', type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=None)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--cosine', action='store_true', default=True)
    parser.add_argument('--tmax', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--mix_embeds', action='store_true', default=False)
    parser.add_argument('--test_dir', type=str, default='./test')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--smoke_test', type=int, default=0, help='stop after 1 batch per epoch')
    parser.add_argument('--reset_csv', action='store_true', default=False, help='reset benchmark_results.csv before running')

    args = parser.parse_args()

    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        args.local_rank = local_rank
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{ip}:{port}",
            world_size=hosts,
            rank=rank)
        torch.cuda.set_device(local_rank)

    logger = BenchmarkLogger(reset=args.reset_csv)
    timer = Timer()

    for ii in range(args.itr):
        setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_ds{}_ef{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data,
            args.seq_len, args.label_len, args.token_len,
            args.learning_rate, args.batch_size, args.weight_decay,
            args.mlp_hidden_dim, args.mlp_hidden_layers,
            args.cosine, args.mix_embeds, args.d_state, args.expand_factor,
            args.des, ii)

        exp = Exp_ComboEficiencia(args)

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        timer.start()
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        metrics = exp.test(setting)
        total_time = timer.stop()

        test_loader_len = len(exp._get_data(flag='test')[1])
        avg_latency = total_time / test_loader_len if test_loader_len > 0 else 0.0

        row = logger.log(
            opt_id=args.opt_id,
            opt_name=args.opt_name,
            args=args,
            metrics=metrics,
            total_time_s=total_time,
            avg_latency_s=avg_latency,
            gpu_time_s=timer.gpu_elapsed,
            wall_time_s=timer.wall_elapsed,
        )

        print("\n=== Benchmark Results ===")
        print("opt_id:        {}".format(args.opt_id))
        print("opt_name:      {}".format(args.opt_name))
        print("total_time:    {:.4f}s".format(total_time))
        print("test_mse:      {:.8f}".format(metrics['mse']))
        print("test_mae:      {:.8f}".format(metrics['mae']))
        print("test_rmse:     {:.8f}".format(metrics['rmse']))
        print("=========================\n")

        torch.cuda.empty_cache()
