import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

torch.cuda.empty_cache()

from benchmark_logger import BenchmarkLogger, Timer
from utils.metrics import metric
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP


class MLAAttention(nn.Module):
    def __init__(self, d_model, n_heads, kv_compression_ratio=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_dim = d_model // kv_compression_ratio

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.kv_dim)
        self.v_proj = nn.Linear(d_model, self.kv_dim)
        self.kv_up = nn.Linear(self.kv_dim, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        bs, seq, d = x.shape
        q = self.q_proj(x).reshape(bs, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.kv_up(self.k_proj(x)).reshape(bs, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.kv_up(self.v_proj(x)).reshape(bs, seq, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bs, seq, d)
        return self.o_proj(out)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=d_inner,
        )
        self.x_proj = nn.Linear(d_inner, d_state * 2)
        self.dt_proj = nn.Linear(d_state, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)

        nn.init.xavier_uniform_(self.in_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2 ** -0.5)

    def forward(self, x):
        residual = x
        z, x_hidden = self.in_proj(x).chunk(2, dim=-1)
        x_hidden = self.conv1d(x_hidden.transpose(1, 2)).transpose(1, 2)
        x_hidden = F.silu(x_hidden)
        delta, B = self.x_proj(x_hidden).split(self.d_state, dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self._ssm_scan(x_hidden, delta, B)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return y + residual

    def _ssm_scan(self, x, delta, B):
        bs, seq, d_inner = x.shape
        A = -torch.exp(torch.rand(self.d_state, device=x.device))
        h = torch.zeros(bs, d_inner, self.d_state, device=x.device)
        out = []
        for t in range(seq):
            delta_t = delta[:, t, :].unsqueeze(-1)
            B_t = B[:, t, :].unsqueeze(1)
            h = h + delta_t * (A.unsqueeze(0).unsqueeze(0) * h + B_t * x[:, t, :].unsqueeze(-1))
            out.append((h @ torch.eye(self.d_state, device=x.device).unsqueeze(0)).sum(dim=-1))
        return torch.stack(out, dim=1)


class MTPDecoder(nn.Module):
    def __init__(self, d_model, token_len, horizons, mlp_hidden_dim, mlp_hidden_layers, dropout, activation):
        super().__init__()
        self.horizons = horizons
        self.decoders = nn.ModuleList([
            MLP(d_model, token_len, mlp_hidden_dim, mlp_hidden_layers, dropout, activation)
            for _ in horizons
        ])

    def forward(self, x):
        return [dec(x) for dec in self.decoders]


class ComboAtencaoPuraModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        torch.set_float32_matmul_precision('high')

        self.token_len = configs.token_len
        self.n_heads = getattr(configs, 'n_heads', 12)
        self.kv_compression_ratio = getattr(configs, 'kv_compression_ratio', 2)
        self.mamba_layers = getattr(configs, 'mamba_layers', 4)
        self.mtp_horizons = getattr(configs, 'mtp_horizons', [1, 2, 4])

        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"

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
        self.hidden_dim = self.gpt2.config.n_embd
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

        if configs.mlp_hidden_layers == 0:
            self.encoder = nn.Linear(self.token_len, self.hidden_dim)
        else:
            self.encoder = MLP(self.token_len, self.hidden_dim,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)

        n_layers = len(self.gpt2.h)
        self.mamba_replace = min(self.mamba_layers, n_layers)
        replace_idxs = set(range(n_layers - self.mamba_replace, n_layers))

        self.mamba_blocks = nn.ModuleList()
        self.mla_attentions = nn.ModuleList()
        for i in range(n_layers):
            if i in replace_idxs:
                self.mamba_blocks.append(MambaBlock(self.hidden_dim))
                self.mla_attentions.append(None)
            else:
                self.mamba_blocks.append(None)
                self.mla_attentions.append(
                    MLAAttention(self.hidden_dim, self.n_heads, self.kv_compression_ratio))

        self.replace_idxs = replace_idxs

        self.mtp = MTPDecoder(self.hidden_dim, self.token_len,
                              self.mtp_horizons,
                              configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                              configs.dropout, configs.mlp_activation)

        if configs.mlp_hidden_layers == 0:
            self.decoder = nn.Linear(self.hidden_dim, self.token_len)
        else:
            self.decoder = MLP(self.hidden_dim, self.token_len,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)

    def _hybrid_forward(self, hidden):
        for i, layer in enumerate(self.gpt2.h):
            if i in self.replace_idxs:
                hidden = self.mamba_blocks[i](hidden)
            else:
                hidden_attn = layer(hidden)
                hidden_attn = hidden_attn[0] if isinstance(hidden_attn, tuple) else hidden_attn
                mla = self.mla_attentions[i]
                if mla is not None:
                    hidden_attn = mla(hidden_attn) + hidden_attn
                hidden = hidden_attn
        return hidden

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mtp_all=False):
        with torch.cuda.amp.autocast():
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

            bs, seq_len, n_vars = x_enc.shape
            x_enc = x_enc.permute(0, 2, 1).contiguous()
            x_enc = x_enc.reshape(bs * n_vars, -1)

            fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
            token_num = fold_out.shape[1]

            times_embeds = self.encoder(fold_out)
            if self.mix:
                times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
                x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
                times_embeds = times_embeds + self.add_scale * x_mark_enc

            times_embeds = times_embeds.to(dtype=self.gpt2.dtype)
            hidden = self._hybrid_forward(times_embeds)
            hidden = hidden.to(dtype=torch.float32)

            if mtp_all:
                mtp_outs = self.mtp(hidden)
                results = []
                for head_out in mtp_outs:
                    d_out = head_out.reshape(bs, n_vars, -1)
                    d_out = d_out.permute(0, 2, 1).contiguous()
                    d_out = d_out * stdev[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1)
                    d_out = d_out + means[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1)
                    results.append(d_out)
                return results

            decoded = self.decoder(hidden)

            dec_out = decoded.reshape(bs, n_vars, -1)
            dec_out = dec_out.permute(0, 2, 1).contiguous()

            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mtp_all=False)


class Exp_ComboAtencaoPura(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _acquire_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Using GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device

    def _build_model(self):
        model = ComboAtencaoPuraModel(self.args).float()
        model = model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p_list.append(p)
                print(n, p.dtype, p.shape)
        model_optim = torch.optim.Adam([{'params': p_list}],
                                       lr=self.args.learning_rate,
                                       weight_decay=self.args.weight_decay)
        print('learning rate = {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    outputs = outputs[:, :, :]
                    batch_y = batch_y[:, :, :].to(self.device, non_blocking=True)
                    loss = criterion(outputs, batch_y)
                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if self.args.smoke_test: break

        total_loss = np.average(total_loss, weights=total_count) if (len(total_loss) > 0 and sum(total_count) > 0) else 0.0
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

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

                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    loss = criterion(outputs, batch_y)
                loss_val += loss.item()
                count += 1
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()
                if self.args.smoke_test: break

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = loss_val / count

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss_eval = self.vali(test_data, test_loader, criterion)

            print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss_eval))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.cosine:
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    model_optim, T_max=self.args.tmax, eta_min=1e-8).step()
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            print("loading model from {}".format(best_model_path))
            load_item = torch.load(best_model_path)
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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

                    with torch.cuda.amp.autocast():
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
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))

        return {'mse': mse, 'mae': mae, 'rmse': rmse}


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='AutoTimes - Combo Atencao Pura (MLA+Mamba+MTP)')

    parser.add_argument('--opt_id', type=str, default='opt15', help='optimization id')
    parser.add_argument('--opt_name', type=str, default='Combo Atencao Pura (MLA+Mamba+MTP)', help='optimization name')

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

    parser.add_argument('--n_heads', type=int, default=12, help='number of attention heads for MLA')
    parser.add_argument('--kv_compression_ratio', type=int, default=2, help='KV compression ratio (latent dim = d_model / ratio)')
    parser.add_argument('--mamba_layers', type=int, default=4, help='number of Mamba SSM blocks replacing GPT-2 layers')
    parser.add_argument('--mtp_horizons', type=int, nargs='+', default=[1, 2, 4], help='MTP prediction horizons')
    parser.add_argument('--smoke_test', type=int, default=0, help='stop after 1 batch per epoch')
    parser.add_argument('--reset_csv', action='store_true', default=False, help='reset benchmark_results.csv before running')

    args = parser.parse_args()

    logger = BenchmarkLogger(reset=args.reset_csv)
    timer = Timer()

    for ii in range(args.itr):
        setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data,
            args.seq_len, args.label_len, args.token_len,
            args.learning_rate, args.batch_size, args.weight_decay,
            args.mlp_hidden_dim, args.mlp_hidden_layers,
            args.cosine, args.mix_embeds, args.des, ii)

        exp = Exp_ComboAtencaoPura(args)

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
