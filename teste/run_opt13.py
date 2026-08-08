import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import gc
from benchmark_logger import BenchmarkLogger, Timer
from utils.metrics import metric
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP


class SMETimesStats(nn.Module):
    def __init__(self, token_len):
        super().__init__()
        self.token_len = token_len

    def forward(self, folded):
        mean = folded.mean(dim=-1, keepdim=True)
        var = folded.var(dim=-1, keepdim=True, unbiased=False)
        t = torch.arange(self.token_len, device=folded.device).float().view(1, 1, -1)
        t_centered = t - t.mean()
        trend = (folded * t_centered).sum(dim=-1, keepdim=True) / ((t_centered ** 2).sum() + 1e-5)
        rng = folded.max(dim=-1, keepdim=True).values - folded.min(dim=-1, keepdim=True).values
        return torch.cat([mean, var, trend, rng], dim=-1)


class AdaptiveFusionGate(nn.Module):
    def __init__(self, d_model, stats_dim=4):
        super().__init__()
        self.stats_proj = nn.Linear(stats_dim, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model + stats_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, times_embeds, stats):
        stats_p = self.stats_proj(stats)
        g = self.gate(torch.cat([times_embeds, stats], dim=-1))
        return times_embeds + g * stats_p


class RAGMemoryBank(nn.Module):
    def __init__(self, d_model, memory_size=1024, top_k=3):
        super().__init__()
        self.memory_size = memory_size
        self.top_k = top_k
        self.register_buffer('memory', torch.randn(memory_size, d_model))
        self.memory = F.normalize(self.memory, dim=-1)
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))
        self.proj = nn.Linear(d_model * 2, d_model)

    def query(self, x):
        flat = x.reshape(-1, x.shape[-1])
        mem_norm = F.normalize(self.memory, dim=-1)
        flat_norm = F.normalize(flat, dim=-1)
        sim = flat_norm @ mem_norm.T
        vals, idxs = sim.topk(self.top_k, dim=-1)
        retrieved = self.memory[idxs].mean(dim=1)
        retrieved = retrieved.reshape(x.shape[0], x.shape[1], -1)
        return self.proj(torch.cat([x, retrieved], dim=-1))

    def update(self, x):
        with torch.no_grad():
            flat = x.detach().reshape(-1, x.shape[-1])
            n = flat.shape[0]
            p = self.ptr.item()
            if n >= self.memory_size:
                self.memory.copy_(F.normalize(flat[-self.memory_size:], dim=-1))
                self.ptr[0] = 0
            elif p + n <= self.memory_size:
                self.memory[p:p + n] = F.normalize(flat, dim=-1)
                self.ptr[0] = p + n
            else:
                rem = self.memory_size - p
                self.memory[p:] = F.normalize(flat[:rem], dim=-1)
                self.memory[:n - rem] = F.normalize(flat[rem:], dim=-1)
                self.ptr[0] = n - rem


class FinGPTCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, mark_dim=5):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self._mark_proj = None

    def forward(self, hidden, x_mark_tokens):
        if x_mark_tokens is None:
            return hidden
        mark_dim = x_mark_tokens.shape[-1]
        if self._mark_proj is None or self._mark_proj.in_features != mark_dim:
            self._mark_proj = nn.Linear(mark_dim, self.cross_attn.embed_dim).to(hidden.device)
        mark_proj = self._mark_proj(x_mark_tokens)
        out, _ = self.cross_attn(hidden, mark_proj, mark_proj)
        return self.norm(hidden + out)


class ComboSLMContextualModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        torch.set_float32_matmul_precision('high')

        self.token_len = configs.token_len
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.memory_size = getattr(configs, 'memory_size', 1024)
        self.rag_top_k = getattr(configs, 'rag_top_k', 3)
        self.mark_dim = getattr(configs, 'mark_dim', 5)

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

        self.smetimes_stats = SMETimesStats(self.token_len)
        self.fusion_gate = AdaptiveFusionGate(self.hidden_dim)
        self.rag_bank = RAGMemoryBank(self.hidden_dim, self.memory_size, self.rag_top_k)
        self.fingpt_cross = FinGPTCrossAttention(self.hidden_dim, self.n_heads, self.mark_dim)

        if configs.mlp_hidden_layers == 0:
            self.decoder = nn.Linear(self.hidden_dim, self.token_len)
        else:
            self.decoder = MLP(self.hidden_dim, self.token_len,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
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

            stats = self.smetimes_stats(fold_out)
            times_embeds = self.fusion_gate(times_embeds, stats)

            if self.mix:
                times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
                x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
                times_embeds = times_embeds + self.add_scale * x_mark_enc

            rag_context = self.rag_bank.query(times_embeds)
            self.rag_bank.update(times_embeds.detach())

            combined = torch.cat([rag_context, times_embeds], dim=1)
            outputs = self.gpt2(inputs_embeds=combined.to(dtype=self.gpt2.dtype).contiguous()).last_hidden_state
            outputs = outputs.to(dtype=torch.float32)
            outputs = outputs[:, token_num:, :]

            if x_mark_enc is not None:
                x_mark_enc = x_mark_enc[:, :seq_len, :]
                xm_bs, xm_seq, xm_dim = x_mark_enc.shape
                x_mark_exp = x_mark_enc.unsqueeze(1).expand(-1, n_vars, -1, -1)
                x_mark_flat = x_mark_exp.contiguous().view(xm_bs * n_vars, xm_seq, xm_dim)
                if xm_seq >= token_num:
                    x_mark_tokens = F.adaptive_avg_pool1d(
                        x_mark_flat.transpose(1, 2), token_num
                    ).transpose(1, 2)
                else:
                    x_mark_tokens = x_mark_flat.mean(dim=1, keepdim=True).expand(-1, token_num, -1)
                outputs = self.fingpt_cross(outputs, x_mark_tokens)

            dec_out = self.decoder(outputs)
            dec_out = dec_out.reshape(bs, n_vars, -1)
            dec_out = dec_out.permute(0, 2, 1).contiguous()

            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)


class Exp_ComboSLMContextual(object):
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
        model = ComboSLMContextualModel(self.args).float()
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

    parser = argparse.ArgumentParser(description='AutoTimes - Combo SLM Contextual (SMETimes+RAG+FinGPT)')

    parser.add_argument('--opt_id', type=str, default='opt13', help='optimization id')
    parser.add_argument('--opt_name', type=str, default='Combo SLM Contextual (SMETimes+RAG+FinGPT)', help='optimization name')

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

    parser.add_argument('--num_workers', type=int, default=2)
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

    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads for FinGPT cross-attention')
    parser.add_argument('--memory_size', type=int, default=1024, help='RAG memory bank capacity')
    parser.add_argument('--rag_top_k', type=int, default=3, help='top-k retrieval from memory bank')
    parser.add_argument('--mark_dim', type=int, default=5, help='time feature dimension in x_mark_enc')
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

        exp = Exp_ComboSLMContextual(args)

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

        gc.collect()
        torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()
