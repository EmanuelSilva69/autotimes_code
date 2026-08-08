import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from benchmark_logger import BenchmarkLogger, Timer
from utils.metrics import metric
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate

from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class MoEMLP(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=4, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_ff, d_model),
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        bs, seq, d = x.shape
        logits = self.gate(x)
        weights = F.softmax(logits, dim=-1)
        top_w, top_idx = torch.topk(weights, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)

        capacity = (seq * self.top_k + self.n_experts - 1) // self.n_experts
        out = torch.zeros_like(x)

        for e in range(self.n_experts):
            expert_mask = (top_idx == e)
            expert_counts = expert_mask.sum(dim=(1, 2))
            max_c = max(expert_counts.max().item(), 1)

            for b in range(bs):
                c = min(max_c, capacity)
                if c == 0:
                    continue
                indices = (top_idx[b] == e).nonzero(as_tuple=True)
                if indices[0].numel() == 0:
                    continue
                n_sel = min(indices[0].numel(), c)
                rows = indices[0][:n_sel]
                cols = indices[1][:n_sel]
                expert_in = x[b, rows, :]
                expert_out = self.experts[e](expert_in)
                for r, co in zip(rows.tolist(), range(n_sel)):
                    out[b, r, :] += expert_out[co] * top_w[b, r, 0:1]
        return out


class MemoryBank(nn.Module):
    def __init__(self, dim, max_size=1000, top_k=3):
        super().__init__()
        self.dim = dim
        self.max_size = max_size
        self.top_k = top_k
        self.keys = []
        self.values = []
        self._cpu_device = torch.device('cpu')

    def add(self, keys, values):
        keys = keys.detach().cpu()
        values = values.detach().cpu()
        bs = keys.shape[0]
        for i in range(bs):
            self.keys.append(keys[i])
            self.values.append(values[i])
        if len(self.keys) > self.max_size:
            self.keys = self.keys[-self.max_size:]
            self.values = self.values[-self.max_size:]

    def retrieve(self, query):
        if len(self.keys) == 0:
            bs, seq, d = query.shape
            return torch.zeros(bs, seq, d, device=query.device)
        query_flat = query.reshape(-1, query.shape[-1])
        query_norm = F.normalize(query_flat, dim=-1)
        keys_tensor = torch.stack(self.keys, dim=0).to(query.device)
        keys_norm = F.normalize(keys_tensor, dim=-1)
        sim = query_norm @ keys_norm.T
        vals, idxs = sim.topk(min(self.top_k, len(self.keys)), dim=-1)
        values_tensor = torch.stack(self.values, dim=0).to(query.device)
        retrieved = values_tensor[idxs].mean(dim=1)
        retrieved = retrieved.reshape(query.shape[0], query.shape[1], -1)
        return retrieved

    def __len__(self):
        return len(self.keys)


class GRPOLoss(nn.Module):
    def __init__(self, group_size=4, alpha=0.1):
        super().__init__()
        self.group_size = group_size
        self.alpha = alpha

    def forward(self, predictions, targets):
        group = self.group_size
        bs = predictions.shape[0] // group
        preds = predictions.reshape(group, bs, *predictions.shape[1:])
        tgt = targets[:bs]
        losses = []
        for i in range(group):
            losses.append(F.mse_loss(preds[i], tgt, reduction='none'))
        loss_stack = torch.stack(losses, dim=0)
        loss_mean = loss_stack.mean(dim=0)
        loss_std = loss_stack.std(dim=0) + 1e-8
        advantage = (loss_mean - loss_stack) / loss_std
        weighted = loss_stack * (1 - self.alpha * advantage.detach())
        return weighted.mean()


class ComboPreditorModel(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.token_len = configs.token_len

        model_dir = configs.llm_ckp_dir
        if not os.path.isdir(os.path.abspath(model_dir)):
            model_dir = "gpt2"
        self.gpt2 = GPT2Model.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.hidden_dim = self.gpt2.config.n_embd

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

        d_ff = configs.mlp_hidden_dim * 2
        self.n_experts = getattr(configs, 'n_experts', 4)
        self.top_k = getattr(configs, 'top_k', 2)
        self.rag_top_k = getattr(configs, 'rag_top_k', 3)
        self.rag_memory_size = getattr(configs, 'rag_memory_size', 1000)

        self.encoder = MoEMLP(self.token_len, d_ff, self.n_experts, self.top_k)
        self.enc_proj = nn.Linear(self.token_len, self.hidden_dim)
        self.decoder = MoEMLP(self.hidden_dim, d_ff, self.n_experts, self.top_k)
        self.dec_proj = nn.Linear(self.hidden_dim, self.token_len)
        self.rag_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.memory_bank = MemoryBank(self.hidden_dim, self.rag_memory_size, self.rag_top_k)

    def add_to_memory(self, x_enc, x_mark_enc):
        with torch.no_grad():
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

            bs, _, n_vars = x_enc.shape
            x_enc = x_enc.permute(0, 2, 1).contiguous()
            x_enc = x_enc.reshape(bs * n_vars, -1)
            fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
            token_num = fold_out.shape[1]

            encoded = self.encoder(fold_out)
            encoded = self.enc_proj(encoded)
            keys = encoded.reshape(bs, n_vars * token_num, -1).mean(dim=1, keepdim=True)
            values = encoded.reshape(bs, n_vars * token_num, -1).mean(dim=1, keepdim=True)
            self.memory_bank.add(keys, values)

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

        encoded = self.encoder(fold_out)
        encoded = self.enc_proj(encoded)

        retrieved = self.memory_bank.retrieve(encoded)
        rag_input = torch.cat([encoded, retrieved], dim=-1)
        rag_out = self.rag_proj(rag_input)

        hidden = self.gpt2(inputs_embeds=(rag_out.unsqueeze(0) if rag_out.dim() == 2 else rag_out).to(dtype=self.gpt2.dtype).contiguous()).last_hidden_state
        hidden = hidden.to(dtype=torch.float32)

        if hidden.dim() == 3 and hidden.shape[0] == 1 and bs * n_vars > 1:
            hidden = hidden.squeeze(0)
        if hidden.shape[0] != bs * n_vars:
            hidden = hidden[:bs * n_vars]

        dec_out = self.decoder(hidden)
        dec_out = self.dec_proj(dec_out)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)


class Exp_ComboPreditor(object):
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
        model = ComboPreditorModel(self.args).float()
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

    def _select_criterion(self, training=False):
        if training and getattr(self.args, 'loss', 'MSE') == 'GRPO':
            return GRPOLoss(
                group_size=getattr(self.args, 'group_size', 4),
                alpha=getattr(self.args, 'grpo_alpha', 0.1),
            )
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
        train_criterion = self._select_criterion(training=True)
        eval_criterion = nn.MSELoss()

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

                group_size = getattr(self.args, 'group_size', 4)
                outputs_list = []
                for _ in range(group_size):
                    out = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    outputs_list.append(out)
                outputs = torch.cat(outputs_list, dim=0)

                loss = train_criterion(outputs, batch_y)
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

                self.model.add_to_memory(batch_x, batch_x_mark)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = loss_val / count

            vali_loss = self.vali(vali_data, vali_loader, eval_criterion)
            test_loss_eval = self.vali(test_data, test_loader, eval_criterion)

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

    parser = argparse.ArgumentParser(description='AutoTimes - Combo Preditor Avançado (RAG+MoE+GRPO)')

    parser.add_argument('--opt_id', type=str, default='opt11', help='optimization id')
    parser.add_argument('--opt_name', type=str, default='Combo Preditor Avançado (RAG+MoE+GRPO)', help='optimization name')

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
    parser.add_argument('--loss', type=str, default='GRPO')
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

    parser.add_argument('--rag_top_k', type=int, default=3, help='top-k RAG retrieval')
    parser.add_argument('--rag_memory_size', type=int, default=1000, help='RAG memory bank capacity')
    parser.add_argument('--group_size', type=int, default=4, help='GRPO group size')
    parser.add_argument('--grpo_alpha', type=float, default=0.1, help='GRPO advantage scaling')
    parser.add_argument('--n_experts', type=int, default=4, help='number of MoE experts')
    parser.add_argument('--top_k', type=int, default=2, help='MoE top-k experts')
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

        exp = Exp_ComboPreditor(args)

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