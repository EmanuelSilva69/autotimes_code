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

from benchmark_logger import BenchmarkLogger, Timer
from utils.metrics import metric
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP


class MoEWithShared(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=4, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts)
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_ff, d_model),
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        shared_out = self.shared(x)
        logits = self.gate(x)
        weights = F.softmax(logits, dim=-1)
        top_w, top_idx = torch.topk(weights, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        routed_out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = top_idx[:, :, k]
            w = top_w[:, :, k:k+1]
            for e in range(self.n_experts):
                mask = (idx == e)
                if mask.any():
                    expert_in = x[mask]
                    routed_out[mask] += self.experts[e](expert_in) * w[mask]
        return shared_out + routed_out


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


class GRPOLoss(nn.Module):
    def __init__(self, clip_eps=0.2, beta=0.01):
        super().__init__()
        self.clip_eps = clip_eps
        self.beta = beta

    def forward(self, outputs, targets, old_preds=None):
        mse = F.mse_loss(outputs, targets)
        if old_preds is not None:
            ratio = torch.exp(-F.mse_loss(outputs, targets, reduction='none') + F.mse_loss(old_preds, targets, reduction='none'))
            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            grpo = -torch.min(ratio, clipped).mean()
            kl = F.kl_div(
                F.log_softmax(outputs.view(-1), dim=-1),
                F.softmax(targets.view(-1), dim=-1),
                reduction='batchmean',
            )
            return mse + 0.1 * grpo + self.beta * kl
        return mse


class ComboPerdaAdaptativaModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        torch.set_float32_matmul_precision('high')

        self.token_len = configs.token_len
        self.n_experts = getattr(configs, 'n_experts', 4)
        self.top_k = getattr(configs, 'moe_top_k', 2)
        self.mtp_horizons = getattr(configs, 'mtp_horizons', [1, 2, 4])
        self.grpo_group_size = getattr(configs, 'grpo_group_size', 4)

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

        self.moe_encoder = MoEWithShared(self.hidden_dim, configs.mlp_hidden_dim * 2,
                                          self.n_experts, self.top_k)
        self.moe_decoder = MoEWithShared(self.hidden_dim * 2, configs.mlp_hidden_dim * 2,
                                          self.n_experts, self.top_k)

        self.mtp = MTPDecoder(self.hidden_dim, self.token_len,
                              self.mtp_horizons,
                              configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                              configs.dropout, configs.mlp_activation)

        self.moe_mid = MoEWithShared(self.hidden_dim, configs.mlp_hidden_dim * 2,
                                      self.n_experts, self.top_k)

        if configs.mlp_hidden_layers == 0:
            self.decoder = nn.Linear(self.hidden_dim, self.token_len)
        else:
            self.decoder = MLP(self.hidden_dim, self.token_len,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mtp_all=False):
        with torch.cuda.amp.autocast():
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc -= means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

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

            encoded = self.moe_encoder(times_embeds)
            encoded = encoded + times_embeds

            outputs = self.gpt2(inputs_embeds=encoded.to(dtype=self.gpt2.dtype).contiguous()).last_hidden_state
            outputs = outputs.to(dtype=torch.float32)

            mid = self.moe_mid(outputs)
            outputs = outputs + mid

            combined = torch.cat([outputs, times_embeds], dim=-1)
            decoded = self.moe_decoder(combined)

            if mtp_all:
                mtp_outs = self.mtp(decoded)
                results = []
                for head_out in mtp_outs:
                    d_out = head_out.reshape(bs, n_vars, -1)
                    d_out = d_out.permute(0, 2, 1).contiguous()
                    d_out = d_out * stdev[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1)
                    d_out = d_out + means[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1)
                    results.append(d_out)
                return results

            decoded = self.decoder(decoded)

            dec_out = decoded.reshape(bs, n_vars, -1)
            dec_out = dec_out.permute(0, 2, 1).contiguous()

            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, token_num * self.token_len, -1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mtp_all=False)


class Exp_ComboPerdaAdaptativa(object):
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
        model = ComboPerdaAdaptativaModel(self.args).float()
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

    def _mtp_loss(self, mtp_outputs, batch_y, criterion):
        total = 0.0
        for i, h in enumerate(getattr(self.args, 'mtp_horizons', [1, 2, 4])):
            pred = mtp_outputs[i]
            shift = (h - 1) * self.args.token_len
            target_len = pred.shape[1]
            target = batch_y[:, shift:shift + target_len, :].to(self.device, non_blocking=True)
            loss = criterion(pred, target)
            total = total + loss
        return total / len(mtp_outputs)

    def _grpo_train_step(self, batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, model_optim):
        group_size = getattr(self.args, 'grpo_group_size', 4)
        noise_std = getattr(self.args, 'grpo_noise_std', 0.05)

        noise_x = torch.randn(group_size, *batch_x.shape, device=self.device) * noise_std
        noises_x_mark = torch.randn(group_size, *batch_x_mark.shape, device=self.device) * noise_std * 0.1

        group_losses = []
        group_preds = []

        for g in range(group_size):
            noisy_x = batch_x + noise_x[g]
            noisy_x_mark = batch_x_mark + noises_x_mark[g]

            with torch.cuda.amp.autocast():
                mtp_outs = self.model.forecast(noisy_x, noisy_x_mark, None, batch_y_mark, mtp_all=True)
                loss = self._mtp_loss(mtp_outs, batch_y, criterion)

            group_losses.append(loss.detach())
            group_preds.append(mtp_outs[0].detach())

        group_losses = torch.stack(group_losses)
        mean_loss = group_losses.mean()
        std_loss = group_losses.std() + 1e-8
        advantages = (mean_loss - group_losses) / std_loss
        adv_weights = F.softmax(advantages / 0.1, dim=0)

        final_loss = 0.0
        for g in range(group_size):
            with torch.cuda.amp.autocast():
                mtp_outs = self.model.forecast(
                    batch_x + noise_x[g],
                    batch_x_mark + noises_x_mark[g],
                    None, batch_y_mark, mtp_all=True)
                loss = self._mtp_loss(mtp_outs, batch_y, criterion)
            final_loss = final_loss + adv_weights[g].detach() * loss

        return final_loss, mean_loss.item()

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

                loss, base_val = self._grpo_train_step(
                    batch_x, batch_y, batch_x_mark, batch_y_mark,
                    criterion, model_optim)
                loss_val += base_val
                count += 1
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, base_val))
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

    parser = argparse.ArgumentParser(description='AutoTimes - Combo Perda Adaptativa (GRPO+MTP+MoE)')

    parser.add_argument('--opt_id', type=str, default='opt14', help='optimization id')
    parser.add_argument('--opt_name', type=str, default='Combo Perda Adaptativa (GRPO+MTP+MoE)', help='optimization name')

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

    parser.add_argument('--n_experts', type=int, default=4, help='number of experts in MoE')
    parser.add_argument('--moe_top_k', type=int, default=2, help='top-k experts per token')
    parser.add_argument('--mtp_horizons', type=int, nargs='+', default=[1, 2, 4], help='MTP prediction horizons')
    parser.add_argument('--grpo_group_size', type=int, default=4, help='number of noisy passes for GRPO')
    parser.add_argument('--grpo_noise_std', type=float, default=0.05, help='std of noise for GRPO group')
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

        exp = Exp_ComboPerdaAdaptativa(args)

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
