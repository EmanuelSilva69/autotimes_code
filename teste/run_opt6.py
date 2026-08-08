import argparse
import os
import random
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from benchmark_logger import BenchmarkLogger, Timer

warnings.filterwarnings('ignore')


class Exp_Opt6:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()
        self.logger = BenchmarkLogger(reset=args.reset_csv)
        self.timer = Timer()

    def _acquire_device(self):
        if self.args.use_multi_gpu:
            return torch.device(f'cuda:{self.args.local_rank}')
        return torch.device(f'cuda:{self.args.gpu}')

    def _build_model(self):
        from models.autotimes_opt6 import Model
        model = Model(self.args)
        if self.args.use_multi_gpu:
            model = nn.parallel.DistributedDataParallel(
                model.cuda(), device_ids=[self.args.local_rank])
        else:
            model = model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = [p for n, p in self.model.named_parameters() if p.requires_grad]
        return optim.Adam([{'params': p_list}], lr=self.args.learning_rate,
                          weight_decay=self.args.weight_decay)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                outputs = outputs[:, -self.args.token_len:, :]
                batch_y = batch_y[:, -self.args.token_len:, :].to(self.device)
                total_loss.append(criterion(outputs, batch_y).detach().cpu())
                if self.args.smoke_test: break
        self.model.train()
        return np.average(total_loss) if len(total_loss) > 0 else 0.0

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        self.timer.start()
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            loss_val = 0.0
            count = 0
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                        loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

                loss_val += loss.item()
                count += 1
                if self.args.smoke_test: break

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    print(f'\tspeed: {speed:.4f}s/iter; left: {left:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.4f}s")
            train_loss = loss_val / max(count, 1)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.7f}, "
                  f"Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.cosine:
                scheduler.step()
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        total_time = self.timer.stop()

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(
            torch.load(best_model_path), strict=False)

        _, test_metrics = self.test(setting, load_model=False)
        avg_latency = total_time / (self.args.train_epochs * max(train_steps, 1))

        self.logger.log(
            opt_id=self.args.opt_id,
            opt_name=self.args.opt_name,
            args=self.args,
            metrics={
                "train_loss": train_loss,
                "val_loss": vali_loss,
                "mse": test_metrics.get("mse", 0),
                "mae": test_metrics.get("mae", 0),
                "rmse": test_metrics.get("rmse", 0),
            },
            total_time_s=total_time,
            avg_latency_s=avg_latency,
            gpu_time_s=self.timer.gpu_elapsed,
            wall_time_s=self.timer.wall_elapsed,
            extra_info=f"d_state={self.args.d_state}, mamba_layers={self.args.mamba_layers}",
        )
        return self.model

    def test(self, setting, test=0, load_model=True):
        test_data, test_loader = self._get_data('test')

        if test and load_model:
            best_model_path = os.path.join(
                self.args.checkpoints, self.args.test_dir, self.args.test_file_name)
            self.model.load_state_dict(torch.load(best_model_path), strict=False)

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                inference_steps = self.args.test_pred_len // self.args.token_len
                dis = self.args.test_pred_len - inference_steps * self.args.token_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat(
                            [batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j-1:j, :]
                        batch_x_mark = torch.cat(
                            [batch_x_mark[:, 1:, :], tmp], dim=1)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(
                                batch_x, batch_x_mark, None, batch_y_mark)
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, None, batch_y_mark)
                    pred_y.append(outputs[:, -self.args.token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-(self.args.token_len - dis), :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                preds.append(pred_y.detach().cpu())
                trues.append(batch_y.detach().cpu())

                if self.args.visualize and len(preds) == 1:
                    folder_path = os.path.join('./test_results', setting)
                    os.makedirs(folder_path, exist_ok=True)
                    gt = np.concatenate([
                        batch_x[0, :, -1].detach().cpu().numpy(),
                        trues[-1][0, :, -1].numpy()])
                    pd = np.concatenate([
                        batch_x[0, :, -1].detach().cpu().numpy(),
                        preds[-1][0, :, -1].numpy()])
                    visual(gt, pd, os.path.join(folder_path, '0.png'))

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}')

        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(f"{setting}\nmse:{mse}, mae:{mae}\n\n")

        return preds, {"mse": mse, "mae": mae, "rmse": rmse}


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='AutoTimes - Mamba/SSM Hybrid')

    parser.add_argument('--task_name', type=str, required=True,
                        default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True,
                        default='AutoTimes_Opt6')

    parser.add_argument('--data', type=str, required=True, default='ETTm1')
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

    parser.add_argument('--d_state', type=int, default=16,
                        help='SSM state dimension')
    parser.add_argument('--mamba_layers', type=int, default=12,
                        help='number of layers to replace with Mamba blocks')

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
    parser.add_argument('--test_file_name', type=str,
                        default='checkpoint.pth')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--opt_id', type=str, default='opt6')
    parser.add_argument('--opt_name', type=str, default='Opt_6_Mamba_SSM')
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

    timer = Timer()
    if args.is_training:
        for ii in range(args.itr):
            setting = (
                f'{args.task_name}_{args.model_id}_{args.model}_{args.data}'
                f'_sl{args.seq_len}_ll{args.label_len}_tl{args.token_len}'
                f'_lr{args.learning_rate}_bt{args.batch_size}'
                f'_wd{args.weight_decay}_hd{args.mlp_hidden_dim}'
                f'_hl{args.mlp_hidden_layers}_cos{args.cosine}'
                f'_mix{args.mix_embeds}_ds{args.d_state}'
                f'_ml{args.mamba_layers}_{args.des}_{ii}'
            )
            print(f'>>>>>start training: {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            timer.start()
            exp = Exp_Opt6(args)
            exp.train(setting)
            print(f'>>>>>testing: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = (
            f'{args.task_name}_{args.model_id}_{args.model}_{args.data}'
            f'_sl{args.seq_len}_ll{args.label_len}_tl{args.token_len}'
            f'_lr{args.learning_rate}_bt{args.batch_size}'
            f'_wd{args.weight_decay}_hd{args.mlp_hidden_dim}'
            f'_hl{args.mlp_hidden_layers}_cos{args.cosine}'
            f'_mix{args.mix_embeds}_ds{args.d_state}'
            f'_ml{args.mamba_layers}_{args.des}_{ii}'
        )
        timer.start()
        exp = Exp_Opt6(args)
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
