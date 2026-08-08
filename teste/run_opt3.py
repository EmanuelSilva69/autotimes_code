import argparse
import os
import random
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from models.autotimes_opt3 import Model
from benchmark_logger import BenchmarkLogger, Timer

warnings.filterwarnings('ignore')


class Exp_MoE:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _acquire_device(self):
        if self.args.use_multi_gpu:
            return torch.device('cuda:{}'.format(self.args.local_rank))
        return torch.device('cuda:{}'.format(self.args.gpu))

    def _build_model(self):
        model = Model(self.args)
        if self.args.use_multi_gpu:
            model = nn.parallel.DistributedDataParallel(
                model.cuda(), device_ids=[self.args.local_rank]
            )
        else:
            model = model.to(self.device)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        p_list = [p for p in self.model.parameters() if p.requires_grad]
        model_optim = optim.Adam(p_list, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                if is_test:
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:, :].to(self.device)
                else:
                    outputs = outputs[:, :, :]
                    batch_y = batch_y[:, :, :].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.detach().cpu().item())
                total_count.append(batch_x.shape[0])
                if self.args.smoke_test: break
        self.model.train()
        if len(total_loss) == 0 or sum(total_count) == 0:
            return 0.0
        return np.average(total_loss, weights=total_count)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            loss_val = 0.0
            count = 0
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                loss = criterion(outputs, batch_y)
                loss.backward()
                model_optim.step()
                loss_val += loss.item()
                count += 1
                if self.args.smoke_test: break

            train_loss = loss_val / count
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
            print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            setting = self.args.test_dir
            best_model_path = os.path.join(self.args.checkpoints, setting, self.args.test_file_name)
            load_item = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j - 1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    pred_y.append(outputs[:, -self.args.token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-(self.args.token_len - dis), :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                preds.append(pred_y.detach().cpu())
                trues.append(batch_y.detach().cpu())

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        return {'mse': mse, 'mae': mae, 'rmse': rmse}


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='AutoTimes')

    parser.add_argument('--opt_id', type=str, default='opt3')
    parser.add_argument('--opt_name', type=str, default='Mixture of Experts (MoE)')

    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, default='autotimes_opt3')

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

    parser.add_argument('--num_experts', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=2)

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
        print(ip, port, hosts, rank, local_rank, torch.cuda.device_count())
        torch.distributed.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}",
                                             world_size=hosts, rank=rank)
        torch.cuda.set_device(local_rank)

    logger = BenchmarkLogger(reset=args.reset_csv)
    timer = Timer()

    if args.is_training:
        for ii in range(args.itr):
            timer.start()
            exp = Exp_MoE(args)
            setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_ne{}_tk{}_{}_{}'.format(
                args.task_name, args.model_id, args.model, args.data,
                args.seq_len, args.label_len, args.token_len,
                args.learning_rate, args.batch_size,
                args.num_experts, args.top_k,
                args.des, ii)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            metrics = exp.test(setting)
            total_time = timer.stop()

            test_loader_len = len(exp._get_data('test')[1])
            avg_latency = total_time / test_loader_len if test_loader_len > 0 else 0.0
            

            log_entry = logger.log(
                opt_id=args.opt_id,
                opt_name=args.opt_name,
                args=args,
                metrics=metrics,
                total_time_s=total_time,
                avg_latency_s=avg_latency,
                gpu_time_s=timer.gpu_elapsed,
                wall_time_s=timer.wall_elapsed,
                extra_info='num_experts={}, top_k={}'.format(args.num_experts, args.top_k),
            )
            print('Logged: {} - {}'.format(log_entry['opt_id'], log_entry['opt_name']))
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_ne{}_tk{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data,
            args.seq_len, args.label_len, args.token_len,
            args.learning_rate, args.batch_size,
            args.num_experts, args.top_k,
            args.des, ii)
        exp = Exp_MoE(args)
        metrics = exp.test(setting, test=1)
        print('Test results: mse={}, mae={}'.format(metrics['mse'], metrics['mae']))
        torch.cuda.empty_cache()