import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import time
import warnings

from models.autotimes_opt5 import Model as AutoTimes_OPT5
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from benchmark_logger import BenchmarkLogger, Timer

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class Exp_OPT5(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _acquire_device(self):
        if self.args.use_multi_gpu:
            return torch.device('cuda:{}'.format(self.args.local_rank))
        return torch.device('cuda:{}'.format(self.args.gpu)) if torch.cuda.is_available() else torch.device('cpu')

    def _build_model(self):
        model = AutoTimes_OPT5(self.args)
        if self.args.use_multi_gpu:
            from torch.nn.parallel import DistributedDataParallel as DDP
            import torch.distributed as dist
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        else:
            model = model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = [p for n, p in self.model.named_parameters() if p.requires_grad]
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                loss = criterion(outputs, batch_y[:, :, :])
                total_loss.append(loss.detach().cpu().item())
                if self.args.smoke_test: break
        self.model.train()
        return np.average(total_loss) if len(total_loss) > 0 else 0.0

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
                if self.args.smoke_test: break

                self.model.module.add_to_memory(batch_x, batch_x_mark) if self.args.use_multi_gpu else self.model.add_to_memory(batch_x, batch_x_mark)

            vali_loss = self.vali(vali_loader, criterion)
            print("Epoch: {}, cost time: {:.2f}s, Vali Loss: {:.7f}".format(epoch + 1, time.time() - epoch_time, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                preds.append(outputs.detach().cpu())
                trues.append(batch_y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        return mse, mae, rmse


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='AutoTimes OPT5 - Temporal RAG')

    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='test')
    parser.add_argument('--model', type=str, default='AutoTimes_OPT5')

    parser.add_argument('--data', type=str, default='ETTm1')
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
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

    parser.add_argument('--rag_top_k', type=int, default=3)
    parser.add_argument('--rag_memory_size', type=int, default=1000)

    parser.add_argument('--num_workers', type=int, default=4)
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

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--opt_id', type=str, default='opt5')
    parser.add_argument('--opt_name', type=str, default='Opt_5_RAG_Temporal')
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
        print(ip, port, hosts, rank, local_rank)
        import torch.distributed as dist
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)
        torch.cuda.set_device(local_rank)

    logger = BenchmarkLogger(reset=args.reset_csv)
    timer = Timer()
    exp = Exp_OPT5(args)

    setting = '{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_ragk{}_ragm{}_{}'.format(
        args.task_name, args.model_id, args.model,
        args.seq_len, args.label_len, args.token_len,
        args.learning_rate, args.batch_size,
        args.rag_top_k, args.rag_memory_size, args.des)

    if args.is_training:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        timer.start()
        exp = Exp_OPT5(args)
        total_time = timer.stop()

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mse, mae, rmse = exp.test(setting)

        train_data, train_loader = exp._get_data('train')
        n_train_batches = len(train_loader) * args.train_epochs

        logger.log(
            opt_id=args.opt_id,
            opt_name=args.opt_name,
            args=args,
            metrics={'mse': mse, 'mae': mae, 'rmse': rmse},
            total_time_s=total_time,
            avg_latency_s=total_time / max(n_train_batches, 1),
            gpu_time_s=timer.gpu_elapsed,
            wall_time_s=timer.wall_elapsed,
        )
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)

    torch.cuda.empty_cache()