import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from models.autotimes_opt7 import Model as MTPModel
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate
from benchmark_logger import BenchmarkLogger, Timer


def parse_args():
    parser = argparse.ArgumentParser(description='AutoTimes – MTP Opt7')

    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='test')
    parser.add_argument('--model', type=str, default='AutoTimes_Gpt2')

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

    parser.add_argument('--mtp_horizons', type=str, default='1,2,4',
                        help='comma-separated horizon multipliers')
    parser.add_argument('--mtp_aux_weight', type=float, default=0.3,
                        help='weight for each auxiliary head loss')
    parser.add_argument('--opt_id', type=str, default='opt7',
                        help='optimization identifier for logging')
    parser.add_argument('--opt_name', type=str, default='mtp',
                        help='optimization name for logging')
    parser.add_argument('--smoke_test', type=int, default=0,
                        help='stop after 1 batch per epoch')
    parser.add_argument('--reset_csv', action='store_true', default=False, help='reset benchmark_results.csv before running')

    return parser.parse_args()


def build_custom_setting(args, run_idx):
    return '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_mtp{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.seq_len,
        args.label_len,
        args.token_len,
        args.learning_rate,
        args.batch_size,
        args.weight_decay,
        args.mlp_hidden_dim,
        args.mlp_hidden_layers,
        args.cosine,
        args.mix_embeds,
        args.mtp_horizons.replace(',', 'x'),
        args.des, run_idx)


def compute_mtp_loss(outputs, batch_y, criterion, aux_weight):
    total_loss = 0.0
    loss_dict = {}
    items = list(outputs.items())
    main_len = items[0][1].shape[1]
    for i, (key, pred) in enumerate(items):
        pred_len = pred.shape[1]
        horizon = pred_len / main_len if main_len > 0 and i > 0 else 1.0
        target_len = batch_y.shape[1]
        if pred_len <= target_len:
            target = batch_y[:, :pred_len, :]
        else:
            target = batch_y
            pred = pred[:, :target_len, :]
        head_loss = criterion(pred, target)
        loss_dict[key] = head_loss.item()
        if i == 0:
            total_loss = head_loss
        else:
            total_loss = total_loss + aux_weight * head_loss / horizon
    return total_loss, loss_dict


def evaluate_model(model, loader, criterion, aux_weight, args):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(args.gpu if not args.use_multi_gpu else args.local_rank)
            batch_y = batch_y.float().to(args.gpu if not args.use_multi_gpu else args.local_rank)
            batch_x_mark = batch_x_mark.float().to(args.gpu if not args.use_multi_gpu else args.local_rank)
            batch_y_mark = batch_y_mark.float().to(args.gpu if not args.use_multi_gpu else args.local_rank)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, None, batch_y_mark)
                    loss, _ = compute_mtp_loss(outputs, batch_y, criterion, aux_weight)
            else:
                outputs = model(batch_x, batch_x_mark, None, batch_y_mark)
                loss, _ = compute_mtp_loss(outputs, batch_y, criterion, aux_weight)
            total_loss += loss.item() * batch_x.shape[0]
            total_count += batch_x.shape[0]
            if args.smoke_test: break

    return total_loss / max(total_count, 1) if total_count > 0 else 0.0


def train_model(args):
    device = args.gpu if not args.use_multi_gpu else args.local_rank
    torch.cuda.set_device(device)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    model = MTPModel(args).to(device)

    p_list = [p for p in model.parameters() if p.requires_grad]
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.dtype, p.shape)

    optimizer = optim.Adam([{'params': p_list}], lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    path = os.path.join(args.checkpoints, args.setting)
    if not os.path.exists(path):
        os.makedirs(path)

    time_now = time.time()
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(args, verbose=True)

    timer = Timer()
    timer.start()
    batch_latencies = []

    for epoch in range(args.train_epochs):
        epoch_time = time.time()
        iter_count = 0

        model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            batch_t0 = time.time()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            optimizer.zero_grad()

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, None, batch_y_mark)
                    loss, _ = compute_mtp_loss(outputs, batch_y, criterion, args.mtp_aux_weight)
                if args.smoke_test: break
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x, batch_x_mark, None, batch_y_mark)
                loss, _ = compute_mtp_loss(outputs, batch_y, criterion, args.mtp_aux_weight)
                if args.smoke_test: break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            if args.smoke_test: break

            batch_latencies.append(time.time() - batch_t0)

            if (i + 1) % 100 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print("\titers: {}, epoch: {} | loss: {:.7f}".format(i + 1, epoch + 1, loss.item()))
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = evaluate_model(model, train_loader, criterion, args.mtp_aux_weight, args)
        val_loss = evaluate_model(model, vali_loader, criterion, args.mtp_aux_weight, args)

        print("Epoch: {} | Train Loss: {:.7f} Vali Loss: {:.7f}".format(
            epoch + 1, train_loss, val_loss))

        early_stopping(val_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if args.cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=1e-8)
            scheduler.step()
        else:
            adjust_learning_rate(optimizer, epoch + 1, args)

    timer.stop()
    total_time_s = timer.elapsed
    avg_latency = np.mean(batch_latencies) if batch_latencies else 0.0

    best_model_path = os.path.join(path, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_model_path), strict=False)

    return model, total_time_s, avg_latency


def test_model(model, loader, args):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device=args.gpu if not args.use_multi_gpu else args.local_rank)
            batch_y = batch_y.float().to(device=args.gpu if not args.use_multi_gpu else args.local_rank)
            batch_x_mark = batch_x_mark.float().to(device=args.gpu if not args.use_multi_gpu else args.local_rank)
            batch_y_mark = batch_y_mark.float().to(device=args.gpu if not args.use_multi_gpu else args.local_rank)

            outputs = model(batch_x, batch_x_mark, None, batch_y_mark)
            main_pred = outputs["main"]

            main_pred = main_pred[:, -args.token_len:, :]
            batch_y = batch_y[:, -args.token_len:, :]

            main_pred = torch.nan_to_num(main_pred, nan=0.0, posinf=1e4, neginf=-1e4)

            preds.append(main_pred.detach().cpu())
            trues.append(batch_y.detach().cpu())
            if args.smoke_test: break

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    return mse, mae, rmse


if __name__ == '__main__':
    args = parse_args()

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    logger = BenchmarkLogger(reset=args.reset_csv)
    timer = Timer()

    for ii in range(args.itr):
        args.setting = build_custom_setting(args, ii)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.setting))

        model, total_time_s, avg_latency = train_model(args)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.setting))

        _, test_loader = data_provider(args, 'test')
        mse, mae, rmse = test_model(model, test_loader, args)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))

        if args.mtp_horizons:
            horizons_str = args.mtp_horizons
        else:
            horizons_str = "1,2,4"
        extra_info = f"mtp_horizons={horizons_str},aux_weight={args.mtp_aux_weight}"

        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
        }

        logger.log(
            opt_id=args.opt_id,
            opt_name=args.opt_name,
            args=args,
            metrics=metrics,
            total_time_s=total_time_s,
            avg_latency_s=avg_latency,
            gpu_time_s=timer.gpu_elapsed,
            wall_time_s=timer.wall_elapsed,
            extra_info=extra_info,
        )

        torch.cuda.empty_cache()