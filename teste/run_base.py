import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse, time, random, numpy as np, torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from benchmark_logger import BenchmarkLogger, Timer

if __name__ == '__main__':
    random.seed(2021); torch.manual_seed(2021); np.random.seed(2021)
    parser = argparse.ArgumentParser(description='AutoTimes - Baseline (original, sem otimizações)')
    parser.add_argument('--task_name', default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', default='test')
    parser.add_argument('--model', default='AutoTimes_Gpt2')
    parser.add_argument('--data', default='ETTh1')
    parser.add_argument('--root_path', default='./dataset/ETT-small/')
    parser.add_argument('--data_path', default='ETTh1.csv')
    parser.add_argument('--test_data_path', default='ETTh1.csv')
    parser.add_argument('--checkpoints', default='./checkpoints/')
    parser.add_argument('--drop_last', action='store_true', default=True)
    parser.add_argument('--val_set_shuffle', action='store_false', default=True)
    parser.add_argument('--drop_short', action='store_true', default=False)
    parser.add_argument('--seq_len', type=int, default=672)
    parser.add_argument('--label_len', type=int, default=576)
    parser.add_argument('--token_len', type=int, default=96)
    parser.add_argument('--test_seq_len', type=int, default=672)
    parser.add_argument('--test_label_len', type=int, default=576)
    parser.add_argument('--test_pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', default='Monthly')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--llm_ckp_dir', default='gpt2')
    parser.add_argument('--mlp_hidden_dim', type=int, default=256)
    parser.add_argument('--mlp_hidden_layers', type=int, default=2)
    parser.add_argument('--mlp_activation', default='tanh')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--des', default='Exp')
    parser.add_argument('--loss', default='MSE')
    parser.add_argument('--lradj', default='type1')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--cosine', action='store_true', default=True)
    parser.add_argument('--tmax', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--mix_embeds', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--test_dir', default='./test')
    parser.add_argument('--test_file_name', default='checkpoint.pth')
    parser.add_argument('--opt_id', type=str, default='base')
    parser.add_argument('--opt_name', type=str, default='Baseline (original, sem otimizações)')
    parser.add_argument('--smoke_test', type=int, default=0, help='stop after 1 batch per epoch')
    parser.add_argument('--reset_csv', action='store_true', default=False, help='reset benchmark_results.csv before running')

    args = parser.parse_args()
    if args.smoke_test:
        args.train_epochs = 1
        args.patience = 10
    logger = BenchmarkLogger(reset=args.reset_csv); timer = Timer()

    for ii in range(args.itr):
        setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_ll{args.label_len}_tl{args.token_len}'
        timer.start()
        exp = Exp_Long_Term_Forecast(args)
        print(f'>>>>>>> start training : {setting} >>>>>>>>>')
        exp.train(setting)
        print(f'>>>>>>> testing : {setting} <<<<<<<<<')
        mse, mae, rmse = exp.test(setting, test=0)
        total_time = timer.stop()
        test_loader_len = len(exp._get_data('test')[1])
        avg_latency = total_time / test_loader_len if test_loader_len > 0 else 0.0
        logger.log(args.opt_id, args.opt_name, args, {'mse': mse, 'mae': mae, 'rmse': rmse}, total_time, avg_latency,
                    gpu_time_s=timer.gpu_elapsed, wall_time_s=timer.wall_elapsed)
        torch.cuda.empty_cache()
