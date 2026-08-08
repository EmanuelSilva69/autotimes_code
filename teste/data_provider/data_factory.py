import os
from data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom, Dataset_M4, Dataset_Solar, Dataset_TSF, Dataset_TSF_ICL
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_hour,
    'ETTm2': Dataset_ETT_hour,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'Solar': Dataset_Solar,
    'tsf': Dataset_TSF,
    'tsf_icl': Dataset_TSF_ICL
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size 
    elif flag == 'val':
        shuffle_flag = args.val_set_shuffle
        drop_last = False
        batch_size = args.batch_size 
    else:
        shuffle_flag = True
        drop_last = args.drop_last
        batch_size = args.batch_size

    if flag in ['train', 'val']:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.token_len],
            seasonal_patterns=args.seasonal_patterns,
            drop_short=args.drop_short,
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.test_seq_len, args.test_label_len, args.test_pred_len],
            seasonal_patterns=args.seasonal_patterns,
            drop_short=args.drop_short,
        )
    if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
        print(flag, len(data_set))

    workers = 0 if os.name == 'nt' else max(args.num_workers, 4)
    prefetch_val = 2 if workers > 0 else None

    if args.use_multi_gpu:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(data_set, 
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=True,
            prefetch_factor=prefetch_val,
            drop_last=drop_last,
            )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=workers,
            pin_memory=True,
            prefetch_factor=prefetch_val,
            drop_last=drop_last)
    return data_set, data_loader