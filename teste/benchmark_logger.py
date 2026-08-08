import csv
import os
import time
import uuid
import torch
import numpy as np
from datetime import datetime
from pathlib import Path


class BenchmarkLogger:
    CSV_HEADER = [
        "run_uuid", "opt_id", "opt_name", "timestamp",
        "dataset", "base_model",
        "seq_len", "label_len", "token_len", "pred_len",
        "batch_size", "learning_rate", "epochs",
        "total_time_s", "gpu_time_s", "wall_time_s", "avg_latency_s_per_batch",
        "peak_vram_gb",
        "train_loss", "val_loss",
        "test_mse", "test_mae", "test_rmse",
        "gpu_name", "pytorch_version", "extra_info"
    ]

    def __init__(self, csv_path=None, reset=False):
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")
        self.csv_path = csv_path
        self.run_uuid = uuid.uuid4().hex[:12]
        if reset:
            self.reset_csv()
        else:
            self._init_csv()

    def reset_csv(self):
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADER)

    def _init_csv(self):
        if not os.path.isfile(self.csv_path):
            self.reset_csv()

    def get_peak_vram(self):
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        return 0.0

    def get_gpu_name(self):
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "cpu"

    def log(self, opt_id, opt_name, args, metrics, total_time_s, avg_latency_s, extra_info="", gpu_time_s=None, wall_time_s=None):
        peak_vram = self.get_peak_vram()
        row = {
            "run_uuid": self.run_uuid,
            "opt_id": opt_id,
            "opt_name": opt_name,
            "timestamp": datetime.now().isoformat(),
            "dataset": args.data,
            "base_model": args.model,
            "seq_len": args.seq_len,
            "label_len": args.label_len,
            "token_len": args.token_len,
            "pred_len": args.token_len,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.train_epochs,
            "total_time_s": round(gpu_time_s, 4) if gpu_time_s is not None else round(total_time_s, 4),
            "gpu_time_s": round(gpu_time_s, 4) if gpu_time_s is not None else round(total_time_s, 4),
            "wall_time_s": round(wall_time_s, 4) if wall_time_s is not None else round(total_time_s, 4),
            "avg_latency_s_per_batch": round(avg_latency_s, 6),
            "peak_vram_gb": round(peak_vram, 4),
            "train_loss": round(metrics.get("train_loss", 0.0), 8),
            "val_loss": round(metrics.get("val_loss", 0.0), 8),
            "test_mse": round(metrics.get("mse", 0.0), 8),
            "test_mae": round(metrics.get("mae", 0.0), 8),
            "test_rmse": round(metrics.get("rmse", 0.0), 8),
            "gpu_name": self.get_gpu_name(),
            "pytorch_version": torch.__version__,
            "extra_info": str(extra_info),
        }
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADER)
            writer.writerow(row)

        test_mse = row.get("test_mse", 0.0)
        test_mae = row.get("test_mae", 0.0)
        test_rmse = row.get("test_rmse", 0.0)
        print("\n=== Benchmark Results ===")
        print(f"opt_id:        {opt_id}")
        print(f"opt_name:      {opt_name}")
        print(f"wall_time:     {total_time_s:.4f}s")
        print(f"gpu_time:      {row['gpu_time_s']:.4f}s")
        print(f"test_mse:      {test_mse:.8f}")
        print(f"test_mae:      {test_mae:.8f}")
        print(f"test_rmse:     {test_rmse:.8f}")
        print("=========================\n")

        return row


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.elapsed = 0.0
        self.wall_elapsed = 0.0
        self.gpu_elapsed = 0.0
        self._wall_start = None
        self._gpu_start = None
        self._gpu_end = None

    def start(self):
        torch.cuda.synchronize()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._wall_start = time.time()
        self._gpu_start = torch.cuda.Event(enable_timing=True)
        self._gpu_start.record()
        return self

    def stop(self):
        self._gpu_end = torch.cuda.Event(enable_timing=True)
        self._gpu_end.record()
        torch.cuda.synchronize()
        self.gpu_elapsed = self._gpu_start.elapsed_time(self._gpu_end) / 1000.0
        self.wall_elapsed = time.time() - self._wall_start
        self.elapsed = self.wall_elapsed
        return self.elapsed
