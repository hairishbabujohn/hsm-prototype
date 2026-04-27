"""
Utilities: seeding, logging, AMP helpers.
"""
import csv
import os
import random
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Logger:
    """Logs metrics to CSV and optionally tensorboard/wandb."""

    def __init__(self, log_dir: str, use_wandb: bool = False, use_tb: bool = False, run_name: str = ""):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self._writer = None
        self._csv_file = None
        self._fields = None
        self.use_wandb = use_wandb
        self.use_tb = use_tb

        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                self._tb_writer = None
                self.use_tb = False
        else:
            self._tb_writer = None

        if use_wandb:
            try:
                import wandb
                wandb.init(project="hsm-prototype", name=run_name or log_dir)
                self._wandb = wandb
            except Exception:
                self.use_wandb = False
                self._wandb = None
        else:
            self._wandb = None

    def log(self, metrics: dict, step: int):
        # CSV
        if self._fields is None:
            self._fields = ["step"] + sorted(metrics.keys())
            self._csv_file = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fields)
            self._writer.writeheader()

        row = {"step": step, **metrics}
        # Fill missing fields
        for f in self._fields:
            if f not in row:
                row[f] = ""
        self._writer.writerow(row)
        self._csv_file.flush()

        if self._tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, step)

        if self._wandb is not None:
            self._wandb.log({**metrics, "step": step})

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb is not None:
            self._wandb.finish()


@contextmanager
def amp_context(enabled: bool = False, device_type: str = "cuda"):
    """AMP autocast context manager."""
    if enabled and torch.cuda.is_available():
        with torch.autocast(device_type=device_type):
            yield
    else:
        yield


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    """Linear warmup + cosine decay scheduler."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
