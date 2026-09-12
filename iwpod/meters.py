"""Smoothed training meters (YOLOX-style MeterBuffer, single-GPU port).

Tracks per-window avg for noisy timings and latest values for losses, plus a
global average for stable ETA estimates. No distributed support — this repo is
single-process.
"""
import functools
from collections import defaultdict, deque

import numpy as np
import torch


def gpu_mem_usage():
    """Current GPU allocation for this process in MB (0.0 off-CUDA)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 * 1024)


def gpu_mem_peak():
    """Peak GPU allocation since last reset in MB (0.0 off-CUDA)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def host_mem_usage():
    """Host process RSS in GB (falls back to system used on psutil absence)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1 << 30)
    except Exception:
        return 0.0


class AverageMeter:
    """Series of values with windowed avg and global avg access."""

    def __init__(self, window_size=50):
        self._deque = deque(maxlen=window_size)
        self._total = 0.0
        self._count = 0

    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean().item()
        value = float(value)
        self._deque.append(value)
        self._count += 1
        self._total += value

    @property
    def avg(self):
        if not self._deque:
            return 0.0
        return float(np.mean(list(self._deque)))

    @property
    def global_avg(self):
        return self._total / max(self._count, 1)

    @property
    def latest(self):
        return self._deque[-1] if self._deque else 0.0

    def reset(self):
        self._deque.clear()
        self._total = 0.0
        self._count = 0

    def clear(self):
        self._deque.clear()


class MeterBuffer(defaultdict):
    """Dict of AverageMeters with filtered views (e.g. 'loss', 'time')."""

    def __init__(self, window_size=20):
        super().__init__(functools.partial(AverageMeter, window_size=window_size))

    def reset(self):
        for v in self.values():
            v.reset()

    def get_filtered_meter(self, filter_key="time"):
        return {k: v for k, v in self.items() if filter_key in k}

    def update(self, values=None, **kwargs):
        if values is None:
            values = {}
        values.update(kwargs)
        for k, v in values.items():
            self[k].update(v)

    def clear_meters(self):
        for v in self.values():
            v.clear()
