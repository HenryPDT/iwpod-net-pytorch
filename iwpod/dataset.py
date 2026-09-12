import os

import cv2
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from iwpod.constants import is_real_plate
from iwpod.label import Shape, readShapes
from iwpod.sampler import augment_sample, labels2output_map
from iwpod.utils import image_files_from_folder

# Degenerate placeholder; diagonal ~1.4e-4, below REAL_PLATE_DIAG.
FAKE_PTS = np.array([[0.5, 0.5001, 0.5001, 0.5], [0.5, 0.5, 0.5001, 0.5001]])
assert not is_real_plate(FAKE_PTS)


def image_label_loader(data_path):
    """Load (jpg_path, shapes) entries — LAZY, images stay on disk.

    Eager imread costs ~1MB+ per image (~7.6GB for 9k images); __getitem__
    reads on demand instead.
    - image with non-empty .txt  -> labeled sample
    - image with empty/missing .txt -> background sample (fake degenerate
      plate, trained as all-negative output map)
    Returns (entries, stats).
    """
    files = image_files_from_folder(data_path)
    fakeshape = Shape(FAKE_PTS)
    entries_out = []
    stats = {"labeled": 0, "empty": 0, "missing": 0}
    for file in files:
        labfile = os.path.splitext(file)[0] + '.txt'
        if os.path.isfile(labfile):
            shapes = readShapes(labfile)
            if len(shapes) > 0:
                stats["labeled"] += 1
                entries_out.append((file, shapes))
            else:
                stats["empty"] += 1
                entries_out.append((file, [fakeshape]))
        else:
            stats["missing"] += 1
            entries_out.append((file, [fakeshape]))

    logger.info(f"{len(entries_out)} images: {stats['labeled']} labeled, "
                f"{stats['empty']} empty-txt background, {stats['missing']} missing-txt background")

    return entries_out, stats


def _mem_available_gb():
    """Available RAM in GiB (Linux /proc, else total physical as fallback)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    try:
        import os as _os
        return _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError):
        return float("inf")


def estimate_cache_gb(entries, samples=32):
    """Extrapolate raw-pixel footprint from a random subset."""
    import random as _random
    rng = _random.Random(0)
    idx = rng.sample(range(len(entries)), min(samples, len(entries)))
    tot = 0
    for i in idx:
        img = cv2.imread(entries[i][0])
        if img is not None:
            tot += img.nbytes
    return tot / max(1, len(idx)) * len(entries) / 1024 ** 3


class ALPRDataset(Dataset):
    def __init__(self, data_path=None, dim=208, stride=16, scales=None, entries=None,
                 cache=None):
        """cache: None (lazy reads, default — safe on any RAM size) or "ram".

        "ram" preloads all decoded images (fastest on a training server;
        needs ~1.2x raw pixels free — 7,395 crops measured 5.4GB raw) and
        refuses to start if the estimate exceeds available RAM.
        """
        self.dim = dim
        self.stride = stride
        # Active resolution for dataset-level multi-scale (train loop sets
        # .scale per epoch so every batch stacks; None == self.dim).
        self.scale = None
        if entries is not None:
            self.data = entries
        else:
            self.data, _ = image_label_loader(data_path)
        self.cache = cache
        self._ram = None
        if cache == "ram":
            est = estimate_cache_gb(self.data)
            avail = _mem_available_gb()
            logger.info(f"RAM cache estimate: {est:.1f} GiB raw pixels ({avail:.1f} GiB available)")
            if est > avail:
                raise RuntimeError(
                    f"RAM cache needs ~{est:.1f} GiB but only {avail:.1f} GiB available: "
                    f"free memory or drop --cache (lazy reads).")
            self._ram = []
            for jpg_path, _shapes in self.data:
                img = cv2.imread(jpg_path)
                if img is None:
                    raise RuntimeError(f"Cannot read image '{jpg_path}' (moved or corrupt?)")
                self._ram.append(img)
            self.data = [(None, shapes) for _, shapes in self.data]
        elif cache is not None:
            raise ValueError(f"Unknown cache mode '{cache}': use 'ram' or omit.")

    def __len__(self):
        return len(self.data)

    def _read(self, index):
        if self.cache == "ram":
            return self._ram[index]
        jpg_path = self.data[index][0]
        img = cv2.imread(jpg_path)
        if img is None:
            raise RuntimeError(f"Cannot read image '{jpg_path}' (moved or corrupt?)")
        return img

    def __getitem__(self, index):
        d = self.scale or self.dim
        img = self._read(index)
        shapes = self.data[index][1]
        aug, llp, ptslist = augment_sample(img, shapes, d)
        y = labels2output_map(llp, ptslist, d, self.stride, alpha=0.5)

        inputs = torch.from_numpy(aug).permute(2, 0, 1).float()
        targets = torch.from_numpy(y).permute(2, 0, 1).float()
        return inputs, targets
