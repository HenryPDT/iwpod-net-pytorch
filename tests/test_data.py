"""Dataset loader tests: labeled / empty-txt / missing-txt handling."""
import cv2
import numpy as np
import pytest

from iwpod.dataset import ALPRDataset, estimate_cache_gb, image_label_loader


def _img(path, w=120, h=80):
    cv2.imwrite(str(path), (np.random.rand(h, w, 3) * 255).astype(np.uint8))


def test_loader_counts_and_background(tmp_path):
    _img(tmp_path / "a.jpg")
    (tmp_path / "a.txt").write_text("4,0.1,0.4,0.4,0.1,0.2,0.2,0.5,0.5,car,\n")
    _img(tmp_path / "b.jpg")
    (tmp_path / "b.txt").write_text("")
    _img(tmp_path / "c.jpg")  # no txt at all
    entries, stats = image_label_loader(str(tmp_path))
    assert len(entries) == 3
    assert stats == {"labeled": 1, "empty": 1, "missing": 1}
    # background entries carry the degenerate fake plate
    assert len(entries[1][1]) == 1 and len(entries[2][1]) == 1


def _labeled_dir(tmp_path, n=4):
    d = tmp_path / "ds"
    d.mkdir(exist_ok=True)
    for i in range(n):
        _img(d / f"s{i:02d}.jpg")
        (d / f"s{i:02d}.txt").write_text("4,0.1,0.4,0.4,0.1,0.2,0.2,0.5,0.5,,\n")
    return str(d)


def test_estimate_cache_gb(tmp_path):
    d = _labeled_dir(tmp_path)
    entries, _ = image_label_loader(d)
    est = estimate_cache_gb(entries)
    assert 0 < est < 1.0  # 4 tiny images: megabytes, not gigabytes


def test_cache_ram_preloads(tmp_path):
    d = _labeled_dir(tmp_path)
    ds = ALPRDataset(d, dim=128, cache="ram")
    assert ds._ram is not None and len(ds._ram) == 4
    x, y = ds[1]
    assert x.shape == (3, 128, 128) and y.shape[0] == 9


def test_cache_invalid_mode(tmp_path):
    d = _labeled_dir(tmp_path)
    with pytest.raises(ValueError, match="Unknown cache mode"):
        ALPRDataset(d, dim=128, cache="tape")
