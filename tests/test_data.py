"""Dataset loader tests: labeled / empty-txt / missing-txt handling."""
import cv2
import numpy as np
import pytest

from iwpod.dataset import ALPRDataset, estimate_cache_gb, image_label_loader
from iwpod.label import ShapeParseError, parse_shape_line
from iwpod.utils import image_files_from_folder


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


def test_parse_shape_line_rejects_garbage():
    with pytest.raises(ShapeParseError):
        parse_shape_line("4,0.1,oops", lineno=3)
    pts, n, text = parse_shape_line("4,0.1,0.4,0.4,0.1,0.2,0.2,0.5,0.5,car,", lineno=1)
    assert n == 4 and text == "car" and pts.shape == (2, 4)


def test_loader_nested_subdirs(tmp_path):
    for scene, stem in (("Access_Control", "a"), ("Cahill_Entry_Front", "c")):
        d = tmp_path / scene
        d.mkdir()
        _img(d / f"{stem}.jpg")
        (d / f"{stem}.txt").write_text("4,0.1,0.4,0.4,0.1,0.2,0.2,0.5,0.5,car,\n")
    empty = tmp_path / "layover_only"
    empty.mkdir()
    _img(empty / "bg.jpg")
    (empty / "bg.txt").write_text("")
    entries, stats = image_label_loader(str(tmp_path))
    assert len(entries) == 3
    assert stats == {"labeled": 2, "empty": 1, "missing": 0}
    paths = {e[0] for e in entries}
    assert any("Access_Control" in p for p in paths)
    assert any("Cahill_Entry_Front" in p for p in paths)


def test_image_files_from_folder_recursive(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _img(tmp_path / "a" / "x.jpg")
    _img(tmp_path / "b" / "y.png")
    _img(tmp_path / "root.jpg")
    files = image_files_from_folder(str(tmp_path))
    assert len(files) == 3
    assert any(p.endswith("root.jpg") for p in files)
    assert any("a" in p and p.endswith("x.jpg") for p in files)
    assert any("b" in p and p.endswith("y.png") for p in files)


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
