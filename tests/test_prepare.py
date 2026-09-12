"""prepare-data tests: per-scene 80/20, validation, manifest, determinism."""
import cv2
import numpy as np
import pytest

from iwpod.prepare import find_scenes, parse_quad_file, prepare_dataset

QUAD = "4,0.1,0.4,0.4,0.1,0.2,0.2,0.5,0.5,car,\n"


def _img(path, w=120, h=80):
    cv2.imwrite(str(path), (np.random.rand(h, w, 3) * 255).astype(np.uint8))


def _scene(root, name, n_good, n_empty=0, n_missing=0, n_bad=0):
    d = root / name
    d.mkdir(parents=True)
    i = 0

    def pair(txt_content):
        nonlocal i
        stem = f"f{i:03d}"
        i += 1
        _img(d / f"{stem}.jpg")
        if txt_content is not None:
            (d / f"{stem}.txt").write_text(txt_content)
        return stem

    for _ in range(n_good):
        pair(QUAD)
    for _ in range(n_empty):
        pair("")
    for _ in range(n_missing):
        pair(None)
    for _ in range(n_bad):
        pair("not,a,quad\n")
    return d


def test_find_scenes_recursive(tmp_path):
    (tmp_path / "a" / "deep").mkdir(parents=True)
    _img(tmp_path / "a" / "deep" / "x.jpg")
    (tmp_path / "empty_dir").mkdir()
    scenes = find_scenes(str(tmp_path))
    assert scenes == [str(tmp_path / "a" / "deep")]


def test_per_scene_split_and_validation(tmp_path):
    src = tmp_path / "raw"
    _scene(src, "big", n_good=10, n_empty=1, n_missing=1, n_bad=1)
    _scene(src, "small", n_good=5)
    out = tmp_path / "prep"
    stats = prepare_dataset(str(src), str(out), ratio=0.8, seed=42,
                            image_reader=lambda p: (120, 80))
    # per-scene 80/20 of valid pairs: big 12 (10 good + empty + missing) -> 10/2;
    # small 5 -> 4/1. The 1 bad file is dropped before splitting.
    big = next(s for s in stats["scenes"] if s["scene"] == "big")
    small = next(s for s in stats["scenes"] if s["scene"] == "small")
    assert (big["train"], big["val"]) == (10, 2)
    assert (small["train"], small["val"]) == (4, 1)
    assert stats["invalid"] == 1
    assert stats["background"] == 2  # empty + missing txt
    assert stats["train"] == 14 and stats["val"] == 3
    # prefixed, loadable by the training loader layout
    names = [p.name for p in (out / "train").iterdir()]
    assert all(n.startswith(("big__", "small__")) for n in names if n != "split_manifest.csv")
    assert (out / "split_manifest.csv").is_file()


def test_determinism(tmp_path):
    src = tmp_path / "raw"
    _scene(src, "s", n_good=20)
    kw = dict(ratio=0.8, seed=7, image_reader=lambda p: (120, 80))
    a = prepare_dataset(str(src), str(tmp_path / "o1"), **kw)
    b = prepare_dataset(str(src), str(tmp_path / "o2"), **kw)
    la = sorted(p.name for p in (tmp_path / "o1" / "train").iterdir())
    lb = sorted(p.name for p in (tmp_path / "o2" / "train").iterdir())
    assert la == lb and a["train"] == b["train"]


def test_overwrite_refused(tmp_path):
    src = tmp_path / "raw"
    _scene(src, "s", n_good=4)
    out = tmp_path / "o"
    prepare_dataset(str(src), str(out), image_reader=lambda p: (120, 80))
    with pytest.raises(RuntimeError, match="exists"):
        prepare_dataset(str(src), str(out), image_reader=lambda p: (120, 80))
    # ...unless asked
    prepare_dataset(str(src), str(out), overwrite=True, image_reader=lambda p: (120, 80))


def test_pixel_coords_normalized(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("4,12,48,48,12,16,16,40,40,,\n")
    quads, err = parse_quad_file(str(f), 120, 80)
    assert err is None
    assert quads[0][0][0] == [0.1, 0.4, 0.4, 0.1]
    assert quads[0][0][1] == [0.2, 0.2, 0.5, 0.5]


def test_label_preserved_through_canonicalization(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("4,0.1,0.4,0.4,0.1,0.2,0.2,0.5,0.5,bike,\n")
    quads, err = parse_quad_file(str(f), 120, 80)
    assert err is None
    assert quads[0][1] == "bike"


def test_parse_shape_line_rejects_garbage(tmp_path):
    import pytest

    from iwpod.label import ShapeParseError, parse_shape_line
    with pytest.raises(ShapeParseError):
        parse_shape_line("4,0.1,oops", lineno=3)
    pts, n, text = parse_shape_line("4,0.1,0.4,0.4,0.1,0.2,0.2,0.5,0.5,car,", lineno=1)
    assert n == 4 and text == "car" and pts.shape == (2, 4)


def test_bad_quad_rejected(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("4,0.1,0.4,99999.0,0.1,0.2,0.2,0.5,0.5,,\n")
    quads, err = parse_quad_file(str(f), 120, 80)
    assert quads is None and err is not None


def test_ratio_and_defaults(tmp_path):
    src = tmp_path / "raw"
    _scene(src, "s", n_good=10)
    out1 = tmp_path / "o1"
    # default ratio=0.8: 10 items -> 8 train, 2 val
    s1 = prepare_dataset(str(src), str(out1), image_reader=lambda p: (120, 80))
    assert s1["train"] == 8 and s1["val"] == 2

    out2 = tmp_path / "o2"
    # explicit ratio=0.7 -> 7 train, 3 val
    s2 = prepare_dataset(str(src), str(out2), ratio=0.7, image_reader=lambda p: (120, 80))
    assert s2["train"] == 7 and s2["val"] == 3


def test_prepare_nested_scene_ids_do_not_collide(tmp_path):
    src = tmp_path / "raw"
    (src / "a" / "cam").mkdir(parents=True)
    (src / "b" / "cam").mkdir(parents=True)
    for d in (src / "a" / "cam", src / "b" / "cam"):
        cv2.imwrite(str(d / "x.jpg"), np.zeros((40, 60, 3), np.uint8))
        (d / "x.txt").write_text(QUAD)
    out = tmp_path / "prep"
    stats = prepare_dataset(str(src), str(out), ratio=1.0, seed=0,
                            image_reader=lambda p: (60, 40))
    names = {s["scene"] for s in stats["scenes"]}
    assert "a__cam" in names and "b__cam" in names
    train_names = [p.name for p in (out / "train").iterdir() if p.suffix == ".jpg"]
    assert any(n.startswith("a__cam__") for n in train_names)
    assert any(n.startswith("b__cam__") for n in train_names)


def test_invalid_ratios(tmp_path):
    src = tmp_path / "raw"
    _scene(src, "s", n_good=4)
    out = tmp_path / "o"
    with pytest.raises(ValueError, match="ratio"):
        prepare_dataset(str(src), str(out), ratio=1.5, image_reader=lambda p: (120, 80))
    with pytest.raises(ValueError, match="ratio"):
        prepare_dataset(str(src), str(out), ratio=-0.1, image_reader=lambda p: (120, 80))

