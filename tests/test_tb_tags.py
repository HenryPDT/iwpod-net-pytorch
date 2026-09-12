"""TensorBoard tag contract: COCO trio (map/map50/map75), no per-threshold spray."""
from iwpod.cli.train import tb_epoch_tags


def test_no_duplicate_iou_scalars():
    tags = tb_epoch_tags()
    assert "val/iou50" not in tags
    assert "val/iou70" not in tags
    assert not any(t.startswith("val/iou/") for t in tags)
    assert len(tags) == len(set(tags))


def test_coco_trio_present():
    tags = tb_epoch_tags()
    assert "val/map50" in tags and "val/map75" in tags and "val/map50-95" in tags


def test_map75_matches_ap_curve():
    # map75 must come from the AP curve (threshold-free), not recall@IoU.
    import numpy as np

    from iwpod.eval_core import map_scores
    m = map_scores(np.array([0.9, 0.4]), np.array([0.8, 0.2]), np.array([True, True]))
    assert m["curve"][0.75] <= m["map50"] + 1e-9


def test_is_better_prefers_map():
    from iwpod.cli.train import is_better
    base = dict(map=0.5, iou70=0.4, iou50=0.6, recall=0.8, vloss=1.0)
    assert is_better({**base, "map": 0.6}, base)
    assert not is_better(base, {**base, "map": 0.6})
    # tie on map falls through to iou70
    assert is_better({**base, "iou70": 0.5}, base)
    # tie on map+iou70 falls through to iou50
    assert is_better({**base, "iou50": 0.7}, base)
