"""CLI smoke: every subcommand wires up and prints help."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("sub", ["train", "eval", "infer", "export"])
def test_cli_help(sub):
    r = subprocess.run([sys.executable, "-m", "iwpod", sub, "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "--help" in r.stdout or "usage" in r.stdout


def test_top_help_lists_subcommands():
    r = subprocess.run([sys.executable, "-m", "iwpod", "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    for sub in ("train", "eval", "infer", "export"):
        assert sub in r.stdout


def _base_cfg():
    from iwpod.cli.train import default_config_path, load_cfg
    return load_cfg(default_config_path())


def test_validate_config_accepts_bundled():
    from iwpod.cli.train import validate_config
    validate_config(_base_cfg())  # must not raise


def test_validate_config_rejects_bad_geometry_and_scheduler():
    import copy

    import pytest

    from iwpod.cli.train import validate_config
    bad_dim = _base_cfg()
    bad_dim["dim"] = 100
    with pytest.raises(RuntimeError, match="multiple of 32"):
        validate_config(bad_dim)
    bad_scale = _base_cfg()
    bad_scale["multi_scale"] = [100]
    with pytest.raises(RuntimeError, match="multiples of 32"):
        validate_config(bad_scale)
    bad_sched = _base_cfg()
    bad_sched["scheduler"] = "onecycle"
    with pytest.raises(RuntimeError, match="cosine\\|none"):
        validate_config(bad_sched)
    bad_stride = _base_cfg()
    bad_stride["stride"] = 8
    with pytest.raises(RuntimeError, match="label-encoding"):
        validate_config(bad_stride)
    neg_loss = _base_cfg()
    neg_loss["loss"] = dict(copy.deepcopy(neg_loss.get("loss") or {}), w_loc=-1.0)
    with pytest.raises(RuntimeError, match="non-negative"):
        validate_config(neg_loss)


def test_loss_kwargs_tolerates_missing_keys():
    from iwpod.cli.train import loss_kwargs
    kw = loss_kwargs({})
    assert kw["w_cls"] == 1.0 and kw["ohem_neg_ratio"] == 3.0
    kw = loss_kwargs({"w_loc": 2.0})
    assert kw["w_loc"] == 2.0 and kw["w_cls"] == 1.0


def test_cache_flag_defaults_to_ram():
    import argparse

    from iwpod.cli import train as train_cli
    ap = argparse.ArgumentParser()
    train_cli.register(ap)
    assert ap.parse_args(["--cache"]).cache == "ram"
    assert ap.parse_args(["--cache", "ram"]).cache == "ram"
    assert ap.parse_args([]).cache is None


def test_v3_flags_default_to_v2_behavior():
    import argparse

    from iwpod.cli import train as train_cli
    ap = argparse.ArgumentParser()
    train_cli.register(ap)
    a = ap.parse_args([])
    assert a.backbone is None and a.head is None
    assert a.simam is None
    assert a.detail_boost is None
    assert not hasattr(a, "loc_type") and not hasattr(a, "copy_paste_p")
    b = ap.parse_args(["--backbone", "rep", "--head", "dw", "--simam",
                       "--detail-boost", "0.5"])
    assert (b.backbone, b.head, b.simam) == ("rep", "dw", True)
    assert b.detail_boost == 0.5


def test_v3_overrides_land_in_config_snapshot():
    from iwpod.cli.train import apply_cli_overrides, default_config_path, load_cfg
    import argparse

    from iwpod.cli import train as train_cli
    ap = argparse.ArgumentParser()
    train_cli.register(ap)
    cfg = load_cfg(default_config_path())
    a = ap.parse_args(["--backbone", "rep", "--simam"])
    apply_cli_overrides(cfg, a)
    assert cfg["model"]["backbone"] == "rep"
    assert cfg["model"]["use_simam"] is True
    assert cfg["model"]["arch_version"] == "v3-s16"
    # defaults untouched without flags
    cfg2 = load_cfg(default_config_path())
    apply_cli_overrides(cfg2, ap.parse_args([]))
    assert cfg2["model"]["backbone"] == "orig"
    assert cfg2["loss"]["w_loc"] == 1.0


def test_arch_preset_bundles():
    import argparse

    from iwpod.cli import train as train_cli
    from iwpod.cli.train import apply_cli_overrides, default_config_path, load_cfg
    ap = argparse.ArgumentParser()
    train_cli.register(ap)
    cfg = load_cfg(default_config_path())
    apply_cli_overrides(cfg, ap.parse_args(["--arch", "v3"]))
    assert cfg["model"]["backbone"] == "rep"
    assert cfg["model"]["head"] == "dw"
    assert cfg["model"]["use_simam"] is True
    assert cfg["model"]["arch_version"] == "v3-s16"
    assert cfg["loss"]["w_loc"] == 1.0
    assert cfg["augment"]["detail_boost"] == 0.5
    assert "copy_paste_p" not in cfg["augment"]  # removed: FP suspect, never isolated
    # granular flag overrides the preset
    cfg = load_cfg(default_config_path())
    apply_cli_overrides(cfg, ap.parse_args(["--arch", "v3", "--head", "orig", "--no-simam"]))
    assert cfg["model"]["head"] == "orig"
    assert cfg["model"]["backbone"] == "rep"  # preset value kept
    assert cfg["model"]["use_simam"] is False
    assert cfg["model"]["arch_version"] == "v3-s16"
    # v2 preset is a no-op reset
    cfg = load_cfg(default_config_path())
    apply_cli_overrides(cfg, ap.parse_args(["--arch", "v2"]))
    assert cfg["model"]["backbone"] == "orig"
    assert cfg["model"]["arch_version"] == "v2"


def test_v3_overrides_reject_bad_values():
    import argparse

    import pytest

    from iwpod.cli import train as train_cli
    from iwpod.cli.train import apply_cli_overrides, default_config_path, load_cfg, validate_config
    ap = argparse.ArgumentParser()
    train_cli.register(ap)
    cfg = load_cfg(default_config_path())
    a = ap.parse_args(["--detail-boost", "1.5"])
    with pytest.raises(RuntimeError, match="detail-boost"):
        apply_cli_overrides(cfg, a)
    bad = load_cfg(default_config_path())
    bad["loss"]["loc_type"] = "lpwing"  # removed loss: stale configs fail loud
    with pytest.raises(RuntimeError, match="was removed"):
        validate_config(bad)
    bad2 = load_cfg(default_config_path())
    bad2["augment"]["copy_paste_p"] = 0.3  # removed augment: same treatment
    with pytest.raises(RuntimeError, match="was removed"):
        validate_config(bad2)



