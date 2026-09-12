"""CLI smoke: every subcommand wires up and prints help."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("sub", ["prepare-data", "train", "eval", "infer", "export"])
def test_cli_help(sub):
    r = subprocess.run([sys.executable, "-m", "iwpod", sub, "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "--help" in r.stdout or "usage" in r.stdout


def test_top_help_lists_subcommands():
    r = subprocess.run([sys.executable, "-m", "iwpod", "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    for sub in ("prepare-data", "train", "eval", "infer", "export"):
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
