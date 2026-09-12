"""Run-dir behavior: increment, command/config artifacts."""
import os

from iwpod import runs


def test_increment_on_collision(tmp_path):
    base = str(tmp_path)
    d1, n1 = runs.resolve_run_dir(base, "exp1")
    assert n1 == "exp1" and os.path.isdir(d1)
    d2, n2 = runs.resolve_run_dir(base, "exp1")
    assert n2 == "exp1_2" and os.path.isdir(d2)
    d3, _ = runs.resolve_run_dir(base, "exp1")
    assert d3.endswith("exp1_3")


def test_command_and_config_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir, _ = runs.resolve_run_dir(str(tmp_path), "exp1")
    monkeypatch.setattr("sys.argv", ["iwpod", "train", "--epochs", "5"])
    runs.write_train_command(run_dir)
    runs.write_config_snapshot(run_dir, {"epochs": 5})
    cmd = open(os.path.join(run_dir, "train_command.txt")).read()
    assert "iwpod train --epochs 5" in cmd
    assert "epochs: 5" in open(os.path.join(run_dir, "config.yaml")).read()
