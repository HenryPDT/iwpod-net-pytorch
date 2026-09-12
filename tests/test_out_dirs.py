"""Out-dir conventions: eval -> out/eval/<stem>, infer -> out/detect/<stem>."""
from iwpod.runs import default_task_dir


def test_default_task_dir_uses_weights_stem():
    assert default_task_dir("eval", "out/train/exp1/exp1_best.pth") == \
        "out/eval/exp1_best"
    assert default_task_dir("detect", "/a/b/model.pth") == "out/detect/model"
    assert default_task_dir("eval", "m.pth") == "out/eval/m"


def test_eval_log_dir_defaults_to_none_meaning_auto():
    import argparse

    from iwpod.cli import evaluate as ev
    ap = argparse.ArgumentParser()
    ev.register(ap)
    assert ap.parse_args(["--weights", "w.pth", "--data", "d"]).log_dir is None
    assert ap.parse_args(["--weights", "w.pth", "--data", "d",
                          "--log-dir", "/tmp/x"]).log_dir == "/tmp/x"


def test_infer_output_defaults_to_none_meaning_auto():
    import argparse

    from iwpod.cli import infer as inf
    ap = argparse.ArgumentParser()
    inf.register(ap)
    assert ap.parse_args(["--weights", "w.pth", "--input", "d"]).output is None
    assert ap.parse_args(["--weights", "w.pth", "--input", "d",
                          "--output", "/tmp/y"]).output == "/tmp/y"
