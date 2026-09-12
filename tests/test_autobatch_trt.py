"""Auto-batch probe + export-trt builder tests (no GPU/TRT needed)."""
import torch

from iwpod.autobatch import auto_batch_size


def test_probe_cpu_fallback_returns_none():
    def model_fn():
        from iwpod.model import IWPODNet
        return IWPODNet()

    x = torch.rand(1, 3, 64, 64)
    y = torch.zeros(1, 9, 4, 4)
    assert auto_batch_size(model_fn, lambda o, t: o.sum(), (x, y),
                           torch.device("cpu")) is None


def test_export_trt_missing_binary():
    import argparse

    from iwpod.cli import export_trt
    ns = argparse.Namespace(onnx="m.onnx", size=[384], batch=8, fp16=True,
                            int8=False, calib=None, workspace=2048,
                            output=None, trtexec="definitely-not-a-binary-xyz")
    try:
        export_trt.build_command(ns)
    except RuntimeError as e:
        assert "not found" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_export_trt_command_shape(tmp_path, monkeypatch):
    import argparse

    from iwpod.cli import export_trt
    fake = tmp_path / "trtexec"
    fake.write_text("#!/bin/sh\necho hi\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path), prepend=":")
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"fake")
    ns = argparse.Namespace(onnx=str(onnx), size=[384], batch=8, fp16=True,
                            int8=False, calib=None, workspace=2048,
                            output=None, trtexec="trtexec")
    monkeypatch.setattr(export_trt, "onnx_is_dynamic", lambda p: True)
    monkeypatch.setattr(export_trt, "_workspace_flag", lambda t, w: f"--workspace={w}")
    cmd, engine = export_trt.build_command(ns)
    assert cmd[0].endswith("trtexec")
    assert "--fp16" in cmd
    assert "--optShapes=input:8x3x384x384" in cmd
    assert engine == str(tmp_path / "m.engine")


def test_export_trt_static_onnx_skips_profiles(tmp_path, monkeypatch):
    import argparse

    from iwpod.cli import export_trt
    fake = tmp_path / "trtexec"
    fake.write_text("#!/bin/sh\necho hi\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path), prepend=":")
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"fake")
    ns = argparse.Namespace(onnx=str(onnx), size=[416], batch=8, fp16=True,
                            int8=False, calib=None, workspace=2048,
                            output=None, trtexec="trtexec")
    monkeypatch.setattr(export_trt, "onnx_is_dynamic", lambda p: False)
    monkeypatch.setattr(export_trt, "_workspace_flag", lambda t, w: f"--workspace={w}")
    cmd, _engine = export_trt.build_command(ns)
    assert not any(a.startswith("--minShapes=") for a in cmd)
    assert not any(a.startswith("--optShapes=") for a in cmd)


def test_export_trt_int8_needs_calib(tmp_path, monkeypatch):
    import argparse

    from iwpod.cli import export_trt
    fake = tmp_path / "trtexec"
    fake.write_text("#!/bin/sh\necho hi\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path), prepend=":")
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"fake")
    ns = argparse.Namespace(onnx=str(onnx), size=[320], batch=4, fp16=False,
                            int8=True, calib=None, workspace=1024,
                            output=None, trtexec="trtexec")
    try:
        export_trt.build_command(ns)
    except RuntimeError as e:
        assert "--calib" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
