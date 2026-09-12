"""ONNX export smoke (skips if onnx/ort missing)."""
import pytest
import torch

pytest.importorskip("onnx")


def test_parse_size_default_416():
    from iwpod.cli.export import parse_size
    from iwpod.constants import DEPLOY_SIZE
    assert parse_size(None) == (DEPLOY_SIZE, DEPLOY_SIZE)
    assert parse_size([256, 320]) == (256, 320)


def test_export_static_and_passthrough(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    from iwpod.ckpt import save_ckpt
    from iwpod.cli import export as export_cli
    from iwpod.model import IWPODNet

    m = IWPODNet(raw_logits=True)
    w = tmp_path / "m.pth"
    save_ckpt(str(w), m, epoch=0, meta={"dim": 256})
    out = tmp_path / "m.onnx"
    ns = type("N", (), dict(
        weights=str(w), size=[256], dynamic=False, dynamic_shape=False, batch=1,
        align=32, opset=17, simplify=False, check_parity=True, fuse=False,
        output=str(out), with_sigmoid=False, with_passthrough=True,
    ))()
    export_cli.run(ns)
    assert out.is_file()
    sess = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    assert names[0] == "lpd_pred" and "pass_through_output" in names
    x = torch.rand(1, 3, 256, 256).numpy()
    pred, pass_ = sess.run(None, {"input": x})
    assert pred.shape[1] == 7
    assert pass_.shape == (1, 3, 256, 256)


def test_export_dynamic_batch2(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    from iwpod.ckpt import save_ckpt
    from iwpod.cli import export as export_cli
    from iwpod.model import IWPODNet

    m = IWPODNet(raw_logits=True)
    w = tmp_path / "m.pth"
    save_ckpt(str(w), m, epoch=0, meta={"dim": 64})
    out = tmp_path / "m.onnx"
    ns = type("N", (), dict(
        weights=str(w), size=[64], dynamic=True, dynamic_shape=False, batch=1,
        align=32, opset=17, simplify=False, check_parity=True, fuse=False,
        output=str(out), with_sigmoid=False, with_passthrough=False,
    ))()
    export_cli.run(ns)
    sess = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.rand(2, 3, 64, 64).numpy()
    pred = sess.run(["lpd_pred"], {"input": x})[0]
    assert pred.shape[0] == 2 and pred.shape[1] == 7


def test_fuse_runs_without_changing_eval_path(tmp_path):
    from iwpod.model import IWPODNet
    m = IWPODNet(raw_logits=True).eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        a = m(x)
    m.fuse()
    with torch.no_grad():
        b = m(x)
    assert torch.allclose(a, b, atol=1e-4)
