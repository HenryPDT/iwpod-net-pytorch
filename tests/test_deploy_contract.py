"""Deploy contract encoded as tests (live gates stay in DOWNSTREAM_GUIDE §5)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "deepstream" / "config_infer_secondary_iwpod.txt"
GUIDE = ROOT / "deepstream" / "DOWNSTREAM_GUIDE.md"


def test_config_onnx_matches_product_recipe():
    txt = CFG.read_text()
    assert "iwpod_416.onnx" in txt
    assert "iwpodv2_416_w_passthrough.onnx" not in txt
    assert "network-mode=2" in txt
    assert "maintain-aspect-ratio=1" in txt
    assert "[user-configs]" not in txt
    assert "iwpod-threshold" not in txt
    assert "\nsymmetric-padding=" not in txt


def test_guide_documents_letterbox_not_passthrough():
    g = GUIDE.read_text()
    assert "iwpod_416.onnx" in g
    assert "NvBufSurface" in g
    assert "Do **not** pass `--with-passthrough`" in g
    assert "kMinConfidence" in g or "compile-time" in g
    assert "Xavier NX" in g
    assert "lpr.lpd_model" in g


def test_python_decode_uses_aabb_and_get_perspective_transform():
    src = (ROOT / "iwpod" / "decode.py").read_text()
    assert "getPerspectiveTransform" in src
    assert "pts.min(axis=1)" in src
    assert "findHomography" not in src
