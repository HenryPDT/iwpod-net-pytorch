"""Deploy contract encoded as tests (Xavier NX live gates stay in DOWNSTREAM_GUIDE §7)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "deepstream" / "config_infer_secondary_iwpod.txt"
GUIDE = ROOT / "deepstream" / "DOWNSTREAM_GUIDE.md"


def test_config_onnx_matches_phase1_recipe():
    txt = CFG.read_text()
    assert "iwpodv2_416_w_passthrough.onnx" in txt
    assert "network-mode=0" in txt
    assert "maintain-aspect-ratio=1" in txt
    assert "[user-configs]" not in txt
    assert "iwpod-threshold" not in txt
    assert "\nsymmetric-padding=" not in txt


def test_guide_documents_nchw_passthrough_and_fp32():
    g = GUIDE.read_text()
    assert "pass_through_output [B,3,H,W]" in g or "NCHW planar" in g
    assert "network-mode=0" in g
    assert "kMinConfidence" in g or "compile-time" in g
    assert "Xavier NX" in g


def test_python_decode_uses_aabb_and_get_perspective_transform():
    src = (ROOT / "iwpod" / "decode.py").read_text()
    assert "getPerspectiveTransform" in src
    assert "pts.min(axis=1)" in src
    assert "findHomography" not in src
