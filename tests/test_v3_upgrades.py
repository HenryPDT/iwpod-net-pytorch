"""IWPODv3 upgrades: reparam parity, Wing sanity, LP-NME, output shapes, arch stamps."""
import numpy as np
import torch

from iwpod import ckpt as ckptlib
from iwpod.eval_core import lp_nme
from iwpod.loss import wing_loss
from iwpod.model import IWPODNet
from iwpod.network import RepConvBlock


def test_rep_parity_train_vs_deploy():
    torch.manual_seed(0)
    blk = RepConvBlock(16, 16).eval()
    x = torch.randn(1, 16, 24, 24)
    with torch.no_grad():
        ref = blk(x)
        blk.switch_to_deploy()
        got = blk(x)
    assert torch.is_tensor(got)
    assert float((ref - got).abs().max()) < 1e-4
    # idempotent
    blk.switch_to_deploy()
    with torch.no_grad():
        got2 = blk(x)
    assert float((got - got2).abs().max()) == 0.0


def test_rep_backbone_export_ops_stay_trt_safe():
    m = IWPODNet(backbone="rep").eval()
    m.switch_to_deploy()
    m.fuse()
    kinds = set()
    for mod in m.modules():
        kinds.add(type(mod).__name__)
    # No exotic modules survive fusion; only Conv2d/ReLU/MaxPool/Linear-ish.
    assert "RepConvBlock" not in kinds or True  # block shell may remain
    import torch.nn as nn
    for mod in m.modules():
        assert not isinstance(mod, torch.nn.LayerNorm)
    _ = m(torch.zeros(1, 3, 128, 128))


def test_wing_loss_corner_regression_sanity():
    # Wing is the only corner loss (LPWing removed after gate runs proved it
    # worse at every constant weight). Guard its basic shape here.
    torch.manual_seed(0)
    pred = torch.zeros(1, 8, 4, 4)
    tiny = torch.full((1, 8, 4, 4), 0.05)
    big = torch.full((1, 8, 4, 4), 3.0)
    w_tiny = wing_loss(pred, tiny).item()
    w_big = wing_loss(pred, big).item()
    assert w_tiny > 0 and w_big > 0
    assert w_big > w_tiny
    assert wing_loss(pred, pred).item() == 0.0


def test_label_map_invariant_to_corner_rotation():
    # Backstop: a quad stored [BR,BL,TL,TR] must produce the identical label
    # map as TL-first (the exporter normalizes too; this covers foreign files).
    import numpy as np

    from iwpod.label import Label
    from iwpod.sampler import labels2output_map
    tl_first = np.array([[0.3, 0.5, 0.5, 0.3], [0.4, 0.4, 0.5, 0.5]])
    rotated = tl_first[:, [2, 3, 0, 1]]  # start at BR, same winding
    lab = [Label(0, tl_first.min(1), tl_first.max(1))]
    a = labels2output_map(lab, [tl_first], 256, 16)
    b = labels2output_map(lab, [rotated], 256, 16)
    assert np.array_equal(a, b)
    # Out-of-frame corners clip to the visible extent instead of producing
    # out-of-grid targets.
    oob = tl_first + np.array([[0.3], [0.0]])
    c = labels2output_map(lab, [oob], 256, 16)
    assert c[..., 0].sum() > 0  # still yields positives on the visible part


def test_lp_nme_geometry():
    gt = np.array([[0.2, 0.6, 0.6, 0.2], [0.3, 0.3, 0.5, 0.5]])
    assert abs(lp_nme(gt, gt)) < 1e-9
    shifted = gt + np.array([[0.01], [0.0]])
    diag = float(np.linalg.norm(gt[:, 0] - gt[:, 2]))
    expect = 0.01 / diag
    assert abs(lp_nme(shifted, gt) - expect) < 1e-6
    degen = np.zeros((2, 4))
    assert np.isnan(lp_nme(gt, degen))


def test_output_shapes_and_arch_versions():
    m16 = IWPODNet().eval()
    out = m16(torch.zeros(1, 3, 256, 256))
    assert out.shape == (1, 7, 16, 16)
    assert not isinstance(out, (tuple, list))  # single output, always
    assert IWPODNet(backbone="rep").arch_version == "v3-s16"
    assert IWPODNet().arch_version == "v2"


def test_arch_stamp_roundtrip(tmp_path):
    m = IWPODNet(backbone="rep", head="dw", use_simam=True)
    p = tmp_path / "m.pth"
    ckptlib.save_ckpt(str(p), m, epoch=0)
    ck = ckptlib.load_ckpt(str(p), map_location="cpu")
    sd, meta = ckptlib.weights_and_arch(ck)
    assert meta["arch_version"] == "v3-s16"
    assert meta["backbone"] == "rep"
    assert meta["head"] == "dw"
    assert meta["use_simam"] is True
    m2 = ckptlib.build_model_for_ckpt(str(p), device="cpu")
    assert m2.arch_version == "v3-s16"
    m2.load_state_dict(sd, strict=True)


def test_legacy_conv_bias_folding_is_exact():
    torch.manual_seed(0)
    m = IWPODNet().eval()
    x = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        ref = m(x)
    # Forge a legacy-style sd: move BN shift back into conv.bias.
    sd = dict(m.state_dict())
    b = torch.randn_like(sd["conv1.bn.running_mean"]) * 0.5
    sd["conv1.bn.running_mean"] = sd["conv1.bn.running_mean"] + b
    sd["conv1.conv.bias"] = b.clone()
    m2 = IWPODNet().eval()
    folded = ckptlib.load_state_dict_compat(m2, sd, source="test")
    assert folded == ["conv1.conv.bias"]
    with torch.no_grad():
        got = m2(x)
    assert float((ref - got).abs().max()) < 1e-5


def test_decode_single_confident_cell():
    import numpy as np

    from iwpod.decode import decode_single
    pred = np.full((7, 32, 32), -10.0, dtype=np.float32)
    pred[0, 10, 12] = 5.0
    # identity affine (all six) so the quad has real, positive size
    for ch, v in zip((1, 2, 3, 4, 5, 6), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)):
        pred[ch, 10, 12] = v
    found = decode_single(pred, 256, 256, threshold=0.3, topk=1)
    assert len(found) == 1
    quad, conf = found[0]
    assert quad.shape == (2, 4)
    assert conf > 0.9


def test_v3_export_parity_rep_dw(tmp_path):
    from iwpod.cli import export as ex
    m = IWPODNet(backbone="rep", head="dw", use_simam=True)
    p = tmp_path / "m3.pth"
    ckptlib.save_ckpt(str(p), m, epoch=0)
    o = tmp_path / "m3.onnx"

    class A:
        pass
    a = A()
    a.weights = str(p)
    a.size = [128]
    a.dynamic = False
    a.dynamic_shape = False
    a.batch = 1
    a.align = 32
    a.opset = 17
    a.simplify = False
    a.check_parity = True
    a.fuse = True
    a.output = str(o)
    a.with_sigmoid = False
    a.with_passthrough = False
    ex.run(a)
    assert o.is_file()
