"""Loss + checkpoint tests (small tensors, CPU)."""
import torch

from iwpod.ckpt import build_model_for_ckpt, save_ckpt
from iwpod.loss import iwpodnet_loss, iwpodnet_loss_v2


def test_legacy_loss_finite():
    yt = torch.zeros(1, 9, 13, 13)
    yt[:, 0, 6, 6] = 1.0
    yp = torch.rand(1, 7, 13, 13) * 0.1 + 0.45
    assert torch.isfinite(iwpodnet_loss(yt, yp).mean())


def test_v2_loss_finite_all_negative():
    yt = torch.zeros(2, 9, 16, 16)  # pure background batch must not NaN
    yp = torch.randn(2, 7, 16, 16) * 0.5
    result = iwpodnet_loss_v2(yt, yp)
    assert result.total.shape == (2,)
    assert torch.isfinite(result.total).all()
    for comp in (result.cls, result.dice, result.loc):
        assert torch.isfinite(comp)


def test_v2_loss_result_matches_total_mean():
    # LossResult refactor must be numerically identical to the old scalar API.
    torch.manual_seed(0)
    yt = torch.zeros(2, 9, 16, 16)
    yt[:, 0, 4:8, 4:8] = 1.0
    yp = torch.randn(2, 7, 16, 16)
    result = iwpodnet_loss_v2(yt, yp, w_cls=1.0, w_dice=0.5, w_loc=1.0)
    expected = 1.0 * result.cls + 0.5 * result.dice + 1.0 * result.loc
    assert torch.isclose(result.total.mean(), expected, atol=1e-6)


def test_ckpt_roundtrip_and_legacy_load(tmp_path):
    # save_ckpt stamps raw_logits; build_model_for_ckpt loads any ckpt
    # (new or legacy sigmoid, whose linear weights are identical).
    from iwpod.model import IWPODNet
    m = IWPODNet(raw_logits=True)
    p = str(tmp_path / "m.pth")
    save_ckpt(p, m, epoch=3, best=1.5)
    m2 = build_model_for_ckpt(p, raw_logits=True, device="cpu")
    assert m2.end_block.raw_logits is True
    for a, b in zip(m.state_dict().values(), m2.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    legacy = str(tmp_path / "legacy.pth")
    torch.save({"model_state_dict": m.state_dict()}, legacy)
    build_model_for_ckpt(legacy, raw_logits=False, device="cpu")
