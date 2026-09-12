"""Checkpoint helpers (single dense architecture).

``raw_logits`` is weight-identical either way (sigmoid lives outside the linear
layer), so checkpoints stamp it for documentation and every loader builds the
same model. There is only one architecture, so mismatch is impossible.
``arch`` also stamps the label-encoding contract (stride/side) plus the train
resolution (dim) so export/DeepStream mismatches fail loud instead of
silently shifting quads.
"""
import os

import torch

from iwpod.constants import NET_STRIDE, SIDE


def load_ckpt(path, map_location="cpu"):
    """Load a full checkpoint dict once (weights + resume payload)."""
    return torch.load(path, map_location=map_location, weights_only=False)


def weights_and_arch(ck):
    """Split a loaded checkpoint into (state_dict, arch meta)."""
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    meta = ck.get("arch", {}) if isinstance(ck, dict) else {}
    return sd, meta


def load_ckpt_weights(path, map_location="cpu"):
    # Local training checkpoints (trusted); weights_only=False restores the
    # full payload (optimizer/scaler/RNG) on torch>=2.6.
    ck = load_ckpt(path, map_location=map_location)
    return weights_and_arch(ck)


def build_model_for_ckpt(ckpt_path, raw_logits=True, device="cpu"):
    from iwpod.model import IWPODNet

    model = IWPODNet(raw_logits=raw_logits).to(device).eval()
    sd, _ = load_ckpt_weights(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    return model


def save_ckpt(path, model, optimizer=None, epoch=-1, best=None, extra=None, meta=None):
    arch = {
        "raw_logits": bool(getattr(getattr(model, "end_block", None), "raw_logits", True)),
        "stride": NET_STRIDE,
        "side": SIDE,
    }
    if meta:
        arch.update(meta)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "arch": arch,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if best is not None:
        payload["best"] = best
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)
