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


def load_state_dict_compat(model, sd, *, source="checkpoint"):
    """Load weights, tolerating legacy TF-ported `*.conv.bias` keys.

    Legacy checkpoints (e.g. the 10k-epoch weights) store a bias for every
    `ConvBatch.conv`, but the model defines those convs with `bias=False`.
    This is numerically exact: a per-channel constant bias `b` only shifts
    BN's input, so folding `running_mean -= b` reproduces the legacy
    forward bit-for-bit (variance is shift-invariant). Anything else still
    fails strict so real mismatches stay loud.
    Returns the list of folded keys.
    """
    model_keys = set(model.state_dict().keys())
    folded = [k for k in sd
              if k.endswith(".conv.bias") and k not in model_keys]
    if folded:
        sd = dict(sd)
        for k in folded:
            bn_mean_key = k[: -len("conv.bias")] + "bn.running_mean"
            if bn_mean_key not in sd:
                raise RuntimeError(
                    f"{source}: legacy key {k} has no {bn_mean_key} to fold into")
            sd[bn_mean_key] = sd[bn_mean_key] - sd[k]
            del sd[k]
        try:
            from loguru import logger as _logger
            _logger.warning(f"{source}: folded {len(folded)} legacy conv.bias keys "
                            f"into bn.running_mean (numerically exact)")
        except ImportError:
            pass
    model.load_state_dict(sd, strict=True)
    return folded


def build_model_for_ckpt(ckpt_path, raw_logits=True, device="cpu"):
    from iwpod.model import IWPODNet

    ck = load_ckpt(ckpt_path, map_location=device)
    sd, meta = weights_and_arch(ck)
    kwargs = {}
    if meta.get("backbone"):
        kwargs["backbone"] = meta["backbone"]
    if meta.get("head"):
        kwargs["head"] = meta["head"]
    if meta.get("use_simam") is not None:
        kwargs["use_simam"] = meta["use_simam"]
    if meta.get("arch_version"):
        kwargs["arch_version"] = meta["arch_version"]
    model = IWPODNet(raw_logits=raw_logits, **kwargs).to(device).eval()
    load_state_dict_compat(model, sd, source=ckpt_path)
    return model


def save_ckpt(path, model, optimizer=None, epoch=-1, best=None, extra=None, meta=None):
    arch = {
        "raw_logits": bool(getattr(getattr(model, "end_block", None), "raw_logits", True)),
        "stride": NET_STRIDE,
        "side": SIDE,
        "arch_version": getattr(model, "arch_version", "v2"),
        "backbone": getattr(model, "backbone", "orig"),
        "head": getattr(model, "head_kind", "orig"),
        "use_simam": bool(getattr(model, "use_simam", False)),
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
