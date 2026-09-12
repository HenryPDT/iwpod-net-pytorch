"""IWPOD-NET v2: quad license-plate detection (PyTorch, DeepStream-ready)."""

from iwpod.constants import ANCHOR_HALF, DEPLOY_SIZE, NET_STRIDE, SIDE
from iwpod.model import IWPODNet

__all__ = ["IWPODNet", "NET_STRIDE", "SIDE", "ANCHOR_HALF", "DEPLOY_SIZE"]
