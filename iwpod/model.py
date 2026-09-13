import torch
import torch.nn as nn

from .constants import NET_STRIDE, SIDE
from .network import ConvBatch, DWConvBatch, RepConvBlock, ResBlock, SimAM

#: Architecture versions stamped into checkpoints (ckpt.py) and checked on
#: export. v2 = original single-stride-16 dense net. v3-s16 = Rep/DW/SimAM
#: upgrades, same single output.
ARCH_V2 = "v2"
ARCH_V3_S16 = "v3-s16"


def _conv(in_ch, out_ch, backbone="orig"):
    if backbone == "rep":
        return RepConvBlock(in_ch, out_ch)
    return ConvBatch(in_ch, out_ch, 3)


class EndBlockIWPODNet(nn.Module):
    """Decoupled heads. `raw_logits=True` exports objectness WITHOUT sigmoid
    (recommended v2: sigmoid runs in DeepStream parser, threshold tunable,
    INT8/FP16-friendly). Legacy checkpoints used sigmoid-in-graph."""

    def __init__(self, in_channels, raw_logits=False, head="orig", use_simam=False):
        super(EndBlockIWPODNet, self).__init__()
        self.raw_logits = raw_logits
        self.head_kind = head
        Conv = DWConvBatch if head == "dw" else ConvBatch
        self.prob_conv1 = Conv(in_channels, 64, 3) if head == "orig" else Conv(in_channels, 64)
        self.prob_conv2 = Conv(64, 32, 3, activation='linear') if head == "orig" else Conv(64, 32, activation='linear')
        self.prob_conv3 = nn.Conv2d(32, 1, 3, padding=1)
        self.bbox_conv1 = Conv(in_channels, 64, 3) if head == "orig" else Conv(in_channels, 64)
        self.bbox_conv2 = Conv(64, 32, 3, activation='linear') if head == "orig" else Conv(64, 32, activation='linear')
        self.bbox_conv3 = nn.Conv2d(32, 6, 3, padding=1)
        self.simam = SimAM() if use_simam else None

    def forward(self, x):
        if self.simam is not None:
            x = self.simam(x)
        x_probs = self.prob_conv1(x)
        x_probs = self.prob_conv2(x_probs)
        x_probs = self.prob_conv3(x_probs)
        if not self.raw_logits:
            x_probs = torch.sigmoid(x_probs)
        x_bbox = self.bbox_conv1(x)
        x_bbox = self.bbox_conv2(x_bbox)
        x_bbox = self.bbox_conv3(x_bbox)
        return torch.cat((x_probs, x_bbox), 1)


class IWPODNet(nn.Module):
    STRIDE = NET_STRIDE
    SIDE = SIDE

    def __init__(self, raw_logits=False, backbone="orig", head="orig",
                 use_simam=False, arch_version=None):
        super(IWPODNet, self).__init__()
        self.raw_logits = raw_logits
        self.backbone = backbone
        self.head_kind = head
        self.use_simam = use_simam
        self.arch_version = arch_version or (
            ARCH_V3_S16 if (backbone != "orig" or head != "orig" or use_simam)
            else ARCH_V2)
        self.conv1 = _conv(3, 16, backbone)
        self.conv2 = _conv(16, 16, backbone)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = _conv(16, 32, backbone)
        self.res1 = ResBlock(32, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = _conv(32, 64, backbone)
        self.res2 = ResBlock(64, 64)
        self.res3 = ResBlock(64, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv5 = _conv(64, 64, backbone)
        self.res4 = ResBlock(64, 64)
        self.res5 = ResBlock(64, 64)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv6 = _conv(64, 128, backbone)
        self.res6 = ResBlock(128, 128)
        self.res7 = ResBlock(128, 128)
        self.res8 = ResBlock(128, 128)
        self.res9 = ResBlock(128, 128)
        self.end_block = EndBlockIWPODNet(128, raw_logits=raw_logits,
                                          head=head, use_simam=use_simam)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.res1(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool3(x)
        x = self.conv5(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.pool4(x)
        x = self.conv6(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        return self.end_block(x)

    def switch_to_deploy(self):
        """Fuse Rep branches (reparam) in-place. Idempotent, parity-checked."""
        from .network import RepConvBlock
        for m in self.modules():
            if isinstance(m, RepConvBlock):
                m.switch_to_deploy()
        return self

    def fuse(self):
        """Fold every Conv-BN (incl. deployed Rep) into a single Conv2d. In-place."""
        self.switch_to_deploy()
        from .network import fuse_conv_bn_tree
        fuse_conv_bn_tree(self)
        return self


# Backwards-compat alias (old WPOD 8-ch compat head builder lives in export script).
IWPODNetV2 = IWPODNet
