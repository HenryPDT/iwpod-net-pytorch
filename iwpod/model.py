import torch
import torch.nn as nn

from .constants import NET_STRIDE, SIDE
from .network import ConvBatch, ResBlock


class EndBlockIWPODNet(nn.Module):
    """Decoupled heads. `raw_logits=True` exports objectness WITHOUT sigmoid
    (recommended v2: sigmoid runs in DeepStream parser, threshold tunable,
    INT8/FP16-friendly). Legacy checkpoints used sigmoid-in-graph."""

    def __init__(self, in_channels, raw_logits=False):
        super(EndBlockIWPODNet, self).__init__()
        self.raw_logits = raw_logits
        self.prob_conv1 = ConvBatch(in_channels, 64, 3)
        self.prob_conv2 = ConvBatch(64, 32, 3, activation='linear')
        self.prob_conv3 = nn.Conv2d(32, 1, 3, padding=1)
        self.bbox_conv1 = ConvBatch(in_channels, 64, 3)
        self.bbox_conv2 = ConvBatch(64, 32, 3, activation='linear')
        self.bbox_conv3 = nn.Conv2d(32, 6, 3, padding=1)

    def forward(self, x):
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

    def __init__(self, raw_logits=False):
        super(IWPODNet, self).__init__()
        self.raw_logits = raw_logits
        self.conv1 = ConvBatch(3, 16, 3)
        self.conv2 = ConvBatch(16, 16, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBatch(16, 32, 3)
        self.res1 = ResBlock(32, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = ConvBatch(32, 64, 3)
        self.res2 = ResBlock(64, 64)
        self.res3 = ResBlock(64, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBatch(64, 64, 3)
        self.res4 = ResBlock(64, 64)
        self.res5 = ResBlock(64, 64)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv6 = ConvBatch(64, 128, 3)
        self.res6 = ResBlock(128, 128)
        self.res7 = ResBlock(128, 128)
        self.res8 = ResBlock(128, 128)
        self.res9 = ResBlock(128, 128)
        self.end_block = EndBlockIWPODNet(128, raw_logits=raw_logits)

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
        x = self.end_block(x)
        return x

    def fuse(self):
        """Fold every Conv-BN into a single Conv2d (export/TRT latency). In-place."""
        from .network import fuse_conv_bn_tree
        fuse_conv_bn_tree(self)
        return self


# Backwards-compat alias (old WPOD 8-ch compat head builder lives in export script).
IWPODNetV2 = IWPODNet
