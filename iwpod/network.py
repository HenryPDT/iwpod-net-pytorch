import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, inner_layers=1):
        super(ResBlock, self).__init__()
        layers = []
        for _ in range(inner_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, filter_size, padding=filter_size // 2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, filter_size, padding=filter_size // 2))
        layers.append(nn.BatchNorm2d(out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return F.relu(x + self.layers(x), inplace=True)


class ConvBatch(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, activation='relu', padding='same', stride=(1, 1)):
        super(ConvBatch, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, filter_size, padding=filter_size // 2, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation == 'relu':
            return F.relu(x, inplace=True)
        else:
            return x

    def fuse(self):
        """Fold BN into conv for TensorRT/export latency. Returns fused nn.Conv2d."""
        conv, bn = self.conv, self.bn
        w = conv.weight
        mean, var = bn.running_mean, bn.running_var
        eps = bn.eps
        gamma = bn.weight if bn.weight is not None else torch.ones_like(mean)
        beta = bn.bias if bn.bias is not None else torch.zeros_like(mean)
        std = torch.sqrt(var + eps)
        w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)
        b_fused = beta + ((conv.bias if conv.bias is not None else 0) - mean) * gamma / std
        fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                          stride=conv.stride, padding=conv.padding, bias=True)
        fused.weight.data.copy_(w_fused)
        fused.bias.data.copy_(b_fused)
        return fused


def fuse_conv_bn_tree(module):
    """Replace every ConvBatch child with a fused Conv2d (+ ReLU if needed)."""
    for name, child in list(module.named_children()):
        if isinstance(child, ConvBatch):
            fused = child.fuse()
            if child.activation == "relu":
                setattr(module, name, nn.Sequential(fused, nn.ReLU(inplace=True)))
            else:
                setattr(module, name, fused)
        else:
            fuse_conv_bn_tree(child)
    return module
