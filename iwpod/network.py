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
        elif isinstance(child, RepConvBlock):
            child.switch_to_deploy()
            fuse_conv_bn_tree(child)
        else:
            fuse_conv_bn_tree(child)
    return module


def _fuse_bn(conv_w, conv_b, bn):
    """Fold BN into (weight, bias). Returns (w_fused, b_fused) tensors."""
    mean, var = bn.running_mean, bn.running_var
    eps = bn.eps
    gamma = bn.weight if bn.weight is not None else torch.ones_like(mean)
    beta = bn.bias if bn.bias is not None else torch.zeros_like(mean)
    std = torch.sqrt(var + eps)
    w_fused = conv_w * (gamma / std).reshape(-1, 1, 1, 1)
    if conv_b is None:
        conv_b = torch.zeros_like(mean)
    b_fused = beta + (conv_b - mean) * gamma / std
    return w_fused, b_fused


class RepConvBlock(nn.Module):
    """RepVGG-style reparam block (TRT8.5/10 + OpenVINO safe).

    Train: 3x3 conv-BN + 1x1 conv-BN (+ identity-BN when in==out, stride 1).
    Infer: single 3x3 conv (+ ReLU). Call switch_to_deploy() before export;
    output is bit-identical up to fp32 rounding (verified by test_rep_parity).
    """

    def __init__(self, in_channels, out_channels, stride=1, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.activation = activation
        self.deploy = False
        pad = 1
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, stride=self.stride,
                                 padding=pad, bias=False)
        self.bn3x3 = nn.BatchNorm2d(out_channels)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=self.stride,
                                 padding=0, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)
        self.has_id = (in_channels == out_channels and self.stride == (1, 1))
        if self.has_id:
            self.bn_id = nn.BatchNorm2d(out_channels)
        else:
            self.bn_id = None
        # Deploy-time single branch (populated by switch_to_deploy).
        self.reparam_conv = None

    def forward(self, x):
        if self.deploy and self.reparam_conv is not None:
            y = self.reparam_conv(x)
            return F.relu(y, inplace=True) if self.activation == "relu" else y
        y3 = self.bn3x3(self.conv3x3(x))
        y1 = self.bn1x1(self.conv1x1(x))
        y = y3 + y1
        if self.has_id:
            y = y + self.bn_id(x)
        if self.activation == "relu":
            return F.relu(y, inplace=True)
        return y

    @torch.no_grad()
    def _fuse_branch(self, conv, bn, k3):
        w, b = _fuse_bn(conv.weight, None, bn)
        if conv.kernel_size == (1, 1):
            w = F.pad(w, [1, 1, 1, 1])
        elif conv.kernel_size != (3, 3):  # pragma: no cover
            raise RuntimeError(f"RepConvBlock supports 3x3/1x1 only, got {conv.kernel_size}")
        return w, b

    @torch.no_grad()
    def switch_to_deploy(self):
        """Fuse branches into a single 3x3 conv. Idempotent.

        Drops the training branches afterwards: the converted block is
        infer/export-only (do not save its state_dict for training resume;
        export loads weights BEFORE calling fuse()).
        """
        if self.deploy and self.reparam_conv is not None:
            return
        w3, b3 = self._fuse_branch(self.conv3x3, self.bn3x3, True)
        w1, b1 = self._fuse_branch(self.conv1x1, self.bn1x1, True)
        w, b = w3 + w1, b3 + b1
        if self.has_id:
            n, c = self.out_channels, self.in_channels
            w_id = torch.zeros((n, c, 3, 3), dtype=w.dtype, device=w.device)
            for i in range(n):
                w_id[i, i, 1, 1] = 1.0
            gamma = self.bn_id.weight if self.bn_id.weight is not None else torch.ones(n, device=w.device)
            beta = self.bn_id.bias if self.bn_id.bias is not None else torch.zeros(n, device=w.device)
            std = torch.sqrt(self.bn_id.running_var + self.bn_id.eps)
            w_id = w_id * (gamma / std).reshape(-1, 1, 1, 1)
            b_id = beta - self.bn_id.running_mean * gamma / std
            w, b = w + w_id.to(w.dtype), b + b_id.to(b.dtype)
        fused = nn.Conv2d(self.in_channels, self.out_channels, 3, stride=self.stride,
                          padding=1, bias=True).to(w.device, w.dtype)
        fused.weight.data.copy_(w)
        fused.bias.data.copy_(b)
        self.reparam_conv = fused
        self.deploy = True
        # Drop training branches so param counts / state_dicts reflect
        # the deployed graph (load training checkpoints BEFORE converting).
        for attr in ("conv3x3", "bn3x3", "conv1x1", "bn1x1", "bn_id"):
            if getattr(self, attr, None) is not None:
                delattr(self, attr)


class SimAM(nn.Module):
    """Parameter-free attention (mean/var + sigmoid). TRT/OpenVINO native."""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        # Trace-safe: spatial mean (biased var) instead of sum/(H*W-1);
        # identical numerics within 1/N and no data-dependent python scalars.
        d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        v = d.mean(dim=(2, 3), keepdim=True)
        e = d / (4 * (v + self.e_lambda)) + 0.5
        return x * torch.sigmoid(e)


class DWConvBatch(nn.Module):
    """Depthwise-separable 3x3 (DW + PW + BN). Keeps ReLU/BN-only infer ops."""

    def __init__(self, in_channels, out_channels, activation="relu", stride=(1, 1)):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=stride,
                            groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        if self.activation == "relu":
            return F.relu(x, inplace=True)
        return x
