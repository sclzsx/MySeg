from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


class DeepLabV3Plus(nn.Module):
    """
    DeepLab v3+: Dilated ResNet with multi-grid + improved ASPP + decoder
    """

    def __init__(self, num_classes, divisor=1, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=16):
        super(DeepLabV3Plus, self).__init__()

        # Stride and dilation
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # Encoder
        ch = [64 * 2 ** p for p in range(6)]
        self.layer1 = _Stem(ch[0]//divisor)
        self.layer2 = _ResLayer(n_blocks[0], ch[0]//divisor, ch[2]//divisor, s[0], d[0])
        self.layer3 = _ResLayer(n_blocks[1], ch[2]//divisor, ch[3]//divisor, s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3]//divisor, ch[4]//divisor, s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[4]//divisor, ch[5]//divisor, s[3], d[3], multi_grids)
        self.aspp = _ASPP(ch[5]//divisor, 256//divisor, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch//divisor, 256//divisor, 1, 1, 0, 1))

        # Decoder
        self.reduce = _ConvBnReLU(256//divisor, 48//divisor, 1, 1, 0, 1)
        self.fc2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", _ConvBnReLU(304//divisor, 256//divisor, 3, 1, 1, 1)),
                    ("conv2", _ConvBnReLU(256//divisor, 256//divisor, 3, 1, 1, 1)),
                    ("conv3", nn.Conv2d(256//divisor, num_classes, kernel_size=1)),
                ]
            )
        )

    def forward(self, x):#224
        # print(x.shape)
        h = self.layer1(x)#57
        # print(h.shape)
        h = self.layer2(h)#57
        # print(h.shape)
        h_ = self.reduce(h)#57
        # print(h_.shape)
        h = self.layer3(h)#29
        # print(h.shape)
        h = self.layer4(h)#15
        # print(h.shape)
        h = self.layer5(h)#15
        # print(h.shape)
        h = self.aspp(h)#15
        # print(h.shape)
        h = self.fc1(h)
        h = F.interpolate(h, size=h_.shape[2:], mode="bilinear", align_corners=False)
        # print(h.shape, h_.shape)
        h = torch.cat((h, h_), dim=1)
        h = self.fc2(h)
        # print(h.shape)
        h = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)
        # print(h.shape)
        return h

try:
    from encoding.nn import SyncBatchNorm

    _BATCH_NORM = SyncBatchNorm
except:
    _BATCH_NORM = nn.BatchNorm2d

_BOTTLENECK_EXPANSION = 4


class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else lambda x: x  # identity
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(3, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))


if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    divisor = 1
    # h = 1080 // 2 // 16 * 16
    # w = 1920 // 2 // 16 * 16
    h = 224
    w = 224

    net = DeepLabV3Plus(num_classes=2, divisor=divisor).cuda()
    image = (3, h, w)
    f, p = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f, p)
    out = net(torch.randn(1, 3, h, w).cuda())
    print(out.shape)
