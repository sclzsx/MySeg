'''MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from Nets.attentions import SpatialAttention, ChannelAttention


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        # print(out.shape)
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=2, attention=False, part=True):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        self.bneck1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
        )
        self.ca4 = ChannelAttention(24, ratio=2)
        self.sa4 = SpatialAttention()
        self.bneck2 = nn.Sequential(
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
        )
        self.ca8 = ChannelAttention(40, ratio=4)
        self.sa8 = SpatialAttention()
        self.bneck3 = nn.Sequential(
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
        )
        self.bneck4 = nn.Sequential(
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )
        self.ca32 = ChannelAttention(160, ratio=16)
        self.sa32 = SpatialAttention()
        self.part = part
        self.attention = attention
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()
        self.ada_pool = nn.AdaptiveAvgPool2d(1)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # print('ss', out.shape)  # 3, 6, 13, 15

        out = self.hs1(self.bn1(self.conv1(x)))
        x_1_4 = self.bneck1(out)
        if self.attention:
            x_1_4 = self.ca4(x_1_4) * x_1_4
            x_1_4 = self.sa4(x_1_4) * x_1_4

        x_1_8 = self.bneck2(x_1_4)
        if self.attention:
            x_1_8 = self.ca8(x_1_8) * x_1_8
            x_1_8 = self.sa8(x_1_8) * x_1_8

        x_1_32 = self.bneck3(x_1_8)
        x_1_32 = self.bneck4(x_1_32)
        if self.attention:
            x_1_32 = self.ca32(x_1_32) * x_1_32
            x_1_32 = self.sa32(x_1_32) * x_1_32

        if self.part:
            return x_1_4, x_1_8, x_1_32
        else:
            # print(x_1_4.shape, x_1_8.shape, x_1_32.shape)
            out = self.hs2(self.bn2(self.conv2(x_1_32)))
            out = self.ada_pool(out)
            # out = F.avg_pool2d(out, 7)
            # print(out.shape)
            out = out.view(out.size(0), -1)
            out = self.hs3(self.bn3(self.linear3(out)))
            out = self.linear4(out)
            return out


from Nets.my_deeplabv3p import ASPP, CBR
import torch.nn.functional as F


class MobileNetV3_Large_Seg(nn.Module):
    def __init__(self, num_classes, attention=False, back_bone_name='m3large'):
        super().__init__()
        if back_bone_name == 'm3large':
            self.backbone = MobileNetV3_Large(num_classes, attention=attention, part=True)
        else:
            self.backbone = MobileNetV3_Small(num_classes, attention=attention, part=True)
        self.attention = attention
        self.conv1x1_4 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=1), hswish())
        self.ASPP = ASPP(in_c=40, out_c=64)
        self.conv1x1_8 = nn.Sequential(nn.Conv2d(64 * 5, 64, kernel_size=1), hswish())
        self.conv1x1_16 = nn.Sequential(nn.Conv2d(160, 64, kernel_size=1), hswish())
        self.dconv_8 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

        self.conv1x1 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1), hswish())

        self.ca = ChannelAttention(64, ratio=16)
        self.sa = SpatialAttention()

    def forward(self, x):
        x_1_4, x_1_8, x_1_32 = self.backbone(x)
        x_1_4 = self.conv1x1_4(x_1_4)
        x_1_8 = self.conv1x1_8(self.ASPP(x_1_8))
        x_1_32 = self.conv1x1_16(x_1_32)
        H1, W1 = x_1_8.shape[2], x_1_8.shape[3]
        x_1_32 = F.interpolate(x_1_32, size=(H1, W1), mode="bilinear", align_corners=False)
        x_1_8 = x_1_8 + x_1_32

        x_1_8 = self.dconv_8(x_1_8)
        # H2, W2 = x_1_4.shape[2], x_1_4.shape[3]
        # x_1_8 = F.interpolate(x_1_8, size=(H2, W2), mode="bilinear", align_corners=False)

        out = self.conv1x1(torch.cat([x_1_4, x_1_8], dim=1))
        if self.attention:
            out = self.ca(out) * out
            out = self.sa(out) * out

        out = F.interpolate(out, size=x.shape[-2], mode="bilinear", align_corners=False)

        return out


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=2, attention=False, part=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        self.part = part
        self.attention = attention
        self.bneck1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),  # 56
        )
        self.ca4 = ChannelAttention(16, ratio=4)
        self.sa4 = SpatialAttention()
        self.bneck2 = nn.Sequential(
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),  # 28
        )
        self.ca8 = ChannelAttention(24, ratio=4)
        self.sa8 = SpatialAttention()
        self.bneck3 = nn.Sequential(
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),  # 14
        )
        self.ca16 = ChannelAttention(48, ratio=4)
        self.sa16 = SpatialAttention()
        self.bneck4 = nn.Sequential(
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),  # 7
        )
        self.ca32 = ChannelAttention(96, ratio=4)
        self.sa32 = SpatialAttention()
        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

        self.ada_pool = nn.AdaptiveAvgPool2d(1)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        x_1_4 = self.bneck1(out)
        if self.attention:
            x_1_4 = self.ca4(x_1_4) * x_1_4
            x_1_4 = self.sa4(x_1_4) * x_1_4
        x_1_8 = self.bneck2(x_1_4)
        if self.attention:
            x_1_8 = self.ca8(x_1_8) * x_1_8
            x_1_8 = self.sa8(x_1_8) * x_1_8
        x_1_16 = self.bneck3(x_1_8)
        if self.attention:
            x_1_16 = self.ca16(x_1_16) * x_1_16
            x_1_16 = self.sa16(x_1_16) * x_1_16
        x_1_32 = self.bneck4(x_1_16)
        if self.attention:
            x_1_32 = self.ca32(x_1_32) * x_1_32
            x_1_32 = self.sa32(x_1_32) * x_1_32
        if self.part:
            return x_1_4, x_1_8, x_1_16, x_1_32
        else:
            out = self.hs2(self.bn2(self.conv2(x_1_32)))
            # out = F.avg_pool2d(out, 7)
            out = self.ada_pool(out)
            out = out.view(out.size(0), -1)
            out = self.hs3(self.bn3(self.linear3(out)))
            out = self.linear4(out)
            return out


def ps(tensor):
    print(tensor.shape)


# class HR_MobileNetV3_Small(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.heada = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), hswish())
#         self.headb = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), hswish())
#         self.headc = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), hswish())
#
#         self.bneck1a = nn.Sequential(Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2))  # 56
#         self.bneck2a = nn.Sequential(Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2), Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1))  # 28
#         self.bneck3a = nn.Sequential(
#             Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
#             Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
#             Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
#             Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
#             Block(5, 48, 144, 48, hswish(), SeModule(48), 1))  # 14
#         self.bneck4a = nn.Sequential(
#             Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
#             Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
#             Block(5, 96, 576, 96, hswish(), SeModule(96), 1))  # 7
#
#         self.bneck3b = nn.Sequential(
#             Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
#             Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
#             Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
#             Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
#             Block(5, 48, 144, 48, hswish(), SeModule(48), 1))  # 14
#         self.bneck4b = nn.Sequential(
#             Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
#             Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
#             Block(5, 96, 576, 96, hswish(), SeModule(96), 1))  # 7
#
#         self.bneck4c = nn.Sequential(
#             Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
#             Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
#             Block(5, 96, 576, 96, hswish(), SeModule(96), 1))  # 7
#
#         self.init_params()
#         self.down_2 = upsample(times=0.5)
#         self.down_8 = upsample(times=0.125)
#         self.c1 = nn.Conv2d(48, 16, 1)
#         self.c2 = nn.Conv2d(96, 16, 1)
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         xa = self.heada(x)  # 16,112
#         xb = self.headb(self.down_2(x))  # 16,56
#         xc = self.headc(self.down_8(x))  # 16, 14
#         # print(xa.shape, xb.shape, xc.shape)
#
#         # bk4c = self.bneck4c(self.down_4(xc))
#         #
#         # bk3b = self.bneck3b(self.down_2(xb))
#         # bk4b = self.bneck3c(self.down_2(bk3b))
#
#         bk1a = self.bneck1a(xa)
#         bk1a = bk1a + xb # 16 56
#         # ps(bk1a)
#         bk3a = self.bneck3a(self.bneck2a(bk1a))#48 14
#         # ps(bk3a)
#         bk3a = self.c1(bk3a) + xc # 16, 14
#         # ps(bk3a)
#         bk4a = self.bneck4a(bk3a)
#         # ps(bk4a)
#         bk4a = self.c2(bk4a) + xc
#
#         # print(x1.shape, x2.shape, x4.shape)
#
#         # x4_1 = self.bneck1(x4)
#         # x4_2 = self.bneck1(x4_1)
#         #
#         # x2_1 = self.bneck1(x2)
#         # x2_2 = self.bneck1(x2_1 + x4_2)
#         # x2_3 = self.bneck1(x2_2 + x4_1)
#         #
#         # x1_1 = self.bneck1(x1)
#         # x1_2 = self.bneck1(x1_1)
#         # x1_3 = self.bneck1(x1_2 + x2_1)
#         # x1_4 = self.bneck1(x1_3 + x2_2)
#         # x1_4 = x1_4 + x2_3
#
#         return x, x, x, x


class MobileNetV3_Small_Seg(nn.Module):
    def __init__(self, num_classes, attention=False):
        super().__init__()
        self.backbone = MobileNetV3_Small(num_classes, attention=attention, part=True)

        self.conv1x1_8 = nn.Sequential(nn.Conv2d(24, 16, kernel_size=1), hswish())

        self.conv1x1_16 = nn.Sequential(nn.Conv2d(48, 16, kernel_size=1), hswish())
        self.ASPP = ASPP(in_c=16, out_c=8)
        self.conv1x1_aspp = nn.Sequential(nn.Conv2d(8 * 5, 16, kernel_size=1), hswish())

        self.conv1x1_32 = nn.Sequential(nn.Conv2d(96, 16, kernel_size=1), hswish())

        self.dconv_16 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0, bias=True)
        self.dconv_8 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0, bias=True)

        self.conv1x1 = nn.Sequential(nn.Conv2d(16 * 2, 16, kernel_size=1), hswish())

    def forward(self, x):
        x_1_4, x_1_8, x_1_16, x_1_32 = self.backbone(x)
        # print(x_1_4.shape, x_1_8.shape, x_1_16.shape, x_1_32.shape)
        # torch.Size([4, 16, 132, 240]) torch.Size([4, 24, 66, 120]) torch.Size([4, 48, 33, 60]) torch.Size([4, 96, 17, 30])
        x_1_8 = self.conv1x1_8(x_1_8)
        x_1_16 = self.conv1x1_16(x_1_16)
        x_1_16 = self.conv1x1_aspp(self.ASPP(x_1_16))
        x_1_32 = self.conv1x1_32(x_1_32)

        H16, W16 = x_1_16.shape[2], x_1_16.shape[3]
        x_1_32 = F.interpolate(x_1_32, size=(H16, W16), mode="bilinear", align_corners=False)
        x_1_16 = self.conv1x1(torch.cat([x_1_32, x_1_16], dim=1))
        x_1_8 = self.dconv_16(x_1_16) + x_1_8
        x_1_4 = self.dconv_8(x_1_8) + x_1_4

        H, W = x.shape[2], x.shape[3]
        out = F.interpolate(x_1_4, size=(H, W), mode="bilinear", align_corners=False)

        return out


# class HR_MobileNetV3_Small_Seg(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.backbone = HR_MobileNetV3_Small()
#
#         self.conv1x1_8 = nn.Sequential(nn.Conv2d(24, 16, kernel_size=1), hswish())
#
#         self.conv1x1_16 = nn.Sequential(nn.Conv2d(48, 16, kernel_size=1), hswish())
#         self.ASPP = ASPP(in_c=16, out_c=8)
#         self.conv1x1_aspp = nn.Sequential(nn.Conv2d(8 * 5, 16, kernel_size=1), hswish())
#
#         self.conv1x1_32 = nn.Sequential(nn.Conv2d(96, 16, kernel_size=1), hswish())
#
#         self.dconv_16 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0, bias=True)
#         self.dconv_8 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0, bias=True)
#
#         self.conv1x1 = nn.Sequential(nn.Conv2d(16 * 2, 16, kernel_size=1), hswish())
#
#     def forward(self, x):
#         x_1_4, x_1_8, x_1_16, x_1_32 = self.backbone(x)
#         # print(x_1_4.shape, x_1_8.shape, x_1_16.shape, x_1_32.shape)
#         # torch.Size([4, 16, 132, 240]) torch.Size([4, 24, 66, 120]) torch.Size([4, 48, 33, 60]) torch.Size([4, 96, 17, 30])
#         x_1_8 = self.conv1x1_8(x_1_8)
#         x_1_16 = self.conv1x1_16(x_1_16)
#         x_1_16 = self.conv1x1_aspp(self.ASPP(x_1_16))
#         x_1_32 = self.conv1x1_32(x_1_32)
#
#         H16, W16 = x_1_16.shape[2], x_1_16.shape[3]
#         x_1_32 = F.interpolate(x_1_32, size=(H16, W16), mode="bilinear", align_corners=False)
#         x_1_16 = self.conv1x1(torch.cat([x_1_32, x_1_16], dim=1))
#         x_1_8 = self.dconv_16(x_1_16) + x_1_8
#         x_1_4 = self.dconv_8(x_1_8) + x_1_4
#
#         H, W = x.shape[2], x.shape[3]
#         out = F.interpolate(x_1_4, size=(H, W), mode="bilinear", align_corners=False)
#
#         return out


class upsample(nn.Module):
    def __init__(self, type="bilinear", times=2.0, dst_hw=None):
        super().__init__()
        self.times = times
        self.type = type
        self.dst_hw = dst_hw

    def forward(self, x):
        if self.dst_hw is not None:
            H, W = self.dst_hw[0], self.dst_hw[1]
        else:
            H, W = int(x.shape[2] * self.times), int(x.shape[3] * self.times)
        x_new = F.interpolate(x, size=(H, W), mode=self.type, align_corners=False)
        return x_new


class MobileNetV3_Small_FPN_Seg(nn.Module):
    def __init__(self, num_classes, attention=False):
        super().__init__()
        self.backbone = MobileNetV3_Small(num_classes, attention=attention, part=True)
        self.up8 = upsample()
        self.up16 = upsample()

        self.conv1x1_8 = nn.Sequential(nn.Conv2d(24, 16, kernel_size=1), hswish())
        self.conv1x1_16 = nn.Sequential(nn.Conv2d(48, 16, kernel_size=1), hswish())
        self.conv1x1_32 = nn.Sequential(nn.Conv2d(96, 16, kernel_size=1), hswish())

        self.ASPP_16 = nn.Sequential(ASPP(in_c=16, out_c=8), nn.Conv2d(8 * 5, 16, kernel_size=1), hswish())

        self.dconv = nn.Sequential(nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0, bias=True),
                                   nn.BatchNorm2d(16), hswish(),
                                   nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0, bias=True))

        self.classify = nn.Sequential(nn.Conv2d(16 * 2, 16, kernel_size=3), hswish(), nn.Conv2d(16, num_classes, kernel_size=3))

    def forward(self, x):
        x_1_4, x_1_8, x_1_16, x_1_32 = self.backbone(x)  # channels: 16, 24, 48, 96
        # print(x_1_16.shape)
        # con1x1
        x_1_8 = self.conv1x1_8(x_1_8)
        x_1_16 = self.conv1x1_16(x_1_16)
        x_1_32 = self.conv1x1_32(x_1_32)

        # FPN
        x_1_32 = F.interpolate(x_1_32, size=x_1_16.shape[2:], mode="bilinear", align_corners=False)
        x_1_16 = x_1_32 + x_1_16
        x_1_8 = self.up16(x_1_16) + x_1_8
        x_1_4 = self.up8(x_1_8) + x_1_4

        # ASPP
        x_1_16 = self.ASPP_16(x_1_16)
        x_1_16 = self.dconv(x_1_16)

        # merge
        x_1_4 = self.classify(torch.cat([x_1_4, x_1_16], dim=1))

        return F.interpolate(x_1_4, size=x.shape[2:], mode="bilinear", align_corners=False)


class MobileNetV3_Small_PANet_Seg(nn.Module):
    def __init__(self, num_classes, attention=False):
        super().__init__()
        self.backbone = MobileNetV3_Small(num_classes, attention=attention, part=True)
        self.up8 = upsample()
        self.up16 = upsample()

        self.down4 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), hswish())
        self.down8 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), hswish())

        self.conv1x1_8 = nn.Sequential(nn.Conv2d(24, 16, kernel_size=1), hswish())
        self.conv1x1_16 = nn.Sequential(nn.Conv2d(48, 16, kernel_size=1), hswish())
        self.conv1x1_32 = nn.Sequential(nn.Conv2d(96, 16, kernel_size=1), hswish())

        self.ASPP_16 = nn.Sequential(ASPP(in_c=16, out_c=8), nn.Conv2d(8 * 5, 16, kernel_size=1), hswish())

        self.dconv = nn.Sequential(nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0, bias=True),
                                   nn.BatchNorm2d(16), hswish(),
                                   nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0, bias=True))

        self.classify = nn.Sequential(nn.Conv2d(16 * 2, 16, kernel_size=3), hswish(), nn.Conv2d(16, num_classes, kernel_size=3))

    def forward(self, x):
        x_1_4, x_1_8, x_1_16, x_1_32 = self.backbone(x)  # channels: 16, 24, 48, 96
        # print(x_1_16.shape)
        # con1x1
        x_1_8 = self.conv1x1_8(x_1_8)
        x_1_16 = self.conv1x1_16(x_1_16)
        x_1_32 = self.conv1x1_32(x_1_32)

        # PANet-FPN
        x_1_32 = F.interpolate(x_1_32, size=x_1_16.shape[2:], mode="bilinear", align_corners=False)
        x_1_16 = x_1_32 + x_1_16
        x_1_8 = self.up16(x_1_16) + x_1_8
        x_1_4 = self.up8(x_1_8) + x_1_4
        x_1_8 = self.down4(x_1_4) + x_1_8
        x_1_16 = self.down8(x_1_8) + x_1_16

        # ASPP
        x_1_16 = self.ASPP_16(x_1_16)
        x_1_16 = self.dconv(x_1_16)

        # merge
        x_1_4 = self.classify(torch.cat([x_1_4, x_1_16], dim=1))

        return F.interpolate(x_1_4, size=x.shape[2:], mode="bilinear", align_corners=False)


class MobileNetV3_Small_BiFPN_Seg(nn.Module):
    def __init__(self, num_classes, attention=False):
        super().__init__()
        self.backbone = MobileNetV3_Small(num_classes, attention=attention, part=True)
        self.up8 = upsample()
        self.up8_2 = upsample()
        self.up16 = upsample()

        self.down4 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), hswish())
        self.down8 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), hswish())

        self.conv1x1_8 = nn.Sequential(nn.Conv2d(24, 16, kernel_size=1), hswish())
        self.conv1x1_16 = nn.Sequential(nn.Conv2d(48, 16, kernel_size=1), hswish())
        self.conv1x1_32 = nn.Sequential(nn.Conv2d(96, 16, kernel_size=1), hswish())

        self.ASPP_16 = nn.Sequential(ASPP(in_c=16, out_c=8), nn.Conv2d(8 * 5, 16, kernel_size=1), hswish())

        self.dconv = nn.Sequential(nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0, bias=True),
                                   nn.BatchNorm2d(16), hswish(),
                                   nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0, bias=True))

        self.classify = nn.Sequential(nn.Conv2d(16 * 3, 16, kernel_size=3), hswish(), nn.Conv2d(16, num_classes, kernel_size=3))

    def forward(self, x):
        c_1_4, x_1_8, x_1_16, x_1_32 = self.backbone(x)  # channels: 16, 24, 48, 96
        # print(x_1_16.shape)
        # con1x1
        c_1_8 = self.conv1x1_8(x_1_8)
        c_1_16 = self.conv1x1_16(x_1_16)
        c_1_32 = self.conv1x1_32(x_1_32)

        # PANet-FPN
        p_1_32 = F.interpolate(c_1_32, size=c_1_16.shape[2:], mode="bilinear", align_corners=False)
        p_1_16 = p_1_32 + c_1_16
        p_1_8 = self.up16(p_1_16) + c_1_8
        p_1_4 = self.up8(p_1_8) + c_1_4
        p_1_8 = self.down4(p_1_4) + p_1_8
        p_1_16 = self.down8(p_1_8) + p_1_16

        # BiFPN
        p_1_16 = c_1_16 + p_1_16
        p_1_8 = c_1_8 + p_1_8
        p_1_4 = c_1_4 + p_1_4

        # ASPP
        p_1_16 = self.ASPP_16(p_1_16)
        p_1_16 = self.dconv(p_1_16)

        # merge
        p_1_8 = self.up8_2(p_1_8)
        p_1_4 = self.classify(torch.cat([p_1_4, p_1_8, p_1_16], dim=1))
        # return p_1_4
        return F.interpolate(p_1_4, size=x.shape[2:], mode="bilinear", align_corners=False)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    h, w = 224, 224
    img = torch.randn(4, 3, h, w)
    # net = MobileNetV3_Small(1000, attention=False, part=False)
    # net = MobileNetV3_Large(1000, attention=False, part=False)
    # net = MobileNetV3_Small_Seg(2, attention=True)
    # net = MobileNetV3_Small_FPN_Seg(2, attention=False)
    # net = MobileNetV3_Small_PANet_Seg(2, attention=False)
    net = MobileNetV3_Small_BiFPN_Seg(2, attention=False)
    # net = HR_MobileNetV3_Small_Seg(2)
    out = net(img)
    print('out shape', out.shape)

    image = (3, h, w)
    f, p = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f, p)

    # 390086820.0 2835790
    # 388749946.0 2835790
