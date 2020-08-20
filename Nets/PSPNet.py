import torch
import torch.nn.functional as F

from Nets import resnet
from torchvision import models
from itertools import chain
import logging
import torch.nn as nn
import numpy as np
import os
import torch
import torch.nn as nn
import numpy as np
import math
import PIL


def dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters(): p.requires_grad = b


def apply_leaf(m, f):
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'
    # return summary(self, input_shape=(2, 3, 224, 224))


class _PSPModule(nn.Module):
    def __init__(self, in_channels, divisor, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // divisor // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(BaseModel):
    def __init__(self, num_classes, divisor=1, in_channels=3, backbone='resnet152', pretrained=True, use_aux=True, freeze_bn=False, freeze_backbone=False):
        super(PSPNet, self).__init__()
        # TODO: Use synch batchnorm
        norm_layer = nn.BatchNorm2d
        model = getattr(resnet, backbone)(pretrained, norm_layer=norm_layer, )
        m_out_sz = model.fc.in_features
        self.use_aux = use_aux

        self.initial = nn.Sequential(*list(model.children())[:4])
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64 // divisor, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.master_branch = nn.Sequential(
            _PSPModule(m_out_sz // divisor, divisor=divisor, bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(m_out_sz // divisor // 4, num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(m_out_sz // divisor // 2, m_out_sz // divisor // 4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz // divisor // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz // divisor // 4, num_classes, kernel_size=1)
        )

        initialize_weights(self.master_branch, self.auxiliary_branch)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)

        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear')
        output = output[:, :, :input_size[0], :input_size[1]]

        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            aux = aux[:, :, :input_size[0], :input_size[1]]
            return output, aux
        return output

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(),
                     self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


## PSP with dense net as the backbone
# class PSPDenseNet(BaseModel):
#     def __init__(self, num_classes, in_channels=3, backbone='densenet201', pretrained=True, use_aux=True, freeze_bn=False, **_):
#         super(PSPDenseNet, self).__init__()
#         self.use_aux = use_aux
#         model = getattr(models, backbone)(pretrained)
#         m_out_sz = model.classifier.in_features
#         aux_out_sz = model.features.transition3.conv.out_channels
#
#         if not pretrained or in_channels != 3:
#             # If we're training from scratch, better to use 3x3 convs
#             block0 = [nn.Conv2d(in_channels, 64, 3, stride=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
#             block0.extend(
#                 [nn.Conv2d(64, 64, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)] * 2
#             )
#             self.block0 = nn.Sequential(
#                 *block0,
#                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#             )
#             initialize_weights(self.block0)
#         else:
#             self.block0 = nn.Sequential(*list(model.features.children())[:4])
#
#         self.block1 = model.features.denseblock1
#         self.block2 = model.features.denseblock2
#         self.block3 = model.features.denseblock3
#         self.block4 = model.features.denseblock4
#
#         self.transition1 = model.features.transition1
#         # No pooling
#         self.transition2 = nn.Sequential(
#             *list(model.features.transition2.children())[:-1])
#         self.transition3 = nn.Sequential(
#             *list(model.features.transition3.children())[:-1])
#
#         for n, m in self.block3.named_modules():
#             if 'conv2' in n:
#                 m.dilation, m.padding = (2, 2), (2, 2)
#         for n, m in self.block4.named_modules():
#             if 'conv2' in n:
#                 m.dilation, m.padding = (4, 4), (4, 4)
#
#         self.master_branch = nn.Sequential(
#             _PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=nn.BatchNorm2d),
#             nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
#         )
#
#         self.auxiliary_branch = nn.Sequential(
#             nn.Conv2d(aux_out_sz, m_out_sz // 4, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(m_out_sz // 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
#         )
#
#         initialize_weights(self.master_branch, self.auxiliary_branch)
#         if freeze_bn: self.freeze_bn()
#
#     def forward(self, x):
#         input_size = (x.size()[2], x.size()[3])
#
#         x = self.block0(x)
#         x = self.block1(x)
#         x = self.transition1(x)
#         x = self.block2(x)
#         x = self.transition2(x)
#         x = self.block3(x)
#         x_aux = self.transition3(x)
#         x = self.block4(x_aux)
#
#         output = self.master_branch(x)
#         output = F.interpolate(output, size=input_size, mode='bilinear')
#
#         if self.training and self.use_aux:
#             aux = self.auxiliary_branch(x_aux)
#             aux = F.interpolate(aux, size=input_size, mode='bilinear')
#             return output, aux
#         return output
#
#     def get_backbone_params(self):
#         return chain(self.block0.parameters(), self.block1.parameters(), self.block2.parameters(),
#                      self.block3.parameters(), self.transition1.parameters(), self.transition2.parameters(),
#                      self.transition3.parameters())
#
#     def get_decoder_params(self):
#         return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())
#
#     def freeze_bn(self):
#         for module in self.modules():
#             if isinstance(module, nn.BatchNorm2d): module.eval()


# output = self.model(data)
# if self.config['arch']['type'][:3] == 'PSP':
# 	assert output[0].size()[2:] == target.size()[1:]
# 	assert output[0].size()[1] == self.num_classes
# 	loss = self.loss(output[0], target)
# 	loss += self.loss(output[1], target) * 0.4
# 	output = output[0]

if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    divisor = 1
    h = 1080 // 4 // 16 * 16
    w = 1920 // 4 // 16 * 16

    net = PSPNet(num_classes=2, divisor=divisor, in_channels=3, backbone='resnet18', pretrained=False, use_aux=False, freeze_bn=False, freeze_backbone=False).cuda()
    out = net(torch.randn(2, 3, h, w).cuda())
    print(out.shape)

    # image = (3, h, w)
    # f, p = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=False, verbose=False)
    # print(f, p)
