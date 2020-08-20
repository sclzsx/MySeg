from Nets.Deeplabv1 import DeepLabV1
from Nets.Deeplabv2 import DeepLabV2
from Nets.Deeplabv3 import DeepLabV3
from Nets.Deeplabv3p import DeepLabV3Plus
from Nets.ENet import ENet
from Nets.FCN import VGGNet, FCNs, FCN8s
from Nets.UNet import UNet
from Nets.SegNet import SegNet
from Nets.LaneNet0508 import LaneNet0508
from Nets.lanenet_mod import LaneNet0508_mod
from Nets.SFNet import sf_resnet50, sf_resnet18_mod
from Nets.LaneNet_CBAM import LaneNet_CBAM
import torch
import torch.nn as nn
from torchstat import stat
from thop import profile, clever_format
from ptflops import get_model_complexity_info
import torchvision.models as models
from torchsummary import summary


def get_optimizer(net, optim_name):
    if optim_name == 'adam':
        optimizer = torch.optim.Adam(net.parameters())
    else:
        optimizer = torch.optim.Adam(net.parameters())
    return optimizer


def get_criterion(out_channels, class_weights=None):
    if out_channels == 1:
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    return criterion


def choose_net(name, out_channels):
    if name == 'unet':
        return UNet(n_classes=out_channels)
    elif name == 'segnet':
        return SegNet(label_nbr=out_channels)
    elif name == 'deeplabv1':
        return DeepLabV1(n_classes=out_channels, n_blocks=[3, 4, 23, 3])
    elif name == 'deeplabv2':
        return DeepLabV2(n_classes=out_channels, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
    elif name == 'deeplabv3':
        return DeepLabV3(n_classes=out_channels, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=8, )
    elif name == 'deeplabv3p':
        return DeepLabV3Plus(num_classes=out_channels)
    elif name == 'fcn8':
        return FCN8s(pretrained_net=VGGNet(requires_grad=True, show_params=False), n_class=out_channels)
    elif name == 'fcn':
        return FCNs(num_classes=out_channels)
    elif name == 'enet':
        return ENet(num_classes=out_channels)
    elif name == 'lanenet0508':
        return LaneNet0508(num_classes=out_channels)
    elif name == 'sfnet50':
        return sf_resnet50()
    elif name == 'lanenet_mod':
        return LaneNet0508_mod(num_classes=out_channels)
    elif name == 'lanenet_cbam':
        return LaneNet_CBAM(num_classes=out_channels)
    elif name == 'lanenet_mod_cbam':
        from Nets.lanenet_mod_cbam import LaneNet_mod_cbam
        return LaneNet_mod_cbam(num_classes=out_channels)
    elif name == 'mobilenetv3_small':
        from Nets.mobilenetv3 import MobileNetV3_Small_Seg
        return MobileNetV3_Small_Seg(num_classes=out_channels, attention=False)
    elif name == 'mobilenetv3_small_cbam':
        from Nets.mobilenetv3 import MobileNetV3_Small_Seg
        return MobileNetV3_Small_Seg(num_classes=out_channels, attention=True)
    elif name == 'mobilenetv3_small_fpn':
        from Nets.mobilenetv3 import MobileNetV3_Small_FPN_Seg
        return MobileNetV3_Small_FPN_Seg(num_classes=out_channels, attention=False)
    elif name == 'mobilenetv3_small_panet':
        from Nets.mobilenetv3 import MobileNetV3_Small_PANet_Seg
        return MobileNetV3_Small_PANet_Seg(num_classes=out_channels, attention=False)
    elif name == 'mobilenetv3_small_bifpn':
        from Nets.mobilenetv3 import MobileNetV3_Small_BiFPN_Seg
        return MobileNetV3_Small_BiFPN_Seg(num_classes=out_channels, attention=False)

    elif name == 'enet_2':
        return ENet(num_classes=out_channels, divisor=2)
    elif name == 'enet_4':
        return ENet(num_classes=out_channels, divisor=4)
    elif name == 'lanenet0508_2':
        return LaneNet0508(num_classes=out_channels, divisor=2)
    elif name == 'deeplabv3p_16':
        return DeepLabV3Plus(num_classes=out_channels, divisor=16)
    elif name == 'sfnet18_16':
        return sf_resnet18_mod()


if __name__ == '__main__':
    net_names = [
        # 'fcn',
        # 'fcn8',
        # 'enet', 'enet_mod',
        # 'lanenet',
        # 'lanenet_deconv',
        # 'lanenet0508'
        'sfnet50'
    ]
    resizes = [
        # (320, 320),
        # (224, 224)
        (528, 960)
    ]

    # batch_cal_comlexity(net_names, resizes, out_channels=2, method=0)
    summary(choose_net(net_names[0], 2).cuda(), (3, resizes[0][0], resizes[0][1]))
