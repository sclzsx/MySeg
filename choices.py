from Nets.Deeplabv1 import DeepLabV1
from Nets.Deeplabv2 import DeepLabV2
from Nets.Deeplabv3 import DeepLabV3
from Nets.Deeplabv3p import DeepLabV3Plus
from Nets.ENet import ENet
from Nets.FCN import VGGNet, FCNs, FCN8s
from Nets.UNet import UNet
from Nets.SegNet import SegNet
from Nets.ENet_mod import ENet_mod, LANENet, LANENet_deconv
from Nets.LaneNet0508 import LaneNet0508
import torch
from torchstat import stat
from thop import profile, clever_format
from ptflops import get_model_complexity_info


def choose_net(name, out_channels):
    if name == 'unet':
        return UNet(n_classes=out_channels)
    elif name == 'fcn':
        return FCNs(pretrained_net=VGGNet(requires_grad=True, show_params=False), n_class=out_channels)
    elif name == 'fcn8':
        return FCN8s(pretrained_net=VGGNet(requires_grad=True, show_params=False), n_class=out_channels)
    elif name == 'enet':
        return ENet(num_classes=out_channels)
    elif name == 'segnet':
        return SegNet(label_nbr=out_channels)
    elif name == 'deeplabv1':
        return DeepLabV1(n_classes=out_channels, n_blocks=[3, 4, 23, 3])
    elif name == 'deeplabv2':
        return DeepLabV2(n_classes=out_channels, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
    elif name == 'deeplabv3':
        return DeepLabV3(n_classes=out_channels, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=8, )
    elif name == 'deeplabv3p':
        return DeepLabV3Plus(n_classes=out_channels, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=16, )
    elif name == 'enet_mod':
        return ENet_mod(num_classes=out_channels)
    elif name == 'lanenet':
        assert out_channels == 2
        return LANENet()
    elif name == 'lanenet_deconv':
        assert out_channels == 2
        return LANENet_deconv()
    elif name == 'lanenet0508':
        return LaneNet0508(num_classes=out_channels)


def cal_complexity_with_thop(net_name, resize, out_channels):
    net = choose_net(net_name, out_channels)
    print(net)
    image = torch.randn(1, 3, resize[0], resize[1])
    flops, params = profile(net, (image,))
    flops, params = clever_format([flops, params], '%.2f')
    return flops, params


def cal_complexity_with_ptflops(net_name, resize, out_channels):
    net = choose_net(net_name, out_channels)
    image = (3, resize[0], resize[1])
    macs, params = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=True, verbose=True)
    return macs, params


def cal_complexity_with_stat(net_name, resize, out_channels):
    net = choose_net(net_name, out_channels)
    image = torch.randn(3, resize[0], resize[1])
    stat(net, (image,))


def batch_cal_comlexity(net_names, resizes, out_channels, method):
    for net_name in net_names:
        for resize in resizes:
            if method == 0:
                f, p = cal_complexity_with_ptflops(net_name, resize, out_channels)
                print('Net:{} Input_size:{} flops:{} params:{}'.format(net_name, resize, f, p))
            elif method == 1:
                f, p = cal_complexity_with_thop(net_name, resize, out_channels)
                print('Net:{} Input_size:{} flops:{} params:{}'.format(net_name, resize, f, p))
            elif method == 2:
                cal_complexity_with_stat(net_name, resize, out_channels)
            else:
                print('Total params: %.2fM' % (sum(p.numel() for p in choose_net(net_name, out_channels).parameters()) / 1000000.0))


if __name__ == '__main__':
    net_names = [
        'fcn',
        # 'fcn8',
        # 'enet', 'enet_mod',
        # 'lanenet',
        # 'lanenet_deconv',
        # 'lanenet0508'
    ]
    resizes = [
        (320, 320),
        # (224, 224)
    ]
    batch_cal_comlexity(net_names, resizes, out_channels=2, method=0)

# Net:fcn Input_size:(320, 320) flops:33.54 GMac params:18.64 M
# Net:fcn Input_size:(224, 224) flops:16.44 GMac params:18.64 M
# Net:fcn8 Input_size:(320, 320) flops:33.54 GMac params:18.64 M
# Net:fcn8 Input_size:(224, 224) flops:16.44 GMac params:18.64 M
# Net:enet Input_size:(320, 320) flops:0.77 GMac params:349.21 k
# Net:enet Input_size:(224, 224) flops:0.38 GMac params:349.21 k
# Net:enet_mod Input_size:(320, 320) flops:0.79 GMac params:355.54 k
# Net:enet_mod Input_size:(224, 224) flops:0.39 GMac params:355.54 k
# Net:lanenet Input_size:(320, 320) flops:0.24 GMac params:67.19 k
# Net:lanenet Input_size:(224, 224) flops:0.12 GMac params:67.19 k
# Net:lanenet_deconv Input_size:(320, 320) flops:0.23 GMac params:63.22 k
# Net:lanenet_deconv Input_size:(224, 224) flops:0.11 GMac params:63.22 k
# Net:lanenet0508 Input_size:(320, 320) flops:0.49 GMac params:172.15 k
# Net:lanenet0508 Input_size:(224, 224) flops:0.24 GMac params:172.15 k
