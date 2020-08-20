import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG
from Nets.resnet import resnet18


class FCN32s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']

        score = self.bn1(self.relu(self.deconv1(x5)))
        score = self.bn2(self.relu(self.deconv2(score)))
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score


class FCN16s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']

        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.bn2(self.relu(self.deconv2(score)))
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score


class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']

        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score


class FCNs_yb(nn.Module):
    def __init__(self, num_classes, backbone='vgg', divisor=1):
        super().__init__()
        self.num_classes = num_classes
        if backbone == 'vgg':
            if divisor == 1:
                self.pretrained_net = VGGNet(pretrained=False, model='vgg16')
            elif divisor == 2:
                self.pretrained_net = VGGNet(pretrained=False, model='vgg16_half')
            elif divisor == 4:
                self.pretrained_net = VGGNet(pretrained=False, model='vgg16_quarter')
            elif divisor == 8:
                self.pretrained_net = VGGNet(pretrained=False, model='vgg16_eighth')

            self.relu = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose2d(512 // divisor, 512 // divisor, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn1 = nn.BatchNorm2d(512 // divisor)
            self.deconv2 = nn.ConvTranspose2d(512 // divisor, 256 // divisor, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn2 = nn.BatchNorm2d(256 // divisor)
            self.deconv3 = nn.ConvTranspose2d(256 // divisor, 128 // divisor, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn3 = nn.BatchNorm2d(128 // divisor)
            self.deconv4 = nn.ConvTranspose2d(128 // divisor, 64 // divisor, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn4 = nn.BatchNorm2d(64 // divisor)
            self.deconv5 = nn.ConvTranspose2d(64 // divisor, 32 // divisor, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn5 = nn.BatchNorm2d(32 // divisor)
            self.classifier = nn.Conv2d(32 // divisor, num_classes, kernel_size=1)
        else:
            channels = [64 * 2 ** i // divisor for i in range(4)]
            self.pretrained_net = eval(backbone + '(channels=channels)')

            self.relu = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn1 = nn.BatchNorm2d(channels[2])
            self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn2 = nn.BatchNorm2d(channels[1])
            self.deconv3 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn3 = nn.BatchNorm2d(channels[0])
            self.deconv4 = nn.ConvTranspose2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0)
            self.bn4 = nn.BatchNorm2d(channels[0])
            self.deconv5 = nn.ConvTranspose2d(channels[0], channels[0] // 2, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn5 = nn.BatchNorm2d(channels[0] // 2)
            self.classifier = nn.Conv2d(channels[0] // 2, num_classes, kernel_size=1)
        # classifier is 1x1 conv, to reduce channels from 32 to n_class

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']
        x2 = output['x2']
        x1 = output['x1']

        score = self.bn1(self.relu(self.deconv1(x5)))
        score = score + x4
        score = self.bn2(self.relu(self.deconv2(score)))
        score = score + x3
        score = self.bn3(self.relu(self.deconv3(score)))
        score = score + x2
        score = self.bn4(self.relu(self.deconv4(score)))
        score = score + x1
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        # delete redundant fully-connected layer params, can save memory
        # 去掉vgg最后的全连接层(classifier)
        if remove_fc:
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx, (begin, end) in enumerate(self.ranges):
            # self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) (vgg16 examples)
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),

    'vgg16_half': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg16_quarter': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg16_eighth': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),

    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# Vgg-Net config
# Vgg网络结构配置
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

    'vgg16_half': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'vgg16_quarter': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'],
    'vgg16_eighth': [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 64, 64, 64, 'M'],

    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# make layers using Vgg-Net config(cfg)
# 由cfg构建vgg-Net
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = nn.Sequential(                 # 1x1卷积
                nn.Conv2d(in_channels, v//4, kernel_size=1),
                nn.Conv2d(v//4, v//4, kernel_size=3, padding=1),
                nn.Conv2d(v//4, v, kernel_size=1)
            )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    divisor = 2
    w = h = 512

    net = FCNs_yb(num_classes=2, backbone='resnet18', divisor=divisor)
    image = (3, h, w)
    f, p = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f, p)
    out = net(torch.randn(1, 3, h, w))
    print(out.shape)
