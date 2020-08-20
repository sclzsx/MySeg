import torch
import torch.nn as nn
import torch.nn.functional as F


class CBR(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0, d=1, b_flag=True, r_flag=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=d)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.b_flag = b_flag
        self.r_flag = r_flag

    def forward(self, x):
        x = self.conv(x)
        if self.b_flag:
            x = self.bn(x)
        if self.r_flag:
            x = self.relu(x)
        # print('sssss', x.shape)
        return x


class residual(nn.Module):
    def __init__(self, in_c, out_c, down_flag=False):
        super().__init__()
        if down_flag == True:
            down_stride = 2
        else:
            down_stride = 1
        self.layer_x_1 = CBR(in_c, out_c, k=3, p=1, s=down_stride)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, stride=down_stride)
        self.layer_x_2 = CBR(out_c, out_c, k=3, p=1, r_flag=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_tmp = self.layer_x_1(x)
        # print(x_tmp.shape)
        x_out = self.layer_x_2(x_tmp)
        # print(x_out.shape)
        x_out = x_out + self.conv(x)
        # print(x_out.shape)
        return self.relu(x_out)


class r18(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super().__init__()

        self.layer1 = nn.Sequential(CBR(3, 64, k=7, s=2, p=3), nn.MaxPool2d(3, 2, 1, ceil_mode=True))
        self.layer2 = nn.Sequential(residual(64, 128, down_flag=False), residual(128, 128))
        self.layer3 = nn.Sequential(residual(128, 256, down_flag=True), residual(256, 256))
        self.layer4 = nn.Sequential(residual(256, 512, down_flag=True), residual(512, 512))
        self.layer5 = nn.Sequential(residual(512, 512, down_flag=False), residual(512, 512))

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # print(x.shape)

        x1 = self.layer1(x)
        # print(x1.shape)

        x2 = self.layer2(x1)
        # print(x2.shape)

        x3 = self.layer3(x2)
        # print(x3.shape)

        x4 = self.layer4(x3)
        # print(x4.shape)

        x5 = self.layer5(x4)
        # print(x5.shape)

        pool2 = self.pool2(x5)
        reshape = pool2.view(pool2.size(0), -1)
        line = self.linear(reshape)
        # print(line.shape)

        return {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'line': line}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class image_pool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = CBR(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


class ASPP(nn.Module):
    def __init__(self, in_c=512, out_c=256):
        super().__init__()
        self.conv = CBR(in_c, out_c, k=1)
        self.dilate_1 = CBR(in_c, out_c, k=3, d=6, p=6)
        self.dilate_2 = CBR(in_c, out_c, k=3, d=12, p=12)
        self.dilate_3 = CBR(in_c, out_c, k=3, d=18, p=18)
        self.image_pool = image_pool(in_c, out_c)

    def forward(self, x):
        # print(x.shape)
        x1 = self.conv(x)
        # print(x1.shape)
        x2 = self.dilate_1(x)
        # print(x2.shape)
        x3 = self.dilate_2(x)
        # print(x3.shape)
        x4 = self.dilate_3(x)
        # print(x4.shape)
        x5 = self.image_pool(x)
        # print(x5.shape)
        return torch.cat([x1, x2, x3, x4, x5], dim=1)


class my_deeplabv3p(nn.Module):
    def __init__(self, num_classes, back_bone_name='r18'):
        super().__init__()
        if back_bone_name == 'r18':
            self.backbone = r18(num_classes)
        self.ASPP = ASPP()
        self.conv1x1_low = nn.Conv2d(128, 48, kernel_size=1)
        self.conv1x1_high = nn.Conv2d(256 * 5, 256, kernel_size=1)
        self.conv3x3 = nn.Conv2d(256 + 48, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        backbone = self.backbone(x)
        x2 = backbone['x2']
        x5 = backbone['x5']
        # print(x2.shape, x5.shape)

        high = self.conv1x1_high(self.ASPP(x5))
        low = self.conv1x1_low(x2)

        H, W = x2.shape[2], x2.shape[3]
        high = F.interpolate(high, size=(H, W), mode="bilinear", align_corners=False)
        # print(high.shape, low.shape)
        out = self.conv3x3(torch.cat([low, high], dim=1))
        out = F.interpolate(out, size=x.shape[-2], mode="bilinear", align_corners=False)
        # print(out.shape)

        return out


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    h, w = 224, 224
    img = torch.randn(4, 3, h, w)
    net = my_deeplabv3p(2)
    out = net(img)
    print('out shape', out.shape)

    image = (3, h, w)
    f, p = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f, p)
