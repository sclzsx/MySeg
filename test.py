import os
import cv2
import json
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from dataset import SegDataset
from choices import choose_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_latest_pt(dir):
    max_num = 0
    latest_path = ''
    for path in Path(dir).glob('*.pt'):
        num = int(path.name.split('_')[1].split('.')[0])
        if num > max_num:
            max_num = num
            latest_path = str(path)
    if latest_path != '':
        print('Loading:', latest_path)
        return latest_path
    else:
        print('No pts this dir:', dir)
        return None


def calcuate_np_iou(y_pred, y_true, num_classes):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    intersection = np.diag(current)  # 取混淆矩阵的对角线元素，即各类别tp的数量，各类别的交集的数量
    ground_truth_set = current.sum(axis=1)  # 按列求和，各个类别的gt数
    predicted_set = current.sum(axis=0)  # 按行求和，各个类别的pred数
    union = ground_truth_set + predicted_set - intersection  # 各类别的并集的数量
    IoU = intersection / union.astype(np.float32)  # 各类别的IoU
    return IoU


def add_mask_to_source(source_np, mask_np, color):
    mask_bool = (np.ones(mask_np.shape, dtype='uint8') & mask_np).astype('bool')

    foreground = np.zeros(source_np.shape, dtype='uint8')
    for i in range(3):
        foreground[:, :, i] = color[i]
    foreground = cv2.addWeighted(source_np, 0.5, foreground, 0.5, 0)

    background = source_np.copy()
    for i in range(3):
        foreground[:, :, i] *= mask_bool
        background[:, :, i] *= (~mask_bool)

    return background + foreground


def predict_batch_data(net, out_channels, device, batch_data, batch_label, criterion, erode):
    if batch_label is None:  # 针对图片或视频帧的预测，没有对应的label，随机生成一个和data等大的label
        batch_label = torch.randn(1, out_channels, batch_data.shape[2], batch_data.shape[3])

    assert batch_data.shape[0] == 1 and batch_label.shape[0] == 1  # 为方便显示和计算指标，只处理批次大小为1的数据

    with torch.no_grad():
        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)
        output = net(batch_data)

        if batch_data.shape[3] != output.shape[3]:  # 通过H判定输出是否比输入小，主要针对deeplab系列，将输出上采样至输入大小
            output = F.interpolate(output, size=(batch_data.shape[2], batch_data.shape[3]), mode="bilinear", align_corners=False)

        if criterion is not None:
            loss = criterion(output, batch_label).item()
        else:
            loss = 0

        if out_channels == 1:
            batch_label = batch_label.float()  # labels默认为long，通道为1时采用逻辑损失，需要data和label均为float
            output = torch.sigmoid(output).squeeze().cpu()  # Sigmod回归后去掉批次维N
            prediction_np = np.where(np.array(output) > 0.2, 1, 0)  # 阈值默认为0.5
        else:
            batch_label = batch_label.squeeze(1)  # 交叉熵损失需要去掉通道维C
            prediction_np = np.array(torch.max(output.data, 1)[1].squeeze(0).cpu())  # 取最大值的索引作为标签，并去掉批次维N

        if erode > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
            prediction_np = cv2.dilate(prediction_np.astype('uint8'), kernel)

        data_np = np.array((batch_data.squeeze(0).permute((1, 2, 0))).cpu()) * 255
        label_np = np.array(batch_label.squeeze().cpu())  # 同时去掉批次维N和通道维C（均为1）

        return data_np, label_np, prediction_np, loss


def vis_multi_plot(data_np, label_np, prediction_np, miou, pause=0.2):
    plt.subplot(131)
    plt.axis('off')
    plt.imshow(data_np)
    plt.title('Source', fontsize=10)

    plt.subplot(132)
    plt.axis('off')
    plt.imshow(label_np)
    plt.title('Label', fontsize=10)

    plt.subplot(133)
    plt.axis('off')
    plt.imshow(prediction_np)
    plt.title('Pred mIoU:' + str(miou), fontsize=10)

    plt.pause(pause)


def save_multi_plot(data_np, label_np, prediction_np, miou, save_path):
    plt.subplot(131)
    plt.axis('off')
    plt.imshow(data_np)
    plt.title('Source', fontsize=10)

    plt.subplot(132)
    plt.axis('off')
    plt.imshow(label_np)
    plt.title('Label', fontsize=10)

    plt.subplot(133)
    plt.axis('off')
    plt.imshow(prediction_np)
    plt.title('Pred mIoU:' + str(miou), fontsize=10)

    plt.savefig(save_path, bbox_inches='tight')


def predict_dataset(net, args):
    test_set = SegDataset(args.testset, resize=args.resize, erode=args.erode)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    sum_ = 0
    for i, (datas, labels) in enumerate(test_dataloader):
        data_np, label_np, prediction_np, _ = predict_batch_data(net, args.out_channels, device, datas, labels, criterion=None, erode=args.erode)
        IoUs = calcuate_np_iou(prediction_np, label_np, num_classes=args.out_channels)
        miou = np.mean(IoUs)

        if args.vis:
            vis_multi_plot(data_np, label_np, prediction_np, miou, pause=0.2)

        if i % 400 == 0:
            # save_multi_plot(data_np, label_np, prediction_np, miou, save_path='./Results/' + args.pt_dir + '/' + str(i) + '.png')
            print('Predicted the {}th image, its miou is {}'.format(i, miou))

        sum_ += miou
        # mean_ious.append(miou)


    # for t in mean_ious:

    # mean_iou = np.mean(mean_ious)
    print('Mean IoU of the whole testing set is:', sum_, len(test_dataloader), '\n')

    with open('./Results/' + args.pt_dir + '/Metrics.txt', 'w') as f:
        for key, val in vars(args).items():
            f.write(str(key))
            f.write('\t')
            f.write(str(val))
            f.write('\n')
        f.write(str(mean_iou))


def predict_videos(net, args):
    cap = cv2.VideoCapture(args.video_path)
    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    save_path = './Results/' + os.path.basename(args.video_path)[:-4] + '_' + args.pt_dir + '.avi'
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, video_size)

    for frame_idx in range(total_frame_num - 1):  # 抛弃最后一帧才能有效保存视频
        if frame_idx % 100 == 0:
            print('Processing frame [{}/{}]'.format(frame_idx, total_frame_num))
        ret, frame_cv2 = cap.read()

        # 将frame转为和data同样的类型
        frame_pil = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
        if args.resize is not None:
            transform = transforms.Compose([transforms.Resize(args.resize, interpolation=Image.NEAREST), transforms.ToTensor()])
            frame_cv2 = cv2.resize(frame_cv2, (args.resize[1], args.resize[0]))
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        frame_tensor = (transform(frame_pil) / 255).unsqueeze(0).to(device)

        _, _, prediction_np, _, _ = eval_batch_data(net, args.out_channels, device, frame_tensor, batch_label=None, criterion=None, erode=args.erode)

        show_np = add_mask_to_source(frame_cv2, prediction_np, args.fg_color)

        if args.vis:
            plt.imshow(show_np)
            plt.pause(0.5)

        show_np = cv2.resize(show_np, video_size)
        out.write(show_np)
    out.release()
    print('Already saved to:', save_path)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out-channels", type=int, default=2)
    parser.add_argument("--erode", type=int, default=15)
    parser.add_argument("--resize", type=tuple, default=(512, 512))
    parser.add_argument("--pt-dir", type=str, default='')
    parser.add_argument("--vis", type=bool, default=False)
    parser.add_argument("--fg-color", type=list, default=[255, 158, 53])
    parser.add_argument("--video_path", type=str, default='/workspace/DATA/adas/test.mp4')
    parser.add_argument("--testset", type=str, default='/workspace/DATA/adas/road0717/test')
    return parser.parse_args()


def merge_args_from_json(args, resize, net_name, save_suffix):
    json_path = './Results/' + net_name + '_' + save_suffix + '_size' + str(resize[0]) + '/args.json'
    if not os.path.exists(json_path):
        return args
    else:
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
        args.out_channels = json_dict['out_channels']
        args.erode = json_dict['erode']
        args.resize = (json_dict['resize'][1], json_dict['resize'][0])
        args.pt_dir = json_dict['net_name'] + '_' + json_dict['save_suffix'] + '_size' + str(args.resize[0])
        return args


def do_test(args, test_mode):
    print(vars(args))
    with torch.cuda.device(args.gpu):
        net_name = args.pt_dir.split('_road')[0]
        net = choose_net(net_name, args.out_channels).to(device)
        latest_pt = find_latest_pt('./Results/' + args.pt_dir)
        if latest_pt is not None:
            net.load_state_dict(torch.load(latest_pt))
            net.eval()
            if test_mode == 0:
                predict_dataset(net, args)
            elif test_mode == 1:
                predict_videos(net, args)
        else:
            print('No pts this dir! ', './Results/' + args.pt_dir)


if __name__ == "__main__":
    args = get_args()
    search_experiment = True

    if search_experiment:
        test_dir_suffix = 'road_0803'
        resizes = [(224, 224), (320, 320)]
        test_net_names = ['fcn', 'fcn8', 'enet', 'enet_mod', 'lanenet', 'lanenet_deconv', 'lanenet0508']
        for test_net_name in test_net_names:
            for resize in resizes:
                merge_args_from_json(args, resize, test_net_name, test_dir_suffix)
                do_test(args, test_mode=0)
    else:
        do_test(args, test_mode=0)
