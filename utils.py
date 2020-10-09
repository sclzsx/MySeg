import os
import cv2
import json
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:
        w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class:
        propensity_score = freq_class / total_pixels.
    References: https://arxiv.org/abs/1606.02147
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.
    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def median_freq_balancing(dataloader, num_classes):
    """Computes class weights using median frequency balancing as described
    in https://arxiv.org/abs/1411.4734:
        w_class = median_freq / freq_class,
    where freq_class is the number of pixels of a given class divided by
    the total number of pixels in images where that class is present, and
    median_freq is the median of freq_class.
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    whose weights are going to be computed.
    - num_classes (``int``): The number of classes
    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=num_classes)

        # Create of mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    return med / freq


def resize_tensor(tensor, size):  # Other nearest methods result in misaligned tensor.
    new_tensor = []
    with torch.no_grad():
        for label in tensor.cpu():
            label = label.float().numpy()
            label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
            new_tensor.append(np.asarray(label))
        new_tensor = torch.LongTensor(new_tensor)
    return new_tensor


# from sklearn.metrics import confusion_matrix

# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import torch.nn.functional as F
#
# from dataset import SegDataset
# from choices import choose_net
# import collections
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


def add_mask_to_source_multi_classes(source_np, mask_np, num_classes):
    colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255], [0, 255, 255], [255, 255, 0]]
    foreground_mask_bool = mask_np.astype('bool')
    foreground_mask = mask_np * foreground_mask_bool
    foreground = np.zeros(source_np.shape, dtype='uint8')
    background = source_np.copy()

    for i in range(1, num_classes + 1):
        fg_tmp = np.where(foreground_mask == i, 1, 0)
        fg_tmp_mask_bool = fg_tmp.astype('bool')

        fg_color_tmp = np.zeros(source_np.shape, dtype='uint8')
        fg_color_tmp[:, :] = colors[i]
        for c in range(3):
            fg_color_tmp[:, :, c] *= fg_tmp_mask_bool
        foreground += fg_color_tmp
    foreground = cv2.addWeighted(source_np, 0.5, foreground, 0.5, 0)

    for i in range(3):
        foreground[:, :, i] *= foreground_mask_bool
        background[:, :, i] *= ~foreground_mask_bool

    show = foreground + background
    # plt.imshow(show)
    # plt.pause(0.5)
    return show


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


def subplots(np1, np2=None, np3=None, np4=None, text=None, pause=0.2, save_path=None):
    nps = list()
    nps.append(np1)
    if np2 is not None:
        nps.append(np2)
    if np3 is not None:
        nps.append(np3)
    if np4 is not None:
        nps.append(np4)
    plt.figure(figsize=(6, 3), dpi=200)
    for i in range(len(nps)):
        ax = 100 + len(nps) * 10 + i + 1
        plt.subplot(ax)
        plt.axis('off')
        plt.imshow(nps[i])
    # if text is not None:
    #     plt.suptitle(text)
    if pause > 0:
        plt.pause(pause)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')


def select_frames_from_videos(test_videos):
    names = [i.name for i in Path(test_videos).glob('*.*')]
    for name in names:
        video_path = os.path.join(test_videos, name)
        cap = cv2.VideoCapture(video_path)
        total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        stride = int(total_frame_num * 0.2)
        for idx in range(0, total_frame_num - 1, stride):  # 抛弃最后一帧才能有效保存视频
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            _, frame = cap.read()
            cv2.imwrite('/workspace/DATA/zhatu/videos/previous/' + name[:-4] + str(idx) + '.jpg', frame)


if __name__ == "__main__":
    select_frames_from_videos('/workspace/DATA/zhatu/videos/previous')
