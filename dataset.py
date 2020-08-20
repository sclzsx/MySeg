import os
import cv2
from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import enet_weighing, median_freq_balancing
import torch.nn as nn
from collections import OrderedDict, Counter
import random
from utils import add_mask_to_source_multi_classes, add_mask_to_source


def get_class_weights(loader, out_channels, weighting):
    print('Weighting method is:{}, please wait.'.format(weighting))
    if weighting == 'enet':
        class_weights = enet_weighing(loader, out_channels)
        class_weights = torch.from_numpy(class_weights).float().cuda()
    elif weighting == 'mfb':
        class_weights = median_freq_balancing(loader, out_channels)
        class_weights = torch.from_numpy(class_weights).float().cuda()
    else:
        class_weights = None
    return class_weights


class PILToLongTensor(object):
    def __call__(self, pic):
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(pic)))
        # handle numpy array
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.long()
        # Convert PIL image to ByteTensor
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # Reshape tensor
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # Convert to long and squeeze the channels
        return img.transpose(0, 1).transpose(0, 2).contiguous().long().squeeze_()


class SegDataset(Dataset):
    def __init__(self, dataset_dir, num_classes=2, appoint_size=(512, 512), erode=0, aug=False):
        self.imgs_dir = os.path.join(dataset_dir, 'images')
        self.labels_dir = os.path.join(dataset_dir, 'labels')
        self.names = os.listdir(self.labels_dir)
        self.num_classes = num_classes
        self.appoint_size = appoint_size
        self.erode = erode
        self.aug = aug

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        label_path = os.path.join(self.labels_dir, name)
        img_path = os.path.join(self.imgs_dir, name[:-3] + 'jpg')

        image = cv2.imread(img_path)
        label = Image.open(label_path)
        if self.aug:
            random_down_factor = random.uniform(1, 5)
            new_size = (int(1920 // random_down_factor), int(1080 // random_down_factor))
            image = cv2.resize(image, new_size)
            label = label.resize((new_size[1], new_size[0]), Image.NEAREST)
        # print(image.shape, label.size)

        img_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(self.appoint_size), transforms.ToTensor()])
        img_tensor = img_transform(image)

        label = label.resize((self.appoint_size[1], self.appoint_size[0]), Image.NEAREST)
        if self.erode > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.erode, self.erode))
            label_np = cv2.erode(np.array(label), kernel)
            label = Image.fromarray(label_np)

        if self.num_classes == 1:
            label_transform = transforms.Compose(transforms.ToTensor())
        else:
            label_transform = transforms.Compose([PILToLongTensor()])

        label_tensor = label_transform(label)
        # print(img_tensor.shape, label_tensor.shape, img_tensor.dtype, label_tensor.dtype)
        return img_tensor, label_tensor


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    aug = True
    num_classes = 3
    appoint_size = (528, 960)
    dataset_dir = '/workspace/DATA/adas/roadside0813/train'

    dataset = SegDataset(dataset_dir, num_classes=num_classes, appoint_size=appoint_size, erode=0, aug=aug)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for batch_data in loader:
        pass
        a_data = batch_data[0][0]
        a_label = batch_data[1][0].squeeze(0)
        img_transform = transforms.Compose([transforms.ToPILImage()])
        data_pil = img_transform(a_data)

        data_np = np.array(data_pil)
        data_np = cv2.cvtColor(data_np, cv2.COLOR_BGR2RGB)
        # data_np = np.array(a_data.permute((1, 2, 0)) * 255, dtype='uint8')
        label_np = np.array(a_label)
        # print(a_data.shape, a_data.dtype, a_label.dtype, a_label.shape)
        # print(Counter(np.array(a_label).flatten()))
        # print(Counter(data_np.flatten()))

        plt.subplot(211)
        plt.imshow(data_np)
        plt.subplot(212)
        # plt.imshow(label_np)
        plt.imshow(add_mask_to_source_multi_classes(data_np, label_np, num_classes=num_classes))
        plt.pause(0.5)
