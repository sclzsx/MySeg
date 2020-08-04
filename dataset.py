import os
import cv2
from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SegDataset(Dataset):
    def __init__(self, dataset_dir, resize=None, erode=0):
        self.imgs_dir = os.path.join(dataset_dir, 'images')
        self.labels_dir = os.path.join(dataset_dir, 'labels')
        self.names = os.listdir(self.labels_dir)
        self.resize = resize
        self.erode = erode

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        label_path = os.path.join(self.labels_dir, name)
        img_path = os.path.join(self.imgs_dir, name[:-3] + 'jpg')

        img = Image.open(img_path)
        label = Image.open(label_path)

        if self.resize is not None:
            transform = transforms.Compose([transforms.Resize(self.resize, interpolation=Image.NEAREST), transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor()])

        img_tensor = transform(img)
        label_tensor = transform(label)

        if self.erode > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.erode, self.erode))
            label_np = cv2.erode(np.array(label_tensor.squeeze(0)).astype('uint8'), kernel)
            label_tensor = torch.from_numpy(label_np).unsqueeze(0)

        return img_tensor / 255, label_tensor.clamp(max=1).long()


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    dataset = SegDataset('/workspace/DATA/adas/road0717/train', resize=(512, 512), erode=35)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for batch_data in loader:
        a_data = batch_data[0][0]
        a_label = batch_data[1][0].squeeze(0)

        plt.subplot(121)
        data_np = np.array(a_data.permute((1, 2, 0)) * 255)
        plt.imshow(data_np)

        plt.subplot(122)
        plt.imshow(a_label)

        plt.pause(0.5)
