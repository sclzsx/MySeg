import os
import json
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SegDataset
from choices import choose_net
from test import eval_batch_data, multi_plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resize_tensor(tensor, size):  # Other nearest methods result in misaligned tensor.
    new_tensor = []
    with torch.no_grad():
        for label in tensor.cpu():
            label = label.float().numpy()
            label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
            new_tensor.append(np.asarray(label))
        new_tensor = torch.LongTensor(new_tensor)
    return new_tensor


def train(args):
    trainset = SegDataset(args.trainset, resize=args.resize, erode=args.erode)
    valset = SegDataset(args.trainset, resize=args.resize, erode=args.erode)
    train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)

    save_dir = './Results/' + args.net_name + '_' + args.save_suffix + '_size' + str(args.resize[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + '/args.json', 'w') as f:
        json.dump(vars(args), f)

    writer = SummaryWriter(save_dir)

    net = choose_net(args.net_name, args.out_channels).to(device)

    if args.out_channels == 1:
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(net.parameters())

    batch_num = len(train_dataloader)
    iter_cnt = 0
    for epo in range(args.epoch):
        net.train()
        for batch_id, (batch_data, batch_label) in enumerate(train_dataloader):

            if args.out_channels == 1:
                batch_label = batch_label.float()  # 逻辑损失需要label的类型和data相同，均为float，而不是long
            else:
                batch_label = batch_label.squeeze(1)  # 交叉熵只用1通道的张量，将C维删除

            optimizer.zero_grad()
            output = net(batch_data.to(device))

            if batch_label.shape[2] != output.shape[3]:  # 只比较宽W，主要针对deeplab系列，其网络的输出比输入小
                batch_label = resize_tensor(batch_label, size=(output.shape[2], output.shape[3])).to(device)

            torch.backends.cudnn.enabled = False
            loss = criterion(output, batch_label.to(device))
            torch.backends.cudnn.enabled = True

            loss.backward()
            optimizer.step()

            iter_loss = loss.item()
            print('Epoch:{}\t[{}/{}]\tTrain loss:{}'.format(epo + 1, batch_id + 1, batch_num, iter_loss))
            writer.add_scalar('Train loss', iter_loss, iter_cnt + 1)
            iter_cnt += 1

        if args.eval:
            net.eval()
            with torch.no_grad():
                test_loss = 0
                test_iou = 0
                for batch_data_t, batch_label_t in val_dataloader:
                    data_np, label_np, prediction_np, iou_tmp, loss_tmp = eval_batch_data(net, args.out_channels, device, batch_data_t, batch_label_t, criterion, args.erode)
                    if args.vis:
                        multi_plot(data_np, label_np, prediction_np, iou_tmp, save_path=None)
                    test_loss += loss_tmp
                    test_iou += iou_tmp
                ep_loss_test = test_loss / len(val_dataloader)
                ep_iou_test = test_iou / len(val_dataloader)
                print('Epoch:', epo + 1, '\tTest loss:', ep_loss_test, '\tTest iou:', ep_iou_test)
                writer.add_scalar('Test loss', ep_loss_test, epo + 1)
                writer.add_scalar('Test iou', ep_iou_test, epo + 1)

        if (epo + 1) % 1 == 0:
            save_file = save_dir + '/' + args.net_name + '_{}.pt'.format(epo + 1)
            net.train()
            torch.save(net.state_dict(), save_file)
            print('Saved checkpoint:', save_file)
    writer.close()


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out-channels", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--erode", type=int, default=25)
    parser.add_argument("--resize", type=tuple, default=(300, 300))
    parser.add_argument("--net-name", type=str, default='')
    parser.add_argument("--vis", type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False, help='Train only or do validation each epoch')
    parser.add_argument("--save-suffix", type=str, default='road_0803')
    parser.add_argument("--trainset", type=str, default='/workspace/DATA/adas/road0717/train_withneg')
    parser.add_argument("--valset", type=str, default='/workspace/DATA/adas/road0717/test')
    return parser.parse_args()


def do_train(args):
    with torch.cuda.device(args.gpu):
        train(args)


def search_train(args):
    resizes = [(224, 224), (320, 320)]
    train_net_names = ['fcn', 'fcn8', 'enet', 'enet_mod', 'lanenet', 'lanenet_deconv', 'lanenet0508']

    for net_name in train_net_names[0:2]:
        for resize in resizes:
            args.gpu = 1
            args.net_name = net_name
            args.resize = resize
            do_train(args)

    for net_name in train_net_names[2:4]:
        args.gpu = 2
        args.net_name = net_name
        do_train(args)

    for net_name in train_net_names[4:6]:
        args.gpu = 3
        args.net_name = net_name
        do_train(args)

    for net_name in train_net_names[6:7]:
        args.gpu = 4
        args.net_name = net_name
        do_train(args)


if __name__ == "__main__":
    args = get_args()
    search_experiment = True

    if search_experiment:
        search_train(args)
    else:
        do_train(args)
