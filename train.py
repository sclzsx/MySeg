import os
import json
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
import torch
from apex import amp
from torch.utils.data import DataLoader
from dataset import SegDataset, get_class_weights
from choices import choose_net, get_criterion, get_optimizer
from predictor import eval_dataset_full, predict_images


def get_train_args():
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--net-name", type=str)
    parser.add_argument("--save-suffix", type=str)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--out-channels", type=int)
    parser.add_argument("--erode", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--train-set", type=str)
    parser.add_argument("--val-set", type=str)
    parser.add_argument("--test-images", type=str)
    parser.add_argument("--f16", type=str, default=False)
    parser.add_argument("--train-aug", type=str, default=False)
    parser.add_argument("--op-name", type=str, default='adam')
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--weighting", type=str, default='none')
    parser.add_argument("--eval", type=bool, default=False)
    return parser.parse_args()


def train(args):
    # Prepare training set
    train_set = SegDataset(args.train_set, num_classes=args.out_channels, appoint_size=(args.height, args.width), erode=args.erode, aug=args.train_aug)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train_class_weights = get_class_weights(train_dataloader, out_channels=args.out_channels, weighting=args.weighting)
    if args.eval:
        val_set = SegDataset(args.val_set, num_classes=args.out_channels, appoint_size=(args.height, args.width), erode=0)
        val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
        val_class_weights = get_class_weights(val_dataloader, out_channels=args.out_channels, weighting=args.weighting)
    else:
        val_dataloader, val_class_weights = None, None

    # Prepare save dir
    save_dir = './Results/' + args.save_suffix + '-' + args.net_name + '-h' + str(train_set[0][0].shape[1]) + 'w' + str(train_set[0][0].shape[2]) + '-erode' + str(args.erode) + '-weighting_' + str(
        args.weighting)
    print('Save dir is:{}  Input size is:{}'.format(save_dir, train_set[0][0].shape))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + '/train_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Prepare network
    writer = SummaryWriter(save_dir)
    val_dicts = []
    net = choose_net(args.net_name, args.out_channels).cuda()
    train_criterion = get_criterion(args.out_channels, class_weights=train_class_weights)
    optimizer = get_optimizer(net, args.op_name)

    if args.f16:
        model, optimizer = amp.initialize(net, optimizer, opt_level="O1")  # 这里是“欧一”，不是“零一”

    # Begin to train
    iter_cnt = 0
    for epo in range(args.epoch):
        net.train()
        batch_num = len(train_dataloader)
        for batch_id, (batch_data, batch_label) in enumerate(train_dataloader):
            if args.out_channels == 1:
                batch_label = batch_label.float()  # 逻辑损失需要label的类型和data相同，均为float，而不是long
            else:
                batch_label = batch_label.squeeze(1)  # 交叉熵label的类型采用默认的long，但需要去除C通道维
            optimizer.zero_grad()
            output = net(batch_data.cuda())
            loss = train_criterion(output, batch_label.cuda())
            iter_loss = loss.item()
            print('Epoch:{} Batch:[{}/{}] Train loss:{}'.format(epo + 1, str(batch_id + 1).zfill(3), batch_num, round(iter_loss, 4)))
            writer.add_scalar('Train loss', iter_loss, iter_cnt)
            iter_cnt += 1
            if args.f16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        if args.eval:
            v_loss, (miou, pa) = eval_dataset_full(net.eval(), args.out_channels, val_dataloader, class_weights=val_class_weights, save_dir=None)
            writer.add_scalar('Val loss', v_loss, epo + 1)
            writer.add_scalar('Val miou', miou, epo + 1)
            writer.add_scalar('Val pa', pa, epo + 1)
            val_dict_tmp = {}
            val_dict_tmp.setdefault('epoch', epo + 1)
            val_dict_tmp.setdefault('loss', v_loss)
            val_dict_tmp.setdefault('miou', miou)
            val_dict_tmp.setdefault('pa', pa)
            val_dicts.append(val_dict_tmp)

        if (epo + 1) == args.epoch or (epo + 1) % 25 == 0 or epo == 0:
            save_file = save_dir + '/' + args.net_name + '_{}.pt'.format(epo + 1)
            torch.save(net.state_dict(), save_file)
            print('Saved checkpoint:', save_file)

    writer.close()
    with open(save_dir + '/val_log.json', 'w') as f2:
        json.dump(val_dicts, f2, indent=2)

    if args.eval:
        predict_images(net, args, dst_size=(960, 540), save_dir=save_dir)


def do_train(args):
    with torch.cuda.device(args.gpu):
        train(args)


def get_choices(args, task):
    sizes = [(528, 960)]
    weightings = ['none']

    if task == 0:  # road
        erodes = [15]
        args.train_set = '/workspace/DATA/adas/road0814/train_withneg'
        args.val_set = '/workspace/DATA/adas/road0814/test'

    elif task == 1:  # roadside
        erodes = [0]
        args.train_set = '/workspace/DATA/adas/roadside0813/train_withneg'
        args.val_set = '/workspace/DATA/adas/roadside0813/test'

    else:  # zhatu
        erodes = [5]
        args.train_set = '/workspace/DATA/zhatu/zhatu0814/train_withneg'
        args.val_set = '/workspace/DATA/zhatu/zhatu0814/test'
    return args, sizes, erodes, weightings


def search_train(args):
    args.out_channels = 2
    args.batch_size = 64
    args.epoch = 50

    if args.gpu == 1:
        train_net_names = ['mobilenetv3_small_fpn']
        save_suffix = 'zhatu0820'
        args, sizes, erodes, weightings = get_choices(args, task=2)
    elif args.gpu == 2:
        train_net_names = ['mobilenetv3_small_panet']
        save_suffix = 'zhatu0820'
        args, sizes, erodes, weightings = get_choices(args, task=2)
    elif args.gpu == 3:
        train_net_names = ['mobilenetv3_small_bifpn']
        save_suffix = 'zhatu0820'
        args, sizes, erodes, weightings = get_choices(args, task=2)

    else:
        train_net_names = ['mobilenetv3_small_fpn', 'mobilenetv3_small_panet', 'mobilenetv3_small_bifpn']
        save_suffix = 'road0820'
        args, sizes, erodes, weightings = get_choices(args, task=0)

    for net_name in train_net_names:
        for size in sizes:
            for erode in erodes:
                for weighting in weightings:
                    args.weighting = weighting
                    args.erode = erode
                    args.net_name = net_name
                    args.height = size[0]
                    args.width = size[1]
                    args.save_suffix = save_suffix

                    do_train(args)
                    # try:
                    #     do_train(args)
                    # except:
                    #     pass
                    # continue


if __name__ == "__main__":
    args = get_train_args()
    search_experiment = True

    if search_experiment:
        search_train(args)
    else:
        do_train(args)
