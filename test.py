from pathlib import Path
from argparse import ArgumentParser
import torch
import json
from choices import choose_net
from predictor import qualitative_results_from_dataset, eval_dataset_full, predict_videos, predict_images
from dataset import SegDataset, get_class_weights
from torch.utils.data import DataLoader
import cv2
import os
import xlwt
import itertools
from matplotlib import pyplot as plt
import numpy as np


def get_test_args():
    parser = ArgumentParser()
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--out-channels", type=int)
    parser.add_argument("--erode", type=int)
    parser.add_argument("--pt-dir", type=str)
    parser.add_argument("--test-videos", type=str)
    parser.add_argument("--test-set", type=str)
    parser.add_argument("--test-images", type=str)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weighting", type=str, default='none')
    parser.add_argument("--pt-root", type=str, default='./Results/')
    parser.add_argument("--vis", type=bool, default=False)
    return parser.parse_args()


def find_latest_pt(dir):
    max_num = 0
    latest_path = ''
    for path in Path(dir).glob('*.pt'):
        num = int(path.name.split('_')[-1].split('.')[0])
        if num > max_num:
            max_num = num
            latest_path = str(path)
    if latest_path != '':
        return latest_path
    else:
        print('No pts this dir:', dir)
        return None


def merge_args_from_train_json(args, json_path, verbose=False):
    if not os.path.exists(json_path):
        return args
    with open(json_path, 'r') as f:
        train_d = json.load(f)
        if verbose:
            print(train_d)
    args.weighting = train_d['weighting']
    args.dilate = train_d['erode']
    args.net_name = train_d['net_name']
    args.out_channels = train_d['out_channels']
    args.save_suffix = train_d['save_suffix']
    args.height = train_d['height']
    args.width = train_d['width']
    with open(json_path.replace('train_args', 'test_args'), 'w') as f:
        d = vars(args)
        json.dump(d, f, indent=2)
    if verbose:
        for k, v in d.items():
            print(k, v)
    return args


def do_test(mode, args):
    print('\nTesting: {}. ################################# Mode: {}'.format(args.pt_dir, mode))
    pt_dir = args.pt_root + '/' + args.pt_dir
    args = merge_args_from_train_json(args, json_path=pt_dir + '/train_args.json')
    pt_path = find_latest_pt(pt_dir)
    if pt_path is None:
        return
    print('Loading:', pt_path)
    net = choose_net(args.net_name, args.out_channels).cuda()
    net.load_state_dict(torch.load(pt_path))
    net.eval()
    test_loader, class_weights = None, None
    if mode == 0 or mode == 1:
        test_set = SegDataset(args.test_set, args.out_channels, appoint_size=(args.height, args.width), erode=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
        class_weights = get_class_weights(test_loader, args.out_channels, args.weighting)

    if mode == 0:
        eval_dataset_full(net, args.out_channels, test_loader, class_weights=class_weights, save_dir=pt_dir)

    if mode == 1:
        qualitative_results_from_dataset(net, args, pause=0, save_dir=pt_dir)

    elif mode == 2:
        predict_videos(net, args, partial=True, save_vid=False, dst_size=(960, 540), save_dir=pt_dir)

    elif mode == 3:
        predict_videos(net, args, partial=False, save_vid=True, dst_size=(960, 540), save_dir='/workspace/')

    elif mode == 4:
        predict_images(net, args, dst_size=(960, 540), save_dir='/workspace/zhatu/')


def do_search(args, task=2):
    args.pt_root = './Results/'
    pt_dirs = []

    if task == 0:
        args.test_videos = '/workspace/DATA/adas'
        args.test_set = '/workspace/DATA/adas/road0814/test'
        args.test_images = '/workspace/DATA/adas/hard_frames'
        args.out_channels = 2
        pt_dirs = [i.name for i in Path(args.pt_root).iterdir() if i.is_dir() and 'road' in i.name]

    elif task == 1:
        args.test_videos = '/workspace/DATA/adas'
        args.test_set = '/workspace/DATA/adas/roadside0813/test'
        args.test_images = '/workspace/DATA/adas/hard_frames'
        args.out_channels = 3
        pt_dirs = [i.name for i in Path(args.pt_root).iterdir() if i.is_dir() and 'roadside0817' in i.name]

    elif task == 2:
        args.test_videos = '/workspace/DATA/zhatu/videos/0807'
        args.test_set = '/workspace/DATA/zhatu/zhatu0814/test'
        args.test_images = '/workspace/DATA/zhatu/videos/0807/hard_frames'
        args.out_channels = 2
        pt_dirs = [i.name for i in Path(args.pt_root).iterdir() if i.is_dir() and 'zhatu' in i.name]

    modes = [4]
    for mode in modes:
        for pt_dir in pt_dirs:
            args.pt_dir = pt_dir
            with torch.cuda.device(0):
                try:
                    do_test(mode, args)
                except:
                    print('Error happened!')
                continue


def do_diff(args, thresh_num=6, save_root='./temp_files', task=0):
    if task == 0:
        args.test_videos = '/workspace/DATA/adas'
        args.test_set = '/workspace/DATA/adas/road0717/test'
        args.test_images = '/workspace/DATA/adas/road0717/test/images'
        base_pt_dir = './Results/good/road/lanenet0508_2_road0807_h640_w1152'
        base_predicts, base_shows, base_frames = do_test(base_pt_dir, 5, args)
    else:
        args.test_videos = '/workspace/DATA/zhatu/videos/0807'
        args.test_set = '/workspace/DATA/zhatu/zhatu0806_withneg/test'
        args.test_images = '/workspace/DATA/zhatu/zhatu0806_withneg/test/images'
        base_pt_dir = './Results/good/zhatu/fcn_road0810_h512_w512'
        base_predicts, base_shows, base_frames = do_test(base_pt_dir, 5, args)

    all_diff_info = []
    diff_pt_root = './Results/'
    pt_dirs = [
        'lanenet0508_2_road0813_h640_w1152',
        'lanenet0508_road0813_h640_w1152',
        'lanenet_mod_road0813_h640_w1152',
        'lanenet0508_2_road0812_h720_w720',
        'lanenet0508_road0812_h1024_w1024',
        'lanenet_mod_road0812_h720_w720',
        'lanenet0508_2_road0807_h640_w1152'
    ]
    for diff_pt_dir in pt_dirs:
        tmp_predicts, tmp_shows, _ = do_test(diff_pt_root + diff_pt_dir, 5, args)
        all_diff_info.append((diff_pt_dir, tmp_predicts, tmp_shows))

    # from Nets.FCN_yb import FCNs_yb
    # all_diff_info = []
    # diff_pt_root = './Results/yb/'
    # diff_pt_dirs = ['fcn-VGG-512-8']
    # for diff_pt_dir in diff_pt_dirs:
    #     args.appoint_size = (512, 512)
    #     if 'VGG' in diff_pt_dir.split('-')[1]:
    #         backbone = 'vgg'
    #     else:
    #         backbone = 'resnet18'
    #     divisor = int(diff_pt_dir.split('-')[-1])
    #     net = FCNs_yb(num_classes=2, backbone=backbone, divisor=divisor).cuda()
    #     latest_pt = find_latest_pt(diff_pt_root + diff_pt_dir)
    #     net.load_state_dict(torch.load(latest_pt, map_location='cuda:0'))
    #     net.eval()
    #     argsmode = 5
    #     tmp_predicts, tmp_shows, _ = do_test_yb(args, net)
    #     all_diff_info.append((diff_pt_dir, tmp_predicts, tmp_shows))

    num_pred = len(base_predicts)
    num_diff = len(all_diff_info)
    print('Num of diffs:{}, num of images:{}'.format(num_diff, num_pred))
    assert len(all_diff_info[0][1]) == num_pred
    assert num_diff > 0
    assert num_pred >= thresh_num
    if not save_root:
        os.makedirs(save_root)
    diff_pixels = []
    for i in range(num_pred):
        base_pred = base_predicts[i]
        diff_pixel_num = 0
        for diff_info in all_diff_info:
            diff_pred = diff_info[1][i]
            diff = cv2.absdiff(base_pred, diff_pred)
            diff[diff > 0] = 1
            diff_pixel_num += diff.sum()
        diff_pixels.append(diff_pixel_num)
    adopt_idx = np.argsort(diff_pixels)[-thresh_num:]
    print('Idx of the most different pred:', adopt_idx)
    for idx in adopt_idx:
        cv2.imwrite(save_root + '/frame_' + str(idx) + '.jpg', base_frames[idx])
        cv2.imwrite(save_root + '/base_' + str(idx) + '.jpg', base_shows[idx])
        for diff_info in all_diff_info:
            diff_name = diff_info[0]
            diff_show = diff_info[2][idx]
            path = save_root + '/' + diff_name + '_' + str(idx) + '.jpg'
            cv2.imwrite(path, diff_show)


def write_excel_for_search(root='./Results', suffix='zhatu'):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet(suffix)
    worksheet.write(0, 0, 'id')
    worksheet.write(0, 1, 'net_name')
    worksheet.write(0, 2, 'height')
    worksheet.write(0, 3, 'width')
    worksheet.write(0, 4, 'GFLOPs')
    worksheet.write(0, 5, 'MParams')
    worksheet.write(0, 6, 'erode')
    worksheet.write(0, 7, 'batch_size')
    worksheet.write(0, 8, 'epoch')
    worksheet.write(0, 9, 'avg_loss')
    worksheet.write(0, 10, 'mIoU')
    worksheet.write(0, 11, 'PA')

    dirs = [i for i in Path(root).iterdir() if i.is_dir() and suffix in i.name]
    for i, subdir in enumerate(dirs):
        with open(root + '/' + subdir.name + '/train_args.json', 'r') as f:
            dict = json.load(f)
        net_name = dict['net_name']
        h = dict['height']
        w = dict['width']
        erode = dict['erode']
        bz = dict['batch_size']
        pt_path = find_latest_pt(str(subdir))
        if pt_path is not None:
            epoch = int(pt_path.split('_')[-1].split('.')[0])
            with open(root + '/' + subdir.name + '/metrics.json', 'r') as ff:
                d = json.load(ff)
                gflops = d['GFLOPs'].split(' ')[0]
                mparams = d['Parameters'].split(' ')[0]
                avg_loss = d['Average loss']
                m_IOU = d['Mean IoU']
                PA = d['Pixel accuracy']
        else:
            epoch, gflops, mparams, avg_loss, m_IOU, PA = 0, 0, 0, 0, 0, 0

        worksheet.write(i + 1, 0, i)
        worksheet.write(i + 1, 1, net_name)
        worksheet.write(i + 1, 2, h)
        worksheet.write(i + 1, 3, w)
        worksheet.write(i + 1, 4, gflops)
        worksheet.write(i + 1, 5, mparams)
        worksheet.write(i + 1, 6, erode)
        worksheet.write(i + 1, 7, bz)
        worksheet.write(i + 1, 8, epoch)
        worksheet.write(i + 1, 9, avg_loss)
        worksheet.write(i + 1, 10, m_IOU)
        worksheet.write(i + 1, 11, PA)

    workbook.save(suffix + '.xls')


if __name__ == "__main__":
    args = get_test_args()
    search_experiment = True

    if search_experiment:
        do_search(args)
    else:
        write_excel_for_search()
