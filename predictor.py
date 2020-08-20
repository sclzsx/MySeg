import torch
import time
import numpy as np
import cv2
import os
from torchvision import transforms
from utils import subplots, add_mask_to_source_multi_classes
from pathlib import Path
from dataset import SegDataset
from choices import get_criterion
from matplotlib import pyplot as plt
from metric import SegmentationMetric


def predict_a_batch(net, out_channels, batch_data, batch_label, class_weights, do_criterion, do_metric):
    if batch_label is None:  # 针对图片或视频帧的预测，没有对应的label，随机生成一个和data等大的label
        batch_label = torch.randn(batch_data.shape[0], out_channels, batch_data.shape[2], batch_data.shape[3])
    with torch.no_grad():
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        output = net(batch_data)
        # if batch_data.shape[3] != output.shape[3]:  # 通过H判定输出是否比输入小，主要针对deeplab系列，将输出上采样至输入大小
        #     output = F.interpolate(output, size=(batch_data.shape[2], batch_data.shape[3]), mode="bilinear", align_corners=False)
        if out_channels == 1:
            batch_label = batch_label.float()  # labels默认为long，通道为1时采用逻辑损失，需要data和label均为float
            output = torch.sigmoid(output).squeeze().cpu()  # Sigmod回归后去掉批次维N
            prediction_np = np.where(np.array(output) > 0.5, 1, 0)  # 阈值默认为0.5
        else:
            batch_label = batch_label.squeeze(1)  # 交叉熵损失需要去掉通道维C
            prediction_np = np.array(torch.max(output.data, 1)[1].squeeze(0).cpu())  # 取最大值的索引作为标签，并去掉批次维N

        loss, pa, miou = None, None, None

        criterion = get_criterion(out_channels, class_weights)
        if do_criterion:
            loss = criterion(output.cuda(), batch_label).item()

        if do_metric:
            metric = SegmentationMetric(out_channels)
            metric.update(output, batch_label)
            # metric.update(output.cuda(), batch_label.cuda())
            pa, miou = metric.get()

        return prediction_np, loss, (pa, miou)


def eval_dataset_full(net, out_channels, loader, class_weights=None, save_dir=None):
    mious, pas, losses, batch_data_shape = [], [], [], ()
    for i, (batch_data, batch_label) in enumerate(loader):
        if i == 0:
            batch_data_shape = batch_data.shape
        _, loss, (pa, miou) = predict_a_batch(net, out_channels, batch_data, batch_label, class_weights=class_weights, do_criterion=True, do_metric=True)
        losses.append(loss)
        mious.append(miou)
        pas.append(pa)
        print('Predicted batch [{}/{}], Loss:{}, IoU:{}, PA:{}'.format(i, len(loader), round(loss, 3), round(miou, 3), round(pa, 3)))
    mean_iou = round(float(np.mean(mious)), 3)
    pixel_acc = round(float(np.mean(pas)), 3)
    avg_loss = round(float(np.mean(losses)), 3)
    print('Average loss:{}, Mean IoU:{}, Pixel accuracy:{}'.format(avg_loss, mean_iou, pixel_acc))
    if save_dir is None:
        return avg_loss, (mean_iou, pixel_acc)
    else:
        from ptflops import get_model_complexity_info
        image = (batch_data_shape[1], batch_data_shape[2], batch_data_shape[3])
        GFLOPs, Parameters = get_model_complexity_info(net.cuda(), image, as_strings=True, print_per_layer_stat=False, verbose=False)
        save_dict = {}
        save_dict.setdefault('GFLOPs', GFLOPs)
        save_dict.setdefault('Parameters', Parameters)
        save_dict.setdefault('Average loss', avg_loss)
        save_dict.setdefault('Mean IoU', mean_iou)
        save_dict.setdefault('Pixel accuracy', pixel_acc)
        with open(save_dir + '/metrics.json', 'w') as f:
            import json
            json.dump(save_dict, f)


def qualitative_results_from_dataset(net, args, sample_rate=0.2, pause=0.2, save_dir=None):
    test_set = SegDataset(args.test_set, num_classes=args.out_channels, appoint_size=(args.height, args.width), erode=args.dilate)
    stride = int(len(test_set) * sample_rate)
    for i in range(0, len(test_set), stride):
        data, label = test_set[i]
        batch_data = data.unsqueeze(0).cuda()
        batch_label = label.unsqueeze(0).cuda()
        prediction_np, _, (_, _) = predict_a_batch(net, args.out_channels, batch_data, batch_label, class_weights=None, do_criterion=False, do_metric=False)
        if args.dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.dilate, args.dilate))
            prediction_np = cv2.dilate(prediction_np.astype('uint8'), kernel)

        data_transform = transforms.Compose([transforms.ToPILImage()])
        data_pil = data_transform(data)
        data_np = cv2.cvtColor(np.array(data_pil), cv2.COLOR_BGR2RGB)
        label_np = np.array(label.unsqueeze(0))

        show_np = add_mask_to_source_multi_classes(data_np, prediction_np, args.out_channels)
        save_path = save_dir + '/' + args.pt_dir + str(i) + '.png'
        subplots(data_np, label_np.squeeze(0), prediction_np, show_np, text='data/label/prediction/merge', pause=pause, save_path=save_path)
        print('Processed image', i)


def predict_videos(net, args, partial=True, dst_size=(960, 540), save_vid=False, save_dir=None):
    predicts, shows, frames, times = [], [], [], []
    names = [i.name for i in Path(args.test_videos).glob('*.*')]
    for name in names:
        video_path = os.path.join(args.test_videos, name)
        cap = cv2.VideoCapture(video_path)
        total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        save_name = name[:-4] + '_' + args.pt_dir
        save_path = args.pt_root + args.pt_dir + '/' + save_name + '.avi'
        out, stride = None, 1
        if partial:
            stride = int(total_frame_num * 0.2)
        if save_dir is not None and save_vid:
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'P', '4', '2'), fps, dst_size)

        for idx in range(0, total_frame_num - 1, stride):  # 抛弃最后一帧才能有效保存视频
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            torch.cuda.synchronize()
            _, frame = cap.read()
            start = time.time()

            img_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((args.height, args.width)), transforms.ToTensor()])
            img_tensor = img_transform(frame).unsqueeze(0)
            prediction_np, _, _ = predict_a_batch(net, args.out_channels, img_tensor, class_weights=None, batch_label=None, do_criterion=False, do_metric=False)
            prediction_np = prediction_np.astype('uint8')
            if args.dilate > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.dilate, args.dilate))
                prediction_np = cv2.dilate(prediction_np, kernel)
            dst_frame = cv2.resize(frame, dst_size)
            dst_prediction = cv2.resize(prediction_np, dst_size)
            # show_np = add_mask_to_source(dst_frame, dst_prediction, args.fg_color)
            dst_show = add_mask_to_source_multi_classes(dst_frame, dst_prediction, args.out_channels)

            torch.cuda.synchronize()
            end = time.time()
            cost_time = end - start

            if save_dir is None:
                predicts.append(dst_prediction)
                shows.append(dst_show)
                frames.append(dst_frame)

            times.append(cost_time)
            print('Processed Video:{} [{}/{}]\tTime:{}'.format(name, idx, total_frame_num, cost_time))
            if save_dir is not None:
                if save_vid:
                    out.write(dst_show)
                else:
                    cv2.imwrite(save_dir + '/' + save_name + '_frame' + str(idx) + '.jpg', dst_show)
                    plt.imshow(dst_show)
                    plt.pause(0.2)

    avg_time = float(np.mean(times))
    fps = int(1 / avg_time)
    print('Processing time per frame:{}\t\tFPS:{}'.format(round(avg_time, 3), fps))
    if save_dir is None:
        return predicts, shows, frames


def predict_images(net, args, dst_size=(960, 540), save_dir=None):
    times = []
    paths = [i for i in Path(args.test_images).glob('*.jpg')]
    for path in paths:
        frame = cv2.imread(str(path))
        start = time.time()

        img_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((args.height, args.width)), transforms.ToTensor()])
        img_tensor = img_transform(frame).unsqueeze(0)
        prediction_np, _, _ = predict_a_batch(net, args.out_channels, img_tensor, class_weights=None, batch_label=None, do_criterion=False, do_metric=False)
        prediction_np = prediction_np.astype('uint8')
        if args.dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.dilate, args.dilate))
            prediction_np = cv2.dilate(prediction_np, kernel)
        dst_frame = cv2.resize(frame, dst_size)
        dst_prediction = cv2.resize(prediction_np, dst_size)
        dst_show = add_mask_to_source_multi_classes(dst_frame, dst_prediction, args.out_channels)

        torch.cuda.synchronize()
        end = time.time()
        cost_time = end - start
        times.append(cost_time)
        print('Processed image:{}\t\tTime:{}'.format(path.name, cost_time))
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(save_dir + '/test_image-' + args.pt_dir + '-' + path.name + '.jpg', dst_show)
            # plt.imshow(dst_show)
            # plt.pause(0.2)
