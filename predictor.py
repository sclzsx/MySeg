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
import json
from tqdm import tqdm
from scipy import interpolate


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

            # fg_idx = 1
            # net_output = torch.abs(output.cpu())  # NCHW
            # fg = torch.max(net_output.data, 1)[1].squeeze(0)
            # sum_map = torch.sum(net_output.squeeze(0), dim=0) + 1e-6
            # sum_map = np.array(sum_map)
            # fg_mask = np.array(fg).astype('bool')
            # select_map = np.array(net_output.squeeze(0)[fg_idx])
            # score_map = (select_map / sum_map) * fg_mask
            # prediction_np = np.where(score_map > 0.9, 1, 0)

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
            json.dump(save_dict, f, indent=2)


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
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    predicts, shows, frames, times = [], [], [], []
    names = [i.name for i in Path(args.test_videos).glob('*.*')]
    for name in names:
        video_path = os.path.join(args.test_videos, name)
        cap = cv2.VideoCapture(video_path)
        total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        save_name = name[:-4] + '_' + args.pt_dir
        save_path = save_dir + '/' + save_name + '.avi'
        out, stride = None, 1
        if partial:
            stride = int(total_frame_num * 0.2)
        if save_dir is not None and save_vid:
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'P', '4', '2'), fps, dst_size)
        stride = 1

        info = []
        info_file_name = save_dir + '/' + save_name + '.json'
        with open(info_file_name, 'w') as f:
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

                d = {}
                d.setdefault('video_name', name)
                d.setdefault('frame_id', idx)
                if 1 in prediction_np:
                    d.setdefault('foreground', 'y')
                else:
                    d.setdefault('foreground', 'n')
                info.append(d)

                dst_frame = cv2.resize(frame, dst_size)
                dst_prediction = cv2.resize(prediction_np, dst_size)
                # show_np = add_mask_to_source(dst_frame, dst_prediction, args.fg_color)
                dst_show = add_mask_to_source_multi_classes(dst_frame, dst_prediction, args.out_channels)
                # plt.imshow(dst_show)
                # plt.pause(0.5)
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
            json.dump(info, f, indent=2)

    avg_time = float(np.mean(times))
    fps = int(1 / avg_time)
    print('Processing time per frame:{}\t\tFPS:{}'.format(round(avg_time, 3), fps))
    if save_dir is None:
        return predicts, shows, frames


def save_all_negs_from_videos(net, args, save_dir=None):
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    names = [i.name for i in Path(args.test_videos).glob('*.*')]
    for name in names:
        video_path = os.path.join(args.test_videos, name)
        cap = cv2.VideoCapture(video_path)
        total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        save_name = name[:-4] + '_' + args.pt_dir
        save_path = save_dir + '/' + save_name
        stride = 1
        for idx in range(0, total_frame_num - 1, stride):  # 抛弃最后一帧才能有效保存视频
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            _, frame = cap.read()
            img_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((args.height, args.width)), transforms.ToTensor()])
            img_tensor = img_transform(frame).unsqueeze(0).cuda()
            with torch.no_grad():
                output = net(img_tensor)
                _, pred = torch.max(output.data, 1)
                prediction_np = np.array(pred.squeeze(0).cpu()).astype('uint8')
                if 1 in prediction_np:
                    cv2.imwrite(save_path + str(idx) + '.jpg', frame)


def get_score_info_from_net_output(net, batch_data, fg_id):
    with torch.no_grad():
        net_output = net(batch_data).cpu()
        score_map = torch.nn.functional.softmax(net_output, dim=1).squeeze(0)[fg_id]
        fg = torch.max(net_output.data, 1)[1].squeeze(0)

        score_map = np.array(score_map)
        fg = np.array(fg).astype('bool')

        fg_score_map = score_map * fg
        fg_score_vector = [i for i in fg_score_map.flatten() if i > 0]

        if len(fg_score_vector) > 0:
            mean = np.mean(fg_score_vector)
            median = np.median(fg_score_vector)
            return mean, median
        else:
            return None, None


def get_fg_scores_from_net_output(net, batch_data, fg_id):
    with torch.no_grad():
        net_output = net(batch_data).cpu()
        score_map = torch.nn.functional.softmax(net_output, dim=1).squeeze(0)[fg_id]
        fg = torch.max(net_output.data, 1)[1].squeeze(0)
        fg = np.array(fg).astype('uint8')

        # fg = cv2.medianBlur(fg, 5)
        # fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        # if dilate > 0:
        #     fg = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate)))

        object_scores = []
        binary, contours, hierarchy = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_mask = np.zeros(fg.shape, np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 1, -1)

            object_score_map = np.array(score_map) * contour_mask.astype('bool')

            object_score_vector = [i for i in object_score_map.flatten() if i > 0]

            assert len(object_score_vector) > 0

            object_scores.append(np.mean(object_score_vector))

        return object_scores


def get_fg_value_from_net_output(net, batch_data, fg_id, lower_bound, th_stride=0.1, measure='mean'):
    upper_bound = lower_bound + th_stride
    if upper_bound > 1:
        upper_bound = 1

    with torch.no_grad():
        net_output = net(batch_data).cpu()
        score_map = torch.nn.functional.softmax(net_output, dim=1).squeeze(0)[fg_id]
        score_map = np.array(score_map).flatten()

        interval_mask = np.where((score_map > lower_bound) & (score_map <= upper_bound), 1, 0)
        if measure == 'mean':
            return np.mean(interval_mask)
        elif measure == 'var':
            return np.var(interval_mask)
        else:
            return len(interval_mask[interval_mask > 0]) / 1000


def get_score_info(net, args, save_dir=None, fg_id=1, type='pos'):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    means, medians = [], []
    if type == 'pos':
        test_set = SegDataset(args.test_set, num_classes=args.out_channels, appoint_size=(args.height, args.width), erode=args.dilate)
        for i in range(0, len(test_set), 1):
            data, label = test_set[i]
            batch_data = data.unsqueeze(0).cuda()
            mean, median = get_score_info_from_net_output(net, batch_data, fg_id)
            if mean is not None and median is not None:
                means.append(mean)
                medians.append(median)
    else:
        stride = 200
        names = [i.name for i in Path(args.test_videos).glob('*.*')]
        for name in names:
            video_path = os.path.join(args.test_videos, name)
            cap = cv2.VideoCapture(video_path)
            total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for idx in range(0, total_frame_num - 1, stride):  # 抛弃最后一帧才能有效保存视频
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                _, frame = cap.read()
                img_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((args.height, args.width)), transforms.ToTensor()])
                batch_data = img_transform(frame).unsqueeze(0).cuda()

                mean, median = get_score_info_from_net_output(net, batch_data, fg_id)
                if mean is not None and median is not None:
                    means.append(mean)
                    medians.append(median)

    save_dict = {}
    save_dict.setdefault('mean', np.mean(means).astype('float'))
    save_dict.setdefault('median', np.median(medians).astype('float'))
    if type == 'pos':
        path = save_dir + '/score_info_from_pos.json'
    else:
        path = save_dir + '/score_info_from_neg.json'
    with open(path, 'w') as f:
        json.dump(save_dict, f, indent=2)


def get_pos_neg_thresh_curves(net, args, save_dir=None, fg_id=1, measure='mean'):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    def normalize_list(list):
        max_ = max(list)
        min_ = min(list)
        return [(i - min_) / (max_ - min_) for i in list]

    pos = []
    neg = []
    dicts = []
    th_stride = 0.1
    lower_bounds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for lower_bound in lower_bounds:
        values_pos, values_neg = [], []
        test_set = SegDataset(args.test_set, num_classes=args.out_channels, appoint_size=(args.height, args.width), erode=args.dilate)
        for i in range(0, len(test_set), 1):
            data, label = test_set[i]
            batch_data = data.unsqueeze(0).cuda()
            value = get_fg_value_from_net_output(net, batch_data, fg_id, lower_bound, th_stride=th_stride, measure=measure)
            values_pos.append(value)
        mean_pos = np.mean(values_pos).astype('float')

        stride = 200
        names = [i.name for i in Path(args.test_videos).glob('*.*')]
        for name in names:
            video_path = os.path.join(args.test_videos, name)
            cap = cv2.VideoCapture(video_path)
            total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for idx in range(0, total_frame_num - 1, stride):  # 抛弃最后一帧才能有效保存视频
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                _, frame = cap.read()
                img_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((args.height, args.width)), transforms.ToTensor()])
                batch_data = img_transform(frame).unsqueeze(0).cuda()
                value = get_fg_value_from_net_output(net, batch_data, fg_id, lower_bound, th_stride=th_stride, measure=measure)
                values_neg.append(value)
        mean_neg = np.mean(values_neg).astype('float')

        save_dict = {}
        save_dict.setdefault('interval', (lower_bound, lower_bound + th_stride))
        save_dict.setdefault('mean_pos', mean_pos)
        save_dict.setdefault('mean_neg', mean_neg)
        dicts.append(save_dict)

        pos.append(mean_pos)
        neg.append(mean_neg)

    def intergrate_list(list):
        for i in range(len(list)):
            tmp = list[:i]
            list[i] += sum(tmp)
        return list

    pos = intergrate_list(pos)
    neg = intergrate_list(neg)

    # pos = normalize_list(pos)
    # neg = normalize_list(neg)

    save_dict1 = {}
    save_dict1.setdefault('pos', pos)
    save_dict1.setdefault('neg', neg)
    dicts.append(save_dict1)

    with open(save_dir + '/' + measure + '-pos_neg.json', 'w') as f:
        json.dump(dicts, f, indent=2)

    def show_curves(x, pos, neg):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("thresh")
        lns1 = ax.plot(x, pos, '-', label='pos')
        # ax.set_ylim(0, 1)
        plt.yticks([])
        ax2 = ax.twinx()
        lns2 = ax2.plot(x, neg, '-r', label='neg')
        # ax2.set_ylim(0, 1)
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        plt.yticks([])
        plt.savefig(save_dir + '/' + measure + '-pos_neg.jpg')
        plt.show()

    show_curves(lower_bounds, pos, neg)


def normalize_list(list):
    max_ = max(list)
    min_ = min(list)
    return [(i - min_) / (max_ - min_) for i in list]


def intergrate_list(list):  # 逐区间地积分
    for i in range(len(list)):
        tmp = list[:i]
        list[i] += sum(tmp)
    return list


def get_obj_num_each_interval(obj_scores, lower_bounds, th_stride, intergrate=True):
    obj_scores = np.array(obj_scores)

    obj_num_each_interval = []  # 各个阈值区间的正样本数
    obj_score_each_interval = []  # 各个阈值区间的正样本分数
    for lower_bound in tqdm(lower_bounds):
        upper_bound = lower_bound + th_stride

        mask = np.where((obj_scores > lower_bound) & (obj_scores <= upper_bound))
        obj_this_interval = obj_scores[mask]
        if len(obj_this_interval) == 0:
            mean_score_this_interval = 0.0
        else:
            mean_score_this_interval = np.mean(obj_this_interval)  # 满足该阈值区间的正样本分数

        obj_num_this_interval = len(obj_this_interval)  # 满足该阈值区间的正样本数

        obj_num_each_interval.append(obj_num_this_interval)
        obj_score_each_interval.append(mean_score_this_interval)

    if intergrate:
        obj_num_each_interval = intergrate_list(obj_num_each_interval)  # 各阈值区间积分后的正样本数

    return normalize_list(obj_num_each_interval)  # 缩放到01区间


def draw_two_curves(x, y1, y2, label1, label2, labelx, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(labelx)
    lns1 = ax.plot(x, y1, '-', label=label1)
    ax.set_ylim(0, 1)
    plt.yticks([])
    ax2 = ax.twinx()
    lns2 = ax2.plot(x, y2, '-r', label=label2)
    ax2.set_ylim(0, 1)
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    plt.yticks([])
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def get_curves_object_detection(net, args, save_dir, fg_id=1):
    pos_scores, neg_scores = [], []  # 无视阈值，所有图片、所有检出区域的平均分数

    test_set = SegDataset(args.test_set, num_classes=args.out_channels, appoint_size=(args.height, args.width), erode=args.dilate)
    for img_id in tqdm(range(0, len(test_set), 1)):
        data, label = test_set[img_id]
        batch_data = data.unsqueeze(0).cuda()
        obj_scores = get_fg_scores_from_net_output(net, batch_data, fg_id)  # 该张图片中各个检出区域的平均分数
        pos_scores = pos_scores + obj_scores

    for name in [i.name for i in Path(args.test_videos).glob('*.*')]:
        video_path = os.path.join(args.test_videos, name)
        cap = cv2.VideoCapture(video_path)
        for idx in tqdm(range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 200)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            _, frame = cap.read()
            img_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((args.height, args.width)), transforms.ToTensor()])
            batch_data = img_transform(frame).unsqueeze(0).cuda()
            obj_scores = get_fg_scores_from_net_output(net, batch_data, fg_id)  # 该张图片中各个检出区域的平均分数
            neg_scores = neg_scores + obj_scores

    th_stride = 0.1
    lower_bounds = [i / 10 for i in range(0, 10, int(th_stride * 10))]

    pos_num_intergrate = get_obj_num_each_interval(pos_scores, lower_bounds, th_stride, intergrate=False)
    neg_num_intergrate = get_obj_num_each_interval(neg_scores, lower_bounds, th_stride, intergrate=False)
    save_path = save_dir + '/num.png'
    draw_two_curves(lower_bounds, pos_num_intergrate, neg_num_intergrate, 'pos', 'neg', 'thresh', save_path)

    pos_num_intergrate = get_obj_num_each_interval(pos_scores, lower_bounds, th_stride, intergrate=True)
    neg_num_intergrate = get_obj_num_each_interval(neg_scores, lower_bounds, th_stride, intergrate=True)
    save_path = save_dir + '/num_int.png'
    draw_two_curves(lower_bounds, pos_num_intergrate, neg_num_intergrate, 'pos', 'neg', 'thresh', save_path)


def predict_images(net, args, dst_size=(960, 540), save_dir=None):
    if not args.test_images:
        print('Test image path is not specific!')
        return

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            for file in Path(save_dir).glob('*.*'):
                os.remove(str(file))

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
            cv2.imwrite(save_dir + '/test_image-' + args.pt_dir + '-' + path.name + '.jpg', dst_show)
        else:
            plt.imshow(dst_show)
            plt.pause(0.5)
