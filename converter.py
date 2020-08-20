import random
import shutil
from PIL import Image
import numpy as np
import sys
import os
from pathlib import Path
import cv2
import json
import os
# import warnings
from PIL import Image
# import yaml
from labelme import utils
import base64
from pathlib import Path
import PIL
from matplotlib import pyplot as plt
from utils import add_mask_to_source_multi_classes
import collections
import Augmentor


def labelme_jsons_to_masks():
    # json_dir = '/workspace/DATA/zhatu/zhatu0806/jsons'
    # image_dir = '/workspace/DATA/zhatu/zhatu0806/images'
    # out_dir = '/workspace/DATA/zhatu/zhatu0806/labels'
    json_dir = '/workspace/DATA/adas/road0717/roadside/V1.1'
    image_dir = '/workspace/DATA/adas/road0717/roadsize/image'
    out_dir = '/workspace/DATA/adas/road0717/roadside/labels'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    json_paths = [i for i in Path(json_dir).rglob('*.json')]
    print(len(json_paths))
    for json_path in json_paths:
        file_name = json_path.name[:-5]
        data = json.load(open(str(json_path)))
        try:
            imageData = data.get('imageData')
            if not imageData:
                image_path = os.path.join(image_dir, data['imagePath'])
                with open(image_path, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
                img = utils.img_b64_to_arr(imageData)
                lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data['shapes'])
                Image.fromarray(lbl).save(os.path.join(out_dir, '{}.png'.format(file_name)))
        except OSError:
            print(json_path)
            pass
        continue


def check_label_in_jsons():
    classes = ('background', 'roads', 'roadside')
    # classes = ('background', 'roads', 'roadside', 'ground_mark', 'zebra_crs')

    image_dir = '/workspace/DATA/adas/roadside0813/done/image'
    json_dir = '/workspace/DATA/adas/roadside0813/done/V1.1'

    out_images_dir = '/workspace/DATA/adas/roadside0813/train/images'
    out_labels_dir = '/workspace/DATA/adas/roadside0813/train/labels'

    # if not os.path.exists(out_images_dir):
    #     os.makedirs(out_labels_dir)
    # if not os.path.exists(out_labels_dir):
    #     os.makedirs(out_labels_dir)

    class_dict = dict(zip(classes, range(len(classes))))
    json_paths = [i for i in Path(json_dir).rglob('*.json')]
    for json_path in json_paths:
        file_name = json_path.name[:-5]
        image = cv2.imread(image_dir + '/' + file_name + '.jpg')
        # cv2.imwrite(image_dir + '/' + file_name + '.jpg', image)
        shutil.copy(image_dir + '/' + file_name + '.jpg', out_images_dir)
        with open(str(json_path), 'r') as f:
            data = json.load(open(str(json_path)))
        top_layer = np.zeros((1080, 1920), dtype='uint8')
        bottom_layer = np.zeros((1080, 1920), dtype='uint8')
        print(file_name)
        for shape in data['shapes']:
            class_name = shape['label']
            if 'zebra' in class_name:
                class_name = 'zebra_crs'
            if class_name in classes:
                mask = PIL.Image.fromarray(np.zeros((1080, 1920), dtype=np.uint8))
                draw = PIL.ImageDraw.Draw(mask)
                xy = [tuple(point) for point in shape['points']]
                draw.polygon(xy=xy, outline=1, fill=1)
                mask = np.array(mask, dtype=bool)
                layer_tmp = np.full((1080, 1920), class_dict[class_name], dtype='uint8') * mask
                if class_name == 'roads':
                    bottom_layer += layer_tmp
                else:
                    top_layer += layer_tmp
        full_mask = top_layer + bottom_layer * (~top_layer.astype('bool'))
        # print(file_name, collections.Counter(np.array(full_mask).flatten()))
        full_mask = Image.fromarray(full_mask.astype('uint8')).convert('L')
        full_mask.save(os.path.join(out_labels_dir, file_name + '.png'))
        # show = add_mask_to_source_multi_classes(image, full_mask, len(classes))
        # print(show.shape)
        # t = Image.open(os.path.join(out_labels_dir, file_name + '.png'))
        # plt.imshow(t)
        # plt.pause(0.5)


def labelme_jsons_to_masks2():
    image_dir = '/workspace/DATA/adas/road0717/roadside/image'
    json_dir = '/workspace/DATA/adas/road0717/roadside/V1.1'
    out_image_dir = '/workspace/DATA/adas/road0717/roadside/images'
    out_dir = '/workspace/DATA/adas/road0717/roadside/labels'
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # if not os.path.exists(out_image_dir):
    #     os.makedirs(out_image_dir)
    json_paths = [i for i in Path(json_dir).rglob('*.json')]
    print(len(json_paths))
    for json_path in json_paths:
        file_name = json_path.name[:-5]
        data = json.load(open(str(json_path)))
        img_shape = (1080, 1920, 3)
        lbl, lbl_names = utils.shape.labelme_shapes_to_label(img_shape, data['shapes'])
        Image.fromarray(lbl).save(os.path.join(out_dir, '{}.png'.format(file_name)))
        shutil.copy(image_dir + '/' + file_name + '.jpg', out_image_dir)


def mkdir(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)


def split_train_val():
    dir = '/workspace/DATA/adas/roadside0813/train'
    mask_names = [i.name for i in Path(dir + '/labels').glob('*.png')]
    random.shuffle(mask_names)
    test_mask_names = mask_names[:int(len(mask_names) * 0.05)]
    test_img = dir + '/test/images'
    test_mask = dir + '/test/labels'
    mkdir(test_img)
    mkdir(test_mask)
    for i in test_mask_names:
        print(i, i[:-3] + 'jpg')
        shutil.move(dir + '/labels/' + i, test_mask)
        shutil.move(dir + '/images/' + i[:-3] + 'jpg', test_img)


def create_hard_neg():
    # hard_img_dir = '/workspace/DATA/zhatu/negs/total_neg_images'
    # save_mask_dir = '/workspace/DATA/zhatu/negs/total_neg_masks'
    # # p_mask = Path(save_mask_dir)
    # if not os.path.exists(save_mask_dir):
    #     os.makedirs(save_mask_dir)
    # for image_path in Path(hard_img_dir).glob('*.jpg'):
    #     name = image_path.name
    #     image = Image.open(str(image_path))
    #     size = image.size
    #     mask_np = np.zeros((size[1], size[0]), dtype='uint8')
    #     mask = Image.fromarray(mask_np.astype('uint8')).convert('L')
    #     mask.save(os.path.join(save_mask_dir, name[:-3] + 'png'))
    #     assert image.size == mask.size

    neg_images_source_dir = '/workspace/DATA/coco/green_train'
    save_root = '/workspace/DATA/adas/road0814'
    neg_images_dir = save_root + '/neg/images'
    neg_labels_dir = save_root + '/neg/labels'
    mkdir(neg_images_dir)
    mkdir(neg_labels_dir)
    for neg_img_path in Path(neg_images_source_dir).glob('*.jpg'):
        neg_img = cv2.imread(str(neg_img_path))
        neg_mask = np.zeros(neg_img.shape[:2], dtype='uint8')
        shutil.copy(str(neg_img_path), neg_images_dir)
        neg_mask = Image.fromarray(neg_mask.astype('uint8')).convert('L')
        neg_mask.save(os.path.join(neg_labels_dir, neg_img_path.name[:-3] + 'png'))
        print('Saved from:', neg_img_path)


def find_green_images(save_dir='/workspace/DATA/coco/green_train', source_dir='/workspace/DATA/coco/coco/train2017', resize=False):
    mkdir(save_dir)
    for p in Path(source_dir).glob('*.jpg'):
        img = cv2.imread(str(p))
        if img.shape[2] == 3:
            if resize:
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            max_idxs_of_bgr = []
            for i in range(3):
                b_hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                b_hist = [j[0] for j in b_hist]
                max_idx_of_b = b_hist.index(max(b_hist))
                max_idxs_of_bgr.append(max_idx_of_b)
            max_channel_idx = max_idxs_of_bgr.index(max(max_idxs_of_bgr))
            if max_channel_idx == 1 and max_idxs_of_bgr[max_channel_idx] > 200:
                if resize:
                    cv2.imwrite(save_dir + '/' + p.name, img)
                else:
                    shutil.copy(str(p), save_dir)


def del_tmp_files(root):
    pngs = [i for i in Path(root).rglob('*.png')]
    jpgs = [i for i in Path(root).rglob('*.jpg')]
    avis = [i for i in Path(root).rglob('*.avi')]
    for f in pngs:
        os.remove(str(f))
    for f in jpgs:
        os.remove(str(f))
    for f in avis:
        os.remove(str(f))


def merge_images_into_video():
    root = '/workspace/DATA/adas/road0717/test/images'
    img_list = [str(i) for i in Path(root).iterdir() if i.is_file()]
    video = '/workspace/DATA/adas/road0717/test/testset.avi'
    video_w, video_h = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter(video, fourcc, 10, (video_w, video_h))
    for img in img_list:
        frame = cv2.imread(img)
        vw.write(frame)
        print('write', img)
    vw.release()


def convert_labelme_to_masks_super(class_info, image_dir, json_dir, save_root, hw=(1080, 1920)):
    mkdir(save_root)

    ######################check all classes#######################
    all_classes = set()
    json_paths = [i for i in Path(json_dir).rglob('*.json')]
    for json_path in json_paths:
        with open(str(json_path), 'r') as f:
            data = json.load(open(str(json_path)))
            for shape in data['shapes']:
                class_name = shape['label']
                all_classes.add(class_name)
    print('All classes of these jsons:', all_classes)
    # sys.exit()

    ######################convert jsons to masks#######################
    all_masks_dir = save_root + '/all_masks'
    mkdir(all_masks_dir)
    adopt_classes = class_info['adopt_classes']
    assert len(adopt_classes[0]) > 0
    dispute_classes = class_info['dispute_classes']
    if dispute_classes is not None:
        assert len(dispute_classes) == 2
    top_classes = class_info['top_classes']
    class_dict = dict(zip(adopt_classes, range(1, len(adopt_classes) + 1)))
    print('Adopt classes:', class_dict)
    json_paths = [i for i in Path(json_dir).rglob('*.json')]
    for json_path in json_paths:
        file_name = json_path.name[:-5]
        with open(str(json_path), 'r') as f:
            data = json.load(open(str(json_path)))
        top_layer = np.zeros(hw, dtype='uint8')
        bottom_layer = np.zeros(hw, dtype='uint8')
        for shape in data['shapes']:
            class_name = shape['label']
            if class_name in dispute_classes:
                class_name = dispute_classes[0]
            if class_name in adopt_classes:
                mask = PIL.Image.fromarray(np.zeros(hw, dtype=np.uint8))
                draw = PIL.ImageDraw.Draw(mask)
                xy = [tuple(point) for point in shape['points']]
                draw.polygon(xy=xy, outline=1, fill=1)
                mask = np.array(mask, dtype=bool)
                layer_tmp = np.full(hw, class_dict[class_name], dtype='uint8') * mask
                if class_name in top_classes:
                    top_layer += layer_tmp
                else:
                    bottom_layer += layer_tmp
        full_mask = top_layer + bottom_layer * (~top_layer.astype('bool'))
        counter = collections.Counter(full_mask.flatten())
        # total_label = sum([v for k, v in counter.items()])
        # assert total_label == hw[0] * hw[1]
        if len(counter) - 1 > len(adopt_classes):
            print('Error file and NOT USE it:', file_name, counter)
        else:
            print('OK file:', file_name, counter)
            full_mask_pil = Image.fromarray(full_mask.astype('uint8')).convert('L')
            save_path = os.path.join(all_masks_dir, file_name + '.png')
            full_mask_pil.save(save_path)
            vis = False
            if vis:
                image = cv2.imread(image_dir + '/' + file_name + '.jpg')
                show = add_mask_to_source_multi_classes(image, full_mask, len(adopt_classes))
                plt.subplot(131)
                plt.imshow(show)
                plt.subplot(132)
                plt.imshow(full_mask)
                plt.subplot(133)
                plt.imshow(Image.open(save_path))
                plt.pause(0.5)


def split_dataset_and_add_neg(image_dir, mask_dir, save_root, neg_image_dir=None):
    if neg_image_dir is not None:
        train_images_dir = save_root + '/train_withneg/images'
        train_labels_dir = save_root + '/train_withneg/labels'
    else:
        train_images_dir = save_root + '/train/images'
        train_labels_dir = save_root + '/train/labels'
    test_images_dir = save_root + '/test/images'
    test_labels_dir = save_root + '/test/labels'
    mkdir(train_images_dir)
    mkdir(train_labels_dir)
    mkdir(test_images_dir)
    mkdir(test_labels_dir)

    ######################create test set#######################
    mask_names = [i.name for i in Path(mask_dir).glob('*.png')]
    random.shuffle(mask_names)
    test_mask_names = mask_names[:int(len(mask_names) * 0.1)]
    for test_name in test_mask_names:
        print('Moving name to testset:', test_name)
        shutil.copy(mask_dir + '/' + test_name, test_labels_dir)
        try:
            shutil.copy(image_dir + '/' + test_name[:-3] + 'jpg', test_images_dir)
        except FileNotFoundError:
            pass
        continue

    ######################create train set#######################
    if neg_image_dir is not None:
        for neg_img_path in Path(neg_image_dir).glob('*.jpg'):
            neg_img = cv2.imread(str(neg_img_path))
            shutil.copy(str(neg_img_path), train_images_dir)
            neg_mask = np.zeros(neg_img.shape[:2], dtype='uint8')
            neg_mask_pil = Image.fromarray(neg_mask)
            neg_mask_pil.save(train_labels_dir + '/' + neg_img_path.name[:-3] + 'png')
    for mask_path in Path(mask_dir).glob('*.png'):
        if mask_path.name not in test_mask_names:
            print('Moving name to trainset:', mask_path.name)
            shutil.copy(str(mask_path), train_labels_dir)
            try:
                shutil.copy(image_dir + '/' + mask_path.name[:-3] + 'jpg', train_images_dir)
            except FileNotFoundError:
                pass
            continue

    delete_unmatched_file(test_images_dir, test_labels_dir)
    delete_unmatched_file(train_images_dir, train_labels_dir)


def check(dir):
    for path in Path(dir).glob('*.*'):
        data = Image.open(str(path))
        print(len(data.split()))
        plt.imshow(data)
        plt.pause(0.5)


def delete_unmatched_file(data_dir, label_dir):
    data_names = [i.name[:-3] for i in Path(data_dir).glob('*.jpg')]
    label_names = [i.name[:-3] for i in Path(label_dir).glob('*.png')]
    adopt_names = [i for i in data_names if i in label_names]
    for name in data_names:
        if name not in adopt_names:
            print('Delete:', name)
            os.remove(data_dir + '/' + name + 'jpg')
    for name in label_names:
        if name not in adopt_names:
            print('Delete:', name)
            os.remove(label_dir + '/' + name + 'png')


def augmentation(label_dir):
    def do_aug(img_and_output_dir, label_dir):
        assert len([i for i in Path(label_tmp_dir).glob('*.*')]) == 1
        p = Augmentor.Pipeline(img_and_output_dir)
        p.ground_truth(label_dir)
        p.zoom_random(probability=0.6, percentage_area=0.9)
        p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
        p.flip_left_right(probability=0.5)
        p.zoom_random(probability=0.6, percentage_area=0.7)
        p.sample(20)

    data_tmp_dir = '/workspace/trash/data_tmp'
    label_tmp_dir = '/workspace/trash/label_tmp'
    mkdir(data_tmp_dir)
    mkdir(label_tmp_dir)
    for i in Path(data_tmp_dir).glob('*.*'):
        os.remove(str(i))
    for i in Path(label_tmp_dir).glob('*.*'):
        os.remove(str(i))

    for label_path in Path(label_dir).glob('*.png'):
        data_tmp = cv2.imread(str(label_path).replace('labels', 'images')[:-3] + 'jpg')
        cv2.imwrite(data_tmp_dir + '/' + label_path.name, data_tmp)
        shutil.copy(str(label_path), label_tmp_dir)

        do_aug(data_tmp_dir, label_tmp_dir)
        for path in Path(data_tmp_dir + '/output').glob('*.*'):
            if 'groundtruth' in path.name:
                new_name = path.name.split('tmp_')[-1]
                shutil.move(str(path), new_name)
                shutil.move(new_name, '/workspace/DATA/adas/roadside0813/train_aug/labels')
            else:
                new_name = path.name.split('original_')[-1][:-3] + 'jpg'
                shutil.move(str(path), new_name)
                shutil.move(new_name, '/workspace/DATA/adas/roadside0813/train_aug/images')

        for i in Path(data_tmp_dir).rglob('*.*'):
            os.remove(str(i))
        for i in Path(label_tmp_dir).rglob('*.*'):
            os.remove(str(i))


if __name__ == '__main__':
    pass
    # del_tmp_files('./Results')
    # merge_images_into_video()
    # labelme_jsons_to_masks2()
    # split_train_val()
    # create_hard_neg()
    # check_label_in_jsons()
    # find_green_images()
    # delete_unmatched_file(data_dir='/workspace/DATA/adas/roadside0813/train/images', label_dir='/workspace/DATA/adas/roadside0813/train/labels')
    # check('/workspace/DATA/adas/road0814/test/labels')
    # augmentation('/workspace/DATA/adas/roadside0813/train_aug/labels')

    # class_info = {'adopt_classes': ('roads',), 'dispute_classes': ('zebra_crs', 'zebra-crs'), 'top_classes': ('roadside',)}
    image_dir = '/workspace/DATA/adas/road0717/images'
    json_dir = '/workspace/DATA/adas/road0717/done'
    save_root = '/workspace/DATA/adas/road0814_2'
    # convert_labelme_to_masks_super(class_info=class_info, image_dir=image_dir, json_dir=json_dir, save_root=save_root)
    neg_image_dir = '/workspace/DATA/coco/green_train'
    mask_dir = '/workspace/DATA/adas/road0814/all_masks'
    split_dataset_and_add_neg(image_dir=image_dir, mask_dir=mask_dir, save_root=save_root, neg_image_dir=None)

    # class_info = {'adopt_classes': ('soil',), 'dispute_classes': ('soil', 'little soil'), 'top_classes': ('soil',)}
    # image_dir = '/workspace/DATA/zhatu/zhatu0814_api/image'
    # json_dir = '/workspace/DATA/zhatu/zhatu0814_api/V1.1'
    # save_root = '/workspace/DATA/zhatu/zhatu0814'
    # # convert_labelme_to_masks_super(class_info=class_info, image_dir=image_dir, json_dir=json_dir, save_root=save_root)
    # neg_image_dir = '/workspace/DATA/zhatu/negs/total_neg_images'
    # mask_dir = '/workspace/DATA/zhatu/zhatu0814/all_masks'
    # split_dataset_and_add_neg(image_dir=image_dir, mask_dir=mask_dir, save_root=save_root, neg_image_dir=neg_image_dir)

    # class_info = {'adopt_classes': ('roads', 'roadside'), 'dispute_classes': ('zebra_crs', 'zebra-crs'), 'top_classes': ('roadside',)}
    # image_dir = '/workspace/DATA/adas/roadside0813/done/image'
    # json_dir = '/workspace/DATA/adas/roadside0813/done/V1.1'
    # save_root = '/workspace/DATA/adas/roadside0813'
    # convert_labelme_to_masks_super(class_info=class_info, image_dir=image_dir, json_dir=json_dir, save_root=save_root)
    # neg_image_dir = None
    # mask_dir = '/workspace/DATA/adas/roadside0813/all_masks'
    # split_dataset_and_add_neg(image_dir=image_dir, mask_dir=mask_dir, save_root=save_root, neg_image_dir=neg_image_dir)
