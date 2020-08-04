import random
import shutil
from PIL import Image
import numpy as np
import os
from pathlib import Path
import cv2


def mkdir(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)


def split_train_val():
    dir = '//192.168.133.15/workspace/sunxin/DATA/adas/road0717'
    mask_names = [i.name for i in Path(dir + '/labels').glob('*.png')]
    random.shuffle(mask_names)
    test_mask_names = mask_names[:int(len(mask_names) * 0.1)]
    test_img = dir + '/test/images'
    test_mask = dir + '/test/labels'
    mkdir(test_img)
    mkdir(test_mask)
    for i in test_mask_names:
        print(i, i[:-3] + 'jpg')
        shutil.move(dir + '/labels/' + i, test_mask)
        shutil.move(dir + '/images/' + i[:-3] + 'jpg', test_img)


def create_hard_neg():
    hard_img_dir = 'D:/DATA/Greens2'
    save_mask_dir = 'D:/DATA/labels'
    p_mask = Path(save_mask_dir)
    if not os.path.exists(save_mask_dir):
        os.makedirs(save_mask_dir)
    for image_path in Path(hard_img_dir).glob('*.jpg'):
        name = image_path.name
        image = Image.open(str(image_path))
        size = image.size
        mask_np = np.zeros((size[1], size[0]), dtype='uint8')
        mask = Image.fromarray(mask_np.astype('uint8')).convert('L')
        mask.save(os.path.join(save_mask_dir, name[:-3] + 'png'))
        assert image.size == mask.size


def find_green_images():
    for p in Path('D:/DATA/COCO/train2017').glob('*.jpg'):
        img = cv2.imread(str(p))
        img_bgr = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        max_idxs_of_bgr = []
        for i in range(3):
            b_hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
            b_hist = [j[0] for j in b_hist]
            max_idx_of_b = b_hist.index(max(b_hist))
            max_idxs_of_bgr.append(max_idx_of_b)
        max_channel_idx = max_idxs_of_bgr.index(max(max_idxs_of_bgr))
        if max_channel_idx == 1 and max_idxs_of_bgr[max_channel_idx] > 200:
            cv2.imwrite('D:/DATA/Greens2/' + p.name, img_bgr)


if __name__ == '__main__':
    pass
