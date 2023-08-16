import os
import random
import shutil

from tqdm import tqdm


def split_img(img_path, label_path, split_list):
    Data = './DataSet'

    if not os.path.exists(Data):
        os.makedirs(Data, exist_ok=True)
    else:
        shutil.rmtree(Data)
        os.makedirs(Data, exist_ok=True)

    train_img_dir = Data + '/images/train'
    val_img_dir = Data + '/images/val'
    test_img_dir = Data + '/images/test'

    train_label_dir = Data + '/labels/train'
    val_label_dir = Data + '/labels/val'
    test_label_dir = Data + '/labels/test'

    # 创建文件夹
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    train, val, test = split_list
    all_img = os.listdir(img_path)
    all_img_path = [os.path.join(img_path, img) for img in all_img if img.split('.')[0]+'.txt' in os.listdir(label_path)]

    train_img = random.sample(all_img_path, int(train * len(all_img_path)))
    train_label = [toLabelPath(img, label_path) for img in train_img]
    for i in tqdm(range(len(train_img)), desc='train ', ncols=80, unit='img'):
        _copy(train_img[i], train_img_dir)
        _copy(train_label[i], train_label_dir)
        all_img_path.remove(train_img[i])

    val_img = random.sample(all_img_path, int(val / (val + test) * len(all_img_path)))
    val_label = [toLabelPath(img, label_path) for img in val_img]
    for i in tqdm(range(len(val_img)), desc='val ', ncols=80, unit='img'):
        _copy(val_img[i], val_img_dir)
        _copy(val_label[i], val_label_dir)
        all_img_path.remove(val_img[i])

    test_img = all_img_path
    test_label = [toLabelPath(img, label_path) for img in test_img]
    for i in tqdm(range(len(test_img)), desc='test ', ncols=80, unit='img'):
        _copy(test_img[i], test_img_dir)
        _copy(test_label[i], test_label_dir)


def _copy(from_path, to_path):
    shutil.copy(from_path, to_path)


def toLabelPath(img_path, label_path):
    img = img_path.split('/')[-1]
    label = img.split('.jpg')[0] + '.txt'
    return os.path.join(label_path, label)


def main():
    img_path = '/home/cai/Documents/object_detection/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages'
    label_path = '/home/cai/Documents/object_detection/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/yolo_annotations'
    split_list = [0.7, 0.2, 0.1]  # 数据集划分比例[train:val:test]
    split_img(img_path, label_path, split_list)


if __name__ == '__main__':
    main()
