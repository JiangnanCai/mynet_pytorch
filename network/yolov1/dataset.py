import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from module.data_augment import BasicDataAugmentation
import os
import numpy as np
import cv2


class Voc2012(Dataset):
    images_root = '/home/cai/Documents/object_detection/DataSet/images/'
    labels_root = '/home/cai/Documents/object_detection/DataSet/labels/'

    def __init__(self,
                 is_train,
                 keyword='train',  # train or val
                 img_size=448,
                 num_grids=7,
                 num_bboxes=2,
                 num_classes=20):
        super(Voc2012, self).__init__()

        self.is_train = is_train
        self.keyword = keyword
        self.image_size = img_size
        self.S = num_grids
        self.B = num_bboxes
        self.C = num_classes

        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        self.mean = np.array(mean_rgb, dtype=np.float32)

        self.to_tensor = transforms.ToTensor()

        self.image_paths = [os.path.join(self.images_root + self.keyword, image_path)
                            for image_path in os.listdir(self.images_root + self.keyword)]

        self.labels = self.get_flatten_labels()

    def get_flatten_labels(self):
        labels = []
        for image_path in self.image_paths:
            label_path = os.path.join(self.labels_root + self.keyword, image_path.split('/')[-1].split('.')[0] + '.txt')
            with open(label_path, 'r') as r:
                lines = r.readlines()
            lines = np.array([list(map(float, line.strip('\n').split())) for line in lines])  # 从txt里提取出来转成float矩阵
            labels.append(lines)
        return labels

    def __getitem__(self, item):
        img = cv2.imread( self.image_paths[item])
        cls, bboxes = self.labels[item][:, 0], self.labels[item][:, 1:]

        if self.is_train:
            # img, bboxes = BasicDataAugmentation.random_flip(img, bboxes)
            img = BasicDataAugmentation.random_blur(img)
            img = BasicDataAugmentation.randon_brightness(img)

        inputs = self.label2input(cls, bboxes)

        # for debug ----------------------------------------------------------------------------------------------------
        # height, width = img.shape[:2]
        # for i in range(self.S):
        #     for j in range(self.S):
        #         if inputs[i, j, 4] == 1.:
        #             print(i, j, inputs[i, j, :])
        #             c_x, c_y, w, h = inputs[i, j, :][0] * width, inputs[i, j, :][1] * height, inputs[i, j, :][2] * width, inputs[i, j, :][3] * height
        #             x1, y1 = int(c_x - 1/2 * w), int(c_y - 1/2 * h)
        #             x2, y2 = int(c_x + 1/2 * w), int(c_y + 1/2 * h)
        #             img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # --------------------------------------------------------------------------------------------------------------

        img = cv2.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - self.mean) / 255.0
        img = self.to_tensor(img)

        return img, inputs

    def __len__(self):
        return len(self.image_paths)

    def label2input(self, cls, bboxes):
        # 从标注的数据格式转成输入的7x7x30
        S, B, C = self.S, self.B, self.C
        N = 5 * B + C
        inputs = torch.zeros(S, S, N)
        grids_size = 1. / float(S)
        for num_obj in range(bboxes.shape[0]):
            bbox, label = bboxes[num_obj], int(cls[num_obj]) - 1
            ij = np.ceil(bbox[:2] / grids_size) - 1.0
            i, j = int(ij[0]), int(ij[1])
            for k in range(B):
                s = 5 * k
                inputs[j, i, s:s+2] = torch.FloatTensor(bbox[:2])
                inputs[j, i, s+2:s+4] = torch.FloatTensor(bbox[2:4])
                inputs[j, i, s+4] = 1.0
            inputs[j, i, 5*B + label] = 1.0
        return inputs


if __name__ == '__main__':
    voc = Voc2012(is_train=True, keyword='train')
    data_loader = DataLoader(voc, batch_size=1, shuffle=False, num_workers=0)
    data_iter = iter(data_loader)
    for i in range(100):
        image, target = next(data_iter)
        print(image.size(), target.size())  # 应该是 torch.Size([1, 3, 448, 448]) torch.Size([1, 7, 7, 30])
