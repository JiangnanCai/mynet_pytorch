import random

import cv2
import numpy as np


class BasicDataAugmentation(object):

    @staticmethod
    def random_flip(img: np.ndarray, bboxes: np.ndarray):
        # 随机翻转
        case = random.choice([1, 2, 3, 4])
        if case == 1:
            return img, bboxes
        elif case == 2:
            img = np.fliplr(img)
            bboxes[:, 0] = 1 - bboxes[:, 0]
        elif case == 3:
            img = np.flipud(img)
            bboxes[:, 1] = 1 - bboxes[:, 1]
        else:
            img = np.flipud(np.fliplr(img))
            bboxes[:, 0], bboxes[:, 1] = 1 - bboxes[:, 0], 1 - bboxes[:, 1]
        img = np.ascontiguousarray(img)
        return img, bboxes

    @staticmethod
    def random_scale(img, bboxes):
        if random.random() < 0.5:
            return img, bboxes

        scale = random.uniform(0.7, 1.3)
        h, w = img.shape[:2]
        img = cv2.resize(img, dsize=(int(w * scale), h), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        bboxes[:, 0], bboxes[:, 2] = bboxes[:, 0], bboxes[:, 2] * scale

        max_size = max(h, w)
        img = cv2.copyMakeBorder(img, int((max_size-h)/2), int((max_size-h)/2),
                                 int((max_size-w)/2), int((max_size-w)/2),
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])

        bboxes[:, 1], bboxes[:, 3] = bboxes[:, 1] * h / max_size + (max_size-h)/(2*max_size), bboxes[:, 3] * h / max_size
        bboxes[:, 0], bboxes[:, 2] = bboxes[:, 0] * w / max_size + (max_size-w)/(2*max_size), bboxes[:, 2] * w / max_size
        return img, bboxes

    @staticmethod
    def random_blur(img):
        if random.random() < 0.5:
            return img
        kernel_size = random.choice([2, 3, 4, 5])
        bgr = cv2.blur(img, (kernel_size, kernel_size))
        return bgr

    @staticmethod
    def random_brightness(img):
        if random.random() < 0.5:
            return img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    @staticmethod
    def random_saturation(img):
        if random.random() < 0.5:
            return img

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr
