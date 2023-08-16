import random

import cv2
import numpy as np


class BasicDataAugmentation(object):

    @staticmethod
    def random_flip(img: np.ndarray, bboxes: np.ndarray):
        if random.random() < 0.5:
            return img, bboxes

        img = np.fliplr(img)
        bboxes[:, 0], bboxes[:, 1] = 1 - bboxes[:, 0], 1 - bboxes[:, 1]

        return img, bboxes

    @staticmethod
    def random_scale(img, bboxes):
        pass

    @staticmethod
    def random_blur(img):
        if random.random() < 0.5:
            return img
        kernel_size = random.choice([2, 3, 4, 5])
        bgr = cv2.blur(img, (kernel_size, kernel_size))
        return bgr

    @staticmethod
    def randon_brightness(img):
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
