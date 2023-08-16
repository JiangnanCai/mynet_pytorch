
import cv2

img = cv2.imread("/home/cai/Documents/object_detection/DataSet/images/train/2007_001154.jpg")
h, w = img.shape[:2]

label = []
with open('/home/cai/Documents/object_detection/DataSet/labels/train/2007_001154.txt', 'r') as f:
    for label in f:
        label = label.split(' ')
        label = [float(x.strip()) for x in label]
        print(label[0])

        pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))
        pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))

        cv2.rectangle(img, pt1, pt2, (0, 0, 255, 2), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
