import os
import shutil
import xml.etree.ElementTree as ET

classes = {
    "background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
    "bottle": 5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
    "cow": 10, "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14,
    "person": 15, "pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19,
    "tvmonitor": 20
}


def convert(size, box):
    dw, dh = 1./size[0], 1./size[1]
    x, y = (box[0] + box[1])/2.0,  (box[2] + box[3])/2.0
    w, h = box[1] - box[0], box[3] - box[2]
    x, y = x * dw, y * dh
    w, h = w * dw, h * dh
    return x, y, w, h


def voc2yolo(voc_path, yolo_save_path):
    with open(voc_path, encoding='utf-8') as r:
        tree = ET.parse(r)
    root = tree.getroot()

    size = root.find('size')
    width, height = int(size.find('width').text), int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')

        if cls not in classes.keys() or int(difficult) == 1:
            continue

        cls_id = classes[cls]
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((width, height), b)

        with open(yolo_save_path, 'a') as f:
            f.write(f"{cls_id} " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    voc_dataset_dir = os.path.join(project_dir, 'VOCtrainval_11-May-2012/VOCdevkit/VOC2012')
    voc_anno_dir = os.path.join(voc_dataset_dir, 'Annotations')
    yolo_anno_dir = os.path.join(voc_dataset_dir, 'yolo_annotations')

    if not os.path.exists(yolo_anno_dir):
        os.makedirs(yolo_anno_dir, exist_ok=True)
    else:
        shutil.rmtree(yolo_anno_dir)
        os.makedirs(yolo_anno_dir, exist_ok=True)

    for file in os.listdir(voc_anno_dir):
        anno_path = os.path.join(voc_anno_dir, file)
        yolo_path = os.path.join(yolo_anno_dir, file.replace(file.split('.')[-1], 'txt'))
        voc2yolo(anno_path, yolo_path)

