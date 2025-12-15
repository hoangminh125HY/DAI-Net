import json
import torch
from PIL import Image
import torch.utils.data as data
import numpy as np
import random
from utils.augmentations import preprocess


class WIDERDetection(data.Dataset):
    """docstring for WIDERDetection"""

    def __init__(self, list_file, mode='train'):
        super(WIDERDetection, self).__init__()
        self.mode = mode
        self.fnames = []  # Danh sách ảnh
        self.boxes = []   # Danh sách bounding boxes
        self.labels = []  # Danh sách labels

        # Đọc tệp JSON
        with open(list_file) as f:
            data = json.load(f)

        # Lặp qua các ảnh trong tệp JSON
        for image_info in data['images']:
            image_path = image_info['file_name']
            num_faces = len(data['annotations'])  # Đếm số lượng annotations cho ảnh này

            box = []
            label = []
            for annotation in data['annotations']:
                if annotation['image_id'] == image_info['id']:
                    x, y, w, h = annotation['bbox']
                    c = annotation['category_id']
                    if w <= 0 or h <= 0:
                        continue
                    box.append([x, y, x + w, y + h])
                    label.append(c)

            if len(box) > 0:
                self.fnames.append(image_path)
                self.boxes.append(box)
                self.labels.append(label)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, target, img_path, h, w = self.pull_item(index)
        return img, target, img_path

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')

            im_width, im_height = img.size
            boxes = self.annotransform(
                np.array(self.boxes[index]), im_width, im_height)
            label = np.array(self.labels[index])
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()
            img, sample_labels = preprocess(
                img, bbox_labels, self.mode, image_path)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break 
            else:
                index = random.randrange(0, self.num_samples)

        return torch.from_numpy(img), target, image_path, im_height, im_width
        

    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    paths = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        paths.append(sample[2])
    return torch.stack(imgs, 0), targets, paths


if __name__ == '__main__':
    from config import cfg
    dataset = WIDERDetection(cfg.FACE.TRAIN_FILE)  # Đảm bảo đường dẫn tệp JSON chính xác
    dataset.pull_item(14)
