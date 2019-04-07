from __future__ import print_function, division
import os
import torch
import re
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageDraw

from utils import nms, get_label_map_from_pbtxt, get_inverse_label_map_from_pbtxt, IoU, parametrize, unparametrize

def softmax(z):
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

class ToothImageDataset(Dataset):
    """Dataset of dental panoramic x-rays"""

    INPUT_SIZE = (1600, 800)
    OUTPUT_SIZE = (100, 50)
    OUTPUT_CELL_SIZE = float(INPUT_SIZE[0]) / float(OUTPUT_SIZE[0])

    ANCHOR_STANDARD_SIZE = 16

    # anchors constants
    ANCHORS_RATIOS = [0.25, 0.5, 1.0]
    ANCHORS_SCALES = [3, 4, 5]

    NUMBER_ANCHORS_WIDE = OUTPUT_SIZE[0]
    NUMBER_ANCHORS_HEIGHT = OUTPUT_SIZE[1]

    NEGATIVE_THRESHOLD = 0.3
    POSITIVE_THRESHOLD = 0.7

    ANCHOR_SAMPLING_SIZE = 256

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images under VOC format.
        """
        self.root_dir = root_dir
        self.label_map_path = os.path.join(root_dir, 'pascal_label_map.pbtxt')
        self.tooth_images_paths = os.listdir(os.path.join(root_dir, 'Annotations'))
        self.label_map = self.get_label_map(self.label_map_path)
        self.inverse_label_map = self.get_inverse_label_map(self.label_map_path)

        self.anchor_dimensions = self.get_anchor_dimensions()
        self.anchor_number = len(self.anchor_dimensions)

    def __len__(self):
        return len(self.tooth_images_paths)

    def __getitem__(self, i):
        image = self.get_image(i)
        bboxes = self.get_truth_bboxes(i)
        anchors = self.get_image_anchors()
        # image input is grayscale, convert to rgb
        im = np.expand_dims(np.stack((resize(image, self.INPUT_SIZE),)*3), axis=0)
        truth_bbox, positives, negatives = self.get_positive_negative_anchors(anchors, bboxes)
        reg_target = parametrize(anchors, truth_bbox)

        indices = np.array([i for i in range(len(anchors.reshape((-1, 4))))])
        selected_indices, positive_indices = self.get_selected_indices_sample(indices, positives, negatives)

        n = self.NUMBER_ANCHORS_WIDE * self.NUMBER_ANCHORS_HEIGHT * self.anchor_number
        cls_truth = np.zeros((n, 2))
        cls_truth[np.arange(n), positives.reshape(n).astype(int)] = 1.0
        return torch.from_numpy(im), torch.from_numpy(reg_target.reshape((-1, 4))), torch.from_numpy(cls_truth), selected_indices, positive_indices

    def get_anchor_dimensions(self):
        dimensions = []
        for r in self.ANCHORS_RATIOS:
            for s in self.ANCHORS_SCALES:
                width = s * np.sqrt(r)
                height = s * np.sqrt(1.0/r)
                dimensions.append((width, height))
        return dimensions

    def get_image(self, i):
        path = os.path.join(self.root_dir, 'JPEGImages', str(i) + '.png')
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return self.preprocess_image(img)

    def preprocess_image(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        cl = clahe.apply(img)
        return cl

    def get_selected_indices_sample(self, indices, positives, negatives):
        positive_indices = indices[positives.reshape(-1)]
        negative_indices = indices[negatives.reshape(-1)]
        random_positives = np.random.permutation(positive_indices)[:self.ANCHOR_SAMPLING_SIZE // 2]
        random_negatives = np.random.permutation(negative_indices)[:self.ANCHOR_SAMPLING_SIZE - len(random_positives)]
        selected_indices = np.concatenate((random_positives, random_negatives))
        return selected_indices, positive_indices

    def get_anchors_at_position(self, pos):
        # dimensions of anchors: (self.anchor_number, 4)
        # each anchor is [xa, ya, xb, yb]
        x, y = pos
        anchors = np.zeros((self.anchor_number, 4))
        for i in range(self.anchor_number):
            center_x = self.OUTPUT_CELL_SIZE * (float(x) + 0.5)
            center_y = self.OUTPUT_CELL_SIZE * (float(y) + 0.5)

            width = self.anchor_dimensions[i][0] * self.ANCHOR_STANDARD_SIZE
            height = self.anchor_dimensions[i][1] * self.ANCHOR_STANDARD_SIZE

            top_x = center_x - width / 2.0
            top_y = center_y - height / 2.0
            anchors[i, :] = [top_x, top_y, top_x + width, top_y + height]
        return anchors

    def get_image_anchors(self):
        anchors = np.zeros((self.NUMBER_ANCHORS_WIDE, self.NUMBER_ANCHORS_HEIGHT, self.anchor_number, 4))

        for i in range(self.NUMBER_ANCHORS_WIDE):
            for j in range(self.NUMBER_ANCHORS_HEIGHT):
                anchors_pos = self.get_anchors_at_position((i, j))
                anchors[i, j, :] = anchors_pos

        return anchors

    def get_truth_bboxes(self, i):
        path = os.path.join(self.root_dir, 'Annotations', str(i) + '.xml')
        tree = ET.parse(path)
        root = tree.getroot()

        # we need to resize the bboxes to the INPUT_SIZE
        size = root.find('size')
        height = int(size.find('height').text)
        width = int(size.find('width').text)
        width_ratio = float(width) / float(self.INPUT_SIZE[0])
        height_ratio = float(height) / float(self.INPUT_SIZE[1])

        raw_boxes = [child for child in root if child.tag == 'object']
        bboxes = np.array([[[int(d.text) for d in c] for c in object if c.tag == 'bndbox'] for object in raw_boxes])
        if not len(bboxes):
            return np.array([])

        bboxes = bboxes.reshape(-1, bboxes.shape[-1])
        for i in [0, 2]:
            bboxes[:, i] = bboxes[:, i] / width_ratio
        for i in [1, 3]:
            bboxes[:, i] = bboxes[:, i] / height_ratio
        return bboxes

    def get_positive_negative_anchors(self, anchors, bboxes):
        if not len(bboxes):
            ious = np.zeros(anchors.shape[:3])
            positives = ious > self.POSITIVE_THRESHOLD
            negatives = ious < self.NEGATIVE_THRESHOLD
            return np.array([]), positives, negatives
        anchors = anchors.reshape((-1, 4))
        ious = np.zeros((anchors.shape[0], len(bboxes)))

        # TODO improve speed with a real numpy formula
        for i in range(ious.shape[0]):
            for j in range(ious.shape[1]):
                ious[i, j] = IoU(anchors[i], bboxes[j])
        best_bbox_for_anchor = np.argmax(ious, axis=1)
        best_anchor_for_bbox = np.argmax(ious, axis=0)
        max_iou_per_anchor = np.amax(ious, axis=1)

        # truth box for each anchor
        truth_bbox = bboxes[best_bbox_for_anchor, :]

        # Selecting all ious > POSITIVE_THRESHOLD
        positives = max_iou_per_anchor > self.POSITIVE_THRESHOLD
        # Adding max iou for each ground truth box
        positives[best_anchor_for_bbox] = True
        negatives = max_iou_per_anchor < self.NEGATIVE_THRESHOLD
        return truth_bbox, positives, negatives

    def get_label_map(self, label_map_path):
        return get_label_map_from_pbtxt(label_map_path)

    def get_inverse_label_map(self, label_map_path):
        return get_inverse_label_map_from_pbtxt(label_map_path)

    def get_resized_image(self, i):
        image = self.get_image(i)
        temp_im = Image.fromarray(image).resize(self.INPUT_SIZE)
        im = Image.new('RGB', temp_im.size)
        im.paste(temp_im)
        return im

    def visualise_proposals_on_image(self, reg, cls, i):
        im = self.get_resized_image(i)

        draw = ImageDraw.Draw(im)

        anchors = self.get_image_anchors()
        bboxes = unparametrize(anchors, reg).reshape((-1, 4))
        print(cls)
        cls = softmax(cls)
        print(cls)
        cls[cls <= 0.] = 0.0
        cls = np.argmax(cls, axis=1)
        # for bbox in bboxes[np.where(cls == 1)]:
        #     draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline = 'red')

        bboxes, cls = nms(bboxes, cls, 0.6)

        for bbox in bboxes[np.where(cls == 1)]:
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline = 'blue')

        im.show()

    def visualise_anchors_on_image(self, i):
        im = self.get_resized_image(i)

        draw = ImageDraw.Draw(im)

        bboxes = self.get_truth_bboxes(i)
        for bbox in bboxes:
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline = 'blue')

        anchors = self.get_image_anchors()
        truth_bbox, positives, negatives = self.get_positive_negative_anchors(anchors, bboxes)
        for bbox in anchors[np.where(positives)]:
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline = 'green')

        im.show()
