from __future__ import print_function, division
import os
import torch
import re
import xml.etree.ElementTree as ET
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageDraw
from utils import *

class ToothImageDataset(Dataset):
    """Dataset of dental panoramic x-rays"""

    INPUT_SIZE = (800, 400)
    OUTPUT_SIZE = (30, 40)
    OUTPUT_CELL_SIZE = float(INPUT_SIZE[0]) / float(OUTPUT_SIZE[0])

    # constants about receptive field for anchors
    # precalculated here https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/receptive_field
    RECEPTIVE_FIELD = 100
    EFFECTIVE_STRIDE = 32
    EFFECTIVE_PADDING = 239

    # anchors constants
    ANCHORS_WIDTH_RATIOS = [0.5, 1.0, 2.0]
    ANCHORS_HEIGHT_RATIOS = [0.5, 1.0, 2.0]

    NUMBER_ANCHORS_WIDE = int((INPUT_SIZE[0] + 2 * EFFECTIVE_PADDING - RECEPTIVE_FIELD) / EFFECTIVE_STRIDE) + 1
    NUMBER_ANCHORS_HEIGHT = int((INPUT_SIZE[1] + 2 * EFFECTIVE_PADDING - RECEPTIVE_FIELD) / EFFECTIVE_STRIDE) + 1

    NEGATIVE_THRESHOLD = 0.3
    POSITIVE_THRESHOLD = 0.7

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images under VOC format.
        """
        self.root_dir = root_dir
        self.label_map_path = os.path.join(root_dir, 'pascal_label_map.pbtxt')
        self.tooth_images_paths = os.listdir(os.path.join(root_dir, 'Annotations'))
        self.label_map = self.get_label_map()
        self.inverse_label_map = self.get_inverse_label_map()

        self.anchor_dimensions = self.get_anchor_dimensions()
        self.anchor_number = len(self.anchor_dimensions)

    def __len__(self):
        return len(self.tooth_images_paths)

    def __getitem__(self, idx):
        """
        Sample format: dict {image, anchors, cls_results, p_star}
        """
        # TODO
        # Get the image
        # get the positive and negative anchors

        image = self.get_image(idx)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        return sample

    def get_anchor_dimensions(self):
        dimensions = []
        for w in self.ANCHORS_WIDTH_RATIOS:
            for h in self.ANCHORS_HEIGHT_RATIOS:
                dimensions.append((w, h))
        return dimensions

    def get_image(self, i):
        path = os.path.join(self.root_dir, 'JPEGImages', str(i) + '.png')
        return io.imread(path)

    def get_anchors_at_position(self, pos):
        """
        position (x, y)
        """
        # returns something (self.anchor_number, 4)
        # each anchor is (x, y, w, h)
        x, y = pos
        anchors = np.zeros((self.anchor_number, 4))
        for i in range(self.anchor_number):
            center_x = self.OUTPUT_CELL_SIZE * (float(x) + 0.5)
            center_y = self.OUTPUT_CELL_SIZE * (float(y) + 0.5)

            width = self.anchor_dimensions[i][0] * self.RECEPTIVE_FIELD
            height = self.anchor_dimensions[i][1] * self.RECEPTIVE_FIELD

            top_x = center_x - width / 2.0
            top_y = center_y - height / 2.0
            anchors[i, :] = [top_x, top_y, top_x + width, top_y + height]
        return anchors

    def get_image_anchors(self):
        # TODO
        # returns something (self.NUMBER_ANCHORS_WIDE, self.NUMBER_ANCHORS_HEIGHT, self.anchor_number, 4)
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

        # we need to resize the bboxes according to the INPUT_SIZE
        size = root.find('size')
        height = int(size.find('height').text)
        width = int(size.find('width').text)
        width_ratio = float(width) / float(self.INPUT_SIZE[0])
        height_ratio = float(height) / float(self.INPUT_SIZE[1])

        raw_boxes = [child for child in root if child.tag == 'object']
        classes = [self.inverse_label_map[c[0].text] for c in raw_boxes]
        # TODO be sure that the order is always the same (xmin, ymin, xmax, ymax)
        bboxes = np.array([[[int(d.text) for d in c] for c in object if c.tag == 'bndbox'] for object in raw_boxes])
        bboxes = bboxes.reshape(-1, bboxes.shape[-1])
        for i in [0, 2]:
            bboxes[:, i] = bboxes[:, i] / width_ratio
        for i in [1, 3]:
            bboxes[:, i] = bboxes[:, i] / height_ratio
        return bboxes

    def get_positive_anchors(self, anchors, bboxes):
        ious = np.zeros((self.NUMBER_ANCHORS_WIDE, self.NUMBER_ANCHORS_HEIGHT, self.anchor_number, len(bboxes)))
        for i in range(anchors.shape[0]):
            for j in range(anchors.shape[1]):
                for n in range(anchors.shape[2]):
                    for b in range(len(bboxes)):
                        ious[i, j, n, b] = IoU(anchors[i, j, n], bboxes[b])
        print(ious)
        max_iou_per_anchor = np.amax(ious, axis=3)
        positives = max_iou_per_anchor > self.POSITIVE_THRESHOLD
        negatives = max_iou_per_anchor < self.NEGATIVE_THRESHOLD
        print(positives)
        print(np.max(max_iou_per_anchor))
        pos = np.unravel_index(np.argmax(max_iou_per_anchor), max_iou_per_anchor.shape)
        return anchors[pos]

    def get_label_map(self):
        #TODO: read the pbtxt file instead of hardcoding values
        label_map = {
            1: 'root',
            2: 'implant',
            3: 'restoration',
            4: 'endodontic',
        }
        return label_map

    def get_inverse_label_map(self):
        #TODO: read the pbtxt file instead of hardcoding values
        inverse_label_map = {
            'root': 1,
            'implant': 2,
            'restoration': 3,
            'endodontic': 4,
        }
        return inverse_label_map

    def visualise_anchors_on_image(self, i):
        image = self.get_image(i)
        temp_im = Image.fromarray(image).resize(self.INPUT_SIZE)
        im = Image.new("RGBA", temp_im.size)
        im.paste(temp_im)

        draw = ImageDraw.Draw(im)

        anchors = self.get_image_anchors()
        anchors = anchors.reshape(-1, anchors.shape[-1])
        # for anchor in anchors:
        #     draw.rectangle([anchor[0], anchor[1], anchor[2], anchor[3]], outline="red")

        bboxes = self.get_truth_bboxes(i)
        for bbox in bboxes:
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline = 'blue')

        bbox = self.get_positive_anchors(self.get_image_anchors(), bboxes)
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline = 'green')

        im.show()

dataset = ToothImageDataset('data')
print(len(dataset))
print(dataset.label_map)
dataset.get_truth_bboxes(2)
dataset.visualise_anchors_on_image(2)
