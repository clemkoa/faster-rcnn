from __future__ import print_function, division
import os
import torch
import re
import xml.etree.ElementTree as ET
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ToothImageDataset(Dataset):
    """Dataset of dental panoramic x-rays"""

    INPUT_SIZE = (480, 640)

    # constants about receptive field for anchors
    # precalculated here https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/receptive_field
    RECEPTIVE_FIELD = 1027
    EFFECTIVE_STRIDE = 32
    EFFECTIVE_PADDING = 513

    # anchors constants
    ANCHORS_WIDTHS = [0.5, 1.0, 2.0]
    ANCHORS_HEIGHTS = [0.5, 1.0, 2.0]

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


    def __len__(self):
        return len(self.tooth_images_paths)

    def __getitem__(self, idx):
        """
        Sample format: dict {image, anchors, cls_results, p_star}
        """
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        return sample

    def get_image(self, i):
        path = os.path.join(self.root_dir, 'JPEGImages', str(i) + '.png')
        return io.imread(path)

    def get_anchors_at_position(pos):
        return pos

    def pave_image_with_anchors(self, image):
        # TODO
        number_anchors_wide = (self.INPUT_SIZE[0] + 2 * self.EFFECTIVE_PADDING - self.RECEPTIVE_FIELD) / self.EFFECTIVE_STRIDE
        number_anchors_height = (self.INPUT_SIZE[1] + 2 * self.EFFECTIVE_PADDING - self.RECEPTIVE_FIELD) / self.EFFECTIVE_STRIDE
        return []

    def get_truth_bboxes(self, i):
        path = os.path.join(self.root_dir, 'Annotations', str(i) + '.xml')
        tree = ET.parse(path)
        root = tree.getroot()
        raw_boxes = [child for child in root if child.tag == 'object']
        classes = [self.inverse_label_map[c[0].text] for c in raw_boxes]
        # TODO be sure that the order is always the same (xmin, ymin, xmax, ymax)
        bboxes = [[[int(d.text) for d in c] for c in object if c.tag == 'bndbox'] for object in raw_boxes]
        return bboxes

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


dataset = ToothImageDataset('data')
print(len(dataset))
print(dataset.label_map)
dataset.get_truth_bboxes(2)
dataset.get_truth_bboxes(3)
dataset.get_truth_bboxes(4)
dataset.get_truth_bboxes(5)
dataset.get_truth_bboxes(6)
