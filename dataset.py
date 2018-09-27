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

class ToothImageDataset(Dataset):
    """Dataset of dental panoramic x-rays"""

    INPUT_SIZE = (2000, 1000)
    OUTPUT_SIZE = (30, 40)
    OUTPUT_CELL_SIZE = float(INPUT_SIZE[0]) / float(OUTPUT_SIZE[0])

    # constants about receptive field for anchors
    # precalculated here https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/receptive_field
    RECEPTIVE_FIELD = 100
    EFFECTIVE_STRIDE = 32
    EFFECTIVE_PADDING = 50

    # anchors constants
    ANCHORS_WIDTH_RATIOS = [0.5, 1.0, 2.0]
    ANCHORS_HEIGHT_RATIOS = [0.5, 1.0, 2.0]

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
            anchors[i, :] = [top_x, top_y, width, height]
        return anchors

    def get_image_anchors(self):
        # TODO
        # returns something (number_anchors_wide, number_anchors_height, self.anchor_number, 4)
        number_anchors_wide = int((self.INPUT_SIZE[0] + 2 * self.EFFECTIVE_PADDING - self.RECEPTIVE_FIELD) / self.EFFECTIVE_STRIDE) + 1
        number_anchors_height = int((self.INPUT_SIZE[1] + 2 * self.EFFECTIVE_PADDING - self.RECEPTIVE_FIELD) / self.EFFECTIVE_STRIDE) + 1

        anchors = np.zeros((number_anchors_wide, number_anchors_height, self.anchor_number, 4))

        for i in range(number_anchors_wide):
            for j in range(number_anchors_height):
                anchors_pos = self.get_anchors_at_position((i, j))
                anchors[i, j, :] = anchors_pos

        return anchors

    def get_truth_bboxes(self, i):
        path = os.path.join(self.root_dir, 'Annotations', str(i) + '.xml')
        tree = ET.parse(path)
        root = tree.getroot()
        raw_boxes = [child for child in root if child.tag == 'object']
        classes = [self.inverse_label_map[c[0].text] for c in raw_boxes]
        # TODO be sure that the order is always the same (xmin, ymin, xmax, ymax)
        bboxes = [[[int(d.text) for d in c] for c in object if c.tag == 'bndbox'] for object in raw_boxes]
        return np.array(bboxes)

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

        anchors = self.get_image_anchors()

        # get list of anchors in 2dim
        anchors = anchors[10][10].reshape(-1, anchors.shape[-1])
        temp_im = Image.fromarray(image)
        im = Image.new("RGBA", temp_im.size)
        im.paste(temp_im)
        draw = ImageDraw.Draw(im)
        # for anchor in anchors:
        #     draw.rectangle([anchor[0], anchor[1], anchor[0] + anchor[2], anchor[1] + anchor[3]])

        bboxes = self.get_truth_bboxes(i)
        bboxes = bboxes.reshape(-1, bboxes.shape[-1])
        print('')
        print(bboxes)
        for bbox in bboxes:
            print(bbox)
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline = 'blue')

        im.show()

dataset = ToothImageDataset('data')
print(len(dataset))
print(dataset.label_map)
dataset.get_truth_bboxes(2)
dataset.visualise_anchors_on_image(2)
