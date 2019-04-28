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

from src.utils import nms, get_label_map_from_pbtxt, get_inverse_label_map_from_pbtxt, unparametrize

class ToothImageDataset(Dataset):
    """Dataset of dental panoramic x-rays"""

    INPUT_SIZE = (1600, 800)

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

    def __len__(self):
        return len(self.tooth_images_paths)

    def __getitem__(self, i):
        image = self.get_image(i)
        bboxes, classes = self.get_truth_bboxes(i)
        # image input is grayscale, convert to rgb
        im = np.expand_dims(np.stack((resize(image, self.INPUT_SIZE),)*3), axis=0)
        return im, bboxes, classes

    def get_image(self, i):
        path = os.path.join(self.root_dir, 'JPEGImages', str(i) + '.png')
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return self.preprocess_image(img)

    def preprocess_image(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        cl = clahe.apply(img)
        return cl

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
        classes = [int(self.inverse_label_map[c.text]) for object in raw_boxes for c in object if c.tag == 'name']
        if not len(bboxes):
            return np.array([])

        bboxes = bboxes.reshape(-1, bboxes.shape[-1])
        for i in [0, 2]:
            bboxes[:, i] = bboxes[:, i] / width_ratio
        for i in [1, 3]:
            bboxes[:, i] = bboxes[:, i] / height_ratio
        return bboxes, classes

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

    def visualise_proposals_on_image(self, bboxes, i):
        im = self.get_resized_image(i)
        draw = ImageDraw.Draw(im)

        for bbox in bboxes:
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline = 'blue')

        im.show()
