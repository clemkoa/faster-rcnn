import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from src.utils import nms, IoU, parametrize, unparametrize

class RPN(nn.Module):
    INPUT_SIZE = (1600, 800)
    OUTPUT_SIZE = (100, 50)
    OUTPUT_CELL_SIZE = float(INPUT_SIZE[0]) / float(OUTPUT_SIZE[0])

    # anchors constants
    ANCHORS_RATIOS = [0.25, 0.5, 0.9]
    ANCHORS_SCALES = [4, 6, 8]

    NUMBER_ANCHORS_WIDE = OUTPUT_SIZE[0]
    NUMBER_ANCHORS_HEIGHT = OUTPUT_SIZE[1]

    NEGATIVE_THRESHOLD = 0.3
    POSITIVE_THRESHOLD = 0.6

    ANCHOR_SAMPLING_SIZE = 256

    NMS_THRESHOLD = 0.5
    PRE_NMS_MAX_PROPOSALS = 6000
    POST_NMS_MAX_PROPOSALS = 100

    def __init__(self, model='resnet50', path='resnet50.pt'):
        super(RPN, self).__init__()

        if model == 'resnet50':
            self.in_dim = 1024
            resnet = models.resnet50(pretrained=True)
            self.feature_map = nn.Sequential(*list(resnet.children())[:-3])
        if model == 'vgg16':
            self.in_dim = 512
            vgg = models.vgg16(pretrained=True)
            self.feature_map = nn.Sequential(*list(vgg.children())[:-1])

        self.anchor_dimensions = self.get_anchor_dimensions()
        self.anchor_number = len(self.anchor_dimensions)
        mid_layers = 1024
        self.RPN_conv = nn.Conv2d(self.in_dim, mid_layers, 3, 1, 1)
        # cls layer
        self.cls_layer = nn.Conv2d(mid_layers, 2  * self.anchor_number, 1, 1, 0)
        # reg_layer
        self.reg_layer = nn.Conv2d(mid_layers, 4 * self.anchor_number, 1, 1, 0)

        #initialize layers
        torch.nn.init.normal_(self.RPN_conv.weight, std=0.01)
        torch.nn.init.normal_(self.cls_layer.weight, std=0.01)
        torch.nn.init.normal_(self.reg_layer.weight, std=0.01)

        if os.path.isfile(path):
            self.load_state_dict(torch.load(path))

    def forward(self, x):
        rpn_conv = F.relu(self.RPN_conv(self.feature_map(x)), inplace=True)
        # permute dimensions
        cls_output = self.cls_layer(rpn_conv).permute(0, 2, 3, 1).contiguous().view(1, -1, 2)
        reg_output = self.reg_layer(rpn_conv).permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

        cls_output = F.softmax(cls_output.view(-1, 2), dim=1)
        reg_output = reg_output.view(-1, 4)
        return cls_output, reg_output

    def get_target(self, bboxes):
        anchors = self.get_image_anchors()
        truth_bbox, positives, negatives = self.get_positive_negative_anchors(anchors, bboxes)
        reg_target = parametrize(anchors, truth_bbox)

        n = len(anchors)
        indices = np.array([i for i in range(n)])
        selected_indices, positive_indices = self.get_selected_indices_sample(indices, positives, negatives)

        cls_truth = np.zeros((n, 2))
        cls_truth[np.arange(n), positives.astype(int)] = 1.0
        return torch.from_numpy(reg_target), torch.from_numpy(cls_truth), selected_indices, positive_indices

    def get_anchor_dimensions(self):
        dimensions = []
        for r in self.ANCHORS_RATIOS:
            for s in self.ANCHORS_SCALES:
                width = s * np.sqrt(r)
                height = s * np.sqrt(1.0 / r)
                dimensions.append((width, height))
        return dimensions

    def get_anchors_at_position(self, pos):
        # dimensions of anchors: (self.anchor_number, 4)
        # each anchor is [xa, ya, xb, yb]
        x, y = pos
        anchors = np.zeros((self.anchor_number, 4))
        for i in range(self.anchor_number):
            center_x = self.OUTPUT_CELL_SIZE * (float(x) + 0.5)
            center_y = self.OUTPUT_CELL_SIZE * (float(y) + 0.5)

            width = self.anchor_dimensions[i][0] * self.OUTPUT_CELL_SIZE
            height = self.anchor_dimensions[i][1] * self.OUTPUT_CELL_SIZE

            top_x = center_x - width / 2.0
            top_y = center_y - height / 2.0
            anchors[i, :] = [top_x, top_y, top_x + width, top_y + height]
        return anchors

    def get_proposals(self, reg, cls):
        objects = torch.argmax(cls, dim=1)
        anchors = torch.from_numpy(self.get_image_anchors()).float()
        bboxes = unparametrize(anchors, reg)

        cls = cls.detach().numpy()
        cls = cls[np.where(objects == 1)][:self.PRE_NMS_MAX_PROPOSALS]
        bboxes = bboxes[np.where(objects == 1)][:self.PRE_NMS_MAX_PROPOSALS]
        keep = nms(bboxes.detach().numpy(), cls[:, 1].ravel(), self.NMS_THRESHOLD)
        cls = cls[keep[:self.POST_NMS_MAX_PROPOSALS]]
        return bboxes[keep[:self.POST_NMS_MAX_PROPOSALS]]

    def get_training_proposals(self, truth_bboxes, reg, cls):
        objects = np.argmax(cls, axis=1)

        anchors = self.get_image_anchors()
        predicted_bboxes = unparametrize(anchors, reg).reshape((-1, 4))
        print(predicted_bboxes.shape)
        truth_bbox, positives, negatives = self.get_positive_negative_anchors(anchors, bboxes)

        cls = cls[np.where(objects == 1)][:self.PRE_NMS_MAX_PROPOSALS]
        predicted_bboxes = predicted_bboxes[np.where(objects == 1)][:self.PRE_NMS_MAX_PROPOSALS]

        keep = nms(predicted_bboxes, cls[:, 1].ravel(), self.NMS_THRESHOLD)
        cls = cls[keep[:self.POST_NMS_MAX_PROPOSALS]]
        return predicted_bboxes[keep[:self.POST_NMS_MAX_PROPOSALS]]

    def get_image_anchors(self):
        anchors = np.zeros((self.NUMBER_ANCHORS_WIDE, self.NUMBER_ANCHORS_HEIGHT, self.anchor_number, 4))

        for i in range(self.NUMBER_ANCHORS_WIDE):
            for j in range(self.NUMBER_ANCHORS_HEIGHT):
                anchors_pos = self.get_anchors_at_position((i, j))
                anchors[i, j, :] = anchors_pos

        return anchors.reshape((-1, 4))

    def get_positive_negative_anchors(self, anchors, bboxes):
        if not len(bboxes):
            ious = np.zeros(anchors.shape[:3])
            positives = ious > self.POSITIVE_THRESHOLD
            negatives = ious < self.NEGATIVE_THRESHOLD
            return np.array([]), positives, negatives

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

    def get_selected_indices_sample(self, indices, positives, negatives):
        positive_indices = indices[positives]
        negative_indices = indices[negatives]
        random_positives = np.random.permutation(positive_indices)[:self.ANCHOR_SAMPLING_SIZE // 2]
        random_negatives = np.random.permutation(negative_indices)[:self.ANCHOR_SAMPLING_SIZE - len(random_positives)]
        selected_indices = np.concatenate((random_positives, random_negatives))
        return selected_indices, positive_indices
