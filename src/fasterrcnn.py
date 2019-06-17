import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.rpn import RPN
from src.utils import nms, IoU, parametrize, unparametrize

class FasterRCNN(nn.Module):
    INPUT_SIZE = (1600, 800)
    OUTPUT_SIZE = (100, 50)
    OUTPUT_CELL_SIZE = float(INPUT_SIZE[0]) / float(OUTPUT_SIZE[0])

    NEGATIVE_THRESHOLD = 0.3
    POSITIVE_THRESHOLD = 0.5

    def __init__(self, n_classes, model='resnet50', path='fasterrcnn_resnet50.pt', training=False):
        super(FasterRCNN, self).__init__()

        self.n_roi_sample = 128
        self.pos_ratio = 0.25
        self.pos_iou_thresh = 0.5
        self.neg_iou_thresh_hi = 0.5
        self.neg_iou_thresh_lo = 0.0

        if model == 'resnet50':
            self.in_dim = 1024
            resnet = models.resnet50(pretrained=True)
            self.feature_map = nn.Sequential(*list(resnet.children())[:-3])
        if model == 'vgg16':
            self.in_dim = 512
            vgg = models.vgg16(pretrained=True)
            self.feature_map = nn.Sequential(*list(vgg.children())[:-1])

        self.n_classes = n_classes + 1
        self.in_fc_dim = 7 * 7 * self.in_dim
        self.out_fc_dim = 1024

        rpn_path = path.replace('fasterrcnn_', '')
        self.rpn = RPN(self.in_dim)
        self.fc = nn.Linear(self.in_fc_dim, self.out_fc_dim)
        self.cls_layer = nn.Linear(self.out_fc_dim, self.n_classes)
        self.reg_layer = nn.Linear(self.out_fc_dim, self.n_classes * 4)

        self.training = training

        #initialize layers
        torch.nn.init.normal_(self.fc.weight, std=0.01)
        torch.nn.init.normal_(self.cls_layer.weight, std=0.1)
        torch.nn.init.normal_(self.reg_layer.weight, std=0.01)

        if os.path.isfile(path):
            self.load_state_dict(torch.load(path))

    def forward(self, x):
        feature_map = self.feature_map(x)
        cls, reg = self.rpn(feature_map)
        feature_map = feature_map.view((-1, self.OUTPUT_SIZE[0], self.OUTPUT_SIZE[1]))
        if self.training:
            proposals = self.rpn.get_proposals(reg, cls)
        else:
            proposals = self.rpn.get_proposals(reg, cls)

        all_cls = []
        all_reg = []
        for roi in proposals.int():
            roi[np.where(roi < 0)] = 0
            roi = roi / self.OUTPUT_CELL_SIZE
            roi_feature_map = feature_map[:, roi[0]:roi[2]+1, roi[1]:roi[3]+1]
            pooled_roi = F.adaptive_max_pool2d(roi_feature_map, (7, 7)).view((-1, 50176))
            r = F.relu(self.fc(pooled_roi))
            r_cls = self.cls_layer(r)
            r_reg = self.reg_layer(r).view((self.n_classes, 4))
            all_cls.append(r_cls)
            all_reg.append(r_reg[torch.argmax(r_cls)])

        return torch.stack(all_cls).view((-1, self.n_classes)), torch.stack(all_reg), proposals, cls, reg

    def get_target(self, proposals, bboxes, classes):
        ious = np.zeros((proposals.shape[0], len(bboxes)))
        for i in range(proposals.shape[0]):
            for j in range(len(bboxes)):
                ious[i, j] = IoU(proposals[i], bboxes[j])
        best_bbox_for_proposal = np.argmax(ious, axis=1)
        best_proposal_for_bbox = np.argmax(ious, axis=0)
        max_iou_per_proposal = np.amax(ious, axis=1)

        labels = classes[best_bbox_for_proposal]

        # truth box for each proposal
        truth_bbox_for_roi = bboxes[best_bbox_for_proposal, :]
        truth_bbox = parametrize(proposals.detach().numpy(), truth_bbox_for_roi)

        # Selecting all ious > POSITIVE_THRESHOLD
        positives = max_iou_per_proposal > self.POSITIVE_THRESHOLD
        # TODO: improve the negatives selection
        negatives = max_iou_per_proposal < self.POSITIVE_THRESHOLD
        # Assign 'other' label to negatives
        labels[negatives] = 0

        # Keep positives and negatives
        selected = np.where(positives | negatives)

        return torch.from_numpy(labels[selected]), torch.from_numpy(truth_bbox[selected])

    def get_proposals(self, reg, cls, rpn_proposals):
        # print(cls)
        # print(F.softmax(cls, dim=1))
        # print(cls.shape)
        objects = torch.argmax(F.softmax(cls, dim=1), dim=1)
        bboxes = unparametrize(rpn_proposals, reg)

        return bboxes[np.where(objects != 0)]
