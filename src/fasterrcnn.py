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
    POSITIVE_THRESHOLD = 0.6

    def __init__(self, n_classes, model='resnet50', path='resnet50.pt'):
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

        self.n_classes = n_classes
        self.in_fc_dim = 7 * 7 * self.in_dim
        self.out_fc_dim = 1024

        self.rpn = RPN(model=model, path=path)
        self.fc = nn.Linear(self.in_fc_dim, self.out_fc_dim)
        self.cls_layer = nn.Linear(self.out_fc_dim, n_classes)
        self.reg_layer = nn.Linear(self.out_fc_dim, 4)

    def forward(self, x):
        feature_map = self.feature_map(x).view((-1, 100, 50))
        cls, reg = self.rpn(x)
        proposals = self.rpn.get_proposals(reg, cls)

        all_cls = []
        all_reg = []
        for roi in proposals.int():
            roi = roi / self.OUTPUT_CELL_SIZE
            roi_feature_map = feature_map[:, roi[0]:roi[2]+1, roi[1]:roi[3]+1]
            pooled_roi = F.adaptive_max_pool2d(roi_feature_map, (7, 7)).view((-1, 50176))
            r = self.fc(pooled_roi)
            r_cls = F.softmax(self.cls_layer(r), dim=1)
            r_reg = self.reg_layer(r)
            all_cls.append(r_cls)
            all_reg.append(r_reg)
        return all_cls, all_reg, proposals

    def get_target(self, proposals, bboxes):
        ious = np.zeros((proposals.shape[0], len(bboxes)))
        for i in range(proposals.shape[0]):
            for j in range(len(bboxes)):
                ious[i, j] = IoU(proposals[i], bboxes[j])
        best_bbox_for_proposal = np.argmax(ious, axis=1)
        best_proposal_for_bbox = np.argmax(ious, axis=0)
        max_iou_per_proposal = np.amax(ious, axis=1)

        # truth box for each proposal
        truth_bbox = bboxes[best_bbox_for_proposal, :]

        # Selecting all ious > POSITIVE_THRESHOLD
        positives = max_iou_per_proposal > self.POSITIVE_THRESHOLD
        # Adding max iou for each ground truth box
        positives[best_proposal_for_bbox] = True
        negatives = max_iou_per_proposal < self.NEGATIVE_THRESHOLD
        print(positives)
        return positives, negatives
