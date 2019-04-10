import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from rpn import RPN
from utils import nms

class FasterRCNN(nn.Module):
    def __init__(self, n_classes, model='resnet50', path='resnet50.pt'):
        super(FasterRCNN, self).__init__()

        self.rpn = RPN(model=model, path=path)
        self.cls_score = nn.Linear(2048, self.n_classes)

    def forward(self, x):
        cls, reg = self.rpn(x)
