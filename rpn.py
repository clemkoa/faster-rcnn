import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

        self.in_dim = 512
        self.anchor_number = 9
        self.lamb = 0.001

        # get the feature map from last conv layer
        resnet = models.vgg16(pretrained=True)
        self.feature_map = nn.Sequential(*list(resnet.children())[:-1])
        self.RPN_conv = nn.Conv2d(self.in_dim, 512, 3, 1, 1, bias=True)

        # cls layer
        self.cls_layer = nn.Conv2d(512, 2* self.anchor_number, 1, 1, 0)
        # reg_layer
        self.reg_layer = nn.Conv2d(512, 4 * self.anchor_number, 1, 1, 0)

    def forward(self, x):
        rpn_conv = F.relu(self.RPN_conv(self.feature_map(x)))
        cls_output = self.cls_layer(rpn_conv)
        reg_output = self.reg_layer(rpn_conv)

        cls_output = cls_output.view(50, 25, 9, 2)
        reg_output = reg_output.view(50, 25, 9, 4)
        return cls_output, reg_output
