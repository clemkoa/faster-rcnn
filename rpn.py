import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class RPN(nn.Module):
    def __init__(self, model='resnet50', path='resnet50.pt'):
        super(RPN, self).__init__()

        if model == 'resnet50':
            self.in_dim = 2048
            resnet = models.resnet50(pretrained=True)
            self.feature_map = nn.Sequential(*list(resnet.children())[:-2])
        if model == 'vgg16':
            self.in_dim = 512
            vgg = models.vgg16(pretrained=True)
            self.feature_map = nn.Sequential(*list(vgg.children())[:-1])

        self.anchor_number = 9

        self.RPN_conv = nn.Conv2d(self.in_dim, 2048, 3, 1, 1)
        # cls layer
        self.cls_layer = nn.Conv2d(2048, 2* self.anchor_number, 1, 1, 0)
        # reg_layer
        self.reg_layer = nn.Conv2d(2048, 4 * self.anchor_number, 1, 1, 0)
        torch.nn.init.normal_(self.RPN_conv.weight, std=0.01)
        torch.nn.init.normal_(self.cls_layer.weight, std=0.01)
        torch.nn.init.normal_(self.reg_layer.weight, std=0.01)

        if os.path.isfile(path):
            self.load_state_dict(torch.load(path))

    def forward(self, x):
        rpn_conv = F.relu(self.RPN_conv(self.feature_map(x)), inplace=True)
        cls_output = self.cls_layer(rpn_conv)
        reg_output = self.reg_layer(rpn_conv)

        cls_output = F.sigmoid(cls_output.view(-1, 2))
        reg_output = reg_output.view(-1, 4)
        return cls_output, reg_output
