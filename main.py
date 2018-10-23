import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models
from dataset import *

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

def train():
    lamb = 0.001
    rpn = RPN()
    dataset = ToothImageDataset('data')

    optimizer = optim.SGD(rpn.parameters(), lr = 0.001, momentum=0.9)

    for i in range(1, len(dataset)):
        im, reg_truth, cls_truth, positives, negatives = dataset[i]

        # zero the parameter gradients
        optimizer.zero_grad()

        cls_output, reg_output = rpn(im.float())
        print(reg_output.shape)
        print(reg_truth.shape)
        reg_loss = F.smooth_l1_loss(reg_output, reg_truth)
        cls_loss = F.cross_entropy(cls_output.view((-1, 2)), cls_truth.view(-1))
        print('cls_loss', cls_loss)
        print('reg_loss', reg_loss)
        loss = cls_loss + lamb * reg_loss
        loss.backward()
        optimizer.step()

        running_loss = loss.item()
        print('[%d] loss: %.3f' % (i + 1, loss.item() / 2000))

    print('Finished Training')

train()
