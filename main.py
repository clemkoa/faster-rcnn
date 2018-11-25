import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import *
from rpn import RPN

def train():
    lamb = 0.01
    rpn = RPN()
    optimizer = optim.SGD(rpn.parameters(), lr = 0.001, momentum=0.9)

    dataset = ToothImageDataset('data')

    for i in range(1, len(dataset)):
        im, reg_truth, cls_truth, selected_indices = dataset[i]

        optimizer.zero_grad()

        cls_output, reg_output = rpn(im.float())
        reg_loss = F.smooth_l1_loss(reg_output, reg_truth)
        cls_loss = F.cross_entropy(cls_output.view((-1, 2)), cls_truth.view(-1))
        loss = cls_loss + lamb * reg_loss

        loss.backward()
        optimizer.step()

        print('[%d] loss: %.5f' % (i, loss.item()))

    print('Finished Training')

train()
