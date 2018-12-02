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
    lamb = 10.0
    rpn = RPN()
    optimizer = optim.Adagrad(rpn.parameters(), lr = 0.0001)

    dataset = ToothImageDataset('data')

    for i in range(1, len(dataset)):
        im, reg_truth, cls_truth, selected_indices, positives = dataset[i]

        cls_output, reg_output = rpn(im.float())
        reg_loss = F.smooth_l1_loss(reg_output[positives], reg_truth[positives])
        cls_loss = F.cross_entropy(cls_output.view((-1, 2))[selected_indices], cls_truth.view(-1)[selected_indices])
        loss = cls_loss.mean() + lamb * reg_loss.mean()
        if not len(positives):
            loss = cls_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[%d] loss: %.5f' % (i, loss.item()))

    print('Finished Training')

train()
