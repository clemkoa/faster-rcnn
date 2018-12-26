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
    model = 'resnet50'
    MODEL_PATH = f'{model}.pt'
    rpn = RPN(model=model)
    rpn.load_state_dict(torch.load(MODEL_PATH))
    optimizer = optim.Adagrad(rpn.parameters(), lr = 0.00005)

    dataset = ToothImageDataset('data')

    for i in range(1, len(dataset)):
        im, reg_truth, cls_truth, selected_indices, positives = dataset[i]

        cls_output, reg_output = rpn(im.float())
        # only look at positive boxes for regression loss
        reg_loss = F.smooth_l1_loss(reg_output[positives], reg_truth[positives])
        # look at a sample of positive + negative boxes for classification
        cls_loss = F.cross_entropy(cls_output.view((-1, 2))[selected_indices], cls_truth.view(-1)[selected_indices])
        loss = cls_loss.mean() + lamb * reg_loss.mean()
        if not len(positives):
            loss = cls_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[%d] loss: %.5f' % (i, loss.item()))

        if i % 10 == 0:
            torch.save(rpn.state_dict(), MODEL_PATH)
    print('Finished Training')

def infer():
    MODEL_PATH = 'test.pt'
    rpn = RPN()
    rpn.load_state_dict(torch.load(MODEL_PATH))

    dataset = ToothImageDataset('data')

    for i in range(1, len(dataset)):
        im, reg_truth, cls_truth, selected_indices, positives = dataset[i]

        cls, reg = rpn(im.float())
        dataset.visualise_proposals_on_image(reg.detach().numpy(), cls.detach().numpy(), i)

# infer()
train()
