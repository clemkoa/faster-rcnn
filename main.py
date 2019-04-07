import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import ToothImageDataset
from rpn import RPN

model = 'resnet50'
MODEL_PATH = os.path.join('models', f'{model}.pt')

def train(dataset):
    save_range = 10
    lamb = 10.0

    rpn = RPN(model=model, path=MODEL_PATH)
    optimizer = optim.SGD(rpn.parameters(), lr = 0.1)

    for i in range(1, len(dataset)):
        optimizer.zero_grad()
        im, reg_truth, cls_truth, selected_indices, positives = dataset[i]

        cls_output, reg_output = rpn(im.float())
        # only look at positive boxes for regression loss
        reg_loss = F.smooth_l1_loss(reg_output[positives], reg_truth[positives])
        # look at a sample of positive + negative boxes for classification
        cls_loss = F.binary_cross_entropy(cls_output[selected_indices], cls_truth[selected_indices].float())

        loss = cls_loss + lamb * reg_loss
        if not len(positives):
            loss = cls_loss

        loss.backward()
        optimizer.step()

        print('[%d] loss: %.5f' % (i, loss.item()))

        if i % save_range == 0:
            torch.save(rpn.state_dict(), MODEL_PATH)
    print('Finished Training')

def infer(dataset):
    with torch.no_grad():
        rpn = RPN(model=model, path=MODEL_PATH)

        for i in range(1, len(dataset)):
            im, reg_truth, cls_truth, selected_indices, positives = dataset[i]
            print(reg_truth.shape, cls_truth.shape, selected_indices.shape, positives.shape)

            cls, reg = rpn(im.float())
            dataset.visualise_proposals_on_image(reg.detach().numpy(), cls.detach().numpy(), i)
            # dataset.visualise_sampling_on_image(i)
            return

def main(args):
    dataset = ToothImageDataset('data')
    if args.infer:
        infer(dataset)
    if args.train:
        train(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-i', '--infer', action='store_true')
    args = parser.parse_args()

    main(args)
