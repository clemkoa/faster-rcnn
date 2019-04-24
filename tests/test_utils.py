import pytest
import torch
import numpy as np
from src.utils import IoU, parametrize, unparametrize

def test_iou():
    assert IoU([1, 1, 10, 10], [1, 1, 10, 10]) == 1.0
    assert IoU([0, 0, 10, 10], [0, 0, 10, 9]) == 0.9
    assert IoU([0, 0, 10, 10], [0, 0, 5, 5]) == 0.25
    assert IoU([0, 0, 10, 10], [20, 20, 50, 50]) == 0.0
    assert IoU([0, 0, 1, 1], [0, 0, 10, 10]) == 0.01

def test_parametrize():
    anchors = np.array([[0, 0, 10, 10]])
    bboxes = np.array([[0, 0, 10, 10]])
    assert np.array_equal(parametrize(anchors, bboxes), np.array([[0, 0, 0, 0]]))

    anchors = np.array([[0, 0, 10, 10]])
    bboxes = np.array([[0, 0, 10, 5]])
    assert np.allclose(parametrize(anchors, bboxes), np.array([[0, -0.25, 0, np.log(0.5)]]))

    anchors = np.array([[10, 10, 20, 20]])
    bboxes = np.array([[0, 0, 30, 30]])
    assert np.allclose(parametrize(anchors, bboxes), np.array([[0, 0, np.log(3), np.log(3)]]))

def test_unparametrize():
    anchors = torch.tensor([[0., 0., 10., 10.]])
    predictions = torch.tensor([[0., 0., 0., 0.]])
    assert torch.all(torch.eq(unparametrize(anchors, predictions), torch.tensor([[0., 0., 10., 10.]])))

    anchors = torch.tensor([[0., 0., 10., 10.]])
    predictions = torch.tensor([[0, -0.25, 0, np.log(0.5)]])
    assert torch.allclose(unparametrize(anchors, predictions), torch.tensor([[0., 0., 10., 5.]]))

    anchors = torch.tensor([[10., 10., 20., 20.]])
    predictions = torch.tensor([[0, 0, np.log(3), np.log(3)]])
    assert torch.allclose(unparametrize(anchors, predictions), torch.tensor([[0., 0., 30., 30.]]))
