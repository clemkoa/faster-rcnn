import pytest
import numpy as np
from utils import IoU

def test_iou():
    assert IoU([1, 1, 10, 10], [1, 1, 10, 10]) == 1.0
    assert IoU([0, 0, 10, 10], [0, 0, 10, 9]) == 0.9
    assert IoU([0, 0, 10, 10], [0, 0, 5, 5]) == 0.25
    assert IoU([0, 0, 10, 10], [20, 20, 50, 50]) == 0.0
    assert IoU([0, 0, 1, 1], [0, 0, 10, 10]) == 0.01
