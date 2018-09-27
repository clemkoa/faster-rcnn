import numpy as np

def IoU(anchor, bbox):
    (x1, y1, x2, y2) = anchor
    (x3, y3, x4, y4) = bbox

    intersect_width = max(0.0, min(x2 - 1, x4 - 1) - max(x1, x3))
    intersect_height = max(0.0, min(y2 - 1, y4 - 1) - max(y1, y3))
    intersect = intersect_width * intersect_height
    return intersect / ((y3 - y1) * (x3 - x1) + (y4 - y2) * (x4 - x2) - intersect)
