import re
import numpy as np
import torch

def IoU(anchor, bbox):
    (x1, y1, x2, y2) = anchor
    (x3, y3, x4, y4) = bbox

    intersect_width = max(0.0, min(x2, x4) - max(x1, x3))
    intersect_height = max(0.0, min(y2, y4) - max(y1, y3))
    intersect = intersect_width * intersect_height
    return intersect / ((y2 - y1) * (x2 - x1) + (y4 - y3) * (x4 - x3) - intersect)

def parse_pbtxt(file):
    lines = open(file, 'r+').readlines()
    text = ''.join(lines)
    items = re.findall("item {([^}]*)}", text)
    return [dict(re.findall("(\w*): '*([^\n']*)'*", item)) for item in items]

def get_label_map_from_pbtxt(file):
    items = parse_pbtxt(file)
    result = {}
    for item in items:
        result[int(item['id'])] = item['name']
    return result

def get_inverse_label_map_from_pbtxt(file):
    items = parse_pbtxt(file)
    result = {}
    for item in items:
        result[item['name']] = int(item['id'])
    return result


def nms(dets, cls, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = cls

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep], cls[keep]
