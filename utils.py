import re
import numpy as np

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
