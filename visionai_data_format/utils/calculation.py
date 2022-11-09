from typing import Dict, Tuple


def xywh2xyxy(geometry: list) -> Tuple:
    h = geometry[3]
    w = geometry[2]
    x = geometry[0]
    y = geometry[1]

    x1 = x - w / 2
    x2 = x + w / 2
    y1 = y - h / 2
    y2 = y + h / 2
    return x1, y1, x2, y2


def xyxy2xywh(geometry: Dict) -> Tuple:

    x1 = geometry["x1"]
    y1 = geometry["y1"]
    x2 = geometry["x2"]
    y2 = geometry["y2"]

    w = x2 - x1
    h = y2 - y1
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2

    return x, y, w, h
