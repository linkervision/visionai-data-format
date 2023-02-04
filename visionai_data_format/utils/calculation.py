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


def gen_intervals(intervals: list[int]) -> list[Dict]:

    intervals.sort()

    new_intervals = [(intervals[0], intervals[0])]

    for frame_num in intervals:
        last_start, last_end = new_intervals[-1]
        if last_start <= frame_num <= last_end:
            continue
        if frame_num > last_end and frame_num - last_end == 1:
            new_intervals[-1] = (last_start, frame_num)
        elif frame_num < last_start and last_start - frame_num == 1:
            new_intervals[-1] = (frame_num, last_end)
        else:
            new_intervals.append((frame_num, frame_num))
    interval_result_list: list[Dict] = [
        {
            "frame_start": start,
            "frame_end": end,
        }
        for start, end in new_intervals
    ]

    return interval_result_list
