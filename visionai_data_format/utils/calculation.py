from typing import Dict, Tuple

import numpy as np


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


def cart2hom(pcs_3d: np.array) -> np.array:
    """Input: nx3 points in Cartesian in Velodyne coordinate system
    Output: nx4 points in Homogeneous by pending 1
    """
    n = pcs_3d.shape[0]
    pcs_3d_hom = np.hstack([pcs_3d, np.ones((n, 1))])
    return pcs_3d_hom


def project_ref_to_velo(pcs_3d_ref: np.array, Tr_cam_to_velo: np.array) -> np.array:
    """Input: nx3 points in ref camera coord.
    Output: nx3 points in velodyne coord.
    """
    # nx3 -> nx4
    pcs_3d_ref = cart2hom(pcs_3d_ref)
    # nx4 @ 4x3 = nx3
    return np.dot(pcs_3d_ref, Tr_cam_to_velo.T)


def project_rect_to_ref(pcs_3d_rect: np.array, R0_rect: np.array) -> np.array:
    """Input: nx3 points in rect camera coord.
    Output: nx3 points in ref camera coord.
    """
    # 3x3 @ 3xn = 3xn -> nx3
    return np.dot(np.linalg.inv(R0_rect), pcs_3d_rect.T).T


def project_rect_to_velo(
    pcs_3d_rect: np.array, R0_rect: np.array, Tr_cam_to_velo: np.array
) -> np.array:
    """Input: nx3 points in rect camera coord.
    Output: nx3 points in velodyne coord.
    """
    pcs_3d_ref = project_rect_to_ref(pcs_3d_rect, R0_rect)
    return project_ref_to_velo(pcs_3d_ref, Tr_cam_to_velo)
