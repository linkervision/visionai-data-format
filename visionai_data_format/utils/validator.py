import json
import logging
import os
from typing import Dict, Union

import numpy as np

from visionai_data_format.schemas.bdd_schema import BDDSchema
from visionai_data_format.schemas.coco_schema import COCO
from visionai_data_format.schemas.visionai_schema import VisionAIModel

logger = logging.getLogger(__name__)


def validate_vai(data: Dict) -> Union[VisionAIModel, None]:
    try:
        vai = VisionAIModel(**data)
        logger.info("[validated_vai] Validate success")
        return vai
    except Exception as e:
        logger.error("[validated_vai] Validate failed : " + str(e))
        return None


def validate_bdd(data: Dict) -> Union[BDDSchema, None]:
    try:
        bdd = BDDSchema(**data)
        logger.info("[validate_bdd] Validation success")
        return bdd
    except Exception as e:
        logger.error("[validate_bdd] Validation failed : " + str(e))
        return None


def validate_coco(data: Dict) -> Union[COCO, None]:
    try:
        bdd = COCO(**data)
        logger.info("[validate_coco] Validation success")
        return bdd
    except Exception as e:
        logger.error("[validate_coco] Validation failed : " + str(e))
        return None


def attribute_generator(
    category: str, attribute: Dict, ontology_class_attrs: Dict
) -> Dict:
    if not attribute:
        return dict()

    new_attribute = dict()
    category = category.upper()
    for attr_name, attr_value in attribute.items():
        logger.info(f"attr_name : {attr_name}")
        logger.info(f"attr_value : {attr_value}")
        if attr_name in ontology_class_attrs[category]:
            new_attribute[attr_name] = attr_value

    logger.info(f"[datarow_attribute_generator] new_attribute : {new_attribute}")
    return new_attribute


def save_as_json(data: Dict, file_name: str, folder_name: str = "") -> None:
    try:
        if folder_name:
            os.makedirs(folder_name, exist_ok=True)
        file = open(os.path.join(folder_name, file_name), "w")
        logger.info(
            f"[save_as_json] Save file to {os.path.join(folder_name,file_name)} started "
        )
        json.dump(data, file)
        logger.info(
            f"[save_as_json] Save file to {os.path.join(folder_name,file_name)} success"
        )

    except Exception as e:
        logger.error("[save_as_json] Save file failed : " + str(e))


def read_calib_data(calib_path: str) -> dict[str, np.array]:
    data = {}
    with open(calib_path, encoding="utf8") as f:
        calib_data = f.readlines()
        for line in calib_data:
            if not len(line) or line == "\n":
                continue
            key, value = line.split(":")
            data[key] = np.array([float(val) for val in value.split()])
    return data


def inverse_transformation_matrix(Tr: np.array):
    """Inverse a rigid body transform matrix (3x4 as [R|t])
    [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)
    inv_Tr[:, :3] = Tr[:, :3].T
    inv_Tr[:, 3] = np.dot(-Tr[:, :3].T, Tr[:, 3])
    return inv_Tr  # 3x4


def parse_calib_data(calib_path: str) -> dict[str, Union[list, np.array]]:
    dict_calib: dict[str, np.array] = read_calib_data(calib_path=calib_path)
    R0_rect = (
        None
        if dict_calib is None or dict_calib.get("R0_rect", None) is None
        else dict_calib["R0_rect"].reshape((3, 3))
    )
    Tr_velo_to_cam = (
        None
        if dict_calib is None or dict_calib.get("Tr_velo_to_cam", None) is None
        else dict_calib["Tr_velo_to_cam"].reshape((3, 4))
    )

    Tr_cam_to_velo = (
        None
        if Tr_velo_to_cam is None
        else inverse_transformation_matrix(Tr_velo_to_cam)
    )
    P = (
        None
        if dict_calib is None or dict_calib.get("P2", None) is None
        else dict_calib["P2"].reshape((3, 4))
    )
    matrix_4_4 = (
        []
        if R0_rect is None or Tr_velo_to_cam is None
        else np.vstack(
            [
                inverse_transformation_matrix(np.dot(R0_rect, Tr_velo_to_cam)),
                [0, 0, 0, 1],
            ]
        )
        .flatten()
        .tolist()
    )
    camera_matrix_3x4 = (
        []
        if R0_rect is None or Tr_velo_to_cam is None
        else np.hstack(
            [
                np.dot(np.dot(R0_rect, Tr_velo_to_cam), np.vstack([P, [0, 0, 0, 1]])),
            ]
        )
        .flatten()
        .tolist()
    )

    return {
        "R0_rect": R0_rect,
        "Tr_velo_to_cam": Tr_velo_to_cam,
        "Tr_cam_to_velo": Tr_cam_to_velo,
        "matrix_4_4": matrix_4_4,
        "camera_matrix_3x4": camera_matrix_3x4,
    }
