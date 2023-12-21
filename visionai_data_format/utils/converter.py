import json
import logging
import os
from typing import Optional

from visionai_data_format.schemas.bdd_schema import AtrributeSchema, FrameSchema
from visionai_data_format.schemas.visionai_schema import VisionAI

from .calculation import xywh2xyxy
from .validator import validate_vai

logger = logging.getLogger(__name__)
VERSION = "00"


def convert_vai_to_bdd(
    folder_name: str,
    company_code: int,
    storage_name: str,
    container_name: str,
    annotation_name: str = "groundtruth",
    target_classes: Optional[list] = None,
) -> dict:
    if not os.path.exists(folder_name) or len(os.listdir(folder_name)) == 0:
        logger.info("[convert_vai_to_bdd] Folder empty or doesn't exits")
    else:
        logger.info("[convert_vai_to_bdd] Convert started")

    frame_list = list()
    for sequence_name in sorted(os.listdir(folder_name)):
        if not os.path.isdir(os.path.join(folder_name, sequence_name)):
            continue
        annotation_file = os.path.join(
            folder_name, sequence_name, "annotations", annotation_name, "visionai.json"
        )
        vai_json = json.loads(open(annotation_file).read())
        vai_data = validate_vai(vai_json).visionai
        cur_frame_list = convert_vai_to_bdd_single(
            vai_data=vai_data,
            sequence_name=sequence_name,
            storage_name=storage_name,
            container_name=container_name,
            target_classes=target_classes,
        )
        frame_list += cur_frame_list

    data = {"frame_list": frame_list, "company_code": company_code}
    logger.info("[convert_vai_to_bdd] Convert finished")
    if not frame_list:
        logger.info("[convert_vai_to_bdd] frame_list is empty")
    return data


def convert_vai_to_bdd_single(
    vai_data: VisionAI,
    sequence_name: str,
    storage_name: str,
    container_name: str,
    img_extension: str = ".jpg",
    target_sensor: str = "camera",
    target_classes: Optional[list] = None,
) -> list:
    frame_list = list()
    # only support sensor type is camera/bbox annotation for now
    # TODO converter for lidar annotation
    target_classes_set = set()
    if target_classes is not None:
        target_classes_set = set(target_classes)
    sensor_names = [
        sensor_name
        for sensor_name, sensor_content in vai_data.streams.items()
        if sensor_content.type == target_sensor
    ]
    for frame_key, frame_data in vai_data.frames.items():
        # create emtpy frame for each target sensor
        sensor_frame = {}
        img_name = frame_key + img_extension
        for sensor in sensor_names:
            frame_temp = FrameSchema(
                storage=storage_name,
                dataset=container_name,
                sequence="/".join([sequence_name, "data", sensor]),
                name=img_name,
                lidarPlaneURLs=[img_name],
                labels=[],
            )
            sensor_frame[sensor] = frame_temp.dict()
        idx = 0
        objects = getattr(frame_data, "objects", None) or {}
        for obj_id, obj_data in objects.items():
            class_ = vai_data.objects.get(obj_id).type
            # filter classes if target_classes is not None
            if target_classes is not None:
                if class_ not in target_classes_set:
                    continue
            bboxes = obj_data.object_data.bbox or [] if obj_data.object_data else []
            for bbox in bboxes:
                geometry = bbox.val
                sensor = bbox.stream  # which sensor is the bbox from
                label = dict()
                label["category"] = class_
                label["meta_ds"] = {}
                label["meta_se"] = {}
                x1, y1, x2, y2 = xywh2xyxy(geometry)
                box2d = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                if bbox.confidence_score is not None:
                    label["meta_ds"]["score"] = bbox.confidence_score
                label["box2d"] = box2d

                object_id = {
                    "project": "General",
                    "function": "General",
                    "object": class_,
                    "version": VERSION,
                }
                label["objectId"] = object_id
                label["attributes"] = AtrributeSchema(INSTANCE_ID=idx).dict()
                sensor_frame[sensor]["labels"].append(label)
                idx += 1
        # frame for different sensors is consider a unique frame in bdd
        for _, frame in sensor_frame.items():
            frame_list.append(frame)
    return frame_list
