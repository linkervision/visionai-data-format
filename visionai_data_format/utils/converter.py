import json
import logging
import os
import shutil
from collections import defaultdict
from typing import Optional

from schemas.bdd_schema import AtrributeSchema
from schemas.visionai_schema import (
    Bbox,
    Frame,
    FrameInterval,
    FrameProperties,
    FramePropertyStream,
    Object,
    ObjectData,
    ObjectDataPointer,
    ObjectType,
    ObjectUnderFrame,
    Stream,
    StreamType,
    VisionAI,
)

from .calculation import gen_intervals, xywh2xyxy, xyxy2xywh
from .common import BBOX_NAME, GROUND_TRUTH_FOLDER, IMAGE_EXT, VISIONAI_OBJECT_JSON
from .validator import save_as_json, validate_vai

logger = logging.getLogger(__name__)
VERSION = "00"


def convert_vai_to_bdd(
    root_folder: str,
    company_code: int,
    storage_name: str,
    container_name: str,
    bdd_dest: Optional[str] = None,
) -> dict:
    if not os.path.exists(root_folder) or len(os.listdir(root_folder)) == 0:
        logger.info("[convert_vai_to_bdd] Folder empty or doesn't exits")
    else:
        logger.info("[convert_vai_to_bdd] Convert started")

    frame_list = list()
    for sequence_folder in sorted(os.listdir(root_folder)):
        annotation_path = os.path.join(
            root_folder, sequence_folder, GROUND_TRUTH_FOLDER, VISIONAI_OBJECT_JSON
        )
        with open(annotation_path) as f:
            raw_data = f.read()
        json_format = json.loads(raw_data)
        vai_data = validate_vai(json_format).visionai
        cur_frame_list = convert_vai_to_bdd_single(
            vai_data, sequence_folder, storage_name, container_name, bdd_dest
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
    bdd_dest: Optional[str] = None,
) -> list:
    cur_data = {}
    cur_data["sequence"] = sequence_name
    cur_data["storage"] = storage_name
    cur_data["dataset"] = container_name

    frame_list = list()
    for frame_idx, frame_data in vai_data.frames.items():
        cur_data["name"] = f"{frame_idx}{IMAGE_EXT}"
        labels = []
        idx = 0
        for obj_id, obj_data in frame_data.objects.items():
            classes = vai_data.objects.get(obj_id).type
            bboxes = obj_data.object_data.bbox or [] if obj_data.object_data else []
            for bbox in bboxes:
                geometry = bbox.val

                label = dict()
                label["category"] = classes
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
                    "object": classes,
                    "version": VERSION,
                }
                label["objectId"] = object_id
                label["attributes"] = AtrributeSchema(INSTANCE_ID=idx).dict()
                labels.append(label)
                idx += 1

        cur_data["labels"] = labels
        frame_list.append(cur_data)
        if bdd_dest:
            bdd_dest_image_folder = os.path.join(bdd_dest, sequence_name)
            os.makedirs(bdd_dest_image_folder, exist_ok=True)
            # get only the first sensor from frame_properties
            sensor = list(frame_data.frame_properties.streams.keys())[0]

            shutil.copy(
                frame_data.frame_properties.streams[sensor].uri,
                os.path.join(bdd_dest_image_folder, f"{frame_idx}{IMAGE_EXT}"),
            )

    return frame_list


def convert_bdd_to_vai(
    bdd_src_folder: str,
    bdd_data: dict,
    vai_dest_folder: str,
    sensor_name: str,
    copy_image: bool,
) -> None:
    bdd_frame_list = bdd_data.get("frame_list", None)

    if not bdd_frame_list:
        logger.info(
            "[convert_bdd_to_vai] frame_list is empty, convert_bdd_to_vai will not be executed"
        )
        return
    try:
        logger.info("[convert_bdd_to_vai] Convert started ")
        custom_sequence_number = 0

        bdd_sequence_frame_dict = defaultdict(dict)
        bdd_frame_file_map: dict = {}
        for bdd_frame in bdd_frame_list:
            sequence_num = bdd_frame["sequence"]
            if not sequence_num.isdigit():
                sequence_num = custom_sequence_number
                custom_sequence_number += 1
            else:
                sequence_num = int(sequence_num)

            frame_file_name = bdd_frame["name"]
            frame_num = int(os.path.splitext(frame_file_name)[0])
            labels = bdd_frame["labels"]
            bdd_sequence_frame_dict[sequence_num][frame_num] = labels
            bdd_frame_file_map[(sequence_num, frame_num)] = frame_file_name

        for sequence_num, frame_data in bdd_sequence_frame_dict.items():
            sequence_idx = f"{sequence_num:012d}"
            vai_seq_folder = os.path.join(vai_dest_folder, sequence_idx)
            vai_annot_folder = os.path.join(vai_seq_folder, GROUND_TRUTH_FOLDER)
            os.makedirs(vai_annot_folder, exist_ok=True)

            gt_frame_intervals = gen_intervals(list(frame_data.keys()))
            frame_intervals = [
                FrameInterval(**frame_interval) for frame_interval in gt_frame_intervals
            ]

            frames: dict[str, Frame] = defaultdict(Frame)
            objects: dict[str, Object] = defaultdict(Object)

            for frame_num, labels in frame_data.items():
                frame_idx = f"{frame_num:012d}"
                vai_image_folder = os.path.join(vai_seq_folder, "data", sensor_name)
                dest_image_path = os.path.join(vai_image_folder, frame_idx + IMAGE_EXT)
                new_frame: Frame = Frame(
                    objects=defaultdict(ObjectData),
                    frame_properties=FrameProperties(
                        streams={sensor_name: FramePropertyStream(uri=dest_image_path)}
                    ),
                )

                if not labels:
                    logger.info(
                        f"[convert_bdd_to_vai] No labels in this sequence-frame : {sequence_idx}-{frame_idx}"
                    )
                for label in labels:
                    # TODO: mapping attributes to the VAI
                    attributes = label["attributes"]
                    attributes.pop("cameraIndex", None)
                    attributes.pop("INSTANCE_ID", None)

                    category = label["category"]
                    obj_uuid = label["uuid"]
                    x, y, w, h = xyxy2xywh(label["box2d"])
                    confidence_score = label.get("meta_ds", {}).get("score", None)
                    object_under_frames = {
                        obj_uuid: ObjectUnderFrame(
                            object_data=ObjectData(
                                bbox=[
                                    Bbox(
                                        name="bbox_shape",
                                        val=[x, y, w, h],
                                        stream=sensor_name,
                                        coordinate_system=sensor_name,
                                        confidence_score=confidence_score,
                                    )
                                ]
                            )
                        )
                    }
                    new_frame.objects.update(object_under_frames)

                    objects[obj_uuid] = Object(
                        name=category,
                        type=category,
                        frame_intervals=frame_intervals,
                        object_data_pointers={
                            BBOX_NAME: ObjectDataPointer(
                                type=ObjectType.bbox, frame_intervals=frame_intervals
                            )
                        },
                    )
                frames[frame_idx] = new_frame

                if copy_image:
                    bdd_image_file = bdd_frame_file_map[(sequence_num, frame_num)]
                    os.makedirs(vai_image_folder, exist_ok=True)
                    dest_image_path = os.path.join(
                        vai_image_folder, frame_idx + IMAGE_EXT
                    )
                    shutil.copy(
                        os.path.join(bdd_src_folder, sequence_idx, bdd_image_file),
                        dest_image_path,
                    )

            streams = {sensor_name: Stream(type=StreamType.camera, uri=dest_image_path)}
            coordinate_systems = {
                sensor_name: {
                    "type": "sensor_cs",
                    "parent": "vehicle-iso8855",
                    "children": [],
                }
            }
            vai_data = {
                "visionai": {
                    "frame_intervals": frame_intervals,
                    "objects": objects,
                    "frames": frames,
                    "streams": streams,
                    "metadata": {"schema_version": "1.0.0"},
                    "coordinate_systems": coordinate_systems,
                }
            }
            vai_data = validate_vai(vai_data).dict(exclude_none=True)
            save_as_json(
                vai_data,
                folder_name=vai_annot_folder,
                file_name=VISIONAI_OBJECT_JSON,
            )

        logger.info("[convert_bdd_to_vai] Convert finished")
    except Exception as e:
        logger.error("[convert_bdd_to_vai] Convert failed : " + str(e))
