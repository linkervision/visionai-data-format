import json
import logging
import os
import uuid
from collections import defaultdict

from visionai_data_format.schemas.bdd_schema import AtrributeSchema, FrameSchema
from visionai_data_format.schemas.visionai_schema import (
    Bbox,
    Context,
    DynamicContextData,
    DynamicObjectData,
    Frame,
    FrameInterval,
    FrameProperties,
    FramePropertyStream,
    Object,
    ObjectDataPointer,
    ObjectType,
    ObjectUnderFrame,
    Stream,
    StreamType,
    VisionAI,
)

from .calculation import xywh2xyxy, xyxy2xywh
from .validator import save_as_json, validate_vai

logger = logging.getLogger(__name__)
VERSION = "00"


def convert_vai_to_bdd(
    folder_name: str,
    company_code: int,
    storage_name: str,
    container_name: str,
    annotation_name: str = "groundtruth",
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
            vai_data, sequence_name, storage_name, container_name
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
) -> list:
    frame_list = list()
    # only support sensor type is camera/bbox annotation for now
    # TODO converter for lidar annotation
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
            classes = vai_data.objects.get(obj_id).type
            bboxes = obj_data.object_data.bbox or [] if obj_data.object_data else []
            for bbox in bboxes:
                geometry = bbox.val
                sensor = bbox.stream  # which sensor is the bbox from
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
                sensor_frame[sensor]["labels"].append(label)
                idx += 1
        # frame for different sensors is consider a unique frame in bdd
        for _, frame in sensor_frame.items():
            frame_list.append(frame)
    return frame_list


def convert_bdd_to_vai(
    bdd_data: dict,
    vai_dest_folder: str,
    sensor_name: str,
    sequence_name: str,
    uri_root: str,
    annotation_name: str = "groundtruth",
    img_extention: str = ".jpg",
) -> None:
    frame_list = bdd_data.get("frame_list", None)

    if not frame_list:
        logger.info(
            "[convert_bdd_to_vai] frame_list is empty, convert_bdd_to_vai will not be executed"
        )
        return

    try:
        logger.info("[convert_bdd_to_vai] Convert started ")
        frames: dict[str, Frame] = defaultdict(Frame)
        objects: dict[str, Object] = defaultdict(Object)
        contexts: dict[str, Context] = defaultdict(Context)
        context_cat = {}
        context_poninters = {}
        for i, frame in enumerate(frame_list):
            frame_idx = f"{int(i):012d}"
            labels = frame["labels"]
            frameLabels = frame["frameLabels"]
            url = os.path.join(
                uri_root, sequence_name, "data", sensor_name, frame_idx + img_extention
            )

            frame_data: Frame = Frame(
                objects=defaultdict(DynamicObjectData),
                contexts=defaultdict(DynamicContextData),
                frame_properties=FrameProperties(
                    streams={sensor_name: FramePropertyStream(uri=url)}
                ),
            )
            frame_intervals = [
                FrameInterval(frame_end=int(frame_idx), frame_start=int(frame_idx))
            ]

            if not labels:
                logger.info(
                    f"[convert_bdd_to_vai] No labels in this frame : {frame['name']}"
                )

            for label in labels:
                attributes = label["attributes"]
                attributes.pop("cameraIndex", None)
                attributes.pop("INSTANCE_ID", None)
                frame_obj_attr = defaultdict(list)
                object_data_pointers_attr = defaultdict(str)
                for attr_name, attr_value in attributes.items():
                    if isinstance(attr_value, bool):
                        frame_obj_attr["boolean"].append(
                            {"name": attr_name, "val": attr_value}
                        )
                        object_data_pointers_attr[attr_name] = "boolean"
                    elif isinstance(type(attr_value), int):
                        frame_obj_attr["num"].append(
                            {"name": attr_name, "val": attr_value}
                        )
                        object_data_pointers_attr[attr_name] = "num"
                    # TODO  distinguish text and vec attribute type
                    else:
                        frame_obj_attr["vec"].append(
                            {"name": attr_name, "val": [attr_value]}
                        )
                        object_data_pointers_attr[attr_name] = "vec"

                category = label["category"]
                obj_uuid = label.get("uuid")
                if obj_uuid is None:
                    obj_uuid = str(uuid.uuid4())
                x, y, w, h = xyxy2xywh(label["box2d"])
                confidence_score = label.get("meta_ds", {}).get("score", None)
                object_under_frames = {
                    obj_uuid: ObjectUnderFrame(
                        object_data=DynamicObjectData(
                            bbox=[
                                Bbox(
                                    name="bbox_shape",
                                    val=[x, y, w, h],
                                    stream=sensor_name,
                                    confidence_score=confidence_score,
                                    attributes=frame_obj_attr,
                                )
                            ]
                        )
                    )
                }
                frame_data.objects.update(object_under_frames)

                objects[obj_uuid] = Object(
                    name=category,
                    type=category,
                    frame_intervals=frame_intervals,
                    object_data_pointers={
                        "bbox_shape": ObjectDataPointer(
                            type=ObjectType.BBOX,
                            frame_intervals=frame_intervals,
                            attributes=object_data_pointers_attr,
                        )
                    },
                )

            # frame tagging data (contexts)
            tagging_frame_intervals = [
                FrameInterval(frame_end=int(frame_idx), frame_start=0)
            ]
            for frame_lb in frameLabels:
                context_id = context_cat.get(frame_lb["category"])
                if context_id is None:
                    context_id = str(uuid.uuid4())
                    context_cat[frame_lb["category"]] = context_id
                    contexts.update({context_id: {"context_data": defaultdict(list)}})
                context_attr = frame_lb["attributes"]

                for attr_name, attr_value in context_attr.items():
                    if attr_name in {"cameraIndex", "INSTANCE_ID"}:
                        continue
                    if attr_value is None:
                        continue
                    if isinstance(attr_value, int):
                        context_poninters[attr_name] = {
                            "type": "num",
                            "frame_intervals": tagging_frame_intervals,
                        }
                        frame_data.contexts[context_id]["context_data"]["num"].append(
                            {
                                "name": attr_name,
                                "val": attr_value,
                                "stream": sensor_name,
                            }
                        )
                    elif isinstance(attr_value, bool):
                        context_poninters[attr_name] = {
                            "type": "boolean",
                            "frame_intervals": tagging_frame_intervals,
                        }
                        frame_data.contexts[context_id]["context_data"][
                            "boolean"
                        ].append(
                            {
                                "name": attr_name,
                                "val": attr_value,
                                "stream": sensor_name,
                            }
                        )
                    else:
                        context_poninters[attr_name] = {
                            "type": "vec",
                            "frame_intervals": tagging_frame_intervals,
                        }
                        if isinstance(attr_value, list):
                            frame_data.contexts[context_id]["context_data"][
                                "vec"
                            ].append(
                                {
                                    "name": attr_name,
                                    "val": attr_value,
                                    "stream": sensor_name,
                                }
                            )
                        else:
                            frame_data.contexts[context_id]["context_data"][
                                "vec"
                            ].append(
                                {
                                    "name": attr_name,
                                    "val": [attr_value],
                                    "stream": sensor_name,
                                }
                            )

            frames[frame_idx] = frame_data

        frame_intervals = [FrameInterval(frame_end=int(i), frame_start=int(0))]
        streams = {sensor_name: Stream(type=StreamType.CAMERA)}
        vai_data = {
            "visionai": {
                "frame_intervals": frame_intervals,
                "objects": objects,
                "frames": frames,
                "streams": streams,
                "metadata": {"schema_version": "1.0.0"},
            }
        }
        if not objects:
            vai_data["visionai"].pop("objects")
        vai_data = validate_vai(vai_data).dict(exclude_none=True)
        save_as_json(
            vai_data,
            folder_name=os.path.join(
                vai_dest_folder, sequence_name, "annotations", annotation_name
            ),
            file_name="visionai.json",
        )
        logger.info("[convert_bdd_to_vai] Convert finished")
    except Exception as e:
        logger.error("[convert_bdd_to_vai] Convert failed : " + str(e))
