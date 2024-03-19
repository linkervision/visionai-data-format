import json
import logging
import os
import shutil
import uuid
from collections import defaultdict

from visionai_data_format.converters.base import Converter, ConverterFactory
from visionai_data_format.exceptions import VisionAIErrorCode, VisionAIException
from visionai_data_format.schemas.bdd_schema import BDDSchema
from visionai_data_format.schemas.common import AnnotationFormat, OntologyImageType
from visionai_data_format.schemas.visionai_schema import (
    Bbox,
    Context,
    ContextUnderFrame,
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
)
from visionai_data_format.utils.calculation import xyxy2xywh
from visionai_data_format.utils.validator import (
    save_as_json,
    validate_bdd,
    validate_vai,
)

__all__ = ["BDDtoVAI"]

logger = logging.getLogger(__name__)


@ConverterFactory.register(
    from_=AnnotationFormat.BDDP,
    to_=AnnotationFormat.VISION_AI,
    image_annotation_type=OntologyImageType._2D_BOUNDING_BOX,
)
class BDDtoVAI(Converter):
    @classmethod
    def convert(
        cls,
        input_annotation_path: str,
        output_dest_folder: str,
        camera_sensor_name: str,
        lidar_sensor_name: str,
        source_data_root: str,
        uri_root: str,
        sequence_idx_start: int = 0,
        copy_sensor_data: bool = True,
        n_frame: int = -1,
        annotation_name: str = "groundtruth",
        img_extension: str = ".jpg",
        **kwargs,
    ) -> None:
        try:
            raw_data = json.load(open(input_annotation_path))
            bdd_data = validate_bdd(raw_data).dict()
            # create sequence/frame/mapping
            sequence_frames = defaultdict(list)
            for frame in bdd_data["frame_list"]:
                seq_name = frame["sequence"]
                dataset_name = frame["dataset"]
                storage_name = frame["storage"]
                sequence_frames[(storage_name, dataset_name, seq_name)].append(frame)
            # one bdd file might contain mutiple sequences
            seq_id = sequence_idx_start
            for sequence_key, frame_list in sequence_frames.items():
                if n_frame > 0:
                    frame_count = len(frame_list)
                    if n_frame < frame_count:
                        frame_list = frame_list[:n_frame]
                    n_frame -= len(frame_list)
                sequence_bdd_data = BDDSchema(frame_list=frame_list).dict()
                sequence_name = f"{seq_id:012d}"
                logger.info(f"convert sequence {sequence_key} to {sequence_name}")
                cls.convert_sequence_bdd_to_vai(
                    bdd_data=sequence_bdd_data,
                    vai_dest_folder=output_dest_folder,
                    camera_sensor_name=camera_sensor_name,
                    lidar_sensor_name=lidar_sensor_name,
                    sequence_name=sequence_name,
                    uri_root=uri_root,
                    annotation_name=annotation_name,
                    img_extension=img_extension,
                    copy_sensor_data=copy_sensor_data,
                    source_data_root=source_data_root,
                )
                seq_id += 1
                if n_frame == 0:
                    break
        except Exception as e:
            logger.error("Convert bdd to vai format failed : " + str(e))

    @staticmethod
    def convert_sequence_bdd_to_vai(
        bdd_data: dict,
        vai_dest_folder: str,
        camera_sensor_name: str,
        lidar_sensor_name: str,
        sequence_name: str,
        uri_root: str,
        source_data_root: str,
        annotation_name: str = "groundtruth",
        img_extension: str = ".jpg",
        copy_sensor_data: bool = True,
    ) -> None:
        frame_list = bdd_data.get("frame_list", None)

        if not frame_list:
            logger.info(
                "[convert_bdd_to_vai] frame_list is empty, convert_bdd_to_vai will not be executed"
            )
            return
        # TODO BDD lidar convert
        if lidar_sensor_name:
            raise VisionAIException(error_code=VisionAIErrorCode.VAI_ERR_005)

        try:
            logger.info(
                f"[convert_bdd_to_vai] Convert started (copy sensor data is {copy_sensor_data})"
            )
            frames: dict[str, Frame] = defaultdict(Frame)
            objects: dict[str, Object] = defaultdict(Object)
            contexts: dict[str, Context] = defaultdict(Context)
            context_pointers: dict[str, dict] = defaultdict(dict)
            context_cat: dict[str, str] = {}
            for i, frame in enumerate(frame_list):
                frame_idx = f"{i:012d}"
                if copy_sensor_data:
                    img_source = os.path.join(
                        source_data_root,
                        frame["storage"],
                        frame["sequence"],
                        frame["dataset"],
                        frame["name"],
                    )
                    img_dest_dir = os.path.join(
                        vai_dest_folder, sequence_name, "data", camera_sensor_name
                    )
                    os.makedirs(img_dest_dir, exist_ok=True)
                    shutil.copy2(
                        img_source,
                        os.path.join(img_dest_dir, frame_idx + img_extension),
                    )
                labels = frame.get("labels", [])
                frameLabels = frame.get("frameLabels", [])

                url = os.path.join(
                    uri_root,
                    sequence_name,
                    "data",
                    camera_sensor_name,
                    frame_idx + img_extension,
                )
                frame_data: Frame = Frame(
                    objects=defaultdict(DynamicObjectData),
                    contexts=defaultdict(DynamicContextData),
                    frame_properties=FrameProperties(
                        streams={camera_sensor_name: FramePropertyStream(uri=url)}
                    ),
                )
                frame_intervals = [FrameInterval(frame_end=i, frame_start=i)]

                if not labels:
                    logger.info(
                        f"[convert_bdd_to_vai] No labels in this frame : {frame['sequence']}/{frame['name']}"
                    )
                for label in labels:
                    box2d = label.get("box2d", None)
                    # only convert box2D objects for now
                    if box2d is None:
                        logger.info(
                            f"The label shape of {label} in {frame['sequence']}/{frame['name']} is not box2d"
                        )
                        continue
                    category = label["category"]
                    obj_uuid = label.get("uuid", str(uuid.uuid4()))
                    x, y, w, h = xyxy2xywh(box2d)
                    confidence_score = label.get("meta_ds", {}).get("score", None)
                    # Get object attribute data
                    attributes = label["attributes"]
                    # this two keys are sensor id and object id (not the actual attributes we need)
                    attributes.pop("cameraIndex", None)
                    attributes.pop("INSTANCE_ID", None)
                    frame_obj_attr = defaultdict(list)
                    object_data_pointers_attr = defaultdict(str)
                    for attr_name, attr_value in attributes.items():
                        # ignore attribute with no value (None or empty list/dict)
                        if attr_value is None or not attr_value:
                            continue
                        if isinstance(attr_value, bool):
                            frame_obj_attr["boolean"].append(
                                {"name": attr_name, "val": attr_value}
                            )
                            object_data_pointers_attr[attr_name] = ObjectType.BOOLEAN
                        elif isinstance(attr_value, int):
                            frame_obj_attr["num"].append(
                                {"name": attr_name, "val": attr_value}
                            )
                            object_data_pointers_attr[attr_name] = ObjectType.NUM
                        # TODO  usually we need vec type for str and list attributes
                        # might need to ask user to provide ontology to know which type they want (text / vec)
                        else:
                            object_data_pointers_attr[attr_name] = ObjectType.VEC
                            if isinstance(attr_value, list):
                                frame_obj_attr[ObjectType.VEC].append(
                                    {"name": attr_name, "val": attr_value}
                                )
                            else:
                                frame_obj_attr[ObjectType.VEC].append(
                                    {"name": attr_name, "val": [attr_value]}
                                )

                    object_under_frames = {
                        obj_uuid: ObjectUnderFrame(
                            object_data=DynamicObjectData(
                                bbox=[
                                    Bbox(
                                        name="bbox_shape",
                                        val=[x, y, w, h],
                                        stream=camera_sensor_name,
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
                tagging_frame_intervals = [FrameInterval(frame_end=i, frame_start=0)]
                dynamic_context_data = {}
                for frame_lb in frameLabels:
                    context_id = context_cat.get(frame_lb["category"])
                    if context_id is None:
                        # create empty form of context data for recording attributes
                        context_id = str(uuid.uuid4())
                        context_cat[frame_lb["category"]] = context_id
                        contexts.update(
                            {
                                context_id: {
                                    "name": frame_lb["category"],
                                    "type": "*tagging",
                                }
                            }
                        )
                    # record dynamic_context_data for given frame
                    if context_id not in dynamic_context_data:
                        dynamic_context_data[context_id] = defaultdict(list)
                    context_attr = frame_lb["attributes"]

                    for attr_name, attr_value in context_attr.items():
                        # this two keys are sensor id and object id (not the actual attributes we need)
                        if attr_name in {"cameraIndex", "INSTANCE_ID"}:
                            continue
                        # ingore attribute with no value (None or empty list/dict)
                        if attr_value is None or not attr_value:
                            continue
                        context_item = {
                            "name": attr_name,
                            "val": attr_value,
                            "stream": camera_sensor_name,
                        }
                        if isinstance(attr_value, int):
                            context_pointers[context_id][attr_name] = {
                                "type": ObjectType.NUM,
                                "frame_intervals": tagging_frame_intervals,
                            }
                            dynamic_context_data[context_id]["num"].append(context_item)
                        elif isinstance(attr_value, bool):
                            context_pointers[context_id][attr_name] = {
                                "type": "boolean",
                                "frame_intervals": tagging_frame_intervals,
                            }
                            dynamic_context_data[context_id]["boolean"].append(
                                context_item
                            )
                        else:
                            context_pointers[context_id][attr_name] = {
                                "type": ObjectType.VEC,
                                "frame_intervals": tagging_frame_intervals,
                            }
                            if isinstance(attr_value, list):
                                dynamic_context_data[context_id][ObjectType.VEC].append(
                                    context_item
                                )
                            else:
                                dynamic_context_data[context_id][ObjectType.VEC].append(
                                    {
                                        "name": attr_name,
                                        "val": [attr_value],
                                        "stream": camera_sensor_name,
                                    }
                                )
                # update the contexts of frame_data
                for context_id, context_data in dynamic_context_data.items():
                    context_under_frames = {
                        context_id: ContextUnderFrame(
                            context_data=DynamicContextData(**context_data)
                        )
                    }
                    frame_data.contexts.update(context_under_frames)

                frames[frame_idx] = frame_data

            frame_intervals = [FrameInterval(frame_end=i, frame_start=0)]
            for context_id, context_pointer_value in context_pointers.items():
                contexts[context_id].update(
                    {"context_data_pointers": context_pointer_value}
                )
                contexts[context_id].update({"frame_intervals": frame_intervals})

            streams = {camera_sensor_name: Stream(type=StreamType.CAMERA)}
            vai_data = {
                "visionai": {
                    "frame_intervals": frame_intervals,
                    "objects": objects,
                    "contexts": contexts,
                    "frames": frames,
                    "streams": streams,
                    "metadata": {"schema_version": "1.0.0"},
                }
            }
            if not objects:
                vai_data["visionai"].pop("objects")
            if not contexts:
                vai_data["visionai"].pop("contexts")

            vai_data = validate_vai(vai_data).dict(exclude_none=True)
            save_as_json(
                vai_data,
                folder_name=os.path.join(
                    vai_dest_folder, sequence_name, "annotations", annotation_name
                ),
                file_name="visionai.json",
            )
            logger.info("[convert_bdd_to_vai] Convert finished")
        except VisionAIException:
            logger.exception("Convert bdd to vai format error")
            raise VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_041,
                message_kwargs={
                    "original_format": "BDD",
                    "destination_format": "VisionAI",
                },
            )
        except Exception:
            logger.exception("Convert bdd to vai failed")
            raise VisionAIException(error_code=VisionAIErrorCode.VAI_ERR_999)
