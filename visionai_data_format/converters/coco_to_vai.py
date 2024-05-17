import json
import logging
import os
import shutil
import uuid
from collections import defaultdict
from typing import Optional

from visionai_data_format.converters.base import Converter, ConverterFactory
from visionai_data_format.exceptions import VisionAIErrorCode, VisionAIException
from visionai_data_format.schemas.common import AnnotationFormat, OntologyImageType
from visionai_data_format.schemas.visionai_schema import (
    Bbox,
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
from visionai_data_format.utils.validator import (
    save_as_json,
    validate_coco,
    validate_vai,
)

__all__ = ["COCOtoVAI"]

logger = logging.getLogger(__name__)


@ConverterFactory.register(
    from_=AnnotationFormat.COCO,
    to_=AnnotationFormat.VISION_AI,
    image_annotation_type=OntologyImageType._2D_BOUNDING_BOX,
)
class COCOtoVAI(Converter):
    @classmethod
    def convert(
        cls,
        input_annotation_path: str,
        output_dest_folder: str,
        camera_sensor_name: str,
        source_data_root: str,
        uri_root: str,
        lidar_sensor_name: Optional[str] = None,
        sequence_idx_start: int = 0,
        copy_sensor_data: bool = True,
        n_frame: int = -1,
        annotation_name: str = "groundtruth",
        img_extension: str = ".jpg",
        **kwargs,
    ) -> None:
        try:
            raw_data = json.load(open(input_annotation_path))
            coco_json_data = validate_coco(raw_data).dict()

            class_id_name_map: dict[str, str] = {
                str(class_info["id"]): class_info["name"]
                for class_info in coco_json_data["categories"]
            }
            img_id_annotations_map = defaultdict(list)
            for annot_info in coco_json_data.pop("annotations", {}):
                img_id_annotations_map[annot_info.get("image_id")].append(annot_info)

            img_name_id_map = defaultdict(int)
            for img_info in coco_json_data.get("images"):
                file_name = img_info.get("file_name")
                if not file_name:
                    raise VisionAIException(
                        error_code=VisionAIErrorCode.VAI_ERR_002,
                        message_kwargs={
                            "field_name": "file_name",
                            "required_place": "images",
                        },
                    )
                img_name_id_map[file_name] = img_info["id"]

            for sequence_idx, image_data in enumerate(
                coco_json_data["images"], sequence_idx_start
            ):
                if n_frame > 0:
                    n_frame -= 1
                dest_sequence_name = f"{sequence_idx:012d}"
                image_path = image_data["file_name"]
                old_sequence_idx = os.path.splitext(image_path)[0].split(os.sep)[-1]
                logger.info(
                    f"convert sequence {old_sequence_idx} to {dest_sequence_name}"
                )
                vai_data = cls.convert_coco_to_vai(
                    image_data=image_data,
                    img_name_id_map=img_name_id_map,
                    vai_dest_folder=output_dest_folder,
                    camera_sensor_name=camera_sensor_name,
                    dest_sequence_name=dest_sequence_name,
                    uri_root=uri_root,
                    img_extension=img_extension,
                    copy_sensor_data=copy_sensor_data,
                    source_data_root=source_data_root,
                    class_id_name_map=class_id_name_map,
                    img_id_annotations_map=img_id_annotations_map,
                )
                save_as_json(
                    vai_data,
                    folder_name=os.path.join(
                        output_dest_folder,
                        dest_sequence_name,
                        "annotations",
                        annotation_name,
                    ),
                    file_name="visionai.json",
                )
                if n_frame == 0:
                    break

        except VisionAIException:
            logger.exception("Convert coco to vai format error")
            raise VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_041,
                message_kwargs={
                    "original_format": "COCO",
                    "destination_format": "VisionAI",
                },
            )

        except Exception:
            logger.exception("Convert coco to vai failed")
            raise VisionAIException(error_code=VisionAIErrorCode.VAI_ERR_999)

    @staticmethod
    def convert_coco_to_vai(
        image_data: dict,
        vai_dest_folder: str,
        camera_sensor_name: str,
        dest_sequence_name: str,
        uri_root: str,
        img_name_id_map: dict,
        source_data_root: str,
        class_id_name_map: dict[str, str],
        img_id_annotations_map: dict[str, list[dict]],
        img_extension: str = ".jpg",
        copy_sensor_data: bool = True,
    ) -> dict:
        try:
            image_file_name = image_data["file_name"]
            image_file_path = os.path.join(source_data_root, image_file_name)

            logger.info(
                f"[convert_coco_to_vai] Convert started (copy sensor data is {copy_sensor_data})"
            )
            img_id: int = img_name_id_map[image_file_name]
            frames = {}
            objects = {}
            frame_intervals = []
            frame_num = f"{0:012d}"
            bbox_name = "bbox_shape"

            dest_camera_folder = os.path.join(
                vai_dest_folder, dest_sequence_name, "data", camera_sensor_name
            )
            dest_camera_path = os.path.join(
                dest_camera_folder, frame_num + img_extension
            )
            if copy_sensor_data:
                os.makedirs(dest_camera_folder, exist_ok=True)
                shutil.copy2(image_file_path, dest_camera_path)

            camera_url = os.path.join(uri_root, dest_camera_path)

            # generate frames below visionai
            frames[frame_num] = Frame(
                frame_properties=FrameProperties(
                    streams={camera_sensor_name: FramePropertyStream(uri=camera_url)}
                ),
                objects={},
            )

            # parse coco: annotations
            for idx, annot_info in enumerate(img_id_annotations_map.pop(img_id, [])):
                object_id = str(uuid.uuid4())

                # from [top left x, top left y, width, height] to [center x, center y, width, height]
                top_left_x, top_left_y, width, height = annot_info["bbox"]
                bbox = [
                    float(top_left_x + width / 2),
                    float(top_left_y + height / 2),
                    width,
                    height,
                ]

                objects_under_frames = {
                    object_id: ObjectUnderFrame(
                        object_data=DynamicObjectData(
                            bbox=[
                                Bbox(
                                    name=bbox_name,
                                    val=bbox,
                                    stream=camera_sensor_name,
                                )
                            ]
                        )
                    )
                }
                frames[frame_num].objects.update(objects_under_frames)

                # to vision_ai: objects
                object_under_vai = {
                    object_id: Object(
                        name=class_id_name_map[str(annot_info["category_id"])]
                        + f"{idx:03d}",
                        type=class_id_name_map[str(annot_info["category_id"])],
                        frame_intervals=[
                            FrameInterval(frame_start=0, frame_end=0)  # TODO: fixme
                        ],
                        object_data_pointers={
                            bbox_name: ObjectDataPointer(
                                type=ObjectType.BBOX,
                                frame_intervals=[
                                    FrameInterval(
                                        frame_start=0, frame_end=0
                                    )  # TODO: fixme
                                ],
                            )
                        },
                    )
                }
                objects.update(object_under_vai)
            streams = {camera_sensor_name: Stream(type=StreamType.CAMERA)}
            frame_intervals = [FrameInterval(frame_start=0, frame_end=0)]

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
            logger.info("[convert_coco_to_vai] Convert finished")
            return vai_data
        except Exception as e:
            logger.error("[convert_coco_to_vai] Convert failed : " + str(e))
