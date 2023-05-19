import argparse
import json
import logging
import os
import shutil
import uuid

from visionai_data_format.schemas.visionai_schema import (
    Bbox,
    DynamicObjectData,
    Frame,
    FrameInterval,
    FrameProperties,
    FramePropertyStream,
    Metadata,
    Object,
    ObjectDataPointer,
    ObjectType,
    ObjectUnderFrame,
    SchemaVersion,
    Stream,
    StreamType,
    VisionAI,
    VisionAIModel,
)
from visionai_data_format.utils.common import (
    ANNOT_PATH,
    BBOX_NAME,
    DATA_PATH,
    GROUND_TRUTH_FOLDER,
    IMAGE_EXT,
    LOGGING_DATEFMT,
    LOGGING_FORMAT,
    VISIONAI_OBJECT_JSON,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.DEBUG,
    datefmt=LOGGING_DATEFMT,
)


def _coco_to_vision_ai(
    coco_dict: dict, sensor_name: str, dest_folder: str, src_image_path: str
) -> dict:
    vision_ai_data_dict = {}
    frames = {}
    objects = {}

    # parse coco: images
    image_id_name_dict = {}
    image_name_list = []
    frame_idx = f"{0:012d}"  # only generate one frame per sequential
    frame_num = int(frame_idx)
    for image_info in coco_dict["images"]:
        image_name = image_info["file_name"].split("/")[-1].split(".")[0]
        image_id = image_info["id"]
        image_id_name_dict.update({str(image_id): str(image_name)})
        image_name_list.append(str(image_name))
        new_image_name = f"{int(image_name):012d}"

        dest_image_folder = os.path.join(
            dest_folder, new_image_name, "data", sensor_name
        )
        coco_url = image_info.get("coco_url")
        if not (coco_url) and src_image_path:
            os.makedirs(dest_image_folder, exist_ok=True)
            shutil.copy(
                os.path.join(src_image_path, image_info["file_name"]),
                os.path.join(dest_image_folder, frame_idx + IMAGE_EXT),
            )
        uri = (
            os.path.join(dest_image_folder, frame_idx + IMAGE_EXT)
            if not coco_url
            else coco_url
        )

        # to vision_ai: frames
        frames[image_name] = Frame(
            frame_properties=FrameProperties(
                streams={sensor_name: FramePropertyStream(uri=uri)}
            ),
            objects={},
        )

    # parse coco: categories
    class_id_name_dict = {
        str(class_info["id"]): class_info["name"]
        for class_info in coco_dict["categories"]
    }

    # parse coco: annotations
    for anno in coco_dict["annotations"]:
        object_id = str(uuid.uuid4())

        # from [top left x, top left y, width, height] to [center x, center y, width, height]
        top_left_x, top_left_y, width, height = anno["bbox"]
        bbox = [
            float(top_left_x + width / 2),
            float(top_left_y + height / 2),
            width,
            height,
        ]
        image_name = image_id_name_dict[str(anno["image_id"])]

        # to vision_ai: frames
        # assume there is only one sensor, so image_index always is 0
        objects_under_frames = {
            object_id: ObjectUnderFrame(
                object_data=DynamicObjectData(
                    bbox=[
                        Bbox(
                            name=BBOX_NAME,
                            val=bbox,
                            stream=sensor_name,
                        )
                    ]
                )
            )
        }

        frames[image_name].objects.update(objects_under_frames)

        # to vision_ai: objects
        object_under_objects = {
            object_id: Object(
                name=class_id_name_dict[str(anno["category_id"])],
                type=class_id_name_dict[str(anno["category_id"])],
                frame_intervals=[
                    FrameInterval(frame_start=frame_num, frame_end=frame_num)
                ],
                object_data_pointers={
                    BBOX_NAME: ObjectDataPointer(
                        type=ObjectType.bbox,
                        frame_intervals=[
                            FrameInterval(
                                frame_start=frame_num,
                                frame_end=frame_num,
                            )
                        ],
                    )
                },
            )
        }
        objects.update(object_under_objects)
    streams = {
        sensor_name: {
            "type": "camera",
            "description": "Frontal camera",
        },
    }

    streams = {
        sensor_name: Stream(
            uri=uri,
            type=StreamType.camera,
            description="Frontal camera",
        )
    }
    # to vision_ai:
    for image_name, frame_obj in frames.items():
        current_frame_objects = getattr(frame_obj, "objects", None) or {}
        objects_per_image = {
            object_id: objects[object_id] for object_id in current_frame_objects
        }

        new_image_name = f"{int(image_name):012d}"
        vision_ai_data_dict[new_image_name] = VisionAIModel(
            visionai=VisionAI(
                frame_intervals=[
                    FrameInterval(frame_start=frame_num, frame_end=frame_num)
                ],
                frames={frame_idx: frame_obj},
                objects=objects_per_image,
                metadata=Metadata(schema_version=SchemaVersion.field_1_0_0),
                streams=streams,
                coordinate_systems={
                    sensor_name: {
                        "type": "sensor_cs",
                        "parent": "vehicle-iso8855",
                        "children": [],
                    }
                },
            )
        )
    return vision_ai_data_dict


def coco_to_vision_ai(
    src: str,
    dst: str,
    sensor: str,
    copy_image: bool,
):
    """
    Args:
        src (str): [Path of coco dataset containing 'data' and 'annotations' subfolder, i.e : ~/dataset/coco/]
        dst (str): [Destination path, i.e : ~/vision_ai/]
    """
    logger.info(f"Convert coco to vision_ai from {src} to {dst} started...")

    src_annot_path = os.path.join(src, ANNOT_PATH)
    src_image_path = os.path.join(src, DATA_PATH) if copy_image else None

    # only take one file from annotations folder
    anno_file_name = os.listdir(src_annot_path)[0]

    with open(os.path.join(src_annot_path, anno_file_name)) as f:
        coco_dict = json.load(f)

    vision_ai_dict = _coco_to_vision_ai(coco_dict, sensor, dst, src_image_path)
    for seq_num, vision_ai in vision_ai_dict.items():
        dest_annot_path = os.path.join(dst, seq_num, GROUND_TRUTH_FOLDER)

        os.makedirs(f"{dest_annot_path}", exist_ok=True)

        logger.info(
            f"Saving data to {os.path.join(dest_annot_path,VISIONAI_OBJECT_JSON)}"
        )
        with open(os.path.join(dest_annot_path, VISIONAI_OBJECT_JSON), "w+") as f:
            json.dump(vision_ai.dict(exclude_none=True), f, indent=4)
        logger.info("Save finished")

    logger.info("Convert finished")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--src",
        type=str,
        required=True,
        help="Path of coco dataset containing 'data' and 'annotations' subfolder, i.e : ~/dataset/coco/",
    )
    parser.add_argument(
        "-d",
        "--dst",
        type=str,
        required=True,
        help="Destination path, i.e : ~/vision_ai/",
    )
    parser.add_argument(
        "--sensor", type=str, help="Sensor name, i.e : `camera1`", default="camera1"
    )

    parser.add_argument(
        "--copy-image", action="store_true", help="enable to copy image"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = make_parser()

    coco_to_vision_ai(args.src, args.dst, args.sensor, args.copy_image)
