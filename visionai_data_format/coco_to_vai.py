import argparse
import json
import logging
import os
import shutil
import uuid

from schemas.visionai_schema import (
    Bbox,
    ElementDataPointer,
    Frame,
    FrameInterval,
    FrameObjectData,
    FrameProperties,
    FramePropertyInfo,
    FramePropertyStream,
    Metadata,
    Object,
    ObjectInFrame,
    SchemaVersion,
    Stream,
    TypeElement,
    VisionAI,
    VisionAIModel,
)
from utils.common import ANNOT_PATH

logger = logging.getLogger(__name__)


def _coco_to_vision_ai(coco_dict: dict, sensor_name: str):
    vision_ai_list = []

    frames = {}
    objects = {}

    # parse coco: images
    image_id_name_dict = {}
    image_name_list = []
    for image_info in coco_dict["images"]:
        image_name = image_info["file_name"].split("/")[-1].split(".")[0]
        image_id = image_info["id"]
        image_id_name_dict.update({str(image_id): str(image_name)})
        image_name_list.append(str(image_name))

        # to vision_ai: frames
        frames[str(image_name)] = Frame(
            frame_properties=FrameProperties(
                streams=FramePropertyStream(
                    __root__={
                        sensor_name: FramePropertyInfo(uri=image_info["coco_url"])
                    }
                )
            ),
            objects={},
        )

    # parse coco: categories
    class_id_name_dict = {}
    for class_info in coco_dict["categories"]:
        class_id_name_dict.update({str(class_info["id"]): class_info["name"]})

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
        bbox_name = "2d_shape"
        image_name = image_id_name_dict[str(anno["image_id"])]

        # to vision_ai: frames
        # assume there is only one sensor, so image_index always is 0
        objects_under_frames = {
            object_id: ObjectInFrame(
                object_data=FrameObjectData(
                    bbox=[
                        Bbox(
                            name=bbox_name,
                            val=bbox,
                            stream=sensor_name,
                            coordinate_system=sensor_name,
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
                    FrameInterval(
                        frame_start=int(image_name), frame_end=int(image_name)
                    )
                ],
                object_data_pointers={
                    bbox_name: ElementDataPointer(
                        type=TypeElement.bbox,
                        frame_intervals=[
                            FrameInterval(
                                frame_start=int(image_name),
                                frame_end=int(image_name),
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
    # to vision_ai:
    for image_name in image_name_list:

        objects_per_image = {}
        for object_id in frames[image_name].objects:
            objects_per_image.update({object_id: objects[object_id]})

        vision_ai_list.append(
            VisionAIModel(
                visionai=VisionAI(
                    frame_intervals=[
                        FrameInterval(
                            frame_start=int(image_name), frame_end=int(image_name)
                        )
                    ],
                    frames={image_name: frames[image_name]},
                    objects=objects_per_image,
                    metadata=Metadata(schema_version=SchemaVersion.field_1_0_0),
                    streams=Stream(__root__=streams),
                    coordinate_systems={
                        sensor_name: {
                            "type": "sensor_cs",
                            "parent": "vehicle-iso8855",
                            "children": [],
                        }
                    },
                )
            )
        )

    return vision_ai_list


def coco_to_vision_ai(
    src: str,
    dst: str,
    sensor: str,
):
    """
    Args:
        src (str): [Path of coco dataset containing 'data' and 'annotations' subfolder, i.e : ~/dataset/coco/]
        dst (str): [Destination path, i.e : ~/vision_ai/]
    """
    logger.info(f"coco to vision_ai from [{src}] to [{dst}]")

    # generate annotation json in vision_ai format #
    src_annot_path = os.path.join(src, ANNOT_PATH)
    src_annotations_list = os.listdir(src_annot_path)

    dest_annot_path = os.path.join(dst, ANNOT_PATH)

    for anno_file_name in src_annotations_list:
        with open(f"{src_annot_path}/{anno_file_name}") as f:
            coco_dict = json.load(f)

        vision_ai_list = _coco_to_vision_ai(coco_dict, sensor)

        os.makedirs(f"{dest_annot_path}", exist_ok=True)
        for vision_ai in vision_ai_list:
            image_name = list(vision_ai.visionai.frames.keys())[0]
            anno_json = f"{image_name}.json"
            with open(f"{dest_annot_path}/{anno_json}", "w+") as f:
                json.dump(vision_ai.dict(exclude_none=True), f, indent=4)

    # copy ./data #

    shutil.copytree(
        os.path.join(src, "data"), os.path.join(dst, "data"), dirs_exist_ok=True
    )


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

    return parser.parse_args()


if __name__ == "__main__":
    args = make_parser()

    coco_to_vision_ai(args.src, args.dst, args.sensor)
