import argparse
import json
import logging
import os
import shutil

from PIL import Image as PILImage

from visionai_data_format.schemas.coco_schema import COCO, Annotation, Category, Image
from visionai_data_format.utils.classes import gen_ontology_classes_dict
from visionai_data_format.utils.common import (
    ANNOT_PATH,
    COCO_LABEL_FILE,
    DATA_PATH,
    GROUND_TRUTH_FOLDER,
    IMAGE_EXT,
    LOGGING_DATEFMT,
    LOGGING_FORMAT,
    VISIONAI_JSON,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.DEBUG,
    datefmt=LOGGING_DATEFMT,
)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--src",
        type=str,
        required=True,
        help="Path of vision_ai dataset containing 'data' and 'annotation' subfolder, i.e : ~/vision_ai/train/",
    )
    parser.add_argument(
        "-d", "--dst", type=str, required=True, help="Destination path, i.e : ~/coco/"
    )
    parser.add_argument(
        "-oc",
        "--ontology_classes",
        type=str,
        default="",  # ex: 'cat,dog,horse'
        help="labels (or categories) of the training data",
    )

    parser.add_argument(
        "--copy-image", action="store_true", help="enable to copy image"
    )
    return parser.parse_args()


def _vision_ai_to_coco(
    dest_img_folder: str,
    vision_ai_dict_list: list[dict],
    classes_dict: dict,
    copy_image: bool,
):
    images = []
    annotations = []

    image_id = 0
    anno_id = 0
    category_map = {}

    for vision_ai_dict in vision_ai_dict_list:
        for frame_data in vision_ai_dict["visionai"]["frames"].values():
            img_url = ""
            # assume there is only one camera img url per frame
            for p_v in frame_data["frame_properties"].values():
                sensor = list(p_v.keys())[0]
                if os.path.splitext(p_v[sensor]["uri"])[-1] in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                ]:
                    img_url = p_v[sensor]["uri"]
            dest_coco_url = os.path.join(dest_img_folder, f"{image_id:012d}{IMAGE_EXT}")
            if copy_image:
                shutil.copy(img_url, dest_coco_url)
            img = PILImage.open(img_url)
            img_width, image_height = img.size
            image = Image(
                id=image_id,
                width=img_width,
                height=image_height,
                file_name=f"{image_id:012d}{IMAGE_EXT}",
                coco_url=dest_coco_url
                # assume there is only one sensor, so there is only one img url per frame
            )
            images.append(image)

            if not frame_data.get("objects", None):
                continue

            for object_id, object_v in frame_data["objects"].items():
                # from [center x, center y, width, height] to [top left x, top left y, width, height]
                center_x, center_y, width, height = object_v["object_data"]["bbox"][0][
                    "val"
                ]
                bbox = [
                    float(center_x - width / 2),
                    float(center_y - height / 2),
                    width,
                    height,
                ]
                category = vision_ai_dict["visionai"]["objects"][object_id]["type"]
                if category not in category_map:
                    category_map[category] = len(category_map)

                annotation = Annotation(
                    id=anno_id,
                    image_id=image_id,
                    category_id=category_map[category],
                    bbox=bbox,
                    area=width * height,
                    iscrowd=0,
                )
                annotations.append(annotation)
                anno_id += 1

            image_id += 1

    # add retrieved categories from sequences
    categories = [
        Category(
            id=id,
            name=cls,
        )
        for cls, id in category_map.items()
    ]

    if classes_dict:
        category_map_size = len(category_map)
        for cls, id in classes_dict.items():
            category = Category(
                id=id + category_map_size,
                name=cls,
            )
            categories.append(category)

    coco = COCO(categories=categories, images=images, annotations=annotations)
    return coco


def vision_ai_to_coco(src: str, dst: str, ontology_classes: str, copy_image: bool):
    """
    Args:
        src (str): [Path of vision_ai dataset containing 'data' and 'annotation' subfolder, i.e : ~/vision_ai/train/]
        dst (str): [Destination path, i.e : ~/coco/]
    """
    logger.info(f"vision_ai to coco from {src} to {dst}")

    # generate ./labels.json #

    classes_dict = gen_ontology_classes_dict(ontology_classes)

    sequence_folder_list = os.listdir(src)
    vision_ai_dict_list = []
    logger.info("retrieve visionai annotations started")
    for sequence in sequence_folder_list:
        ground_truth_path = os.path.join(
            src, sequence, GROUND_TRUTH_FOLDER, VISIONAI_JSON
        )
        logger.info(f"retrieve annotation from {ground_truth_path}")
        with open(ground_truth_path) as f:
            vision_ai_dict_list.append(json.load(f))
    logger.info("retrieve visionai annotations finished")

    dest_img_folder = os.path.join(dst, DATA_PATH)
    dest_json_folder = os.path.join(dst, ANNOT_PATH)
    if copy_image:
        # create {dest}/data folder #
        os.makedirs(dest_img_folder, exist_ok=True)
    # create {dest}/annotations folder #
    os.makedirs(dest_json_folder, exist_ok=True)

    logger.info("convert visionai to coco format started")
    coco = _vision_ai_to_coco(
        dest_img_folder,
        vision_ai_dict_list,  # list of vision_ai dicts
        classes_dict,
        copy_image,
    )
    logger.info("convert visionai to coco format finished")

    with open(os.path.join(dest_json_folder, COCO_LABEL_FILE), "w+") as f:
        json.dump(coco.dict(), f, indent=4)


if __name__ == "__main__":
    args = make_parser()

    vision_ai_to_coco(args.src, args.dst, args.ontology_classes, args.copy_image)
