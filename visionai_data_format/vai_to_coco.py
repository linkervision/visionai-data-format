import argparse
import json
import logging
import os
import shutil

from PIL import Image as PILImage
from schemas.coco_schema import Annotation, Category, Coco, Image
from utils.classes import gen_ontology_classes_dict
from utils.common import ANNOT_PATH, DATA_PATH

logger = logging.getLogger(__name__)


def _vision_ai_to_coco(
    vision_ai_dict_list: list[dict],
    src_data_dir: str,
    src_data_basename: str,
    src_data_list: list,
    classes_dict: dict,
):
    images = []
    annotations = []
    categories = []

    image_id = 0
    anno_id = 0
    for vision_ai_dict in vision_ai_dict_list:
        for frame_name, frame_v in vision_ai_dict["visionai"]["frames"].items():
            image_id += 1

            file_name = ""
            for filename in src_data_list:
                # assume filename == f"{frame_name}.{ext}"
                prefix, _ = os.path.splitext(filename)
                if f"{int(prefix):012d}" == frame_name:
                    file_name = filename

            img = PILImage.open(os.path.join(src_data_dir, file_name))
            img_width, image_height = img.size

            img_url = ""
            # assume there is only one camera img url per frame
            for _, p_v in frame_v["frame_properties"].items():
                if os.path.splitext(p_v["uri"])[-1] in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                ]:
                    img_url = p_v["uri"]

            image = Image(
                id=image_id,
                width=img_width,
                height=image_height,
                file_name=os.path.join(src_data_basename, file_name),
                coco_url=img_url
                # assume there is only one sensor, so there is only one img url per frame
            )
            images.append(image)

            if not frame_v.get("objects", None):
                continue

            for object_id, object_v in frame_v["objects"].items():
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
                anno_id += 1
                annotation = Annotation(
                    id=anno_id,
                    image_id=image_id,
                    category_id=classes_dict[
                        vision_ai_dict["visionai"]["objects"][object_id]["type"]
                    ],
                    bbox=bbox,
                    area=width * height,
                    iscrowd=0,
                )
                annotations.append(annotation)

    if classes_dict:
        for cls, id in classes_dict.items():
            category = Category(
                id=id,
                name=cls,
            )
            categories.append(category)

    coco = Coco(categories=categories, images=images, annotations=annotations)
    return coco


def vision_ai_to_coco(
    src: str,
    dst: str,
    ontology_classes: str,
    rm_src: bool = False,
):
    """
    Args:
        src (str): [Path of vision_ai dataset containing 'data' and 'annotation' subfolder, i.e : ~/vision_ai/train/]
        dst (str): [Destination path, i.e : ~/coco/]
    """
    logger.info(f"vision_ai to coco from [{src}] to [{dst}]")

    # generate ./labels.json #

    classes_dict = gen_ontology_classes_dict(ontology_classes)

    src_data_dir = os.path.join(src, DATA_PATH)
    src_data_basename = "data"
    src_data_list = os.listdir(src_data_dir)
    src_data_list.sort()

    src_anno_dir = os.path.join(src, ANNOT_PATH)
    src_anno_list = [f for f in os.listdir(src_anno_dir) if f.endswith(".json")]
    src_anno_list.sort()

    vision_ai_dict_list = []
    for src_anno_file in src_anno_list:
        with open(f"{src_anno_dir}/{src_anno_file}") as f:
            vision_ai_dict_list.append(json.load(f))

    coco = _vision_ai_to_coco(
        vision_ai_dict_list,  # list of vision_ai dicts
        src_data_dir,  # dir of image files
        src_data_basename,  # basename of dir of image files
        src_data_list,  # list of image file names
        classes_dict,
    )

    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "labels.json"), "w+") as f:
        json.dump(coco.dict(), f, indent=4)

    # copy ./data #

    shutil.copytree(
        os.path.join(src, DATA_PATH), os.path.join(dst, DATA_PATH), dirs_exist_ok=True
    )

    # remove original data #
    if rm_src is True:
        shutil.rmtree(src)


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

    return parser.parse_args()


if __name__ == "__main__":
    args = make_parser()

    vision_ai_to_coco(args.src, args.dst, args.ontology_classes)
