import argparse
import copy
import json
import logging
import os

import cv2

logger = logging.getLogger(__name__)


def resize_bbox(annotation, ratio_w, ratio_h):
    x, y, w, h = annotation["bbox"]
    new_x = round(x * ratio_w, 1)
    new_y = round(y * ratio_h, 1)
    new_w = round(w * ratio_w, 1)
    new_h = round(h * ratio_h, 1)
    annotation["bbox"] = [new_x, new_y, new_w, new_h]


def append_ann_json(ori_json, new_json, new_size):
    w, h = new_size
    images = ori_json["images"].copy()

    for annotation in new_json["annotations"]:
        annotation["width"] = w
        annotation["height"] = h

        ratio_w = w
        ratio_h = h
        find = False

        for img in images:
            for k, v in img.items():
                if k == "id" and v == annotation["image_id"]:
                    # Calculate the scaling ratio of each image to calculate the correct position of each bbox
                    ratio_w = w / img["width"]
                    ratio_h = h / img["height"]
                    find = True
                    break
            if find is True:
                break

        resize_bbox(annotation, ratio_w, ratio_h)

        annotation["iscrowd"] = 0

    logger.info("------ann json is done-----")


def resize_json(src: str, dst: str, img_dst: str, new_size: tuple[int, int]):
    ori_json = json.load(open(src))

    w, h = new_size
    new_img_base_dir = img_dst.split("/")[-2]
    new_json = copy.deepcopy(ori_json)

    # Annotation has five parts, and we only resize the three important parts
    for img in new_json["images"]:
        img["width"] = w
        img["height"] = h
        ori_filename = img["file_name"]
        new_filename = os.path.join(new_img_base_dir, os.path.split(ori_filename)[-1])
        img["file_name"] = new_filename

    logger.info("------image json is done-----")

    append_ann_json(ori_json, new_json, new_size)

    with open(dst, "w") as f:
        json.dump(new_json, f, indent=4)

    logger.info("------already wrote json-----")


def resize_image(src: str, dst: str, new_size: tuple[int, int]):
    logger.info("------start image augment-----")

    imgs = os.listdir(src)
    os.makedirs(dst, exist_ok=True)
    for img_name in imgs:
        src_file = os.path.join(src, img_name)
        dst_file = os.path.join(dst, img_name)
        img = cv2.imread(src_file)
        new_img = cv2.resize(img, new_size)
        cv2.imwrite(dst_file, new_img)

    logger.info("------image augmentation is done-----")


def resize_coco(
    label_src: str,
    label_dst: str,
    img_src: str,
    img_dst: str,
    new_size: tuple[int, int],
):
    # TODO: Do acceleration in the future
    resize_json(label_src, label_dst, img_dst, new_size)
    resize_image(img_src, img_dst, new_size)


def make_parser():
    parser = argparse.ArgumentParser("Resize Coco Dataset")
    parser.add_argument(
        "-sj",
        "--source-json",
        required=True,
        type=str,
        help="Path of original json file , i.e : ~/labels.json",
    )
    parser.add_argument(
        "-dj",
        "--dest-json",
        required=True,
        type=str,
        help="Destination path of json file , i.e : ~/resized.json",
    )
    parser.add_argument(
        "-si",
        "--source-image",
        required=True,
        type=str,
        help="Images folder path , i.e : ~/ori_image/",
    )
    parser.add_argument(
        "-di",
        "--dest-image",
        required=True,
        type=str,
        help="Resized images folder path , i.e : ~/resized_image/",
    )
    parser.add_argument(
        "-size", "--size", type=int, default=640, help="Size of resized image"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = make_parser()

    label_src = str(args.source_json)
    label_dst = str(args.dest_json)
    img_src = str(args.source_image)
    img_dst = str(args.dest_image)
    new_w = int(args.size)
    new_h = int(args.size)
    new_size = (new_w, new_h)

    resize_coco(label_src, label_dst, img_src, img_dst, new_size)
