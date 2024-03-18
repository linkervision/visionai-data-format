import json
import logging
import os
import shutil
from typing import Optional

from PIL import Image as PILImage

from visionai_data_format.converters.base import Converter, ConverterFactory
from visionai_data_format.schemas.coco_schema import COCO, Annotation, Category, Image
from visionai_data_format.schemas.common import AnnotationFormat, OntologyImageType
from visionai_data_format.utils.classes import gen_ontology_classes_dict
from visionai_data_format.utils.common import (
    ANNOT_PATH,
    COCO_LABEL_FILE,
    DATA_PATH,
    IMAGE_EXT,
    VISIONAI_JSON,
)

__all__ = ["VAItoCOCO"]

logger = logging.getLogger(__name__)


@ConverterFactory.register(
    from_=AnnotationFormat.VISION_AI,
    to_=AnnotationFormat.COCO,
    image_annotation_type=OntologyImageType._2D_BOUNDING_BOX,
)
class VAItoCOCO(Converter):
    @classmethod
    def convert(
        cls,
        input_annotation_path: str,
        output_dest_folder: str,
        ontology_classes: str,  # ','.join(ontology_classes_list)
        camera_sensor_name: str,
        source_data_root: str,
        uri_root: str,
        lidar_sensor_name: Optional[str] = None,
        sequence_idx_start: int = 0,
        copy_sensor_data: bool = True,
        n_frame: int = -1,
        annotation_name: str = "groundtruth",
        img_extension: str = ".jpg",
    ) -> None:
        logger.info(
            f"vision_ai to coco from {input_annotation_path} to {output_dest_folder}"
        )

        # generate ./labels.json #

        classes_dict = gen_ontology_classes_dict(ontology_classes)

        sequence_folder_list = os.listdir(input_annotation_path)
        vision_ai_dict_list = []
        logger.info("retrieve visionai annotations started")
        for sequence in sequence_folder_list:
            ground_truth_path = os.path.join(
                input_annotation_path, sequence, annotation_name, VISIONAI_JSON
            )
            logger.info(f"retrieve annotation from {ground_truth_path}")
            with open(ground_truth_path) as f:
                vision_ai_dict_list.append(json.load(f))
        logger.info("retrieve visionai annotations finished")

        dest_img_folder = os.path.join(output_dest_folder, DATA_PATH)
        dest_json_folder = os.path.join(output_dest_folder, ANNOT_PATH)
        if copy_sensor_data:
            # create {dest}/data folder #
            os.makedirs(dest_img_folder, exist_ok=True)
        # create {dest}/annotations folder #
        os.makedirs(dest_json_folder, exist_ok=True)

        logger.info("convert visionai to coco format started")
        coco = cls._vision_ai_to_coco(
            dest_img_folder,
            vision_ai_dict_list,  # list of vision_ai dicts
            classes_dict,
            copy_sensor_data,
            camera_sensor_name,
        )
        logger.info("convert visionai to coco format finished")

        with open(os.path.join(dest_json_folder, COCO_LABEL_FILE), "w+") as f:
            json.dump(coco.dict(), f, indent=4)

    @staticmethod
    def convert_single_vision_ai_to_coco(
        dest_img_folder: str,
        vision_ai_dict: dict,
        category_map: dict,
        copy_sensor_data: bool,
        camera_sensor_name: str,
        image_id_start: int = 0,
        anno_id_start: int = 0,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
    ):
        images = []
        annotations = []
        image_id = image_id_start
        anno_id = anno_id_start
        for frame_data in vision_ai_dict["visionai"]["frames"].values():
            dest_coco_url = os.path.join(dest_img_folder, f"{image_id:012d}{IMAGE_EXT}")
            img_url = (
                frame_data["frame_properties"]
                .get("streams", {})
                .get(camera_sensor_name, {})
                .get("uri")
            )
            if copy_sensor_data:
                if os.path.splitext(img_url)[-1] not in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                ]:
                    raise ValueError("The image data type is not supported")
                shutil.copy(img_url, dest_coco_url)
            if img_width is None or img_height is None:
                img = PILImage.open(img_url)
                img_width, img_height = img.size
            image = Image(
                id=image_id,
                width=img_width,
                height=img_height,
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
        return category_map, images, annotations, image_id, anno_id

    @classmethod
    def _vision_ai_to_coco(
        cls,
        dest_img_folder: str,
        vision_ai_dict_list: list[dict],
        classes_dict: dict,
        copy_sensor_data: bool,
        camera_sensor_name: str,
    ):
        images = []
        annotations = []

        image_id_start = 0
        anno_id_start = 0
        category_map = {}

        for vision_ai_dict in vision_ai_dict_list:
            (
                category_map,
                image_update,
                anno_update,
                image_id_start,
                anno_id_start,
            ) = cls.convert_single_vision_ai_to_coco(
                dest_img_folder=dest_img_folder,
                vision_ai_dict=vision_ai_dict,
                classes_dict=classes_dict,
                copy_sensor_data=copy_sensor_data,
                camera_sensor_name=camera_sensor_name,
                image_id_start=image_id_start,
                anno_id_start=anno_id_start,
                category_map=category_map,
            )
            images.extend(image_update)
            annotations.extend(anno_update)

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
