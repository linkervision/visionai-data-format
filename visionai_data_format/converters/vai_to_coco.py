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
    COCO_IMAGE_PATH,
    COCO_LABEL_FILE,
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
        source_data_root: str,
        output_dest_folder: str,
        uri_root: str,
        camera_sensor_name: str,
        ontology_classes: str = "",  # ','.join(ontology_classes_list)
        copy_sensor_data: bool = True,
        n_frame: int = -1,
        annotation_name: str = "groundtruth",
        img_extension: str = IMAGE_EXT,
        **kwargs,
    ) -> None:
        logger.info(
            f"convert VisionAI to coco from {source_data_root} to {output_dest_folder}"
        )
        category_map = gen_ontology_classes_dict(ontology_classes)

        sequence_folder_list = os.listdir(source_data_root)
        visionai_dict_list = []
        logger.info("retrieve visionai annotations started")
        for sequence in sequence_folder_list:
            if not os.path.isdir(os.path.join(source_data_root, sequence)):
                logger.info(
                    f"file {sequence} is ignore since it is not a sequence folder"
                )
                continue
            annotation_path = os.path.join(
                source_data_root,
                sequence,
                "annotations",
                annotation_name,
                VISIONAI_JSON,
            )
            logger.info(f"retrieve annotation from {annotation_path}")
            with open(annotation_path) as f:
                visionai_dict_list.append(json.load(f))

        logger.info("retrieve visionai annotations finished")

        dest_img_folder = os.path.join(output_dest_folder, COCO_IMAGE_PATH)
        dest_json_folder = os.path.join(output_dest_folder, ANNOT_PATH)
        if copy_sensor_data:
            # create {dest}/data folder #
            os.makedirs(dest_img_folder, exist_ok=True)
        # create {dest}/annotations folder #
        os.makedirs(dest_json_folder, exist_ok=True)

        logger.info("convert visionai to coco format started")
        coco = cls._visionai_to_coco(
            dest_img_folder=dest_img_folder,
            visionai_dict_list=visionai_dict_list,  # list of visionai dicts
            copy_sensor_data=copy_sensor_data,
            camera_sensor_name=camera_sensor_name,
            source_data_root=source_data_root,
            uri_root=uri_root,
            category_map=category_map,
            n_frame=n_frame,
            img_extension=img_extension,
        )
        logger.info("convert visionai to coco format finished")

        with open(os.path.join(dest_json_folder, COCO_LABEL_FILE), "w+") as f:
            json.dump(coco.dict(), f, indent=4)

    @staticmethod
    def convert_single_visionai_to_coco(
        dest_img_folder: str,
        visionai_dict: dict,
        category_map: dict,
        copy_sensor_data: bool,
        source_data_root: str,
        uri_root: str,
        camera_sensor_name: str,
        n_frame: int = -1,
        img_extension: str = IMAGE_EXT,
        image_id_start: int = 0,
        anno_id_start: int = 0,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
    ) -> tuple[dict, list, list, int, int, int]:
        """Convert single visionai data to coco format

        Parameters
        ----------
        dest_img_folder : str
        visionai_dict : dict
        category_map : dict
        copy_sensor_data : bool
        source_data_root : str
        uri_root : str
            uri root for target upload for coco uploaded
        camera_sensor_name : str
            for getting the target camera sensor data
        n_frame : int, optional
            number of frame to be converted (-1 means all), by default -1
        img_extension : str, optional
            by default IMAGE_EXT (.jpg)
        image_id_start : int, optional
            by default 0
        anno_id_start : int, optional
            by default 0
        img_width : Optional[int], optional
            by default None
        img_height : Optional[int], optional
            by default None

        Returns
        -------
        tuple[dict, list, list, int, int, int]
            category_map: dict
            images: list of coco images
            annotations: list of coco annotations
            image_id: int (image_id_start for next sequence)
            anno_id: int (anno_id_start for next sequence)
            n_frame: int (number of frames left to be converted)

        Raises
        ------
        ValueError
            image data type is not supported
        """
        images = []
        annotations = []
        image_id = image_id_start
        anno_id = anno_id_start
        for frame_data in visionai_dict["visionai"]["frames"].values():
            if len(images) == n_frame:
                break
            dest_coco_url = os.path.join(
                uri_root, COCO_IMAGE_PATH, f"{image_id:012d}{img_extension}"
            )
            dest_coco_img = os.path.join(
                dest_img_folder, f"{image_id:012d}{img_extension}"
            )
            img_url = (
                frame_data["frame_properties"]
                .get("streams", {})
                .get(camera_sensor_name, {})
                .get("uri")
            )
            source_image_path = "/".join([source_data_root] + (img_url.split("/")[-4:]))
            if copy_sensor_data:
                if os.path.splitext(img_url)[-1] not in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                ]:
                    raise ValueError("The image data type is not supported")
                shutil.copy(source_image_path, dest_coco_img)
            if img_width is None or img_height is None:
                img = PILImage.open(source_image_path)
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
                image_id += 1
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
                category = visionai_dict["visionai"]["objects"][object_id]["type"]
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
        if n_frame != -1:
            n_frame -= len(images)
        return (category_map, images, annotations, image_id, anno_id, n_frame)

    @classmethod
    def _visionai_to_coco(
        cls,
        dest_img_folder: str,
        visionai_dict_list: list[dict],
        copy_sensor_data: bool,
        camera_sensor_name: str,
        source_data_root: str,
        uri_root: str,
        category_map: dict,
        n_frame: int = -1,
        img_extension: str = IMAGE_EXT,
    ) -> COCO:
        images = []
        annotations = []

        image_id_start = 0
        anno_id_start = 0
        for visionai_dict in visionai_dict_list:
            (
                category_map,
                image_update,
                anno_update,
                image_id_start,
                anno_id_start,
                n_frame,
            ) = cls.convert_single_visionai_to_coco(
                dest_img_folder=dest_img_folder,
                visionai_dict=visionai_dict,
                copy_sensor_data=copy_sensor_data,
                source_data_root=source_data_root,
                uri_root=uri_root,
                camera_sensor_name=camera_sensor_name,
                image_id_start=image_id_start,
                anno_id_start=anno_id_start,
                category_map=category_map,
                n_frame=n_frame,
                img_extension=img_extension,
            )
            images.extend(image_update)
            annotations.extend(anno_update)
            if n_frame == 0:
                break

        # generate category objects
        categories = [
            Category(
                id=class_id,
                name=class_name,
            )
            for class_name, class_id in category_map.items()
        ]
        coco = COCO(categories=categories, images=images, annotations=annotations)
        return coco
