import json
import logging
import os
import shutil
from typing import Optional

from PIL import Image as PILImage

from visionai_data_format.converters.base import Converter, ConverterFactory
from visionai_data_format.schemas.common import AnnotationFormat, OntologyImageType
from visionai_data_format.utils.classes import gen_ontology_classes_dict
from visionai_data_format.utils.common import (
    IMAGE_EXT,
    VISIONAI_JSON,
    YOLO_CATEGORY_FILE,
    YOLO_IMAGE_FOLDER,
    YOLO_LABEL_FOLDER,
)

__all__ = ["VAItoYOLO"]

logger = logging.getLogger(__name__)


@ConverterFactory.register(
    from_=AnnotationFormat.VISION_AI,
    to_=AnnotationFormat.YOLO,
    image_annotation_type=OntologyImageType._2D_BOUNDING_BOX,
)
class VAItoYOLO(Converter):
    @staticmethod
    def xywh2nxywh(obj: list, img_w: int, img_h: int) -> list:
        nx = round(float(obj[0]) / img_w, 5)
        ny = round(float(obj[1]) / img_h, 5)
        nw = round(float(obj[2]) / img_w, 5)
        nh = round(float(obj[3]) / img_h, 5)
        return [nx, ny, nw, nh]

    @classmethod
    def convert(
        cls,
        source_data_root: str,
        output_dest_folder: str,
        camera_sensor_name: str,
        ontology_classes: str = "",
        copy_sensor_data: bool = True,
        n_frame: int = -1,
        annotation_name: str = "groundtruth",
        img_extension: str = IMAGE_EXT,
        **kwargs,
    ) -> None:
        logger.info(
            f"convert VisionAI to yolo from {source_data_root} to {output_dest_folder}"
        )

        dest_img_folder = os.path.join(output_dest_folder, YOLO_IMAGE_FOLDER)
        dest_label_folder = os.path.join(output_dest_folder, YOLO_LABEL_FOLDER)
        if copy_sensor_data:
            # create {dest}/images folder #
            os.makedirs(dest_img_folder, exist_ok=True)
        # create {dest}/labels folder #
        os.makedirs(dest_label_folder, exist_ok=True)

        category_map = gen_ontology_classes_dict(ontology_classes)

        sequence_folder_list = os.listdir(source_data_root)
        image_id_start = 0
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
            with open(annotation_path) as f:
                visionai_dict = json.load(f)
            (
                category_map,
                image_labels_map,
                image_id_start,
                n_frame,
            ) = cls.convert_single_visionai_to_yolo(
                dest_img_folder=dest_img_folder,
                visionai_dict=visionai_dict,
                copy_sensor_data=copy_sensor_data,
                source_data_root=source_data_root,
                uri_root=output_dest_folder,
                camera_sensor_name=camera_sensor_name,
                image_id_start=image_id_start,
                category_map=category_map,
                n_frame=n_frame,
                img_extension=img_extension,
            )
            # output frame labels to files
            for img_path, labels in image_labels_map.items():
                label_path = (
                    f"{dest_label_folder}/{img_path.split('/')[-1].split('.')[0] }.txt"
                )
                dump_annotation = "\n".join(labels)
                with open(label_path, "w") as f:
                    f.write(dump_annotation)
        dest_category_path = os.path.join(output_dest_folder, YOLO_CATEGORY_FILE)
        if not category_map:
            logging.info("No annotation objects are found. Category file is empty.")
        dump_classes = "\n".join(list(category_map.keys()))
        with open(dest_category_path, "w") as f:
            f.write(dump_classes)
        logger.info("convert visionai to yolo format finished")

    @classmethod
    def convert_single_visionai_to_yolo(
        cls,
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
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
    ) -> tuple[dict, dict, int, int]:
        """Convert single visionai data to yolo format

        Parameters
        ----------
        dest_img_folder : str
        visionai_dict : dict
        category_map : dict
        copy_sensor_data : bool
        source_data_root : str
        uri_root : str
            uri root for target upload root folder
        camera_sensor_name : str
            for getting the target camera sensor data
        n_frame : int, optional
            number of frame to be converted (-1 means all), by default -1
        img_extension : str, optional
            by default IMAGE_EXT (.jpg)
        image_id_start : int, optional
            by default 0
        img_width : Optional[int], optional
            by default None
        img_height : Optional[int], optional
            by default None

        Returns
        -------
        tuple[dict, dict,int, int]
            category_map: dict
            output_image_label: dict
            image_id: int (image_id_start for next sequence)
            n_frame: int (number of frames left to be converted)

        Raises
        ------
        ValueError
            image data type is not supported
        """
        image_labels_map = {}
        image_id = image_id_start
        for frame_data in visionai_dict["visionai"]["frames"].values():
            if len(image_labels_map) == n_frame:
                break
            dest_yolo_url = os.path.join(
                uri_root, YOLO_IMAGE_FOLDER, f"{image_id:012d}{img_extension}"
            )
            dest_yolo_img = os.path.join(
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
                shutil.copy(source_image_path, dest_yolo_img)
            if img_width is None or img_height is None:
                img = PILImage.open(source_image_path)
                img_width, img_height = img.size
            image_labels_map[dest_yolo_url] = []

            if not frame_data.get("objects", None):
                image_id += 1
                continue

            for object_id, object_v in frame_data["objects"].items():
                # from [center x, center y, width, height] to [n-center x, n-center y, n-width, n-height]
                center_x, center_y, width, height = object_v["object_data"]["bbox"][0][
                    "val"
                ]
                bbox = cls.xywh2nxywh(
                    obj=[center_x, center_y, width, height],
                    img_w=img_width,
                    img_h=img_height,
                )
                category = visionai_dict["visionai"]["objects"][object_id]["type"]
                if category not in category_map:
                    category_map[category] = len(category_map)
                category_id = category_map[category]
                # join category and bbox as string for output to txt file
                image_labels_map[dest_yolo_url].append(
                    " ".join(map(str, [category_id] + bbox))
                )
            image_id += 1
        if n_frame != -1:
            n_frame -= len(image_labels_map[dest_yolo_url])
        return (category_map, image_labels_map, image_id, n_frame)
