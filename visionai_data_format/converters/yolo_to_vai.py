import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

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
from visionai_data_format.utils.validator import save_as_json, validate_vai

__all__ = ["YOLOtoVAI"]

logger = logging.getLogger(__name__)

YOLO_IMAGE_FOLDER = "images"
YOLO_LABEL_FOLDER = "labels"
IMAGE_EXTS = ["jpg", ".jpeg", ".png"]


@ConverterFactory.register(
    from_=AnnotationFormat.YOLO,
    to_=AnnotationFormat.VISION_AI,
    image_annotation_type=OntologyImageType._2D_BOUNDING_BOX,
)
class YOLOtoVAI(Converter):
    @staticmethod
    def nxywh2xywh(obj, img_w, img_h):
        x = int(obj[0] * img_w)
        y = int(obj[1] * img_h)
        w = int(obj[2] * img_w)
        h = int(obj[3] * img_h)

        return x, y, w, h

    @classmethod
    def convert(
        cls,
        camera_sensor_name: str,
        source_data_root: str,
        output_dest_folder: str,
        uri_root: str,
        sequence_idx_start: int = 0,
        copy_sensor_data: bool = True,
        n_frame: int = -1,
        annotation_name: str = "groundtruth",
        img_extension: str = ".jpg",
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        classes_file_name: str = "classes.txt",
        **kwargs,
    ) -> None:
        """convert yolo format data to visionai data format

        Parameters
        ----------
        camera_sensor_name : str
        source_data_root : str
            data root folder of yolo format
        output_dest_folder : str
        uri_root : str
            uri root for target upload VisionAI
        sequence_idx_start : int, optional
            sequence start id, by default 0
        copy_sensor_data : bool, optional
            enable to copy image data, by default True
        n_frame : int, optional
            number of frame to be converted (-1 means all), by default -1
        annotation_name : str, optional
            VisionAI annotation folder name , by default "groundtruth"
        img_extension : str, optional
            image file extension, by default ".jpg"
        img_height : Optional[int], optional
            image height for all images, by default None
        img_width : Optional[int], optional
            image width for all images, by default None
        classes_file_name : str, optional
            txt file contain category names in each line, by default "classes.txt"
        """
        try:
            classes_file_path = os.path.join(source_data_root, classes_file_name)
            if not Path(classes_file_path).exists():
                raise FileNotFoundError(
                    "Please ensure your classes txt file is under source data root"
                )
            with open(classes_file_path) as classes_file:
                classes_list: list = [line.strip() for line in classes_file]
            image_folder_path = Path(source_data_root) / YOLO_IMAGE_FOLDER
            annotation_folder = Path(source_data_root) / YOLO_LABEL_FOLDER

            image_file_paths = []
            # Get image files
            for img_ext in IMAGE_EXTS:
                image_file_paths += list(image_folder_path.rglob(f"*{img_ext}"))
            for sequence_idx, img_file in enumerate(
                image_file_paths, sequence_idx_start
            ):
                if n_frame > 0:
                    n_frame -= 1
                annotation_path = annotation_folder / f"{img_file.stem}.txt"
                # The image may not have any applicable annotation txt file.
                if annotation_path.exists():
                    with open(str(annotation_path)) as anno_file:
                        label_list: list = [line.strip() for line in anno_file]
                else:
                    logging.info(
                        f"{str(img_file)} has not mapping annotation file and is consider as an empty image."
                    )
                    label_list = []
                dest_sequence_name = f"{sequence_idx:012d}"
                if not img_height or not img_width:
                    img = Image.open(str(img_file))
                    img_width, img_height = img.size

                vai_data = cls.convert_yolo_label_vai(
                    image_file_path=str(img_file),
                    label_list=label_list,
                    img_height=img_height,
                    img_width=img_width,
                    vai_dest_folder=output_dest_folder,
                    classes_list=classes_list,
                    camera_sensor_name=camera_sensor_name,
                    dest_sequence_name=dest_sequence_name,
                    uri_root=uri_root,
                    img_extension=img_extension,
                    copy_sensor_data=copy_sensor_data,
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
                    "original_format": "YOLO",
                    "destination_format": "VisionAI",
                },
            )

        except Exception:
            logger.exception("Convert yolo to vai failed")
            raise VisionAIException(error_code=VisionAIErrorCode.VAI_ERR_999)

    @classmethod
    def convert_yolo_label_vai(
        cls,
        image_file_path: str,
        label_list: list,
        classes_list: list,
        img_height: int,
        img_width: int,
        vai_dest_folder: str,
        camera_sensor_name: str,
        dest_sequence_name: str,
        uri_root: str,
        img_extension: str = ".jpg",
        copy_sensor_data: bool = True,
    ) -> dict:
        try:
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
            streams = {camera_sensor_name: Stream(type=StreamType.CAMERA)}
            frame_intervals = [FrameInterval(frame_start=0, frame_end=0)]
            # parse yolo-labels
            for obj_idx, label in enumerate(label_list):
                object_id = str(uuid.uuid4())
                label_items = label.split()
                # yolo format [class_id, center x, center y, width, height]
                class_id = int(label_items[0])
                obj = [float(loc) for loc in label_items[1:]]

                bbox = cls.nxywh2xywh(obj=obj, img_h=img_height, img_w=img_width)

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
                        name=f"{classes_list[class_id]}_{obj_idx}",
                        type=classes_list[class_id],
                        frame_intervals=frame_intervals,
                        object_data_pointers={
                            bbox_name: ObjectDataPointer(
                                type=ObjectType.BBOX,
                                frame_intervals=frame_intervals,
                            )
                        },
                    )
                }
                objects.update(object_under_vai)

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
            logger.info("[convert_yolo_to_vai] Convert finished")
            return vai_data
        except Exception as e:
            logger.error("[convert_yolo_to_vai] Convert failed : " + str(e))
