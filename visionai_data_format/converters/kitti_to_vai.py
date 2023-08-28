import logging
import math
import os
import shutil
import uuid

import cv2
import numpy as np

from visionai_data_format.converters.base import Converter, ConverterFactory
from visionai_data_format.schemas.common import AnnotationFormat, OntologyImageType
from visionai_data_format.schemas.visionai_schema import (
    Bbox,
    Cuboid,
    DynamicObjectData,
    Frame,
    FrameInterval,
    FrameProperties,
    FramePropertyStream,
    Object,
    ObjectDataPointer,
    ObjectType,
    ObjectUnderFrame,
)
from visionai_data_format.utils.calculation import project_rect_to_velo
from visionai_data_format.utils.common import (
    KITTI_BOX_BOTTOM,
    KITTI_BOX_LEFT,
    KITTI_BOX_RIGHT,
    KITTI_BOX_TOP,
    KITTI_CLS_INDEX,
    KITTI_DIM_HEIGHT,
    KITTI_DIM_LENGTH,
    KITTI_DIM_WIDTH,
    KITTI_POS_X,
    KITTI_POS_Y,
    KITTI_POS_Z,
    KITTI_ROT_Y,
    VISIONAI_JSON,
)
from visionai_data_format.utils.validator import (
    parse_calib_data,
    save_as_json,
    validate_vai,
)

__all__ = ["KITTItoVAI"]

logger = logging.getLogger(__name__)


@ConverterFactory.register(
    from_=AnnotationFormat.KITTI,
    to_=AnnotationFormat.VISION_AI,
    image_annotation_type=OntologyImageType._2D_BOUNDING_BOX,
)
class KITTItoVAI(Converter):
    @classmethod
    def convert(
        cls,
        source_data_root: str,
        output_dest_folder: str,
        camera_sensor_name: str,
        lidar_sensor_name: str,
        uri_root: str,
        sequence_idx_start: int = 0,
        copy_sensor_data: bool = True,
        n_frame: int = -1,
        annotation_name: str = "groundtruth",
        img_extension: str = ".jpg",
        **kwargs,
    ) -> None:
        image_folder_path = os.path.join(source_data_root, "data")
        try:
            for sequence_idx, image_path in enumerate(
                os.listdir(image_folder_path), sequence_idx_start
            ):
                if n_frame > 0:
                    n_frame -= 1
                dest_sequence_name = f"{sequence_idx:012d}"
                image_file_path = os.path.join(image_folder_path, image_path)
                old_sequence_idx = os.path.splitext(image_path)[0].split(os.sep)[-1]
                logger.info(
                    f"convert sequence {old_sequence_idx} to {dest_sequence_name}"
                )
                try:
                    cls.convert_kitti_to_vai(
                        vai_dest_folder=output_dest_folder,
                        camera_sensor_name=camera_sensor_name,
                        lidar_sensor_name=lidar_sensor_name,
                        image_file_path=image_file_path,
                        dest_sequence_name=dest_sequence_name,
                        uri_root=uri_root,
                        annotation_name=annotation_name,
                        img_extension=img_extension,
                        copy_sensor_data=copy_sensor_data,
                        source_data_root=source_data_root,
                    )
                except Exception as e:
                    logger.error("[convert_kitti_to_vai] Convert failed : " + str(e))

                if n_frame == 0:
                    break
        except Exception as e:
            logger.error("Convert kitti to vai format failed : " + str(e))

    @staticmethod
    def convert_kitti_to_vai(
        vai_dest_folder: str,
        camera_sensor_name: str,
        lidar_sensor_name: str,
        image_file_path: str,
        uri_root: str,
        dest_sequence_name: str,
        source_data_root: str,
        annotation_name: str = "groundtruth",
        img_extension: str = ".jpg",
        copy_sensor_data: bool = True,
    ) -> None:
        try:
            image_file_path_without_extension, _ = os.path.splitext(image_file_path)
            source_sequence_name = os.path.split(image_file_path_without_extension)[-1]
            pcd_path = (
                None
                if not lidar_sensor_name
                else os.path.join(
                    source_data_root, "pcd", f"{source_sequence_name}.pcd"
                )
            )
            calib_path = (
                None
                if not lidar_sensor_name
                else os.path.join(
                    source_data_root, "calib", f"{source_sequence_name}.txt"
                )
            )
            label_path = os.path.join(
                source_data_root, "labels", f"{source_sequence_name}.txt"
            )

            dict_calib = {} if not calib_path else parse_calib_data(calib_path)

            lidar_coor_system = (
                {}
                if not lidar_sensor_name
                else {
                    lidar_sensor_name: {
                        "type": "sensor_cs",
                        "parent": "",
                        "children": [],
                    }
                }
            )
            cam_coor_system = (
                {}
                if not camera_sensor_name
                else {
                    camera_sensor_name: {
                        "type": "sensor_cs",
                        "parent": "",
                        "children": [],
                        "pose_wrt_parent": None
                        if "matrix_4_4" not in dict_calib
                        else {
                            "matrix4x4": dict_calib["matrix_4_4"],
                        },
                    }
                }
            )

            coor_system = {}
            coor_system.update(lidar_coor_system)
            coor_system.update(cam_coor_system)

            label_data = []
            with open(label_path, encoding="utf8") as f:
                label_data = f.read().split("\n")

            frame_intervals = []
            frames = {}
            objects = {}
            frame_num = (
                f"{0:012d}"  # TODO: change it when using sequential kitti dataset
            )

            frame_properties = {"streams": {}}
            camera_url = ""
            if camera_sensor_name:
                dest_camera_folder = os.path.join(
                    vai_dest_folder, dest_sequence_name, "data", camera_sensor_name
                )
                dest_camera_path = os.path.join(
                    dest_camera_folder, frame_num + img_extension
                )
                camera_url = os.path.join(uri_root, dest_camera_path)
                frame_properties["streams"].update(
                    {
                        camera_sensor_name: FramePropertyStream(uri=camera_url),
                    }
                )
                if copy_sensor_data:
                    os.makedirs(dest_camera_folder, exist_ok=True)
                    shutil.copy2(image_file_path, dest_camera_path)

            if lidar_sensor_name:
                dest_lidar_folder = os.path.join(
                    vai_dest_folder, dest_sequence_name, "data", lidar_sensor_name
                )
                dest_lidar_path = os.path.join(dest_lidar_folder, frame_num + ".pcd")
                lidar_url = os.path.join(uri_root, dest_lidar_path)

                frame_properties["streams"].update(
                    {
                        lidar_sensor_name: FramePropertyStream(uri=lidar_url),
                    }
                )
                if copy_sensor_data:
                    os.makedirs(dest_lidar_folder, exist_ok=True)
                    shutil.copy2(pcd_path, dest_lidar_path)

            frames[frame_num] = Frame(
                frame_properties=FrameProperties(**frame_properties),
                objects={},
            )

            bbox_name = "bbox_shape"
            cuboid_name = "cuboid_shape"
            for label in label_data:
                label_info = label.split(" ")
                if not label_info or label_info[0] == "":
                    continue
                cls = label_info[KITTI_CLS_INDEX]
                # x1,y1,x2,y2 -> x_center, y_center, w, h
                bbox = [
                    (
                        float(label_info[KITTI_BOX_LEFT])
                        + float(label_info[KITTI_BOX_RIGHT])
                    )
                    / 2,
                    (
                        float(label_info[KITTI_BOX_TOP])
                        + float(label_info[KITTI_BOX_BOTTOM])
                    )
                    / 2,
                    float(label_info[KITTI_BOX_RIGHT])
                    - float(label_info[KITTI_BOX_LEFT]),
                    float(label_info[KITTI_BOX_BOTTOM])
                    - float(label_info[KITTI_BOX_TOP]),
                ]

                object_uuid = str(uuid.uuid4())

                object_data_pointers = {}
                if camera_sensor_name:
                    object_data_pointers.update(
                        {
                            bbox_name: ObjectDataPointer(
                                type=ObjectType.BBOX,
                                frame_intervals=[
                                    FrameInterval(
                                        frame_start=0,
                                        frame_end=0,
                                    )
                                ],
                            )
                        }
                    )
                if lidar_sensor_name:
                    object_data_pointers.update(
                        {
                            cuboid_name: ObjectDataPointer(
                                type=ObjectType.CUBOID,
                                frame_intervals=[
                                    FrameInterval(
                                        frame_start=0,
                                        frame_end=0,
                                    )
                                ],
                            )
                        }
                    )
                objects[object_uuid] = Object(
                    name=cls,
                    type=cls,
                    frame_intervals=[
                        FrameInterval(
                            frame_start=0,
                            frame_end=0,
                        )
                    ],
                    object_data_pointers=object_data_pointers,
                )

                pcs_3d_rect = np.array(
                    [
                        [
                            float(label_info[KITTI_POS_X]),
                            float(label_info[KITTI_POS_Y]),
                            float(label_info[KITTI_POS_Z]),
                        ]
                    ],
                    dtype=np.float16,
                )

                R0_rect = dict_calib.get("R0_rect")
                Tr_cam_to_velo = dict_calib.get("Tr_cam_to_velo")
                x, y, z = (
                    (0, 0, 0)
                    if pcs_3d_rect is None or R0_rect is None or Tr_cam_to_velo is None
                    else project_rect_to_velo(pcs_3d_rect, R0_rect, Tr_cam_to_velo)[0]
                )
                z += float(label_info[KITTI_DIM_HEIGHT]) / 2

                frames[frame_num].objects.update(
                    {
                        object_uuid: ObjectUnderFrame(
                            object_data=DynamicObjectData(
                                bbox=(
                                    [
                                        Bbox(
                                            name=bbox_name,
                                            val=bbox,
                                            stream=camera_sensor_name,
                                            coordinate_system=camera_sensor_name,
                                        )
                                    ]
                                ),
                                cuboid=(
                                    None
                                    if not lidar_coor_system
                                    else [
                                        Cuboid(
                                            name=cuboid_name,
                                            val=[
                                                x,
                                                y,
                                                z,
                                                0,
                                                0,
                                                -(
                                                    float(label_info[KITTI_ROT_Y])
                                                    + math.pi / 2
                                                ),
                                                float(label_info[KITTI_DIM_LENGTH]),
                                                float(label_info[KITTI_DIM_WIDTH]),
                                                float(label_info[KITTI_DIM_HEIGHT]),
                                            ],
                                            stream=lidar_sensor_name,
                                            coordinate_system=lidar_sensor_name,
                                        )
                                    ]
                                ),
                            )
                        )
                    }
                )

            # to vision_ai: frame_intervals
            frame_intervals.append(FrameInterval(frame_start=0, frame_end=0))
            for v in objects.values():
                new_interval = []
                for interval in v.frame_intervals:
                    if not new_interval:
                        new_interval.append(interval)
                        continue
                    last_interval = new_interval[-1]
                    if interval.frame_start - last_interval.frame_end <= 1:
                        last_interval.frame_end = interval.frame_end
                    else:
                        new_interval.append(interval)
                v.frame_intervals = []
                v.frame_intervals = new_interval
                new_interval = []
                for interval in v.object_data_pointers["bbox_shape"].frame_intervals:
                    if not new_interval:
                        new_interval.append(interval)
                        continue
                    last_interval = new_interval[-1]
                    if interval.frame_start - last_interval.frame_end <= 1:
                        last_interval.frame_end = interval.frame_end
                    else:
                        new_interval.append(interval)
                v.object_data_pointers["bbox_shape"].frame_intervals = new_interval

            camera_stream_properties = {}
            if dict_calib.get("camera_matrix_3x4"):
                img = cv2.imread(image_file_path)
                img_height, img_width = img.shape[:2]
                camera_stream_properties = {
                    "intrinsics_pinhole": {
                        "camera_matrix_3x4": dict_calib["camera_matrix_3x4"],
                        "height_px": img_height,
                        "width_px": img_width,
                    }
                }
            streams = {
                camera_sensor_name: {
                    "type": "camera",
                    "uri": camera_url,
                    "description": "Frontal camera",
                    "stream_properties": camera_stream_properties,
                }
            }

            if lidar_sensor_name:
                streams.update(
                    {
                        lidar_sensor_name: {
                            "type": "lidar",
                            "description": "Central lidar",
                            "uri": lidar_url,
                        }
                    }
                )

            vai_data = {
                "visionai": {
                    "frame_intervals": frame_intervals,
                    "frames": frames,
                    "objects": objects,
                    "metadata": {"schema_version": "1.0.0"},
                    "coordinate_systems": coor_system,
                    "streams": streams,
                }
            }
            vai_data = validate_vai(vai_data).dict(exclude_none=True)
            save_as_json(
                vai_data,
                folder_name=os.path.join(
                    vai_dest_folder, dest_sequence_name, "annotations", annotation_name
                ),
                file_name=VISIONAI_JSON,
            )
            logger.info(
                f"[convert_kitti_to_vai] Convert sequence {dest_sequence_name} finished"
            )
        except Exception as e:
            logger.error("[convert_kitti_to_vai] Convert failed : " + str(e))
