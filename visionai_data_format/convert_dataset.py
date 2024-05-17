import argparse
import logging
from typing import Optional

from visionai_data_format.converters.base import ConverterFactory
from visionai_data_format.exceptions import VisionAIErrorCode, VisionAIException
from visionai_data_format.schemas.common import AnnotationFormat, OntologyImageType


class DatasetConverter:
    @classmethod
    def run(
        cls,
        input_format: str,
        output_format: str,
        image_annotation_type: str,
        source_data_root: str,
        output_dest_folder: str,
        uri_root: str,
        camera_sensor_name: str,
        lidar_sensor_name: Optional[str] = None,
        input_annotation_path: Optional[str] = None,
        sequence_idx_start: int = 0,
        copy_sensor_data: bool = True,
        n_frame: int = -1,
        annotation_name: str = "groundtruth",
        img_extension: str = ".jpg",
        ontology_classes: str = "",
        classes_file_name: str = "classes.txt",
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
    ):
        """Run Dataset Converter

        Parameters
        ----------
        input_format : str
        output_format : str
        image_annotation_type : str
            label annotation type of images (2d_bounding_box/polygon/point ...)
        source_data_root : str
            source data root for sensor data and calibration data (will use relative path for getting the files inside)
        output_dest_folder : str
        uri_root : str
            uri root for target upload data path
        camera_sensor_name : str
        lidar_sensor_name : str, optional
            lidar sensor name, by default None
        input_annotation_path : str, optional
            annotation file path, by default None
        sequence_idx_start : int, optional
            sequence start id, by default 0
        copy_sensor_data : bool, optional
            whether copy sensor files or not, by default True
        n_frame : int, optional
            number of frame to be converted (-1 means all), by default -1
        annotation_name : str, optional
            output annotation name, by default "groundtruth"
        img_extension : str, optional
            img file extension, by default ".jpg"
        ontology_classes: str, by default: ""
        classes_file_name: str, by default: "classes.txt",
        img_height: int, optional
        img_width: int, optional


        Raises
        ------
        ValueError
            If the selected convert case is not provided
        """
        input_format = AnnotationFormat(input_format)
        output_format = AnnotationFormat(output_format)
        image_annotation_type = OntologyImageType(image_annotation_type)
        converter = ConverterFactory.get(
            from_=input_format,
            to_=output_format,
            image_annotation_type=image_annotation_type,
        )
        if not converter:
            raise VisionAIException(error_code=VisionAIErrorCode.VAI_ERR_001)
        converter.convert(
            input_annotation_path=input_annotation_path,
            output_dest_folder=output_dest_folder,
            uri_root=uri_root,
            camera_sensor_name=camera_sensor_name,
            lidar_sensor_name=lidar_sensor_name,
            sequence_idx_start=sequence_idx_start,
            copy_sensor_data=copy_sensor_data,
            source_data_root=source_data_root,
            n_frame=n_frame,
            annotation_name=annotation_name,
            img_extension=img_extension,
            ontology_classes=ontology_classes,
            classes_file_name=classes_file_name,
            img_height=img_height,
            img_width=img_width,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input_format",
        type=str,
        required=True,
        help="bddp/coco/kitti",
    )
    parser.add_argument(
        "-output_format",
        type=str,
        required=True,
        help="vision_ai",
    )
    parser.add_argument(
        "-image_annotation_type",
        type=str,
        required=True,
        help="2d_bounding_box",
    )
    parser.add_argument(
        "-input_annotation_path",
        type=str,
        default="",
        help="BDD json path or coco annotation dir",
    )
    parser.add_argument(
        "-source_data_root",
        type=str,
        required=True,
        help="Source root for sensor data folder and calibration file",
    )
    parser.add_argument(
        "-output_dest_folder",
        type=str,
        required=True,
        help="Target output format destination folder path",
    )
    parser.add_argument(
        "-uri_root",
        type=str,
        required=True,
        help="uri root for storage i.e: https://azuresorate/container1",
    )
    parser.add_argument(
        "-camera_sensor_name",
        type=str,
        help="Camera Sensor name, i.e : `camera1`",
        default="",
    )
    parser.add_argument(
        "-lidar_sensor_name",
        type=str,
        help="Lidar Sensor name, i.e : `lidar1`",
        default="",
    )
    parser.add_argument(
        "-sequence_idx_start", type=int, help="sequence id start number", default=0
    )
    parser.add_argument(
        "-annotation_name",
        type=str,
        default="groundtruth",
        help=" annotation folder name (default: 'groundtruth')",
    )
    parser.add_argument(
        "-img_extension",
        type=str,
        default=".jpg",
        help="image extention (default: .jpg)",
    )
    parser.add_argument(
        "-img_width",
        type=int,
        required=False,
        help="image width",
    )
    parser.add_argument(
        "-img_height",
        type=int,
        required=False,
        help="image height",
    )
    parser.add_argument(
        "-n_frame",
        type=int,
        help="target convert frame number, -1 means all",
        default=-1,
    )
    parser.add_argument(
        "--ontology_classes",
        type=str,
        default="",
        help="','.join(ontology_classes_list), add this if we required category id follow the specified order ",
    )
    parser.add_argument(
        "-classes_file",
        type=str,
        default="classes.txt",
        help="file for store category names for yolo format",
    )
    parser.add_argument(
        "--copy_sensor_data",
        action="store_true",
        help="enable to copy image/lidar data",
    )
    FORMAT = "%(asctime)s[%(process)d][%(levelname)s] %(name)-16s : %(message)s"
    DATEFMT = "[%d-%m-%Y %H:%M:%S]"

    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG,
        datefmt=DATEFMT,
    )

    args = parser.parse_args()

    if not args.camera_sensor_name and not args.lidar_sensor_name:
        raise VisionAIException(error_code=VisionAIErrorCode.VAI_ERR_002)
    DatasetConverter.run(
        input_format=args.input_format,
        output_format=args.output_format,
        image_annotation_type=args.image_annotation_type,
        input_annotation_path=args.input_annotation_path,
        source_data_root=args.source_data_root,
        output_dest_folder=args.output_dest_folder,
        uri_root=args.uri_root,
        sequence_idx_start=args.sequence_idx_start,
        camera_sensor_name=args.camera_sensor_name,
        lidar_sensor_name=args.lidar_sensor_name,
        annotation_name=args.annotation_name,
        img_extension=args.img_extension,
        n_frame=args.n_frame,
        copy_sensor_data=args.copy_sensor_data,
        ontology_classes=args.ontology_classes,
        classes_file_name=args.classes_file,
        img_width=args.img_width,
        img_height=args.img_height,
    )
