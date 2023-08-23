import argparse
import logging

from visionai_data_format.schemas.common import AnnotationFormat, OntologyImageType
from visionai_data_format.utils.converters import ConverterFactory


class DatasetConverter:
    @classmethod
    def run(
        cls,
        input_format: str,
        output_format: str,
        image_annotation_type: str,
        input_annotation_path: str,
        source_data_root: str,
        output_dest_folder: str,
        uri_root: str,
        sensor_name: str,
        sequence_idx_start: int = 0,
        copy_image: bool = True,
        n_img: int = -1,
        annotation_name: str = "groundtruth",
        img_extention: str = ".jpg",
    ):
        input_format = AnnotationFormat(input_format)
        output_format = AnnotationFormat(output_format)
        image_annotation_type = OntologyImageType(image_annotation_type)
        converter = ConverterFactory.get(
            from_=input_format,
            to_=output_format,
            image_annotation_type=image_annotation_type,
        )
        converter.convert(
            input_annotation_path=input_annotation_path,
            output_dest_folder=output_dest_folder,
            uri_root=uri_root,
            sensor_name=sensor_name,
            sequence_idx_start=sequence_idx_start,
            copy_image=copy_image,
            source_data_root=source_data_root,
            n_img=n_img,
            annotation_name=annotation_name,
            img_extention=img_extention,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-inpupt_format",
        type=str,
        required=True,
        help="BDDP/COCO ",
    )
    parser.add_argument(
        "-output_format",
        type=str,
        required=True,
        help="vision_ai ",
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
        required=True,
        help="BDD json path or coco annotation dir",
    )
    parser.add_argument(
        "-source_data_root",
        type=str,
        required=True,
        help="Source root for sensor data folder",
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
        help="uri root for storage i.e: https://azuresorate/container1",
    )
    parser.add_argument(
        "-sensor", type=str, help="Sensor name, i.e : `camera1`", default="camera1"
    )
    parser.add_argument(
        "-sequence_idx_start", type=int, help="seqnuece id start number", default=0
    )

    parser.add_argument(
        "-annotation_name",
        type=str,
        default="groundtruth",
        help=" annotation folder name (default: 'groundtruth')",
    )
    parser.add_argument(
        "-img_extention",
        type=str,
        default=".jpg",
        help="image extention (default: .jpg)",
    )
    parser.add_argument(
        "-n_img", type=int, help="target convert image number, -1 means all", default=-1
    )
    parser.add_argument(
        "--copy-image", action="store_true", help="enable to copy image"
    )

    FORMAT = "%(asctime)s[%(process)d][%(levelname)s] %(name)-16s : %(message)s"
    DATEFMT = "[%d-%m-%Y %H:%M:%S]"

    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG,
        datefmt=DATEFMT,
    )

    args = parser.parse_args()

    DatasetConverter.run(
        input_format=args.input_format,
        output_format=args.output_format,
        image_annotation_type=args.image_annotation_type,
        input_annotation_path=args.input_annotation_path,
        source_data_root=args.source_data_root,
        output_dest_folder=args.output_dest_folder,
        uri_root=args.uri_root,
        sequence_idx_start=args.sequence_idx_start,
        sensor_name=args.sensor,
        annotation_name=args.annotation_name,
        img_extention=args.img_extention,
        n_img=args.n_img,
        copy_image=args.copy_image,
    )
