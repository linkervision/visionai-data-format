import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, Type

from visionai_data_format.schemas.bdd_schema import BDDSchema
from visionai_data_format.schemas.common import AnnotationFormat, OntologyImageType
from visionai_data_format.utils.converter import convert_bdd_to_vai
from visionai_data_format.utils.validator import validate_bdd

logger = logging.getLogger(__name__)


class Converter(ABC):
    from_: Optional[AnnotationFormat] = None
    to_: Optional[AnnotationFormat] = None

    @classmethod
    @abstractmethod
    def convert(cls, data: Dict, *args, **kwargs) -> Any:
        raise NotImplementedError


class ConverterFactory:
    _MAP: Dict[
        Tuple[AnnotationFormat, AnnotationFormat, OntologyImageType], Type[Converter]
    ] = {}

    @classmethod
    def get(
        cls,
        from_: AnnotationFormat,
        to_: AnnotationFormat,
        image_annotation_type: OntologyImageType,
    ) -> Optional[Type[Converter]]:
        converter_class = cls._MAP.get((from_, to_, image_annotation_type))
        if converter_class:
            converter_class.from_ = from_
            converter_class.to_ = to_
            converter_class.image_annotation_type = image_annotation_type
        return converter_class

    @classmethod
    def _register(
        cls,
        from_: AnnotationFormat,
        to_: AnnotationFormat,
        image_annotation_type: OntologyImageType,
        converter: Type[Converter],
    ) -> None:
        cls._MAP[(from_, to_, image_annotation_type)] = converter

    @staticmethod
    def register(
        from_: AnnotationFormat,
        to_: AnnotationFormat,
        image_annotation_type: OntologyImageType,
    ):
        def wrap(cls):
            ConverterFactory._register(from_, to_, image_annotation_type, cls)
            return cls

        return wrap


@ConverterFactory.register(
    from_=AnnotationFormat.BDDP,
    to_=AnnotationFormat.VISION_AI,
    image_annotation_type=OntologyImageType._2D_BOUNDING_BOX,
)
class BDDtoVAI(Converter):
    @classmethod
    def convert(
        cls,
        input_annotation_path: str,
        output_dest_folder: str,
        sensor_name: str,
        source_data_root: str,
        uri_root: str,
        sequence_idx_start: int = 0,
        copy_image: bool = True,
        n_img: int = -1,
        annotation_name: str = "groundtruth",
        img_extention: str = ".jpg",
    ) -> None:
        try:
            raw_data = json.load(open(input_annotation_path))
            bdd_data = validate_bdd(raw_data).dict()
            # create sequence/frame/mapping
            sequence_frames = defaultdict(list)
            for frame in bdd_data["frame_list"]:
                seq_name = frame["sequence"]
                dataset_name = frame["dataset"]
                storage_name = frame["storage"]
                sequence_frames[(storage_name, dataset_name, seq_name)].append(frame)
            # one bdd file might contain mutiple sequqences
            seq_id = sequence_idx_start
            for sequence_key, frame_list in sequence_frames.items():
                if n_img > 0:
                    frame_count = len(frame_list)
                    if n_img < frame_count:
                        frame_list = frame_list[:n_img]
                    n_img -= len(frame_list)
                sequence_bdd_data = BDDSchema(frame_list=frame_list).dict()
                sequence_name = f"{seq_id:012d}"
                logger.info(f"convert sequence {sequence_key} to {sequence_name}")
                convert_bdd_to_vai(
                    bdd_data=sequence_bdd_data,
                    vai_dest_folder=output_dest_folder,
                    sensor_name=sensor_name,
                    sequence_name=sequence_name,
                    uri_root=uri_root,
                    annotation_name=annotation_name,
                    img_extention=img_extention,
                    copy_image=copy_image,
                    source_data_root=source_data_root,
                )
                seq_id += 1
                if n_img == 0:
                    break
        except Exception as e:
            logger.error("Convert bdd to vai format failed : " + str(e))
