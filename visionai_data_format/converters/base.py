from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type

from visionai_data_format.schemas.common import AnnotationFormat, OntologyImageType

__all__ = ["Converter", "ConverterFactory"]


class Converter(ABC):
    from_: Optional[AnnotationFormat] = None
    to_: Optional[AnnotationFormat] = None
    image_annotation_type: Optional[OntologyImageType] = None

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
