from enum import Enum, EnumMeta
from typing import Any, Optional, Set

from pydantic import BaseModel


class BaseEnumMeta(EnumMeta):
    _value_set: Optional[Set[Any]] = None

    def __contains__(cls, item):
        if cls._value_set is None:
            cls._value_set: Set[Any] = {v.value for v in cls.__members__.values()}

        return item in cls._value_set


class OntologyImageType(str, Enum, metaclass=BaseEnumMeta):
    _2D_BOUNDING_BOX = "2d_bounding_box"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    CLASSIFICATION = "classification"
    POINT = "point"
    POLYGON = "polygon"
    POLYLINE = "polyline"


class OntologyPcdType(str, Enum, metaclass=BaseEnumMeta):
    CUBOID = "cuboid"


class SensorType(str, Enum, metaclass=BaseEnumMeta):
    CAMERA = "camera"
    LIDAR = "lidar"


class AnnotationFormat(str, Enum, metaclass=BaseEnumMeta):
    VISION_AI = "vision_ai"
    COCO = "coco"
    BDDP = "bddp"
    IMAGE = "image"
    KITTI = "kitti"


class DatasetType(str, Enum, metaclass=BaseEnumMeta):
    ANNOTATED_DATA = "annotated_data"
    RAW_DATA = "raw_data"


class ExcludedNoneBaseModel(BaseModel):
    def dict(self, **kwargs):
        exclude_none = kwargs.pop("exclude_none", True)
        exclude_unset = kwargs.pop("exclude_unset", True)
        return super().dict(
            exclude_none=exclude_none, exclude_unset=exclude_unset, **kwargs
        )

    def json(self, **kwargs):
        exclude_none = kwargs.pop("exclude_none", True)
        exclude_unset = kwargs.pop("exclude_unset", True)
        return super().json(
            exclude_none=exclude_none, exclude_unset=exclude_unset, **kwargs
        )
