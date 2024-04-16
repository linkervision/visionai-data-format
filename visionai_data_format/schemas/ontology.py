from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, StrictStr

from visionai_data_format.schemas.common import SensorType


class AttributeType(str, Enum):
    BBOX = "bbox"
    CUBOID = "cuboid"
    POINT2D = "point2d"
    POLY2D = "poly2d"
    IMAGE = "image"
    BOOLEAN = "boolean"
    NUM = "num"
    VEC = "vec"
    TEXT = "text"
    BINARY = "binary"


class Attribute(BaseModel):
    type: AttributeType
    value: Optional[List[Union[str, int, float]]] = None

    class Config:
        use_enum_values = True


class OntologyInfo(BaseModel):
    attributes: Dict[StrictStr, Attribute] = None


class Stream(BaseModel):
    type: SensorType

    class Config:
        use_enum_values = True


class Ontology(BaseModel):
    contexts: Optional[Dict[StrictStr, OntologyInfo]] = None
    objects: Optional[Dict[StrictStr, OntologyInfo]] = None
    streams: Dict[StrictStr, Stream]
    tags: Optional[Dict[StrictStr, OntologyInfo]] = None
