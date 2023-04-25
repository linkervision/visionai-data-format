from enum import Enum
from typing import Dict, List, Optional, Union

from common import SensorType
from pydantic import BaseModel, StrictStr


class AttributeType(str, Enum):
    BOOLEAN = "boolean"
    NUM = "num"
    VEC = "vec"
    TEXT = "text"


class OntologyType(str, Enum):
    POINT = "point"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    BOUNDING_BOX = " bounding_box"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    CLASSIFICATION = "classification"


class Attribute(BaseModel):
    type: AttributeType
    value: Optional[List[Union[float, str, int]]] = None

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
    taggings: Optional[Dict[StrictStr, Attribute]] = None
    streams: Dict[StrictStr, Stream]
    tags: Optional[Dict[StrictStr, OntologyInfo]] = None
    type: str = OntologyType
