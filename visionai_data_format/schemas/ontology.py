from typing import Dict, List, Optional, Union

from common import SensorType
from pydantic import BaseModel, StrictStr


class Attribute(BaseModel):
    type: str
    value: Optional[List[Union[float, str, int]]] = None


class ElementData(BaseModel):
    attributes: Dict[StrictStr, Attribute] = None


class Stream(BaseModel):
    type: SensorType

    class Config:
        use_enum_values = True


class Tag(BaseModel):
    classes: List[str]


class Ontology(BaseModel):
    contexts: Optional[Dict[StrictStr, ElementData]] = None
    objects: Optional[Dict[StrictStr, ElementData]] = None
    taggings: Optional[Dict[StrictStr, ElementData]] = None
    streams: Dict[StrictStr, Stream]
    tags: Optional[Tag] = None
