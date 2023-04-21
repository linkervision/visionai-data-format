from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, validator

from .common import OntologyImageType, OntologyPcdType, SensorType


class AttributeType(str, Enum):
    BOOLEAN = "boolean"
    NUM = "num"
    NUMBER = "number"  # TODO: remove this when BE is read
    OPTION = "option"
    TEXT = "text"


class AttributeOption(BaseModel):
    value: Union[str, float, int, bool]


class Attribute(BaseModel):
    name: str
    options: Optional[List[AttributeOption]] = None
    type: AttributeType

    class Config:
        use_enum_values = True

    @validator("type")
    def option_data_validator(cls, value, values, **kwargs):
        if value == AttributeType.OPTION and not values.get("options"):
            raise ValueError(
                "Need to assign value for `options` "
                + "if the Attribute type is option"
            )
        return value


class OntologyClass(BaseModel):
    name: str
    attributes: Optional[List[Attribute]] = None


class Ontology(BaseModel):
    image_type: Optional[OntologyImageType] = None
    pcd_type: Optional[OntologyPcdType] = None
    classes: Optional[list[OntologyClass]] = None

    class Config:
        use_enum_values = True


class ProjectTag(BaseModel):
    attributes: Optional[list[Attribute]] = None

    class Config:
        use_enum_values = True


class Sensor(BaseModel):
    name: str
    type: SensorType

    class Config:
        use_enum_values = True


class Project(BaseModel):
    ontology: Ontology
    sensors: list[Sensor]
    project_tag: Optional[ProjectTag] = None
