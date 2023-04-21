from typing import List, Optional, Union

from pydantic import BaseModel, validator

from .common import OntologyImageType, OntologyPcdType
from .visionai_schema import AttributeType


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
