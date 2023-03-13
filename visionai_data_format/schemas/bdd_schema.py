# the file is the same as the file converter_tools/bdd_vai_converter/bdd_schema.py
from typing import Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Extra, Field

BDD_VERSION = "1.1.4"


class ObjectIdSchema(BaseModel):
    project: str
    function: str
    object: str = Field(..., alias="object")
    version: str


class MetaDsSchema(BaseModel):
    score: Optional[float]
    coco_url: Optional[str]


class MetaSeSchema(BaseModel):
    status: List[str] = ["INFERENCE_MODEL", "INFERENCE_MODEL"]


class AtrributeSchema(BaseModel):
    cameraIndex: Optional[int] = 0
    INSTANCE_ID: int = 0

    class Config:
        extra = Extra.allow


class Box2dSchema(BaseModel):
    x1: Union[float, int]
    y1: Union[float, int]
    x2: Union[float, int]
    y2: Union[float, int]


def gen_uuid():
    return str(uuid4())


class PolyInfo(BaseModel):
    vertices: List[List[Union[int, float]]]
    closed: bool
    types: Optional[str] = None


class PolygonSchema(BaseModel):
    __root__: List[PolyInfo]


class FrameLabelSchema(BaseModel):
    category: str
    attributes: Optional[AtrributeSchema] = AtrributeSchema().dict()


class CategorySchema(BaseModel):
    category: str
    attributes: Optional[AtrributeSchema] = AtrributeSchema().dict()
    box2d: Optional[Box2dSchema] = None
    poly2d: Optional[PolygonSchema] = None
    point2d: Optional[PolygonSchema] = None
    meta_ds: Optional[MetaDsSchema] = {}
    meta_se: Optional[MetaSeSchema] = {}
    uuid: str = Field(default_factory=gen_uuid)
    objectId: Optional[ObjectIdSchema]

    def dict(self, *args, **kwargs) -> Dict:
        kwargs.pop("exclude_none")
        return super().dict(*args, exclude_none=True, **kwargs)


class FrameSchema(BaseModel):
    name: str
    storage: str
    dataset: str
    sequence: str
    labels: List[CategorySchema]
    frameLabels: Optional[List[FrameLabelSchema]] = []
    meta_ds: Optional[MetaDsSchema] = {}


class BDDSchema(BaseModel):
    bdd_version: str = BDD_VERSION
    company_code: Optional[str] = None
    inference_object: str = "detection"
    meta_ds: Dict = {}
    meta_se: Dict = {}
    frame_list: List[FrameSchema]
