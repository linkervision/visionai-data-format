# the file is the same as the file converter_tools/bdd_vai_converter/bdd_schema.py
from typing import Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, RootModel

BDD_VERSION = "1.1.4"


class ObjectIdSchema(BaseModel):
    project: str
    function: str
    object: str = Field(..., alias="object")
    version: str


class MetaDsSchema(BaseModel):
    score: Optional[float] = None
    coco_url: Optional[str] = None


class ResolutionSchema(BaseModel):
    width: int
    height: int


class MetaSeSchema(BaseModel):
    status: List[str] = ["INFERENCE_MODEL", "INFERENCE_MODEL"]
    resolution: Optional[ResolutionSchema] = None


class AttributeSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    cameraIndex: Optional[int] = 0
    INSTANCE_ID: int = 0


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


class PolygonSchema(RootModel):
    root: List[PolyInfo]


class FrameLabelSchema(BaseModel):
    category: str
    attributes: Optional[AttributeSchema] = AttributeSchema().model_dump()


class SegmentSchema(BaseModel):
    bbox: List[Union[float, int]] = Field(default=[0, 0, 0, 0])
    counts: List[int]
    resolution: List[int] = Field(default=[0, 0])

    model_config = ConfigDict(
        json_schema_extra={
            "properties": {
                "bbox": {"minItems": 4, "maxItems": 4},
                "resolution": {"minItems": 2, "maxItems": 2},
            }
        }
    )


class CategorySchema(BaseModel):
    category: str
    attributes: Optional[AttributeSchema] = AttributeSchema().model_dump()
    box2d: Optional[Box2dSchema] = None
    poly2d: Optional[PolygonSchema] = None
    point2d: Optional[PolygonSchema] = None
    meta_ds: MetaDsSchema = {}
    meta_se: MetaSeSchema = {}
    uuid: str = Field(default_factory=gen_uuid)
    objectId: Optional[ObjectIdSchema] = None
    segment: Optional[SegmentSchema] = None

    def dict(self, *args, **kwargs) -> Dict:
        kwargs.pop("exclude_none")
        return super().model_dump(*args, exclude_none=True, **kwargs)


class FrameSchema(BaseModel):
    name: str
    storage: str
    dataset: str
    sequence: str
    labels: List[CategorySchema]
    frameLabels: List[FrameLabelSchema] = []
    meta_ds: MetaDsSchema = {}
    lidarPlaneURLs: List[str] = []


class BDDSchema(BaseModel):
    bdd_version: str = BDD_VERSION
    company_code: Optional[str] = None
    inference_object: str = "detection"
    meta_ds: Dict = {}
    meta_se: Dict = {}
    frame_list: List[FrameSchema]
