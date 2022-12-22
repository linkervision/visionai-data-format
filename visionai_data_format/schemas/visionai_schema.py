# modify from openlabel_json_schema.py

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Extra, Field, validator


class Type(str, Enum):
    value = "value"


class TypeElement(str, Enum):
    bbox = "bbox"
    cuboid = "cuboid"
    point2d = "point2d"
    poly2d = "poly2d"
    text = "text"
    image = "image"
    mat = "mat"


class TypeMinMax(str, Enum):
    value = "value"
    min = "min"
    max = "max"


class TypeRange(str, Enum):
    values = "values"
    range = "range"


class TypeAttribute(str, Enum):
    boolean = "boolean"
    number = "number"
    text = "text"
    vec = "vec"


class TypeStream(str, Enum):
    camera = "camera"
    lidar = "lidar"
    radar = "radar"
    gps_imu = "gps_imu"
    other = "other"


class ObjectInFrame(BaseModel):
    object_data: FrameObjectData


class FramePropertyInfo(BaseModel):
    uri: str = Field(..., description="the urls of image")

    class Config:
        extra = Extra.allow


class FramePropertyStream(BaseModel):
    __root__: dict[str, FramePropertyInfo]


class FrameProperties(BaseModel):
    streams: FramePropertyStream = Field(default_factory=dict)


class Frame(BaseModel):
    class Config:
        extra = Extra.forbid

    objects: dict[str, ObjectInFrame] = Field(
        default_factory=dict,
        description="This is a JSON object that contains dynamic information on VisionAI objects."
        + " Object keys are strings containing numerical UIDs or 32 bytes UUIDs."
        + ' Object values may contain an "object_data" JSON object.',
    )

    contexts: dict[str, ContextInFrame] = Field(
        default_factory=dict,
        description="This is a JSON object that contains dynamic information on VisionAI contexts."
        + " Context keys are strings containing numerical UIDs or 32 bytes UUIDs."
        + ' Context values may contain an "context_data" JSON object.',
    )

    frame_properties: FrameProperties = Field(
        default_factory=dict,
        description="This is a JSON object which contains information about this frame.",
    )


class FrameInterval(BaseModel):
    class Config:
        extra = Extra.forbid

    frame_start: int = Field(..., description="Initial frame number of the interval.")
    frame_end: int = Field(..., description="Ending frame number of the interval.")


class Attributes(BaseModel):
    class Config:
        extra = Extra.forbid

    boolean: list[Boolean] = Field(default_factory=list)
    number: list[Number] = Field(default_factory=list)
    text: list[Text] = Field(default_factory=list)
    vec: list[Vec] = Field(default_factory=list)


class ObjectDataElement(BaseModel):

    attributes: Attributes = Field(default_factory=dict)
    name: str = Field(
        ...,
        description="This is a string encoding the name of this object data."
        + " It is used as index inside the corresponding object data pointers.",
    )
    stream: str = Field(
        ...,
        description="Name of the stream in respect of which this object data is expressed.",
    )
    coordinate_system: Optional[str] = Field(
        None,
        description="Name of the coordinate system in respect of which this object data is expressed.",
    )
    confidence_score: Optional[float] = Field(
        None,
        description="The confidence score of model prediction of this object."
        + " Ground truth does not have this attribute.",
    )


class ObjectDataTextElement(ObjectDataElement):
    name: Optional[str] = Field(
        "semantic_mask",
        description="This is a string encoding the name of this object data.",
    )
    val: str = Field(
        ..., description="This is a string encoding the path of this object data."
    )

    class Config:
        extra = Extra.allow


class ObjectDataMatrixElement(ObjectDataElement):

    height: int = Field(
        ..., description="This is the height (number of rows) of the matrix."
    )
    width: int = Field(
        ..., description="This is the width (number of columns) of the matrix."
    )
    channels: int = Field(
        ..., description="This is the number of channels of the matrix."
    )

    data_type: str = Field(
        ...,
        description="This is a string declares the type of values of the matrix."
        + " Only `float` or `int` allowed",
    )

    val: list[Union[float, int]] = Field(
        ..., description="This is a list of flattened values of the matrix."
    )

    class Config:
        extra = Extra.allow

    @validator("data_type")
    def validate_data_type(cls, value):
        allowed_type = {"float", "int"}
        if value not in allowed_type:
            raise ValueError("only `float` or `int` allowed for `data_type` field")
        return value

    @validator("val")
    def validate_val(cls, value, values):
        allowed_type = {"float": float, "int": int}
        if not (data_type := values.get("data_type")):
            raise ValueError("Need to define the `data_type`")

        cur_type = allowed_type.get(data_type)
        if not all(type(n) is cur_type for n in value):
            raise ValueError("Val contains not allowed value")

        return value


class Bbox(ObjectDataElement):
    class Config:
        extra = Extra.allow

    val: list[float] = Field(
        ...,
        description="The array of 4 values that define the [x, y, w, h] values of the bbox.",
        max_items=4,
        min_items=4,
    )


class Point2D(ObjectDataElement):
    class Config:
        extra = Extra.allow

    val: list[float] = Field(
        ...,
        description="The array of 2 values that define the [x, y] values of the point.",
        max_items=2,
        min_items=2,
    )


class Poly2D(ObjectDataElement):
    class Config:
        extra = Extra.allow

    val: list[float] = Field(
        ...,
        description="The array of values that define the [x] values of the point"
        + " in the odd index positions and [y] values of the point in the even index positions.",
        min_items=2,
    )
    closed: bool = Field(
        ...,
        description="The boolean value to define whether current polygon is a polygon or a polyline",
    )

    mode: Literal["MODE_POLY2D_ABSOLUTE"] = "MODE_POLY2D_ABSOLUTE"

    @validator("val")
    def val_length_must_be_even(cls, v):
        if len(v) % 2 != 0:
            raise ValueError("Array length must be even number")
        return v


class Cuboid(ObjectDataElement):
    class Config:
        extra = Extra.allow

    val: list[float] = Field(
        default_factory=list,
        description="List of values encoding the position, rotation and dimensions."
        + " Two options are supported, using 9 or 10 values. If 9 values are used,"
        + " the format is (x, y, z, rx, ry, rz, sx, sy, sz), where (x, y, z) encodes the position,"
        + " (rx, ry, rz) encodes the Euler angles that encode the rotation,"
        + " and (sx, sy, sz) are the dimensions of the cuboid in its object coordinate system."
        + " If 10 values are used, then the format is (x, y, z, qx, qy, qz, qw, sx, sy, sz)"
        + " with the only difference of the rotation values which are the 4 values of a quaternion.",
        min_items=9,
        max_items=9,
    )


class Boolean(BaseModel):

    attributes: Attributes = Field(default_factory=dict)
    name: str = Field(
        ...,
        description="This is a string encoding the name of this object data."
        + " It is used as index inside the corresponding object data pointers.",
    )
    type: Optional[Type] = Field(
        None,
        description="This attribute specifies how the boolean shall be considered."
        + " In this schema the only possible option is as a value.",
    )
    val: bool = Field(..., description="The boolean value.")
    coordinate_system: Optional[str] = Field(
        None,
        description="Name of the coordinate system in respect of which this object data is expressed.",
    )

    class Config:
        extra = Extra.allow
        use_enum_values = True


class Number(BaseModel):
    class Config:
        use_enum_values = True
        extra = Extra.allow

    attributes: Attributes = Field(default_factory=dict)
    name: str = Field(
        ...,
        description="This is a string encoding the name of this object data."
        + " It is used as index inside the corresponding object data pointers.",
    )
    type: Optional[TypeMinMax] = Field(
        None,
        description="This attribute specifies whether the number shall be considered "
        + "as a value, a minimum, or a maximum in its context.",
    )
    val: float = Field(..., description="The numerical value of the number.")
    coordinate_system: Optional[str] = Field(
        None,
        description="Name of the coordinate system in respect of which this object data is expressed.",
    )


class Text(BaseModel):
    class Config:
        use_enum_values = True
        extra = Extra.allow

    attributes: Attributes = Field(default_factory=dict)
    name: Optional[str] = Field(
        None,
        description="This is a string encoding the name of this object data."
        + " It is used as index inside the corresponding object data pointers.",
    )
    type: Optional[Type] = Field(
        None,
        description="This attribute specifies how the text shall be considered."
        + " The only possible option is as a value.",
    )
    val: str = Field(..., description="The characters of the text.")
    coordinate_system: Optional[str] = Field(
        None,
        description="Name of the coordinate system in respect of which this object data is expressed.",
    )


class Vec(BaseModel):
    class Config:
        use_enum_values = True
        extra = Extra.allow

    attributes: Attributes = Field(default_factory=dict)
    name: str = Field(
        ...,
        description="This is a string encoding the name of this object data."
        + " It is used as index inside the corresponding object data pointers.",
    )
    type: Optional[TypeRange] = Field(
        None,
        description="This attribute specifies whether the vector shall be"
        + " considered as a descriptor of individual values or as a definition of a range.",
    )
    val: list[Union[float, int, str]] = Field(
        ..., description="The values of the vector (list)."
    )
    coordinate_system: Optional[str] = Field(
        None,
        description="Name of the coordinate system in respect of which this object data is expressed.",
    )


class Object(BaseModel):
    class Config:
        extra = Extra.forbid

    frame_intervals: Optional[list[FrameInterval]] = Field(
        None,
        description="The array of frame intervals where this object exists or is defined.",
    )
    name: str = Field(
        ...,
        description="Name of the object. It is a friendly name and not used for indexing.",
    )
    object_data: ObjectData = Field(default_factory=dict)
    object_data_pointers: dict[str, ElementDataPointer] = Field(default_factory=dict)
    type: str = Field(
        ...,
        description="The type of an object, defines the class the object corresponds to.",
    )


class ObjectData(BaseModel):
    class Config:
        extra = Extra.forbid

    boolean: Optional[list[Boolean]] = Field(
        None, description='List of "boolean" that describe this object.'
    )
    number: Optional[list[Number]] = Field(
        None, description='List of "number" that describe this object.'
    )
    text: Optional[list[Text]] = Field(
        None, description='List of "text" that describe this object.'
    )
    vec: Optional[list[Vec]] = Field(
        None, description='List of "vec" that describe this object.'
    )


class FrameObjectData(BaseModel):
    class Config:
        extra = Extra.forbid

    bbox: Optional[list[Bbox]] = Field(
        None, description='List of "bbox" that describe this object.'
    )
    cuboid: Optional[list[Cuboid]] = Field(
        None, description='List of "cuboid" that describe this object.'
    )
    point2d: Optional[list[Point2D]] = Field(
        None, description='List of "point2d" that describe this object.'
    )
    poly2d: Optional[list[Poly2D]] = Field(
        None, description='List of "poly2d" that describe this object.'
    )
    text: Optional[list[ObjectDataTextElement]] = Field(
        None,
        description='List of "path" that describe this object semantic path location.',
    )
    mat: Optional[list[ObjectDataMatrixElement]] = Field(
        None,
        description='List of "matrix" that describe this object matrix information such as `confidence_score`.',
    )


class ElementDataPointer(BaseModel):
    class Config:
        use_enum_values = True
        extra = Extra.forbid

    attributes: Optional[dict[str, TypeAttribute]] = Field(
        None,
        description="This is a JSON object which contains pointers to the attributes of"
        + ' the element data pointed by this pointer. The attributes pointer keys shall be the "name" of the'
        + " attribute of the element data this pointer points to.",
    )
    frame_intervals: list[FrameInterval] = Field(
        ...,
        description="List of frame intervals of the element data pointed by this pointer.",
    )
    type: Optional[TypeElement] = Field(
        None, description="Type of the element data pointed by this pointer."
    )


class ContextDataPointer(BaseModel):
    class Config:
        use_enum_values = True
        extra = Extra.forbid

    attributes: Optional[dict[str, TypeAttribute]] = Field(
        None,
        description="This is a JSON object which contains pointers to the attributes of"
        + ' the element data pointed by this pointer. The attributes pointer keys shall be the "name" of the'
        + " attribute of the element data this pointer points to.",
    )
    frame_intervals: list[FrameInterval] = Field(
        ...,
        description="List of frame intervals of the element data pointed by this pointer.",
    )
    type: Optional[TypeAttribute] = Field(
        None, description="Type of the element data pointed by this pointer."
    )


class SchemaVersion(str, Enum):
    field_1_0_0 = "1.0.0"


class Metadata(BaseModel):
    class Config:
        use_enum_values = True
        extra = Extra.allow

    schema_version: SchemaVersion = Field(
        default=SchemaVersion.field_1_0_0,
        description="Version number of the VisionAI schema this annotation JSON object follows.",
    )


class StreamProperties(BaseModel):
    class Config:
        extra = Extra.allow


class StreamInfo(BaseModel):
    type: TypeStream
    uri: Optional[str] = ""
    description: Optional[str] = ""
    stream_properties: Optional[StreamProperties] = None

    class Config:
        use_enum_values = True


class Stream(BaseModel):
    __root__: dict[str, StreamInfo] = Field(
        default_factory=dict,
        description="This is the JSON object of VisionAI that contains the streams and their details.",
    )


class CoordinateSystemInfo(BaseModel):
    type: str
    parent: str = ""
    children: list[str] = Field(default_factory=list)

    class Config:
        extra = Extra.allow


class CoordinateSystem(BaseModel):
    __root__: dict[str, CoordinateSystemInfo]


class ContextInFrame(BaseModel):
    context_data: ObjectData = Field(
        default_factory=dict,
    )


class ContextInfo(BaseModel):
    class Config:
        extra = Extra.forbid

    frame_intervals: list[FrameInterval] = Field(
        ...,
        description="The array of frame intervals where this object exists or is defined.",
    )
    name: str = Field(
        ...,
        description="Name of the context. It is a friendly name and not used for indexing.",
    )
    context_data: ObjectData = Field(default_factory=dict)
    context_data_pointers: dict[str, ContextDataPointer] = Field(default_factory=dict)
    type: str = Field(
        ...,
        description="The type of a context, defines the class the context corresponds to.",
    )


class Context(BaseModel):
    __root__: dict[str, ContextInfo]


class VisionAI(BaseModel):
    class Config:
        extra = Extra.forbid

    contexts: Optional[Context] = Field(
        default_factory=dict,
        description="This is the JSON object of VisionAI classified class context."
        + " Object keys are strings containing numerical UIDs or 32 bytes UUIDs.",
    )

    frame_intervals: Optional[list[FrameInterval]] = Field(
        default_factory=dict, description="This is an array of frame intervals."
    )
    frames: Optional[dict[str, Frame]] = Field(
        default_factory=dict,
        description="This is the JSON object of frames that contain the dynamic, timewise, annotations."
        + " Keys are strings containing numerical frame identifiers, which are denoted as master frame numbers.",
    )
    objects: Optional[dict[str, Object]] = Field(
        default_factory=dict,
        description="This is the JSON object of VisionAI objects."
        + " Object keys are strings containing numerical UIDs or 32 bytes UUIDs.",
    )
    coordinate_systems: Optional[CoordinateSystem] = Field(
        default_factory=dict,
        description="This is the JSON object of coordinate system. Object keys are strings."
        + " Values are dictionary containing information of current key device.",
    )

    streams: Stream = Field(default_factory=dict)

    metadata: Metadata = Field(default_factory=Metadata)


class VisionAIModel(BaseModel):
    class Config:
        extra = Extra.forbid

    visionai: VisionAI


Attributes.update_forward_refs()
Context.update_forward_refs()
ContextInfo.update_forward_refs()
ContextInFrame.update_forward_refs()
Frame.update_forward_refs()
Object.update_forward_refs()
ObjectData.update_forward_refs()
ObjectInFrame.update_forward_refs()
VisionAI.update_forward_refs()
