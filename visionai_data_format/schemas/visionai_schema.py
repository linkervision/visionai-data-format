# modify from openlabel_json_schema.py

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  #

from pydantic import (
    BaseModel,
    Extra,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    conlist,
    validator,
)


class Type(str, Enum):
    value = "value"


class ObjectType(str, Enum):
    bbox = "bbox"
    cuboid = "cuboid"
    point2d = "point2d"
    poly2d = "poly2d"
    image = "image"
    mat = "mat"
    boolean = "boolean"
    number = "number"
    vec = "vec"
    text = "text"


class TypeMinMax(str, Enum):
    value = "value"
    min = "min"
    max = "max"


class TypeRange(str, Enum):
    values = "values"
    range = "range"


class AttributeType(str, Enum):
    boolean = "boolean"
    number = "number"
    vec = "vec"
    text = "text"


class StreamType(str, Enum):
    camera = "camera"
    lidar = "lidar"
    radar = "radar"
    gps_imu = "gps_imu"
    other = "other"


class Attributes(BaseModel):
    class Config:
        extra = Extra.forbid

    boolean: List[Boolean] = Field(default_factory=list)
    number: List[Number] = Field(default_factory=list)
    text: List[Text] = Field(default_factory=list)
    vec: List[Vec] = Field(default_factory=list)


class ObjectDataElement(BaseModel):

    attributes: Attributes = Field(default_factory=dict)
    name: StrictStr = Field(
        ...,
        description="This is a string encoding the name of this object data."
        + " It is used as index inside the corresponding object data pointers.",
    )
    stream: StrictStr = Field(
        ...,
        description="Name of the stream in respect of which this object data is expressed.",
    )
    coordinate_system: Optional[StrictStr] = Field(
        None,
        description="Name of the coordinate system in respect of which this object data is expressed.",
    )
    confidence_score: Optional[float] = Field(
        None,
        description="The confidence score of model prediction of this object."
        + " Ground truth does not have this attribute.",
    )


class Matrix(ObjectDataElement):

    height: StrictInt = Field(
        ..., description="This is the height (number of rows) of the matrix."
    )
    width: StrictInt = Field(
        ..., description="This is the width (number of columns) of the matrix."
    )
    channels: StrictInt = Field(
        ..., description="This is the number of channels of the matrix."
    )

    data_type: StrictStr = Field(
        ...,
        description="This is a string declares the type of values of the matrix."
        + " Only `float` or `int` allowed",
    )

    val: List[Union[float, int]] = Field(
        ..., description="This is a list of flattened values of the matrix."
    )

    class Config:
        extra = Extra.allow

    @validator("data_type")
    def validate_data_type_data(cls, value):
        allowed_type = {"float", "int"}
        if value not in allowed_type:
            raise ValueError("only `float` or `int` allowed for `data_type` field")
        return value

    @validator("val")
    def validate_val_data(cls, value, values):
        allowed_type = {"float": float, "int": int}
        data_type = values.get("data_type")
        if not (data_type):
            raise ValueError("Need to define the `data_type`")

        cur_type = allowed_type.get(data_type)
        if not all(type(n) is cur_type for n in value):
            raise ValueError("Val contains not allowed value")

        return value


class Bbox(ObjectDataElement):
    class Config:
        extra = Extra.allow

    val: conlist(
        Union[float, int],
        max_items=4,
        min_items=4,
    )


class Point2D(ObjectDataElement):
    class Config:
        extra = Extra.allow

    val: conlist(
        Union[float, int],
        max_items=2,
        min_items=2,
    )


class Poly2D(ObjectDataElement):
    class Config:
        extra = Extra.allow

    val: conlist(
        Union[float, int],
        min_items=2,
    )
    closed: StrictBool = Field(
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

    val: conlist(
        Union[float, int],
        min_items=9,
        max_items=9,
    )


class Text(BaseModel):
    class Config:
        use_enum_values = True
        extra = Extra.allow

    attributes: Attributes = Field(default_factory=dict)
    name: Optional[StrictStr] = Field(
        None,
        description="This is a string encoding the name of this object data."
        + " It is used as index inside the corresponding object data pointers.",
    )
    type: Optional[Type] = Field(
        None,
        description="This attribute specifies how the text shall be considered."
        + " The only possible option is as a value.",
    )
    val: StrictStr = Field(..., description="The characters of the text.")
    coordinate_system: Optional[StrictStr] = Field(
        None,
        description="Name of the coordinate system in respect of which this object data is expressed.",
    )


class Vec(BaseModel):
    attributes: Attributes = Field(default_factory=dict)
    name: StrictStr = Field(
        ...,
        description="This is a string encoding the name of this object data."
        + " It is used as index inside the corresponding object data pointers.",
    )
    type: Optional[TypeRange] = Field(
        None,
        description="This attribute specifies whether the vector shall be"
        + " considered as a descriptor of individual values or as a definition of a range.",
    )
    val: List[Union[float, int, str]] = Field(
        ..., description="The values of the vector (list)."
    )
    coordinate_system: Optional[StrictStr] = Field(
        None,
        description="Name of the coordinate system in respect of which this object data is expressed.",
    )

    class Config:
        use_enum_values = True
        extra = Extra.allow


class Boolean(BaseModel):

    attributes: Attributes = Field(default_factory=dict)
    name: StrictStr = Field(
        ...,
        description="This is a string encoding the name of this object data."
        + " It is used as index inside the corresponding object data pointers.",
    )
    type: Optional[Type] = Field(
        None,
        description="This attribute specifies how the boolean shall be considered."
        + " In this schema the only possible option is as a value.",
    )
    val: StrictBool = Field(..., description="The boolean value.")
    coordinate_system: Optional[StrictStr] = Field(
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
    name: StrictStr = Field(
        ...,
        description="This is a string encoding the name of this object data."
        + " It is used as index inside the corresponding object data pointers.",
    )
    type: Optional[TypeMinMax] = Field(
        None,
        description="This attribute specifies whether the number shall be considered "
        + "as a value, a minimum, or a maximum in its context.",
    )
    val: Union[float, int] = Field(
        ..., description="The numerical value of the number."
    )
    coordinate_system: Optional[StrictStr] = Field(
        None,
        description="Name of the coordinate system in respect of which this object data is expressed.",
    )


class FrameInterval(BaseModel):
    class Config:
        extra = Extra.forbid

    frame_start: StrictInt = Field(
        ..., description="Initial frame number of the interval."
    )
    frame_end: StrictInt = Field(
        ..., description="Ending frame number of the interval."
    )


class BaseElementData(BaseModel):
    boolean: Optional[List[Boolean]] = Field(
        None, description='List of "boolean" that describe this object.'
    )
    number: Optional[List[Number]] = Field(
        None, description='List of "number" that describe this object.'
    )
    text: Optional[List[Text]] = Field(
        None, description='List of "text" that describe this object.'
    )
    vec: Optional[List[Vec]] = Field(
        None, description='List of "vec" that describe this object.'
    )


class ContextData(BaseElementData):
    class Config:
        extra = Extra.forbid


class Context(BaseModel):
    class Config:
        extra = Extra.forbid

    frame_intervals: List[FrameInterval] = Field(
        ...,
        description="The array of frame intervals where this object exists or is defined.",
    )
    name: StrictStr = Field(
        ...,
        description="Name of the context. It is a friendly name and not used for indexing.",
    )
    context_data: ContextData = Field(default_factory=dict)
    context_data_pointers: Dict[StrictStr, ContextDataPointer] = Field(
        default_factory=dict
    )
    type: StrictStr = Field(
        ...,
        description="The type of a context, defines the class the context corresponds to.",
    )


class StreamProperties(BaseModel):
    class Config:
        extra = Extra.allow


class Stream(BaseModel):
    type: StreamType
    uri: Optional[StrictStr] = ""
    description: Optional[StrictStr] = ""
    stream_properties: Optional[StreamProperties] = None

    class Config:
        use_enum_values = True


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


class ObjectData(BaseElementData):
    class Config:
        extra = Extra.forbid

    bbox: Optional[List[Bbox]] = Field(
        None, description='List of "bbox" that describe this object.'
    )
    cuboid: Optional[List[Cuboid]] = Field(
        None, description='List of "cuboid" that describe this object.'
    )
    point2d: Optional[List[Point2D]] = Field(
        None, description='List of "point2d" that describe this object.'
    )
    poly2d: Optional[List[Poly2D]] = Field(
        None, description='List of "poly2d" that describe this object.'
    )
    mat: Optional[List[Matrix]] = Field(
        None,
        description='List of "matrix" that describe this object matrix information such as `confidence_score`.',
    )


class ObjectUnderFrame(BaseModel):
    object_data: ObjectData = Field(
        default_factory=dict,
    )


class ContextUnderFrame(BaseModel):
    context_data: ContextData = Field(
        default_factory=dict,
    )


class TimeStampElement(BaseModel):
    timestamp: str

    class Config:
        extra = Extra.forbid


class StreamPropertyUnderFrameProperty(BaseModel):
    sync: TimeStampElement

    class Config:
        extra = Extra.allow


class FramePropertyStream(BaseModel):
    uri: str = Field(..., description="the urls of image")
    stream_properties: Optional[
        Dict[StrictStr, StreamPropertyUnderFrameProperty]
    ] = Field(None, description="Additional properties of the stream")

    class Config:
        extra = Extra.allow


class FrameProperties(BaseModel):
    timestamp: Optional[str] = Field(
        None,
        descriptions="A relative or absolute time reference that specifies "
        + "the time instant this frame corresponds to",
    )
    streams: Dict[StrictStr, FramePropertyStream] = Field(default_factory=dict)


class Frame(BaseModel):
    class Config:
        extra = Extra.forbid

    objects: Dict[StrictStr, ObjectUnderFrame] = Field(
        default_factory=dict,
        description="This is a JSON object that contains dynamic information on VisionAI objects."
        + " Object keys are strings containing numerical UIDs or 32 bytes UUIDs."
        + ' Object values may contain an "object_data" JSON object.',
    )

    contexts: Dict[StrictStr, ContextUnderFrame] = Field(
        default_factory=dict,
        description="This is a JSON object that contains dynamic information on VisionAI contexts."
        + " Context keys are strings containing numerical UIDs or 32 bytes UUIDs."
        + ' Context values may contain an "context_data" JSON object.',
    )

    frame_properties: FrameProperties = Field(
        default_factory=dict,
        description="This is a JSON object which contains information about this frame.",
    )


class ElementDataPointer(BaseModel):

    attributes: Optional[Dict[StrictStr, AttributeType]] = Field(
        None,
        description="This is a JSON object which contains pointers to the attributes of"
        + ' the element data pointed by this pointer. The attributes pointer keys shall be the "name" of the'
        + " attribute of the element data this pointer points to.",
    )
    frame_intervals: List[FrameInterval] = Field(
        ...,
        description="List of frame intervals of the element data pointed by this pointer.",
    )


class ContextDataPointer(ElementDataPointer):
    class Config:
        use_enum_values = True
        extra = Extra.forbid

    type: AttributeType = Field(
        ..., description="Type of the element data pointed by this pointer."
    )


class ObjectDataPointer(ElementDataPointer):
    class Config:
        use_enum_values = True
        extra = Extra.forbid

    type: ObjectType = Field(
        ..., description="Type of the element data pointed by this pointer."
    )


class Object(BaseModel):
    class Config:
        extra = Extra.forbid

    frame_intervals: Optional[List[FrameInterval]] = Field(
        None,
        description="The array of frame intervals where this object exists or is defined.",
    )
    name: StrictStr = Field(
        ...,
        description="Name of the object. It is a friendly name and not used for indexing.",
    )
    object_data: ObjectData = Field(default_factory=dict)
    object_data_pointers: Dict[StrictStr, ObjectDataPointer] = Field(
        default_factory=dict
    )
    type: StrictStr = Field(
        ...,
        description="The type of an object, defines the class the object corresponds to.",
    )


class CoordinateSystem(BaseModel):
    type: StrictStr
    parent: StrictStr = ""
    children: List[StrictStr] = Field(default_factory=list)

    class Config:
        extra = Extra.allow


class VisionAI(BaseModel):
    class Config:
        extra = Extra.forbid

    contexts: Dict[StrictStr, Context] = Field(
        default_factory=dict,
        description="This is the JSON object of VisionAI classified class context."
        + " Object keys are strings containing numerical UIDs or 32 bytes UUIDs.",
    )

    frame_intervals: List[FrameInterval] = Field(
        default_factory=list, description="This is an array of frame intervals."
    )
    frames: Dict[StrictStr, Frame] = Field(
        default_factory=dict,
        description="This is the JSON object of frames that contain the dynamic, timewise, annotations."
        + " Keys are strings containing numerical frame identifiers, which are denoted as master frame numbers.",
    )
    objects: Dict[StrictStr, Object] = Field(
        default_factory=dict,
        description="This is the JSON object of VisionAI objects."
        + " Object keys are strings containing numerical UIDs or 32 bytes UUIDs.",
    )
    coordinate_systems: Dict[StrictStr, CoordinateSystem] = Field(
        default_factory=dict,
        description="This is the JSON object of coordinate system. Object keys are strings."
        + " Values are dictionary containing information of current key device.",
    )

    streams: Dict[StrictStr, Stream] = Field(
        default_factory=dict,
        description="This is the JSON object of VisionAI that contains the streams and their details.",
    )

    metadata: Metadata = Field(default_factory=Metadata)


class VisionAIModel(BaseModel):
    class Config:
        extra = Extra.forbid

    visionai: VisionAI


Attributes.update_forward_refs()
Context.update_forward_refs()
ContextDataPointer.update_forward_refs()
ContextUnderFrame.update_forward_refs()
Frame.update_forward_refs()
Object.update_forward_refs()
ObjectData.update_forward_refs()
ObjectUnderFrame.update_forward_refs()
VisionAI.update_forward_refs()
