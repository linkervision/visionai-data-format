# modify from openlabel_json_schema.py

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  #

from pydantic import (
    Extra,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    conlist,
    root_validator,
    validator,
)

from visionai_data_format.schemas.common import ExcludedNoneBaseModel
from visionai_data_format.schemas.ontology import Ontology
from visionai_data_format.schemas.utils.validators import (
    build_ontology_attributes_map,
    validate_contexts,
    validate_objects,
    validate_streams,
    validate_visionai_intervals,
)


class SchemaVersion(str, Enum):
    FIELD_1_0_0 = "1.0.0"


class CoordinateSystemType(str, Enum):
    SENSOR_CS = "sensor_cs"
    LOCAL_CS = "local_cs"


class Type(str, Enum):
    VALUE = "value"


class ObjectType(str, Enum):
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


class TypeMinMax(str, Enum):
    VALUE = "value"
    MIN = "min"
    MAX = "max"


class TypeRange(str, Enum):
    VALUES = "values"
    RANGE = "range"


class AttributeType(str, Enum):
    BOOLEAN = "boolean"
    NUM = "num"
    VEC = "vec"
    TEXT = "text"


class StreamType(str, Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    GPS_IMU = "gps_imu"
    OTHER = "other"


class Attributes(ExcludedNoneBaseModel):
    class Config:
        extra = Extra.forbid

    boolean: List[StaticBoolean] = Field(default_factory=list)
    num: List[StaticNum] = Field(default_factory=list)
    text: List[StaticText] = Field(default_factory=list)
    vec: List[StaticVec] = Field(default_factory=list)


class CoordinateSystemWRTParent(ExcludedNoneBaseModel):
    matrix4x4: List[Union[float, int]]

    @validator("matrix4x4")
    def validate_matrix4x4(cls, value):
        assert (
            len(value) == 16
        ), f"value {value} with length {len(value)} is not allowed"
        return value


class CoordinateSystem(ExcludedNoneBaseModel):
    type: CoordinateSystemType
    parent: StrictStr
    children: List[StrictStr]
    pose_wrt_parent: Optional[CoordinateSystemWRTParent] = Field(default=None)

    class Config:
        extra = Extra.forbid
        use_enum_values = True

    @validator("pose_wrt_parent")
    def validate_pose_wrt_parent(cls, value):
        assert value, f"value {value} is not allowed"
        return value


class FrameInterval(ExcludedNoneBaseModel):
    class Config:
        extra = Extra.forbid

    frame_start: StrictInt = Field(
        ..., description="Initial frame number of the interval."
    )
    frame_end: StrictInt = Field(
        ..., description="Ending frame number of the interval."
    )

    @root_validator
    def validate_frame_range(cls, values):
        frame_start = values.get("frame_start")
        frame_end = values.get("frame_end")

        if frame_start > frame_end:
            raise ValueError(
                "Range is not accepted,"
                + f" frame start : {frame_start} , frame end : {frame_end}"
            )
        return values


class IntrinsicsPinhole(ExcludedNoneBaseModel):
    camera_matrix_3x4: List[float]
    distortion_coeffs_1xN: Optional[List[Union[float, int]]] = None
    height_px: int
    width_px: int

    @validator("distortion_coeffs_1xN")
    def validate_distortion_coeffs_1xN(cls, value):
        assert value, f"value {value} is not allowed"
        return value

    @validator("camera_matrix_3x4")
    def validate_camera_matrix_3x4(cls, value):
        assert (
            value and len(value) == 12
        ), f"value {value} is not allowed, must be a list of 12 float values"
        return value


class Metadata(ExcludedNoneBaseModel):
    class Config:
        use_enum_values = True
        extra = Extra.allow

    schema_version: SchemaVersion = Field(
        description="Version number of the VisionAI schema this annotation JSON object follows.",
    )


class StaticBoolean(ExcludedNoneBaseModel):
    attributes: Optional[Attributes] = None
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

    class Config:
        use_enum_values = True
        extra = Extra.forbid


class DynamicBoolean(StaticBoolean):
    stream: StrictStr = Field(
        ...,
        description="Name of the stream in respect of which this object data is expressed.",
    )
    confidence_score: Optional[float] = Field(
        None,
        description="The confidence score of model prediction of this object."
        + " Ground truth does not have this attribute.",
    )


class StaticNum(ExcludedNoneBaseModel):
    class Config:
        use_enum_values = True
        extra = Extra.forbid

    attributes: Optional[Attributes] = None
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


class DynamicNum(StaticNum):
    stream: StrictStr = Field(
        ...,
        description="Name of the stream in respect of which this object data is expressed.",
    )
    confidence_score: Optional[float] = Field(
        None,
        description="The confidence score of model prediction of this object."
        + " Ground truth does not have this attribute.",
    )


class StaticText(ExcludedNoneBaseModel):
    class Config:
        use_enum_values = True
        extra = Extra.forbid

    attributes: Optional[Attributes] = None
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


class DynamicText(StaticText):
    stream: StrictStr = Field(
        ...,
        description="Name of the stream in respect of which this object data is expressed.",
    )

    confidence_score: Optional[float] = Field(
        None,
        description="The confidence score of model prediction of this object."
        + " Ground truth does not have this attribute.",
    )


class VecBaseNoName(ExcludedNoneBaseModel):
    attributes: Optional[Attributes] = None
    type: Optional[TypeRange] = Field(
        None,
        description="This attribute specifies whether the vector shall be"
        + " considered as a descriptor of individual values or as a definition of a range.",
    )
    val: List[Union[float, int, str]] = Field(
        ..., description="The values of the vector (list)."
    )

    class Config:
        use_enum_values = True
        extra = Extra.forbid


class StaticVec(VecBaseNoName):
    name: StrictStr = Field(
        ...,
        description="This is a string encoding the name of this object data."
        + " It is used as index inside the corresponding object data pointers.",
    )


class DynamicVec(StaticVec):
    stream: StrictStr = Field(
        ...,
        description="Name of the stream in respect of which this object data is expressed.",
    )
    confidence_score: Optional[float] = Field(
        None,
        description="The confidence score of model prediction of this object."
        + " Ground truth does not have this attribute.",
    )


class BaseStaticElementData(ExcludedNoneBaseModel):
    boolean: Optional[List[StaticBoolean]] = Field(
        None, description='List of "boolean" that describe this object.'
    )
    num: Optional[List[StaticNum]] = Field(
        None, description='List of "number" that describe this object.'
    )
    text: Optional[List[StaticText]] = Field(
        None, description='List of "text" that describe this object.'
    )
    vec: Optional[List[StaticVec]] = Field(
        None, description='List of "vec" that describe this object.'
    )


class BaseDynamicElementData(ExcludedNoneBaseModel):
    boolean: Optional[List[DynamicBoolean]] = Field(
        None, description='List of "boolean" that describe this object.'
    )
    num: Optional[List[DynamicNum]] = Field(
        None, description='List of "number" that describe this object.'
    )
    text: Optional[List[DynamicText]] = Field(
        None, description='List of "text" that describe this object.'
    )
    vec: Optional[List[DynamicVec]] = Field(
        None, description='List of "vec" that describe this object.'
    )


class ElementDataPointer(ExcludedNoneBaseModel):
    attributes: Optional[Dict[StrictStr, AttributeType]] = Field(
        None,
        description="This is a JSON object which contains pointers to the attributes of"
        + ' the element data pointed by this pointer. The attributes pointer keys shall be the "name" of the'
        + " attribute of the element data this pointer points to.",
    )
    frame_intervals: Optional[List[FrameInterval]] = Field(
        default=None,
        description="List of frame intervals of the element data pointed by this pointer.",
    )


class ContextDataStatic(BaseStaticElementData):
    class Config:
        extra = Extra.forbid


class DynamicContextData(BaseDynamicElementData):
    class Config:
        extra = Extra.forbid


class ContextDataPointer(ElementDataPointer):
    class Config:
        use_enum_values = True
        extra = Extra.forbid

    type: AttributeType = Field(
        ..., description="Type of the element data pointed by this pointer."
    )


class Context(ExcludedNoneBaseModel):
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
    context_data: Optional[ContextDataStatic] = None
    context_data_pointers: Dict[StrictStr, ContextDataPointer]
    type: StrictStr = Field(
        ...,
        description="The type of a context, defines the class the context corresponds to.",
    )

    @validator("context_data", pre=True)
    def validate_context_data(cls, value):
        assert isinstance(value, dict) and value, f"value {value} is not allowed"
        return value

    @validator("context_data_pointers", pre=True)
    def pre_validate_context_data_pointers(cls, value):
        assert value, f"value {value} is not allowed"
        return value

    @root_validator
    def validate_context_data_relations(cls, values):
        context_data_pointers = values.get("context_data_pointers")
        context_data = values.get("context_data", {})
        if context_data and not context_data_pointers:
            raise ValueError(
                "context data pointers can't be empty with contexts data exists"
            )

        static_contexts_data_name_type_map = {}

        if context_data:
            for obj_type, obj_info_list in context_data:
                if not obj_info_list:
                    continue
                for obj_info in obj_info_list:
                    static_contexts_data_name_type_map.update({obj_info.name: obj_type})
        for obj_name, obj_type in static_contexts_data_name_type_map.items():
            obj_data_dict = context_data_pointers.get(obj_name, {})
            if not obj_data_dict:
                raise ValueError(
                    f"Static contexts data {obj_name}:{obj_type} doesn't found under context data pointer"
                )
            obj_data_pointer_type = getattr(obj_data_dict, "type", "")
            if obj_type != obj_data_pointer_type:
                raise ValueError(
                    f"Static contexts data {obj_name}:{obj_type} doesn't match"
                    + f" with data_pointer {obj_name}:{obj_data_pointer_type}"
                )

        static_context_data_name_set = set(static_contexts_data_name_type_map.keys())
        error_name_list = []

        for obj_name, obj_info in context_data_pointers.items():
            if obj_name not in static_context_data_name_set and not getattr(
                obj_info, "frame_intervals", None
            ):
                error_name_list.append(f"{obj_name}:{obj_info.type}")

        if error_name_list:
            raise ValueError(
                f"Dynamic context data pointer {error_name_list}"
                + " missing frame intervals"
            )
        return values


class ObjectDataPointer(ElementDataPointer):
    class Config:
        use_enum_values = True
        extra = Extra.forbid

    type: ObjectType = Field(
        ..., description="Type of the element data pointed by this pointer."
    )


class ObjectDataStatic(BaseStaticElementData):
    class Config:
        extra = Extra.forbid


class Object(ExcludedNoneBaseModel):
    class Config:
        extra = Extra.forbid

    frame_intervals: List[FrameInterval] = Field(
        ...,
        description="The array of frame intervals where this object exists or is defined.",
    )
    name: Optional[StrictStr] = Field(
        default=None,
        description="Name of the object. It is a friendly name and not used for indexing.",
    )
    object_data: Optional[ObjectDataStatic] = None
    object_data_pointers: Dict[StrictStr, ObjectDataPointer]
    type: StrictStr = Field(
        ...,
        description="The type of an object, defines the class the object corresponds to.",
    )

    @validator("object_data", pre=True)
    def validate_object_data(cls, value):
        assert isinstance(value, dict) and value, f"value {value} is not allowed"
        return value

    @validator("object_data_pointers", pre=True)
    def pre_validate_object_data_pointers(cls, value):
        assert value, f"value {value} is not allowed"
        return value

    @root_validator
    def validate_object_data_relations(cls, values):
        object_data_pointers = values.get("object_data_pointers")
        object_data = values.get("object_data", {})
        if object_data and not object_data_pointers:
            raise ValueError(
                "object data pointers can't be empty with objects data exists"
            )

        static_objects_data_name_type_map = {}

        if object_data:
            for obj_type, obj_info_list in object_data:
                if not obj_info_list:
                    continue
                for obj_info in obj_info_list:
                    static_objects_data_name_type_map.update({obj_info.name: obj_type})
        for obj_name, obj_type in static_objects_data_name_type_map.items():
            obj_data_dict = object_data_pointers.get(obj_name, {})
            if not obj_data_dict:
                raise ValueError(
                    f"Static objects data {obj_name}:{obj_type} doesn't found under object data pointer"
                )
            obj_data_pointer_type = getattr(obj_data_dict, "type", "")
            if obj_type != obj_data_pointer_type:
                raise ValueError(
                    f"Static objects data {obj_name}:{obj_type} doesn't match"
                    + f" with data_pointer {obj_name}:{obj_data_pointer_type}"
                )

        static_object_data_name_set = set(static_objects_data_name_type_map.keys())
        error_name_list = []
        for obj_name, obj_info in object_data_pointers.items():
            if obj_name not in static_object_data_name_set and not getattr(
                obj_info, "frame_intervals", None
            ):
                error_name_list.append(f"{obj_name}:{obj_info.type}")

        if error_name_list:
            raise ValueError(
                f"Dynamic object data pointer {error_name_list}"
                + " missing frame intervals"
            )
        return values


class StreamProperties(ExcludedNoneBaseModel):
    intrinsics_pinhole: IntrinsicsPinhole


class TagData(ExcludedNoneBaseModel):
    vec: List[StaticVec] = Field(...)

    @validator("vec")
    def validate_vec(cls, values):
        assert len(values) == 1, "Only allow one data inside list"
        value = values[0]
        assert value.type == "values", "Value {} is not allowed"
        return values


class Stream(ExcludedNoneBaseModel):
    type: StreamType
    uri: Optional[StrictStr] = ""
    description: Optional[StrictStr] = ""
    stream_properties: Optional[StreamProperties] = None

    class Config:
        use_enum_values = True

    @validator("stream_properties")
    def validate_stream_properties(cls, value):
        assert value, f"value {value} is not allowed"
        return value


class Tag(ExcludedNoneBaseModel):
    ontology_uid: StrictStr
    type: StrictStr
    tag_data: TagData


class TimeStampElement(ExcludedNoneBaseModel):
    timestamp: str

    class Config:
        extra = Extra.forbid

    @validator("timestamp")
    def validate_timestamp(cls, value):
        iso_time_regex = r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2}\.\d{3})(Z|[+-]((\d{2}:\d{2})|(\d{4})))$"
        assert re.match(iso_time_regex, value), f"Wrong timestamp format : {value}"
        return value


class StreamPropertyUnderFrameProperty(ExcludedNoneBaseModel):
    sync: Optional[TimeStampElement] = None


class FramePropertyStream(ExcludedNoneBaseModel):
    uri: str = Field(description="the urls of image")
    stream_properties: Optional[StreamPropertyUnderFrameProperty] = Field(
        None, description="Additional properties of the stream"
    )

    class Config:
        extra = Extra.allow

    @validator("stream_properties")
    def validate_stream_properties(cls, value):
        assert value, f"value {value} is not allowed"
        return value


class FrameProperties(ExcludedNoneBaseModel):
    timestamp: Optional[str] = Field(
        None,
        descriptions="A relative or absolute time reference that specifies "
        + "the time instant this frame corresponds to",
    )
    streams: Dict[StrictStr, FramePropertyStream]


class ObjectDataElement(ExcludedNoneBaseModel):
    attributes: Optional[Attributes] = None
    name: StrictStr = Field(
        description="This is a string encoding the name of this object data."
        + " It is used as index inside the corresponding object data pointers.",
    )
    stream: StrictStr = Field(
        description="Name of the stream in respect of which this object data is expressed.",
    )
    confidence_score: Optional[float] = Field(
        None,
        description="The confidence score of model prediction of this object."
        + " Ground truth does not have this attribute.",
    )


class Bbox(ObjectDataElement):
    class Config:
        extra = Extra.allow

    val: conlist(
        Union[float, int],
        max_items=4,
        min_items=4,
    )


class Cuboid(ObjectDataElement):
    class Config:
        extra = Extra.allow

    val: conlist(
        Union[float, int],
        min_items=9,
        max_items=9,
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


class Point2D(ObjectDataElement):
    class Config:
        extra = Extra.allow

    val: conlist(
        Union[float, int],
        max_items=2,
        min_items=2,
    )


class Binary(ObjectDataElement):
    encoding: Literal["rle"] = Field(
        ..., description="The encoding method. It only supports “rle“ value."
    )
    data_type: Literal[""] = Field(
        ...,
        description="This is a string declares the type of values of the binary."
        + " Only empty string "
        " value allowed",
    )
    val: StrictStr = Field(...)

    @validator("name")
    def validate_name_field(cls, value):
        if value != "semantic_mask":
            raise ValueError("Name value must be `semantic_mask`")
        return value


class DynamicObjectData(ExcludedNoneBaseModel):
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
    binary: Optional[List[Binary]] = Field(
        None,
        description='List of "binary" that describe this object semantic mask info.',
    )


class ObjectUnderFrame(ExcludedNoneBaseModel):
    object_data: DynamicObjectData


class ContextUnderFrame(ExcludedNoneBaseModel):
    context_data: DynamicContextData


class Frame(ExcludedNoneBaseModel):
    class Config:
        extra = Extra.forbid

    objects: Optional[Dict[StrictStr, ObjectUnderFrame]] = Field(
        default=None,
        description="This is a JSON object that contains dynamic information on VisionAI objects."
        + " Object keys are strings containing numerical UIDs or 32 bytes UUIDs."
        + ' Object values may contain an "object_data" JSON object.',
    )

    contexts: Optional[Dict[StrictStr, ContextUnderFrame]] = Field(
        default=None,
        description="This is a JSON object that contains dynamic information on VisionAI contexts."
        + " Context keys are strings containing numerical UIDs or 32 bytes UUIDs."
        + ' Context values may contain an "context_data" JSON object.',
    )

    frame_properties: FrameProperties = Field(
        description="This is a JSON object which contains information about this frame.",
    )


class VisionAI(ExcludedNoneBaseModel):
    class Config:
        extra = Extra.forbid

    contexts: Optional[Dict[StrictStr, Context]] = Field(
        default=None,
        description="This is the JSON object of VisionAI classified class context."
        + " Object keys are strings containing numerical UIDs or 32 bytes UUIDs.",
    )

    @validator("contexts")
    def validate_contexts(cls, value):
        assert value, f"value {value} is not allowed"
        return value

    frame_intervals: List[FrameInterval] = Field(
        description="This is an array of frame intervals."
    )

    frames: Dict[StrictStr, Frame] = Field(
        description="This is the JSON object of frames that contain the dynamic, time-wise, annotations."
        + " Keys are strings containing numerical frame identifiers, which are denoted as master frame numbers.",
    )

    @validator("frames")
    def validate_frames(cls, value):
        frame_keys = list(value.keys())

        assert all(
            len(key) == 12 and key.isdigit() for key in frame_keys
        ), "Key must be a digit with 12 characters length"

        return value

    objects: Optional[Dict[StrictStr, Object]] = Field(
        default=None,
        description="This is the JSON object of VisionAI objects."
        + " Object keys are strings containing numerical UIDs or 32 bytes UUIDs.",
    )

    @validator("objects")
    def validate_objects(cls, value):
        assert value, f"value {value} is not allowed"
        return value

    coordinate_systems: Optional[Dict[StrictStr, CoordinateSystem]] = Field(
        default=None,
        description="This is the JSON object of coordinate system. Object keys are strings."
        + " Values are dictionary containing information of current key device.",
    )

    @validator("coordinate_systems")
    def validate_coordinate_systems(cls, value):
        assert value, f" Value {value} is not allowed"

        for k, v in value.items():
            if v.type == "local_cs" and "iso8855" not in k:
                raise ValueError(
                    f"Can't assign coordinate system {k} with local_cs type"
                )
        return value

    streams: Dict[StrictStr, Stream] = Field(
        description="This is the JSON object of VisionAI that contains the streams and their details.",
    )

    metadata: Metadata

    tags: Optional[Dict[StrictStr, Tag]] = Field(
        default=None,
        description="This is the JSON object of tags. Object keys are strings."
        + " Values are dictionary containing information of current sequence.",
    )

    @validator("tags")
    def validate_tags(cls, value):
        assert value, f" Value {value} is not allowed"
        return value


class VisionAIModel(ExcludedNoneBaseModel):
    class Config:
        extra = Extra.forbid

    visionai: VisionAI

    def validate_with_ontology(self, ontology: Type[Ontology]) -> List[str]:
        validator_map = {
            "contexts": validate_contexts,
            "objects": validate_objects,
        }

        errors: List[str] = []

        tags = ontology.get("tags", {})

        visionai = self.visionai.dict(exclude_unset=True, exclude_none=True)

        err = validate_visionai_intervals(visionai=visionai)
        if err:
            errors.append(err)

        streams_data = ontology["streams"]

        sensor_info: Dict[str, str] = {
            sensor_name: sensor_obj["type"]
            for sensor_name, sensor_obj in streams_data.items()
        }

        has_multi_sensor: bool = len(streams_data) > 1

        has_lidar_sensor: bool = any(
            sensor_type == "lidar" for sensor_type in sensor_info.values()
        )
        ontology_attributes_map = build_ontology_attributes_map(ontology)

        err, visionai_sensor_info = validate_streams(
            visionai=visionai,
            sensor_info=sensor_info,
            has_lidar_sensor=has_lidar_sensor,
            has_multi_sensor=has_multi_sensor,
        )
        if err:
            errors.append(err)

        for ontology_type, ontology_data in ontology.items():
            if not ontology_data or ontology_type not in validator_map:
                continue
            err = validator_map[ontology_type](
                visionai=visionai,
                ontology_data=ontology_data,
                ontology_attributes_map=ontology_attributes_map,
                tags=tags,
                sensor_info=visionai_sensor_info,
                has_multi_sensor=has_multi_sensor,
                has_lidar_sensor=has_lidar_sensor,
            )
            if err:
                errors.append(err)

        return errors


Attributes.update_forward_refs()
Context.update_forward_refs()
ContextDataPointer.update_forward_refs()
ContextUnderFrame.update_forward_refs()
Frame.update_forward_refs()
Object.update_forward_refs()
DynamicObjectData.update_forward_refs()
ObjectUnderFrame.update_forward_refs()
VisionAI.update_forward_refs()
