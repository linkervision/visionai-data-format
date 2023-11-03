from .constants import VisionAIErrorCode

VAI_ERROR_MESSAGES_MAP = {
    VisionAIErrorCode.VAI_ERR_001: "The requested converter is not supported.",
    VisionAIErrorCode.VAI_ERR_002: "Please specify at least one sensor name (camera/lidar).",
    VisionAIErrorCode.VAI_ERR_003: (
        "{data_type} sensors {extra_sensors} doesn't match with"
        + " {root_name} sensor {root_sensors}."
    ),
    VisionAIErrorCode.VAI_ERR_004: "Missing field {field_name} in {required_place}",
    VisionAIErrorCode.VAI_ERR_005: "Doesn't support BDD format conversion with lidar",
    VisionAIErrorCode.VAI_ERR_006: "Invalid frame range, frame start : {frame_start}, frame end : {frame_end}",
    VisionAIErrorCode.VAI_ERR_007: "Missing frame interval {data_type} {data_name}",
    VisionAIErrorCode.VAI_ERR_008: "{root_key} missing data pointers",
    VisionAIErrorCode.VAI_ERR_009: "{root_key} missing data_pointer while `data` exists",
    VisionAIErrorCode.VAI_ERR_010: "Missing {data_status} {root_key} data {data_name} with type"
    + " {data_type} doesn't found",
    VisionAIErrorCode.VAI_ERR_011: (
        "{data_status} {root_key} data {data_name}:{data_type}"
        + " doesn't match with data pointer {object_name}:{object_type}"
    ),
    VisionAIErrorCode.VAI_ERR_012: "Contains extra stream sensors {sensor_name} with type {sensor_type}",
    VisionAIErrorCode.VAI_ERR_013: "value length must be {allowed_type}",
    VisionAIErrorCode.VAI_ERR_014: "{data_name} type must be set as {required_type}",
    VisionAIErrorCode.VAI_ERR_015: "Can't assign coordinate system {coordinate_system_name} with `local_cs` type",
    VisionAIErrorCode.VAI_ERR_016: "{field_name} length doesn't match with needed length : {required_length}",
    VisionAIErrorCode.VAI_ERR_017: (
        "Contains extra attributes {extra_attributes} from ontology class {ontology_class_name}"
        + " attributes : {ontology_class_attribute_name_set}"
    ),
    VisionAIErrorCode.VAI_ERR_018: "Invalid key {root_key}",
    VisionAIErrorCode.VAI_ERR_019: "Missing key {root_key}",
    VisionAIErrorCode.VAI_ERR_020: "Contains extra classes {class_name}",
    VisionAIErrorCode.VAI_ERR_021: "RLE contains extra class indices {class_indices_list}",
    VisionAIErrorCode.VAI_ERR_022: "{data_status} {root_key} data pointer {data_name_list} missing frame intervals",
    VisionAIErrorCode.VAI_ERR_023: "Invalid value {root_key}",
    VisionAIErrorCode.VAI_ERR_024: "Extra frame from frame_intervals : {extra_frames}",
    VisionAIErrorCode.VAI_ERR_025: "Missing frame from frame_intervals : {missing_frames}",
    VisionAIErrorCode.VAI_ERR_026: "Missing field {field_key} with value {field_value} in {required_place}",
    VisionAIErrorCode.VAI_ERR_027: "Empty {root_key} data",
    VisionAIErrorCode.VAI_ERR_028: "Extra attributes {attribute_name} from dynamic "
    + "data pointer in frames {frame_list}.",
    VisionAIErrorCode.VAI_ERR_029: "Extra attributes {attribute_name} from static data pointer.",
    VisionAIErrorCode.VAI_ERR_030: "Missing attributes {attribute_name} from dynamic"
    + " data pointer in frames {frame_list}.",
    VisionAIErrorCode.VAI_ERR_031: "Missing attributes {attribute_name} from static data pointer.",
    VisionAIErrorCode.VAI_ERR_032: "{root_key} {data_uuid} frame interval(s) validate"
    + " {root_key} with frames error, start : {start}, end : {end}",
    VisionAIErrorCode.VAI_ERR_033: "{root_key} {data_uuid} frame interval(s) error, current interval [{start},{end}] "
    + "doesn't match with frames intervals {visionai_frame_intervals}",
    VisionAIErrorCode.VAI_ERR_034: "data pointer {attribute_name} frame interval(s) merge error,"
    + " start : {start}, end : {end}",
    VisionAIErrorCode.VAI_ERR_035: "data pointer {attribute_name} frame interval(s) "
    + "find duplicated with start : {start}, end : {end}",
    VisionAIErrorCode.VAI_ERR_036: "data pointer {attribute_name} frame interval(s) error,"
    + "intervals has duplicate index : {interval_list}",
    VisionAIErrorCode.VAI_ERR_037: "{root_key} {data_uuid} with data pointer {attribute_name} frame interval(s) error,"
    + " current interval [{start},{end}] doesn't match with frames {root_key} intervals {data_uuid_intervals}",
    VisionAIErrorCode.VAI_ERR_038: "{root_key} uuid {data_uuid} with attribute name {attribute_name} is not found.",
    VisionAIErrorCode.VAI_ERR_039: "Frame {frame_num} with class indices {class_list} contains disallowed index "
    + "while only allowed from 0 to {tags_count} classes.",
    VisionAIErrorCode.VAI_ERR_040: "Frame {frame_num} RLE pixel count {pixel_total} doesn't match "
    + "with image with area {image_area}.",
    VisionAIErrorCode.VAI_ERR_041: "Convert format from {original_format} to {destination_format} format error.",
    VisionAIErrorCode.VAI_ERR_999: "Processing Invalid",
}
