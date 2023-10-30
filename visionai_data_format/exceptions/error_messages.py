from .constants import VisionAIErrorCode

VAI_ERROR_MESSAGES_MAP = {
    VisionAIErrorCode.VAI_ERR_001: "The requested converter is not supported.",
    VisionAIErrorCode.VAI_ERR_002: "Please specify at least one sensor name (camera/lidar).",
    VisionAIErrorCode.VAI_ERR_003: (
        "Sensors {extra_sensors} doesn't match with"
        + " project ontology/root sensor {root_sensors}."
    ),
    VisionAIErrorCode.VAI_ERR_004: "Missing field {field_name} in {required_place}",
    VisionAIErrorCode.VAI_ERR_005: "Doesn't support BDD format conversion with lidar",
    VisionAIErrorCode.VAI_ERR_006: "Invalid frame range, frame start : {frame_start}, frame end : {frame_end}",
    VisionAIErrorCode.VAI_ERR_007: "Missing frame interval {data_type} {data_name}",
    VisionAIErrorCode.VAI_ERR_008: "{root_key} missing data pointers",
    VisionAIErrorCode.VAI_ERR_009: "{root_key} missing data_pointer while `data` exists",
    VisionAIErrorCode.VAI_ERR_010: "Missing {data_status} {root_key} data {data_name}:{data_type} doesn't found",
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
    VisionAIErrorCode.VAI_ERR_999: "Processing Invalid",
}
