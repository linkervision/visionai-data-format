from .constants import VisionAIErrorCode

VAI_ERROR_MESSAGES_MAP = {
    VisionAIErrorCode.VAI_ERR_001: "The requested converter is not supported.",
    VisionAIErrorCode.VAI_ERR_002: "No sensor name has been specified in the data.",
    VisionAIErrorCode.VAI_ERR_003: "The frame streams or coordinate system sensors"
    + " do not match the visionai-data-format.",
    VisionAIErrorCode.VAI_ERR_004: "A mandatory field {field_name} in {required_place}.",
    VisionAIErrorCode.VAI_ERR_005: "The BDD format conversion is not supported for lidar data.",
    VisionAIErrorCode.VAI_ERR_006: "The specified frame range is invalid with start"
    + " at {frame_start} and end at {frame_end}.",
    VisionAIErrorCode.VAI_ERR_007: "Frame intervals are missing for the data type "
    + "{data_type} with the name {data_name}.",
    VisionAIErrorCode.VAI_ERR_008: "Expected data pointers for object/context data are missing.",
    VisionAIErrorCode.VAI_ERR_009: "Object/context data is present, but the corresponding data pointer is missing.",
    VisionAIErrorCode.VAI_ERR_010: "Static/dynamic object data named {data_name} with type {type} is not found.",
    VisionAIErrorCode.VAI_ERR_011: (
        "The data type {data_type} for {data_name} does not match with the data "
        + "pointer for {object_name}:{object_type}."
    ),
    VisionAIErrorCode.VAI_ERR_012: "Additional stream sensors {sensor_name} with"
    + " type {sensor_type} are present that are not required by the visionai format.",
    VisionAIErrorCode.VAI_ERR_013: "The length of the value provided does not"
    + " match the required length of {allowed_length}",
    VisionAIErrorCode.VAI_ERR_014: "The type of {data_name} is incorrect and"
    + " must be set to the required type {required_type}",
    VisionAIErrorCode.VAI_ERR_015: "Cannot assign the coordinate system "
    + "{coordinate_system_name} as its type is local_cs.",
    VisionAIErrorCode.VAI_ERR_016: "The length of {field_name} does not "
    + "match the required length: {required_length}.",
    VisionAIErrorCode.VAI_ERR_017: (
        "Extra attributes {extra_attributes} are present that are not "
        + "defined in the ontology class {ontology_class_name}."
    ),
    VisionAIErrorCode.VAI_ERR_018: "The key {root_key} is invalid or not recognized.",
    VisionAIErrorCode.VAI_ERR_019: "A required key {root_key} is missing.",
    VisionAIErrorCode.VAI_ERR_020: "The data contains additional classes "
    + "{class_name} that are not expected.",
    VisionAIErrorCode.VAI_ERR_021: "The Run-Length Encoding (RLE) contains "
    + "additional class indices {class_indices_list} that are unexpected.",
    VisionAIErrorCode.VAI_ERR_022: "Static/dynamic Object/Context data pointer"
    + " {data_name_list} is missing required frame intervals.",
    VisionAIErrorCode.VAI_ERR_023: "An invalid value {root_key} was detected in the data.",
    VisionAIErrorCode.VAI_ERR_024: "An extra frame was detected beyond the "
    + "defined frame intervals: {extra_frames}.",
    VisionAIErrorCode.VAI_ERR_025: "There are missing frames from the defined"
    + " frame intervals: {missing_frames}.",
    VisionAIErrorCode.VAI_ERR_026: "Missing field {field_key} with value "
    + "{required_value} in {required_place}.",
    VisionAIErrorCode.VAI_ERR_027: "The {root_key} data is empty and requires content.",
    VisionAIErrorCode.VAI_ERR_028: "Extra attributes {attribute_name} are present"
    + " in the dynamic data pointer for frames {frame_list}.",
    VisionAIErrorCode.VAI_ERR_029: "Extra attributes {attribute_name} found in the static data pointer.",
    VisionAIErrorCode.VAI_ERR_030: "Essential attributes {attribute_name} are missing from the dynamic data.",
    VisionAIErrorCode.VAI_ERR_031: "Attributes {attribute_name} are absent from the static data pointer.",
    VisionAIErrorCode.VAI_ERR_032: "{root_key} {data_uuid} frame interval(s) have"
    + " validation issues with frames starting from start : {start} to end : {end}",
    VisionAIErrorCode.VAI_ERR_033: "The current interval [{start},{end}] for {root_key}"
    + " {data_uuid} doesn't align with the specified frame intervals {visionai_frame_intervals}.",
    VisionAIErrorCode.VAI_ERR_034: "There's a merge error in the frame intervals for data"
    + " pointer {attribute_name}, starting at {start} and ending at {end}.",
    VisionAIErrorCode.VAI_ERR_035: "The frame interval(s) for data pointer {attribute_name}"
    + " are duplicated with start at {start} and end at {end}.",
    VisionAIErrorCode.VAI_ERR_036: "The frame interval(s) for data pointer {attribute_name}"
    + " have duplicate indices at {interval_list}.",
    VisionAIErrorCode.VAI_ERR_037: "For {root_key} {data_uuid} with data pointer {attribute_name},"
    + " the current interval [{start},{end}] does not match with frames {root_key} intervals {data_uuid_intervals}.",
    VisionAIErrorCode.VAI_ERR_038: "For {root_key} uuid {data_uuid}, the attribute {attribute_name} is not found.",
    VisionAIErrorCode.VAI_ERR_039: "Frame {frame_num} contains class indices "
    + "{class_list} with disallowed index, permissible range is from 0 to {tags_count} classes.",
    VisionAIErrorCode.VAI_ERR_040: "The Run-Length Encoding (RLE) pixel count "
    + "{pixel_total} in frame {frame_num} does not correspond with the area {image_area} of the image.",
    VisionAIErrorCode.VAI_ERR_041: "An error occurred while converting data from"
    + " {original_format} to {destination_format} format.",
    VisionAIErrorCode.VAI_ERR_042: "Missing field {field_key} with value at {type} {attribute_name}"
    + " when sensors contain at least one lidar",
    VisionAIErrorCode.VAI_ERR_999: "An invalid process has been identified.",
}
