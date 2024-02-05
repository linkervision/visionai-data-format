from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

from pydantic import StrictInt, StrictStr

from visionai_data_format.exceptions import VisionAIErrorCode, VisionAIException

from ..ontology import Ontology


def mapping_attributes_type_value(attributes: Dict) -> Dict[str, Set]:
    """mapping attributes"""
    if not attributes:
        return {}
    attributes_map = defaultdict(set)
    for attr_type, data_list in attributes.items():
        if not data_list:
            continue

        for data in data_list:
            name = data.get("name")
            key = f"{name}:{attr_type}"
            option = {}
            if not data.get("val"):
                continue
            if isinstance(
                data.get("val"),
                (
                    str,
                    bool,
                    int,
                    float,
                ),
            ):
                option = {str(data.get("val"))}
            else:
                val_attr_vec = data.get("attributes", {}).get("vec", [])
                if val_attr_vec:
                    probability_list = []
                    for val_attr_data in val_attr_vec:
                        if val_attr_data.get("name", "") == "probability":
                            probability_list = val_attr_data.get("val", [])
                            break

                    data_length = len(data.get("val", []))
                    if data_length != len(probability_list):
                        raise VisionAIException(
                            error_code=VisionAIErrorCode.VAI_ERR_016,
                            message_kwargs={
                                "field_name": "Probability",
                                "required_length": data_length,
                            },
                        )
                option = {str(d) for d in data.get("val")}
            attributes_map[key] |= option
    return attributes_map


def parse_visionai_child_type(
    child_data: Dict[str, Dict], data_key: str
) -> Dict[str, Dict[str, Set]]:
    if not child_data or not data_key:
        return {}

    classes_attributes_map: Dict[str, Dict[str, Set]] = defaultdict(dict)
    for data in child_data.values():
        obj_class = data["type"]
        classes_attributes_map[obj_class].update(
            mapping_attributes_type_value(data.get(data_key, None))
        )

    return classes_attributes_map


def validate_visionai_intervals(visionai: Dict) -> List[VisionAIException]:
    """Validate frame intervals under visionai with its frames

    Parameters
    ----------
    visionai : Dict
        visionai data in dictionary

    Returns
    -------
    Optional[str]
        error message
    """
    error_list: List[VisionAIException] = []

    visionai_frame_interval_set = {
        frame_num
        for frame_interval in visionai["frame_intervals"]
        for frame_num in range(
            frame_interval["frame_start"], frame_interval["frame_end"] + 1
        )
    }
    frame_num_set = {int(frame_num) for frame_num in visionai["frames"].keys()}

    if visionai_frame_interval_set ^ frame_num_set:
        extra_frames = frame_num_set - visionai_frame_interval_set
        missing_frames = visionai_frame_interval_set - frame_num_set
        if extra_frames:
            error_list.append(
                VisionAIException(
                    error_code=VisionAIErrorCode.VAI_ERR_024,
                    message_kwargs={"extra_frames": extra_frames},
                )
            )

        if missing_frames:
            error_list.append(
                VisionAIException(
                    error_code=VisionAIErrorCode.VAI_ERR_025,
                    message_kwargs={"missing_frames": missing_frames},
                )
            )
    return error_list


def validate_classes(
    visionai: Dict,
    root_key: str,
    sub_root_key: str,
    ontology_classes: Set[str],
) -> Tuple[Set[str], Dict[str, Dict[str, Set]]]:
    if not visionai:
        return set()

    classes_attributes_map: Dict[str, Dict[str, Set]] = parse_visionai_child_type(
        child_data=visionai.get(root_key, {}),
        data_key=sub_root_key,
    )

    extra_classes = set(classes_attributes_map.keys()) - ontology_classes
    return extra_classes, classes_attributes_map


def build_ontology_attributes_map(ontology: Ontology) -> Dict[str, Dict]:
    """Mapping ontology root, categories to their attributes and attribute value set

    Parameters
    ----------
    ontology : Ontology
        project ontology

    Returns
    -------
    Dict[str, Dict]
        Mapping for ontology_root / class_name / attribute_name  / attribtue value set

    """
    attributes_map = defaultdict(lambda: defaultdict(dict))
    ontology_keys = {"objects", "contexts"}
    for ontology_root in ontology_keys:
        ontology_info = ontology.get(ontology_root)
        if not ontology_info:
            continue
        for _class_name, _class_data in ontology_info.items():
            attributes_map[ontology_root][_class_name] = defaultdict(set)
            if not _class_data:
                continue
            for attribute_name, attribute_info in _class_data.get(
                "attributes", {}
            ).items():
                key = f"{attribute_name}:{attribute_info['type']}"
                options = {}
                if attribute_info.get("value"):
                    options = {
                        str(val) if isinstance(val, str) else str(val)
                        for val in attribute_info["value"]
                    }
                attributes_map[ontology_root][_class_name][key].update(options)
    return attributes_map


def validate_tags_classes(
    ontology_classes: Set[str],
    tags: Optional[Dict] = None,
) -> Tuple[Optional[VisionAIException], int]:
    """verify tags under visionai data

    Parameters
    ----------
    ontology_classes : Set[str]
        current ontology classes
    tags : Optional[Dict]
        tags under visionai data, could be None

    Returns
    -------
    tuple[str, int]
        a tuple of validation error message and number of classes under tags

    """
    if tags is None:
        return ("", 0)

    if not tags:
        return (
            VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_027,
                message_kwargs={
                    "root_key": "tags",
                },
            ),
            -1,
        )

    tag_segmentation_data: Dict = {}
    for tag_data in tags.values():
        if tag_data["type"] == "semantic_segmentation_RLE":
            tag_segmentation_data = tag_data
            break
    if not tag_segmentation_data:
        return (
            VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_026,
                message_kwargs={
                    "field_key": "type",
                    "field_value": "semantic_segmentation_RLE",
                    "required_place": "tags",
                },
            ),
            -1,
        )

    vec_list = tag_segmentation_data.get("tag_data", {}).get("vec", [])
    if not vec_list:
        return (
            VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_004,
                message_kwargs={"field_name": "vector", "required_place": "tags"},
            ),
            -1,
        )

    # get the first element from vector list
    vec_info: Dict = vec_list[0]
    if vec_info["type"] != "values":
        return (
            VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_014,
                message_kwargs={"data_name": "vector", "required_type": "values"},
            ),
            -1,
        )

    classes_list: List[str] = vec_info["val"]

    classes_set: Set[str] = set(classes_list)

    extra_classes = classes_set - ontology_classes
    if extra_classes:
        return (
            VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_020,
                message_kwargs={"class_name": extra_classes},
            ),
            -1,
        )

    return (None, len(classes_set))


def validate_tags(
    visionai: Dict, tags: Dict, *args, **kwargs
) -> Tuple[Optional[VisionAIException], int]:
    # Validate the tags classes if the visionai contains this key
    ontology_classes: Set[str] = set(tags.keys())

    return validate_tags_classes(
        tags=visionai.get("tags"), ontology_classes=ontology_classes
    )


def validate_attributes(
    classes_attributes_map: Dict[str, Dict[str, Set]],
    attributes: Dict,
    excluded_attributes: Optional[Set] = None,
) -> List[VisionAIException]:
    error_list = []
    for label_class, label_attrs_data in classes_attributes_map.items():
        # already valid the class in previous step
        ontology_attr_name_type_dict: Dict[str, Set] = attributes.get(label_class, {})
        ontology_attr_name_type_set: Set[str] = set(ontology_attr_name_type_dict.keys())
        label_name_type_set: Set[str] = set(label_attrs_data.keys())
        extra_attr = label_name_type_set - ontology_attr_name_type_set
        if extra_attr:
            error_list.append(
                VisionAIException(
                    error_code=VisionAIErrorCode.VAI_ERR_017,
                    message_kwargs={
                        "extra_attributes": extra_attr,
                        "ontology_class_name": label_class,
                    },
                )
            )
            return error_list

        for label_attr_name_type, label_attr_options in label_attrs_data.items():
            # `label_attr_name_type` is combination of attribute name with its type
            #  i.e : `STREAM:text`
            label_attr_name, label_attr_type = label_attr_name_type.split(":")
            # Check whether attribute name in the excluded attributes set
            if (
                excluded_attributes and label_attr_name.lower() in excluded_attributes
            ) or label_attr_type.lower() != "vec":
                continue

            ontology_attr_options: Set[str] = ontology_attr_name_type_dict.get(
                label_attr_name_type, set()
            )

            # Change all attribute values string set
            processed_options = (
                set()
                if not label_attr_options
                else {str(opt) for opt in label_attr_options}
            )
            extra_options = processed_options - ontology_attr_options
            # Raise error in case attribute options of current class ontology is empty
            # or current datarow attributes contain extra attributes
            if not ontology_attr_options or extra_options:
                error_list.append(
                    VisionAIException(
                        error_code=VisionAIErrorCode.VAI_ERR_017,
                        message_kwargs={
                            "extra_attributes": f"{label_attr_name_type}:{extra_options}",
                            "ontology_class_name": label_class,
                        },
                    )
                )
    return error_list


def validate_frame_object_sensors_data(
    data_root_key: str,
    data_child_key: str,
    frames: Dict,
    has_lidar_sensor: bool,
    has_multi_sensor: bool,
    sensor_name_set: Set[str],
) -> Optional[VisionAIException]:
    for frame_obj in frames.values():
        cur_obj_data_type = set()
        cur_obj_stream_sensor = set()
        cur_obj_coor_sensor = set()
        obj_data_dict = frame_obj.get(data_root_key)
        if not obj_data_dict:
            continue
        for obj in obj_data_dict.values():
            for obj_data_info in obj.get(data_child_key, {}).values():
                if not obj_data_info:
                    continue
                for obj_data_info_data in obj_data_info:
                    name = obj_data_info_data.get("name")
                    stream = obj_data_info_data.get("stream")
                    coor_sensor = obj_data_info_data.get("coordinate_system")
                    if name is not None:
                        cur_obj_data_type.add(name)
                    if stream is not None:
                        cur_obj_stream_sensor.add(stream)
                    if coor_sensor is not None:
                        cur_obj_coor_sensor.add(coor_sensor)
        extra = cur_obj_stream_sensor - sensor_name_set
        if has_multi_sensor and extra:
            return VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_003,
                message_kwargs={
                    "data_type": "frame stream",
                    "extra_sensors": extra,
                    "root_sensors": sensor_name_set,
                    "root_name": "visionai streams",
                },
            )
        extra = cur_obj_coor_sensor - sensor_name_set
        if has_lidar_sensor and extra:
            return VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_003,
                message_kwargs={
                    "data_type": "frame coordinate systems",
                    "extra_sensors": extra,
                    "root_sensors": sensor_name_set,
                    "root_name": "visionai coordinate systems",
                },
            )

        frame_properties = frame_obj.get("frame_properties")
        if not frame_properties:
            return VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_019,
                message_kwargs={"root_key": "frame_properties"},
            )

        streams_name_set = set(frame_properties["streams"].keys())

        extra = streams_name_set - sensor_name_set

        if extra:
            return VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_012,
                message_kwargs={"sensor_name": extra, "sensor_type": "camera or lidar"},
            )

    return None


def get_frame_object_attr_type(
    frame_objects: Dict[str, Dict],
    all_objects: Dict[str, Dict],
    subroot_key: str,
) -> Dict[str, Dict[str, Set]]:
    """get frame object/context attributes to compare"""
    if not frame_objects:
        return
    if subroot_key not in {"object_data", "context_data"}:
        raise VisionAIException(
            error_code=VisionAIErrorCode.VAI_ERR_018,
            message_kwargs={"root_key": subroot_key},
        )

    obj_data_ele_set = {"bbox", "poly2d", "point2d", "binary"}

    classes_attributes_map: Dict[str, Dict[str, Set]] = defaultdict(
        lambda: defaultdict(set)
    )
    for obj_id, obj_data in frame_objects.items():
        global_object = all_objects.get(obj_id)
        if not global_object:
            continue

        data = obj_data.get(subroot_key)
        if not data:
            continue

        obj_class = global_object["type"]

        mapped_attributes = {}
        if subroot_key == "object_data":
            for ele_type in obj_data_ele_set:
                ele_obj = data.get(ele_type)
                if not ele_obj:
                    continue
                for ele in ele_obj:
                    mapped_attributes.update(
                        mapping_attributes_type_value(ele.get("attributes"))
                    )
        else:
            mapped_attributes = mapping_attributes_type_value(data)
        for attribute_name_type, attribute_options in mapped_attributes.items():
            classes_attributes_map[obj_class][attribute_name_type].update(
                attribute_options
            )
    return classes_attributes_map


def parse_visionai_frames_objects(
    frames: Dict[str, Dict],
    objects: Dict[str, Dict],
    root_key: str,
) -> Dict[str, Dict[str, Set]]:
    """get object_data/ context_data in visionai frames to compare"""
    if not frames:
        return
    subroot_key = "object_data" if root_key == "objects" else "context_data"
    classes_attributes_map: Dict[str, Dict[str, Set]] = defaultdict(
        lambda: defaultdict(set)
    )
    for data in frames.values():
        obj = data.get(root_key, None)
        if not obj:
            continue
        frame_object_attribute_type_map = get_frame_object_attr_type(
            obj, objects, subroot_key
        )
        # update attribute value set for each frame
        for class_, attribute_data in frame_object_attribute_type_map.items():
            for attribute_name, attribute_values in attribute_data.items():
                classes_attributes_map[class_][attribute_name].update(attribute_values)
    return classes_attributes_map


def parse_data_pointers(
    data_under_vai: Dict, pointer_type: str
) -> Tuple[Dict[Tuple[str, str], Dict], Dict[str, Dict]]:
    """mapping data pointers under visionai with
    object uuid and its name as key

    Parameters
    ----------
    data_under_vai : Dict
        data under visionai, such as `objects` or `contexts` data
    pointer_type : str
        key of data pointer under the objects, such as `object_data_pointers` or `context_data_pointers`

    Returns
    -------
    tuple[Dict[tuple[str, str], Dict], Dict[str, Dict]]
        a tuple of two dictionary, the first dictionary is a dictionary of data pointer type and frame intervals
        with uuid and attribute name combination as the key, the second dictionary is a dictionary of
        object uuid with the list of interval tuple
    """

    if not data_under_vai:
        return {}, {}
    data_pointers: Dict[Tuple[str, str], Dict] = defaultdict(dict)
    data_obj_under_vai_intervals: Dict[str, List] = defaultdict(list)
    for uuid, data in data_under_vai.items():
        for attr_name, attr_ptr_data in data[pointer_type].items():
            data_pointers[(uuid, attr_name)] = {
                "type": attr_ptr_data["type"],
                "frame_intervals": attr_ptr_data["frame_intervals"],
            }
        data_obj_under_vai_intervals[uuid] = []
        for interval in data["frame_intervals"]:
            data_obj_under_vai_intervals[uuid].append(
                (int(interval["frame_start"]), int(interval["frame_end"]))
            )

    return data_pointers, data_obj_under_vai_intervals


def parse_dynamic_attrs(
    frames: Dict, root_key: str, sub_root_key: str
) -> Dict[Tuple[str], dict]:
    """mapping attributes inside frame based on object uuid, attribute name, and frame number

    Parameters
    ----------
    frames : dict
        frames data from visionai
    root_key : str
        key under frame, such as `objects` or `contexts`
    sub_root_key : str
        child key of the root key, such as `object_data` or `context_data`

    Returns
    -------
    Dict[tuple[str], dict]
        dictionary of attribute type and value with uuid, attribute name,
        and frame number combination as the key
    """
    dynamic_attrs: Dict[Tuple[str, str], Dict] = defaultdict(lambda: defaultdict(dict))
    for frame_no, frame_obj in frames.items():
        cur_frame_no = int(frame_no)
        if not frame_obj.get(root_key):
            continue
        for uuid, data in frame_obj[root_key].items():
            for attr_type, attr_list in data[sub_root_key].items():
                for attr in attr_list:
                    dynamic_attrs[(uuid, attr["name"])][cur_frame_no] = {
                        "type": attr_type,
                        "val": attr["val"],
                    }

    return dynamic_attrs


def parse_static_attrs(
    data_under_vai: Dict, sub_root_key: str
) -> Dict[Tuple[str, str], Dict]:
    """mapping data attributes under visionai objects with
    object uuid and its name as key

    Parameters
    ----------
    data_under_vai : dict
        data under visionai, such as `objects` or `contexts` data
    sub_root_key : str
        child key of the root key, such as `object_data` or `context_data`

    Returns
    -------
    Dict[tuple, dict]
        dictionary of attribute type and value with uuid, attribute name,
        and frame number combination as the key
    """

    static_attrs: Dict[Tuple[str, str], Dict] = defaultdict(dict)

    for uuid, data in data_under_vai.items():
        # we need to skip to parsing current contexts/objects static attributes
        # if we meets below requirements:
        # 1. when current contexts/objects doesn't contains `context_data`/`object_data`,
        #    we could skip current contexts/objects
        if sub_root_key not in data:
            continue
        for attr_type, attr_list in data[sub_root_key].items():
            for attr in attr_list:
                static_attrs[(uuid, attr["name"])] = {
                    "type": attr_type,
                    "val": attr["val"],
                }
    return static_attrs


def gen_intervals(range_list: List[int]) -> List[Tuple[int, int]]:
    """given a list of numbers, return its range interval list

    Parameters
    ----------
    range_list : list[int]
        list of numbers

    Returns
    -------
    list[tuple[int, int]]
        list of range intervals in tuple, where first index is the start of the range,
        the second index is the end of the range
    """
    # generate intervals from list
    # [0,1,2,3,5,8,9,12] -> [(0, 3), (5, 5), (8, 9), (12, 12)]
    range_list.sort()
    start, end = range_list[0], range_list[0]
    result_intervals: List[tuple[int, int]] = [(start, end)]
    for frame_num in range_list:
        last_start, last_end = result_intervals[-1]
        if last_start <= frame_num <= last_end:
            continue
        if frame_num > last_end and frame_num - last_end == 1:
            result_intervals[-1] = (last_start, frame_num)
        elif frame_num < last_start and last_start - frame_num == 1:
            result_intervals[-1] = (frame_num, last_end)
        else:
            result_intervals.append((frame_num, frame_num))

    if len(result_intervals) == 1:
        return result_intervals

    # merge intervals in case there is any interval that could overlap
    # [(0, 3), (3, 5), (8, 9), (12, 12)] -> [(0, 5), (8, 9), (12, 12)]
    result_intervals.sort(key=lambda x: x[0])
    new_result_intervals: list[tuple[int, int]] = [result_intervals[0]]
    for start, end in result_intervals[1:]:
        last_start, last_end = new_result_intervals[-1]
        if last_start <= start and end <= last_end:
            continue
        if 0 <= (last_end - start) <= 1:
            new_end = max(end, last_end)
            new_result_intervals[-1] = (last_start, new_end)
        elif 0 <= (last_start - end) <= 1:
            new_result_intervals[-1] = (start, last_end)
        else:
            new_result_intervals.append((start, end))
    return new_result_intervals


def validate_vai_data_frame_intervals(
    root_key: str,
    data_obj_under_vai_intervals: Dict[str, List],
    visionai_frame_intervals: List[Tuple[int, int]],
) -> List[VisionAIException]:
    """validate frame intervals of object/context under visionai

    Parameters
    ----------
    root_key : str
        visionai object key, such as `contexts` or `objects`
    data_obj_under_vai_intervals : Dict[str, list]
        a dictionary of object uuid with the list of interval tuple
    visionai_frame_intervals : list[tuple[int, int]]
        list of interval range from current visionai frames object

    Returns
    -------
    list[VisionAIException]
        return a list of VisionAIException
    """

    error_list: List[VisionAIException] = []
    for data_uuid, data_intervals in data_obj_under_vai_intervals.items():
        for start, end in data_intervals:
            if start > end or start < 0 or end < 0:
                error_list.append(
                    VisionAIException(
                        error_code=VisionAIErrorCode.VAI_ERR_032,
                        message_kwargs={
                            "root_key": root_key,
                            "data_uuid": data_uuid,
                            "start": start,
                            "end": end,
                        },
                    )
                )
                continue

            if any(
                frame_interval[0] <= start <= frame_interval[1]
                and frame_interval[0] <= end <= frame_interval[1]
                for frame_interval in visionai_frame_intervals
            ):
                continue

            error_list.append(
                VisionAIException(
                    error_code=VisionAIErrorCode.VAI_ERR_032,
                    message_kwargs={
                        "root_key": root_key,
                        "data_uuid": data_uuid,
                        "start": start,
                        "end": end,
                    },
                )
            )
    return error_list


def vai_data_data_pointers_intervals(
    root_key: str,
    data_pointers: Dict[Tuple[str, str], Dict],
    data_obj_under_vai_intervals: Dict[str, List],
) -> Tuple[
    VisionAIException,
    Union[
        Dict[Tuple[StrictStr, StrictStr], List[Tuple[StrictInt, StrictInt]]], StrictStr
    ],
]:
    """validate intervals between data pointer and its object frame intervals

    Parameters
    ----------
    root_key : str
        visionai object key, such as `contexts` or `objects`
    data_pointers : Dict[tuple[str, str], dict]
        a dictionary of data pointer type and frame intervals
        with uuid and attribute name combination as the key
    data_obj_under_vai_intervals : Dict[str, list]
        a dictionary of object uuid with the list of interval tuple

    Returns
    -------
    Tuple[List[VisionAIException],Union[Dict[ tuple[str, str], list[tuple[int, int]]],str]]
        return a tuple of error list and its contents,
        error message or a dictionary of interval range with uuid and attr name
        as its key
    """

    error_list: List[VisionAIException] = []
    # merge data pointers intervals
    data_pointers_frames_intervals: Dict[
        Tuple[str, str], List[Tuple[int, int]]
    ] = defaultdict(list)
    for data_key, data_info in data_pointers.items():
        interval_list: List[Tuple[int, int]] = list()
        interval_set: Set[List[Tuple[int, int]]] = set()
        for frame_interval_info in data_info["frame_intervals"]:
            start = int(frame_interval_info["frame_start"])
            end = int(frame_interval_info["frame_end"])
            if start > end or start < 0 or end < 0:
                error_list.append(
                    VisionAIException(
                        error_code=VisionAIErrorCode.VAI_ERR_034,
                        message_kwargs={
                            "attribute_name": data_key,
                            "start": start,
                            "end": end,
                        },
                    )
                )
                continue
            if (start, end) in interval_set:
                error_list.append(
                    VisionAIException(
                        error_code=VisionAIErrorCode.VAI_ERR_035,
                        message_kwargs={
                            "attribute_name": data_key,
                            "start": start,
                            "end": end,
                        },
                    )
                )
                continue
            interval_set.add((start, end))
            interval_list.extend([idx for idx in range(start, end + 1)])
        # validate whether there is any duplicate intervals
        if len(interval_list) == len(set(interval_list)):
            data_pointers_frames_intervals[data_key] = gen_intervals(interval_list)
        else:
            error_list.append(
                VisionAIException(
                    error_code=VisionAIErrorCode.VAI_ERR_036,
                    message_kwargs={
                        "attribute_name": data_key,
                        "interval_list": interval_list,
                    },
                )
            )

    # validate data under vai intervals with data pointers intervals
    for (
        data_pointer_key,
        data_pointer_frame_intervals,
    ) in data_pointers_frames_intervals.items():
        attr_uuid, attr_name = data_pointer_key
        for start, end in data_pointer_frame_intervals:
            if any(
                frame_interval[0] <= start <= frame_interval[1]
                and frame_interval[0] <= end <= frame_interval[1]
                for frame_interval in data_obj_under_vai_intervals[attr_uuid]
            ):
                continue
            error_list.append(
                VisionAIException(
                    error_code=VisionAIErrorCode.VAI_ERR_037,
                    message_kwargs={
                        "root_key": root_key,
                        "data_uuid": attr_uuid,
                        "attribute_name": attr_name,
                        "start": start,
                        "end": end,
                        "data_uuid_intervals": data_obj_under_vai_intervals[attr_uuid],
                    },
                )
            )

    return error_list, data_pointers_frames_intervals


def validate_dynamic_attrs_data_pointer_intervals(
    root_key: str,
    dynamic_attrs: Dict[Tuple[str, str], Dict],
    data_pointers_frames_intervals: Dict[Tuple[str, str], List[Tuple[int, int]]],
) -> List[VisionAIException]:
    """validate dynamic attributes frames intervals with data pointer intervals

    Parameters
    ----------
    root_key : str
        visionai object key, such as `contexts` or `objects`
    dynamic_attrs : Dict[tuple[str, str], dict]
        a dictionary of dynamic attributes that declared under `frames`
    data_pointers_frames_intervals : Dict[ tuple[str, str], list[tuple[int, int]]]
        a dictionary of interval range with uuid and attr name as its key

    Returns
    -------
    List[VisionAIException]
        list of VisionAIException
    """

    dynamic_attrs_frames_intervals: Dict[Tuple[str, str], List[Tuple[int, int]]] = {
        data_key: gen_intervals(list(data_info.keys()))
        for data_key, data_info in dynamic_attrs.items()
    }

    error_list: List[VisionAIException] = []

    for attr_key, attr_intervals in dynamic_attrs_frames_intervals.items():
        data_pointer_frame_intervals = data_pointers_frames_intervals.get(attr_key)
        if not data_pointer_frame_intervals:
            error_list.append(
                VisionAIException(
                    error_code=VisionAIErrorCode.VAI_ERR_038,
                    message_kwargs={
                        "root_key": root_key,
                        "data_uuid": attr_key[0],
                        "attribute_name": attr_key[1],
                    },
                )
            )
            break
        # validate data under frames intervals with data pointers intervals

        for start, end in attr_intervals:
            if any(
                frame_interval[0] <= start <= frame_interval[1]
                and frame_interval[0] <= end <= frame_interval[1]
                for frame_interval in data_pointer_frame_intervals
            ):
                continue

            error_list.append(
                VisionAIException(
                    error_code=VisionAIErrorCode.VAI_ERR_037,
                    message_kwargs={
                        "root_key": root_key,
                        "data_uuid": attr_key[0],
                        "attribute_name": attr_key[1],
                        "start": start,
                        "end": end,
                        "data_uuid_intervals": data_pointer_frame_intervals,
                    },
                )
            )

    return error_list


def validate_dynamic_attrs_data_pointer_semantic_values(
    dynamic_attrs: Dict[Tuple[str, str], Dict],
    tags_count: int,
    img_area: int = -1,
) -> List[VisionAIException]:
    """validate dynamic attributes semantic value

    Parameters
    ----------
    dynamic_attrs : Dict[Tuple[str, str], dict]
        a dictionary of dynamic attributes that declared under `frames`
    tags_count : int
        number of classes inside tags object under visionai
    img_area : int
        area of image from width*height

    Returns
    -------
    Tuple[bool,str]
        return a tuple of boolean status and its error message
    """

    error_list: List[VisionAIException] = []
    for frame_data in dynamic_attrs.values():
        for frame_num, attr_info in frame_data.items():
            if attr_info["type"] != "binary":
                continue
            mask_rle: str = attr_info["val"]

            # retrieve classes from #pixelnumVclass
            pixel_list: List[str] = [data for data in mask_rle.split("#") if data]
            pixel_total: int = 0
            cls_list: List[int] = []

            for data in pixel_list:
                pixel_count, cls_idx = data.rsplit("V", 1)
                pixel_total += int(pixel_count)
                cls_list.append(int(cls_idx))

            if tags_count <= 0 and cls_list:
                error_list.append(
                    VisionAIException(
                        error_code=VisionAIErrorCode.VAI_ERR_018,
                        message_kwargs={"root_key": "tags"},
                    )
                )
            # validate whether annotation class indices are lower or higher than allowed
            if tags_count >= 0 and max(cls_list) >= tags_count or min(cls_list) < 0:
                error_list.append(
                    VisionAIException(
                        error_code=VisionAIErrorCode.VAI_ERR_039,
                        message_kwargs={
                            "frame_num": frame_num,
                            "class_list": list(
                                set(cls_list)
                            ),  # only need to show unique classes
                            "tags_count": tags_count - 1,
                        },
                    )
                )

            # TODO: retrieve saved image area
            if img_area != -1 and pixel_total != img_area:
                error_list.append(
                    VisionAIException(
                        error_code=VisionAIErrorCode.VAI_ERR_040,
                        message_kwargs={
                            "frame_num": frame_num,
                            "pixel_total": pixel_total,
                            "image_area": img_area,
                        },
                    )
                )
    return error_list


def generate_missing_attributes_error_message(
    extra_attribute_names: Set[str],
    missing_attribute_names: Set[str],
    dynamic_attrs: dict,
) -> List[VisionAIException]:
    """Generate error messages of extra or missing attributes from existing dynamic attributes

    Parameters
    ----------
    extra_attribute_names : Set[str]
        names of extra attributes, attributes that don't exist under visionai objects/contexts
    missing_attribute_names : Set[str]
        names of missing attributes, attributes that don't used under visionai frames
    dynamic_attrs : dict
        dynamic attributes data

    Returns
    -------
    list
        VisionAIException error list
    """
    error_list = []
    if extra_attribute_names:
        for attr_name in extra_attribute_names:
            if attr_name in dynamic_attrs:
                error_list.append(
                    VisionAIException(
                        error_code=VisionAIErrorCode.VAI_ERR_028,
                        message_kwargs={
                            "attribute_name": attr_name,
                            "frame_list": list(dynamic_attrs[attr_name].keys()),
                        },
                    )
                )
            else:
                error_list.append(
                    VisionAIException(
                        error_code=VisionAIErrorCode.VAI_ERR_029,
                        message_kwargs={
                            "attribute_name": attr_name,
                        },
                    )
                )

    if missing_attribute_names:
        for attr_name in missing_attribute_names:
            if attr_name in dynamic_attrs:
                error_list.append(
                    VisionAIException(
                        error_code=VisionAIErrorCode.VAI_ERR_030,
                        message_kwargs={
                            "attribute_name": attr_name,
                        },
                    )
                )
            else:
                error_list.append(
                    VisionAIException(
                        error_code=VisionAIErrorCode.VAI_ERR_031,
                        message_kwargs={
                            "attribute_name": attr_name,
                        },
                    )
                )
    return error_list


def validate_visionai_data(
    data_under_vai: Dict,
    frames: Dict[str, Dict],
    root_key: str = "contexts",
    sub_root_key: str = "context_data",
    pointer_type: str = "context_data_pointers",
    tags_count: int = -1,
) -> List[VisionAIException]:
    error_list: List[VisionAIException] = []
    parsed_data_pointers: Tuple[
        Dict[Tuple[str, str], Dict], Dict[str, List]
    ] = parse_data_pointers(
        data_under_vai,
        pointer_type,
    )
    data_pointers, data_obj_under_vai_intervals = parsed_data_pointers

    static_attrs: Dict[Tuple[str, str], Dict] = parse_static_attrs(
        data_under_vai, sub_root_key
    )

    dynamic_attrs: Dict[Tuple[str, str], Dict] = parse_dynamic_attrs(
        frames,
        root_key,
        sub_root_key,
    )

    # the reason why changing static_attrs and dynamic_attrs structure is the key
    # that contains attribute data is attribute type, instead of attribute name
    # e.g "text":[{"name": ..., }, {}, {}, {}], therefore for each look up
    # to check attribute existence costs a lot.

    # retrieve static attributes keys
    static_attrs_keys: Set[Tuple[str, str]] = (
        set() if not static_attrs else set(static_attrs.keys())
    )

    dynamic_attrs_keys: Set[Tuple[str, str]] = (
        set() if not dynamic_attrs else set(dynamic_attrs.keys())
    )

    data_pointers_keys: Set[Tuple[str, str]] = (
        set() if not data_pointers else set(data_pointers.keys())
    )

    # validate if combinations of static and dynamic equals to data pointers
    combination_attrs = static_attrs_keys | dynamic_attrs_keys
    if combination_attrs ^ data_pointers_keys:
        extra_attribute_names: Set[str] = combination_attrs - data_pointers_keys
        missing_attribute_names: Set[str] = data_pointers_keys - combination_attrs
        error_list += generate_missing_attributes_error_message(
            extra_attribute_names=extra_attribute_names,
            missing_attribute_names=missing_attribute_names,
            dynamic_attrs=dynamic_attrs,
        )

    # retrieve frame numbers
    frame_numbers = [int(frame_num) for frame_num in frames.keys()]

    # create frame intervals from frame numbers in case the frames is not continuous
    visionai_frame_intervals: List[Tuple[int, int]] = gen_intervals(frame_numbers)

    # validate data under vai intervals with the frame intervals
    error_list += validate_vai_data_frame_intervals(
        root_key=root_key,
        data_obj_under_vai_intervals=data_obj_under_vai_intervals,
        visionai_frame_intervals=visionai_frame_intervals,
    )

    # validate data under vai intervals with data pointers intervals
    errors, return_content = vai_data_data_pointers_intervals(
        root_key=root_key,
        data_pointers=data_pointers,
        data_obj_under_vai_intervals=data_obj_under_vai_intervals,
    )
    if errors:
        error_list += errors
    else:
        # retrieve data pointers frame intervals
        data_pointers_frames_intervals: Dict[
            Tuple[str, str], List[Tuple[int, int]]
        ] = return_content

        # validate dynamic attributes frames intervals with data pointer intervals
        errors = validate_dynamic_attrs_data_pointer_intervals(
            root_key=root_key,
            dynamic_attrs=dynamic_attrs,
            data_pointers_frames_intervals=data_pointers_frames_intervals,
        )

        if errors:
            error_list += errors

        # validate if current image_type is semantic_segmentation
        if root_key == "objects":
            errors = validate_dynamic_attrs_data_pointer_semantic_values(
                dynamic_attrs=dynamic_attrs,
                tags_count=tags_count,
            )
            if errors:
                error_list += errors

    return error_list


def validate_visionai_children(
    visionai: Dict,
    ontology_data: Dict,
    root_key: str,
    data_key_map: Dict,
    sensor_info: Dict,
    has_lidar_sensor: bool,
    has_multi_sensor: bool,
    ontology_attributes_map: Optional[Dict[str, Dict[str, Set]]] = None,
    tags_count: int = -1,
    *args,
    **kwargs,
) -> List[VisionAIException]:
    error_list: List[VisionAIException] = []

    if not ontology_attributes_map:
        ontology_attributes_map = {}
    ontology_classes = set(ontology_data.keys())
    visionai_frames = visionai.get("frames", {})
    visionai_objects = visionai.get(root_key, {})
    extra_classes, classes_attributes_map = validate_classes(
        visionai=visionai,
        ontology_classes=ontology_classes,
        root_key=root_key,
        sub_root_key=data_key_map["sub_root_key"],
    )

    if extra_classes:
        error_list.append(
            VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_020,
                message_kwargs={"class_name": extra_classes},
            )
        )

    ontology_attribute_exceptions: List[VisionAIException] = validate_attributes(
        classes_attributes_map, ontology_attributes_map
    )
    error_list += ontology_attribute_exceptions
    sensor_name_set = set(sensor_info.keys())
    valid_frame_sensor_error: Optional[
        VisionAIException
    ] = validate_frame_object_sensors_data(
        data_root_key=root_key,
        data_child_key=data_key_map["sub_root_key"],
        frames=visionai_frames,
        has_lidar_sensor=has_lidar_sensor,
        has_multi_sensor=has_multi_sensor,
        sensor_name_set=sensor_name_set,
    )

    if valid_frame_sensor_error:
        error_list.append(valid_frame_sensor_error)

    frames_attributes_map: Dict[str, Dict[str, Set]] = parse_visionai_frames_objects(
        visionai_frames, visionai_objects, root_key
    )
    frame_attribute_exceptions: List[VisionAIException] = validate_attributes(
        frames_attributes_map, ontology_attributes_map
    )
    error_list += frame_attribute_exceptions

    error_list += validate_visionai_data(
        data_under_vai=visionai_objects,
        frames=visionai_frames,
        root_key=root_key,
        sub_root_key=data_key_map["sub_root_key"],
        pointer_type=data_key_map["pointer_type"],
        tags_count=tags_count,
    )

    return error_list


def validate_contexts(
    visionai: Dict,
    ontology_data: Dict,
    ontology_attributes_map: Dict,
    has_lidar_sensor: bool,
    has_multi_sensor: bool,
    sensor_info: Dict,
    tags_count: int = -1,
    *args,
    **kwargs,
) -> List[VisionAIException]:
    root_key = "contexts"
    data_key_map: Dict[str, str] = {
        "sub_root_key": "context_data",
        "pointer_type": "context_data_pointers",
    }

    return validate_visionai_children(
        visionai=visionai,
        ontology_data=ontology_data,
        ontology_attributes_map=ontology_attributes_map,
        root_key=root_key,
        data_key_map=data_key_map,
        sensor_info=sensor_info,
        has_lidar_sensor=has_lidar_sensor,
        has_multi_sensor=has_multi_sensor,
        tags_count=tags_count,
    )


def validate_objects(
    visionai: Dict,
    ontology_data: Dict,
    ontology_attributes_map: Dict,
    tags: Dict,
    has_lidar_sensor: bool,
    has_multi_sensor: bool,
    sensor_info: Dict,
    *args,
    **kwargs,
) -> List[VisionAIException]:
    root_key = "objects"
    data_key_map = {
        "sub_root_key": "object_data",
        "pointer_type": "object_data_pointers",
    }
    tags_count = -1

    error_list: List[VisionAIException] = []
    # We do not need tags for instance_mask, so we set the tags_count to 2 for doing the validation of segmentation
    # For instance_mask, we'll only have 0 for background and 1 for the given category.
    # The RLE will only have V0 and V1.
    for object in ontology_data.values():
        if "instance_mask" in object.get("attributes", {}):
            tags_count = 2
    # validate ontology.tags and visionai.tags for segmentation data
    if tags:
        error_msg, tags_count = validate_tags(visionai=visionai, tags=tags)
        if error_msg:
            error_list += [error_msg]
    error_list += validate_visionai_children(
        visionai=visionai,
        ontology_data=ontology_data,
        ontology_attributes_map=ontology_attributes_map,
        root_key=root_key,
        data_key_map=data_key_map,
        sensor_info=sensor_info,
        has_lidar_sensor=has_lidar_sensor,
        has_multi_sensor=has_multi_sensor,
        tags_count=tags_count,
    )

    return error_list


def validate_streams_obj(
    streams_data: Dict[str, Dict], ontology_sensors: Dict[str, str]
) -> Optional[VisionAIException]:
    if not streams_data:
        return False
    for stream_name, stream_obj in streams_data.items():
        stream_obj_type = stream_obj.get("type", "")
        if (
            stream_name not in ontology_sensors
            or stream_obj_type != ontology_sensors[stream_name]
        ):
            return VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_012,
                message_kwargs={
                    "sensor_name": stream_name,
                    "sensor_type": stream_obj_type,
                },
            )
    return None


def validate_coor_system_obj(
    coord_systems_data: Dict[str, Dict], ontology_sensors_name_set: Set[str]
) -> Optional[VisionAIErrorCode]:
    if not coord_systems_data:
        return VisionAIException(
            error_code=VisionAIErrorCode.VAI_ERR_019,
            message_kwargs={"root_key": "coordinate_systems"},
        )
    data_sensors = {
        sensor_name
        for sensor_name, sensor_info in coord_systems_data.items()
        if sensor_info["type"] != "local_cs"
    }
    extra_sensors = data_sensors - ontology_sensors_name_set
    if extra_sensors:
        return VisionAIException(
            error_code=VisionAIErrorCode.VAI_ERR_003,
            message_kwargs={
                "data_type": "coordinate systems",
                "extra_sensors": extra_sensors,
                "root_sensors": ontology_sensors_name_set,
                "root_name": "project ontology",
            },
        )

    return None


def validate_streams(
    visionai: Dict,
    sensor_info: Dict[str, str],
    has_multi_sensor: bool,
    has_lidar_sensor: bool,
    *args,
    **kwargs,
) -> Tuple[Optional[VisionAIException], Dict[str, str]]:
    if not visionai.get("streams"):
        return (
            VisionAIException(
                error_code=VisionAIErrorCode.VAI_ERR_019,
                message_kwargs={"root_key": "streams"},
            ),
            {},
        )

    # verify the streams based on sensors
    streams_data = visionai.get("streams")
    error: Optional[VisionAIErrorCode] = validate_streams_obj(
        streams_data=streams_data,
        ontology_sensors=sensor_info,
    )
    if error:
        return error, {}
    ontology_sensors_name_set = set(sensor_info.keys())
    if has_multi_sensor and has_lidar_sensor:
        error: Optional[VisionAIErrorCode] = validate_coor_system_obj(
            coord_systems_data=visionai.get("coordinate_systems"),
            ontology_sensors_name_set=ontology_sensors_name_set,
        )
        if error:
            return error, {}

    new_sensor_info = {
        stream_name: stream_obj.get("type", "")
        for stream_name, stream_obj in streams_data.items()
    }

    return None, new_sensor_info


def validate_data_pointers(
    attribute_data: Dict[str, Dict],
    data_under_vai: Dict[str, Dict],
    pointer_type: str = "context_data_pointers",
) -> Tuple[bool, str]:
    """validate attribute names and type under data_pointer with project settings,
    since data_pointer contains all attributes(name and type)
    under this uuid.

    Parameters
    ----------
    attribute_data : Dict[str, dict]
        attribute data from project
    data_under_vai : Dict[str, dict]
        data under visionai, such as `objects` or `contexts` data
    pointer_type : str, optional
        key of data pointer under the objects, by default "context_data_pointers"

    Returns
    -------
    tuple[bool, str]
        the tuple of error status and its message
    """
    attribute_data_name_set = set(attribute_data.keys())

    attribute_data_name_type_set = {
        (attr_name, attr_info["type"])
        for attr_name, attr_info in attribute_data.items()
    }

    for data_uuid, data_info in data_under_vai.items():
        data_pointer = data_info.get(pointer_type)
        if not data_pointer:
            return False, f"UUID {data_uuid} doesn't contains data key {pointer_type}"

        data_pointer_name_set = set(data_pointer.keys())

        extras = data_pointer_name_set - attribute_data_name_set
        if extras:
            return False, f"UUID {data_uuid} have extra attributes : {extras}"

        data_pointer_name_type_set: Set[Tuple[str, str]] = {
            (
                data_pointer_name,
                data_pointer_info["type"]
                if data_pointer_info["type"] != "vec"
                else "option",
            )
            for data_pointer_name, data_pointer_info in data_pointer.items()
        }

        extras: Set[str] = data_pointer_name_type_set - attribute_data_name_type_set

        if extras:
            error_message = "Data pointer type error : Extra <-> Exist\n"
            for extra in extras:
                data_pointer_name, data_pointer_type = extra
                attr_type = attribute_data[data_pointer_name]["type"]
                error_message += f"{data_pointer_name}:{data_pointer_type} <-> {data_pointer_name}:{attr_type}\n"
            return False, error_message

    return True, ""
