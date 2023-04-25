import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

from visionai_data_format.schemas.bdd_schema import BDDSchema
from visionai_data_format.schemas.visionai_schema import (
    Attributes,
    Context,
    CoordinateSystem,
    Frame,
    Object,
    Stream,
    VisionAIModel,
)

logger = logging.getLogger(__name__)


def validate_vai(data: Dict) -> Union[VisionAIModel, None]:
    try:
        vai = VisionAIModel(**data)
        logger.info("[validated_vai] Validate success")
        return vai
    except Exception as e:
        logger.error("[validated_vai] Validate failed : " + str(e))
        return None


def validate_bdd(data: Dict) -> Union[BDDSchema, None]:
    try:
        bdd = BDDSchema(**data)
        logger.info("[validate_bdd] Validation success")
        return bdd
    except Exception as e:
        logger.error("[validate_bdd] Validation failed : " + str(e))
        return None


def attribute_generator(
    category: str, attribute: Dict, ontology_class_attrs: Dict
) -> Dict:

    if not attribute:
        return dict()

    new_attribute = dict()
    category = category.upper()
    for attr_name, attr_value in attribute.items():
        logger.info(f"attr_name : {attr_name}")
        logger.info(f"attr_value : {attr_value}")
        if attr_name in ontology_class_attrs[category]:
            new_attribute[attr_name] = attr_value

    logger.info(f"[datarow_attribute_generator] new_attribute : {new_attribute}")
    return new_attribute


def save_as_json(data: Dict, file_name: str, folder_name: str = "") -> None:
    try:
        if folder_name:
            os.makedirs(folder_name, exist_ok=True)
        file = open(os.path.join(folder_name, file_name), "w")
        logger.info(
            f"[save_as_json] Save file to {os.path.join(folder_name,file_name)} started "
        )
        json.dump(data, file)
        logger.info(
            f"[save_as_json] Save file to {os.path.join(folder_name,file_name)} success"
        )

    except Exception as e:
        logger.error("[save_as_json] Save file failed : " + str(e))


def get_all_attrs_type(attributes: Attributes, all_attributes: Dict[str, Set]) -> None:
    """get attributes and convert to upper case to compare"""
    if not attributes:
        return

    for attr_type, data_list in attributes.items():
        if not data_list:
            continue

        if attr_type == "vec":
            attr_type = "option"

        for data in data_list:
            name = data.get("name").upper()
            key = f"{name}:{attr_type}"
            option = None
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
                option = {str(data.get("val")).upper()}
            else:
                if "attributes" in data:
                    val_attr_vec = data.get("attributes", {}).get("vec", [])
                    if val_attr_vec:
                        probability_list = []
                        for val_attr_data in val_attr_vec:
                            if val_attr_data.get("name", "") == "probability":
                                probability_list = val_attr_data.get("val", [])
                                break

                        data_length = len(data.get("val", []))
                        if data_length != len(probability_list):
                            raise ValueError(
                                "Probability list length "
                                + f" {len(probability_list)} doesn't match needed length : {data_length}"
                            )

                option = {str(d).upper() for d in data.get("val")}
            if not option:
                continue
            if key not in all_attributes:
                all_attributes[key] = option
            else:
                all_attributes[key] |= option

            # Now only support the first layer attribute, won't parse all layers
            # get_all_attrs_type(data.get("attributes", None), attrs)


def parse_visionai_child_type(
    child_data: Dict[str, Union[Object, Context]], label_attributes: Dict, data_key: str
) -> Set:
    if not child_data or not data_key:
        return set()
    all_types = set()
    for data in child_data.values():
        obj_class = data["type"]
        all_types.add(obj_class)
        obj_attributes = label_attributes[obj_class]

        get_all_attrs_type(data.get(data_key, None), obj_attributes)

    return all_types


def validate_classes(
    visionai: Dict,
    root_key: str,
    sub_root_key: str,
    ontology_classes: Set[str],
    label_attributes: Dict,
    is_semantic: bool = False,
) -> Set[str]:

    if not visionai:
        return False

    label_classes: Set[str] = parse_visionai_child_type(
        child_data=visionai.get(root_key, {}),
        label_attributes=label_attributes,
        data_key=sub_root_key,
    )

    if is_semantic:
        # Add an *segmentation_matrix for accepting segmentation objects.
        ontology_classes.add("*segmentation_matrix")
    extra_classes = label_classes - ontology_classes
    logger.debug(f"extra_classes : {extra_classes}")
    return extra_classes


def validate_tags_classes(
    tags: Dict,
    ontology_classes: Set[str],
) -> Tuple[str, int]:
    """verify tags under visionai data

    Parameters
    ----------
    tags : Dict
        tags under visionai data
    ontology_classes : Set[str]
        current ontology classes

    Returns
    -------
    tuple[bool, int]
        a tuple of validation error message and number of classes under tags

    """

    if not tags:
        return 0

    tag_segmentation_data: Dict = {}
    for tag_data in tags.values():
        if tag_data["type"] == "semantic_segmentation_RLE":
            tag_segmentation_data = tag_data
            break
    if not tag_segmentation_data:
        return ("Tag with type `semantic_segmentation_RLE` doesn't found", -1)

    vec_list = tag_segmentation_data.get("tag_data", {}).get("vec", [])
    if not vec_list:
        return ("Can't found vector info inside tag", -1)

    # get the first element from vector list
    vec_info: Dict = vec_list[0]
    if vec_info["type"] != "values":
        return ("Vector type must be `values`", -1)

    classes_list: List[str] = vec_info["val"]

    classes_set: Set[str] = set(classes_list)

    extra_classes = classes_set - ontology_classes
    if extra_classes:
        raise (f"Tag label with classes {extra_classes} doesn't accepted", -1)

    return ("", len(classes_set))


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

        logger.debug(f"parse_data_pointers : {uuid}-{data['type']}")
        for attr_name, attr_ptr_data in data[pointer_type].items():
            data_pointers[(uuid, attr_name)] = {
                "type": attr_ptr_data["type"],
                "frame_intervals": attr_ptr_data["frame_intervals"],
            }
        data_obj_under_vai_intervals[uuid] = []
        for interval in data["frame_intervals"]:
            logger.debug(f"interval : {interval}")
            data_obj_under_vai_intervals[uuid].append(
                (int(interval["frame_start"]), int(interval["frame_end"]))
            )

    return data_pointers, data_obj_under_vai_intervals


def parse_static_attrs(
    data_under_vai: Dict, sub_root_key: str, only_tags: bool = True
) -> Dict[Tuple[str, str], Dict]:
    """mapping data attributes under visionai objects with
    object uuid and its name as key

    Parameters
    ----------
    data_under_vai : dict
        data under visionai, such as `objects` or `contexts` data
    sub_root_key : str
        key under the data
    only_tags : bool, optional
        flag to retrieve only tags related data, by default True

    Returns
    -------
    Dict[tuple, dict]
        dictionary of attribute type and value with uuid, attribute name,
        and frame number combination as the key
    """
    static_attrs: Dict[Tuple[str, str], Dict] = defaultdict(dict)
    for uuid, data in data_under_vai.items():
        if only_tags and data["type"] != "*tagging":
            continue
        elif not only_tags and data["type"] == "*tagging":
            continue
        logger.debug(f"parse_static_attrs : {uuid}-{data['type']}")
        for attr_type, attr_list in data[sub_root_key].items():
            for attr in attr_list:
                static_attrs[(uuid, attr["name"])] = {
                    "type": attr_type,
                    "val": attr["val"],
                }
    return static_attrs


def parse_dynamic_attrs(
    frames: Dict, root_key: str, sub_root_key: str, data_pointers: Dict
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
    data_pointers : dict
        data pointer from `objects` or `contexts` under visionai

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
                    # skip uuid that we doesn't want to validate
                    # TODO: find better implementation
                    if (uuid, attr["name"]) not in data_pointers:
                        continue
                    dynamic_attrs[(uuid, attr["name"])][cur_frame_no] = {
                        "type": attr_type,
                        "val": attr["val"],
                    }

    return dynamic_attrs


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
    result_intervals: list[tuple[int, int]] = [(start, end)]
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
) -> Tuple[bool, str]:
    """_summary_

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
    tuple[bool,str]
        return a tuple of boolean status and its error message
    """
    for data_uuid, data_intervals in data_obj_under_vai_intervals.items():
        for start, end in data_intervals:
            if start > end or start < 0 or end < 0:
                return (
                    False,
                    f"{root_key} {data_uuid} frame interval(s) validate {root_key} with frames error,"
                    f"start : {start}, end : {end}",
                )
            if any(
                frame_interval[0] <= start <= frame_interval[1]
                and frame_interval[0] <= end <= frame_interval[1]
                for frame_interval in visionai_frame_intervals
            ):
                continue

            return (
                False,
                f"{root_key} {data_uuid} frame interval(s) error,"
                + f" current interval {start,end} doesn't match with frames intervals {visionai_frame_intervals}",
            )
    return True, ""


def vai_data_data_pointers_intervals(
    root_key: str,
    data_pointers: Dict[Tuple[str, str], Dict],
    data_obj_under_vai_intervals: Dict[str, List],
) -> Tuple[bool, Union[Dict[Tuple[str, str], List[Tuple[int, int]]], str]]:

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
    tuple[bool,Union[Dict[ tuple[str, str], list[tuple[int, int]]],str]]
        return a tuple of boolean status and its contents,
        error message or a dictionary of interval range with uuid and attr name
        as its key
    """

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
                return (
                    False,
                    f"data pointer {data_key} frame interval(s) merge error,"
                    f" start : {start}, end : {end}",
                )
            if (start, end) in interval_set:
                return (
                    False,
                    f"data pointer {data_key} frame interval(s) error,"
                    f"find duplicated with start : {start}, end : {end}",
                )
            interval_set.add((start, end))
            interval_list.extend([idx for idx in range(start, end + 1)])
        # validate whether there is any duplicate intervals
        if len(interval_list) != len(set(interval_list)):
            return (
                False,
                f"data pointer {data_key} frame interval(s) error,"
                f" intervals has duplicate index : {interval_list}",
            )
        data_pointers_frames_intervals[data_key] = gen_intervals(interval_list)

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
            return (
                False,
                f"{root_key} {attr_uuid} with data pointer {attr_name} frame interval(s) error,"
                + f" current data pointer interval {start,end} doesn't match with {root_key}"
                + f" interval {data_obj_under_vai_intervals[attr_uuid]}",
            )

    return True, data_pointers_frames_intervals


def validate_dynamic_attrs_data_pointer_intervals(
    root_key: str,
    dynamic_attrs: Dict[Tuple[str, str], Dict],
    data_pointers_frames_intervals: Dict[Tuple[str, str], List[Tuple[int, int]]],
) -> Tuple[bool, str]:
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
    tuple[bool,str]
        return a tuple of boolean status and its error message
    """

    dynamic_attrs_frames_intervals: Dict[Tuple[str, str], List[Tuple[int, int]]] = {
        data_key: gen_intervals(list(data_info.keys()))
        for data_key, data_info in dynamic_attrs.items()
    }

    for attr_key, attr_intervals in dynamic_attrs_frames_intervals.items():
        data_pointer_frame_intervals = data_pointers_frames_intervals.get(attr_key)
        if not data_pointer_frame_intervals:
            return (
                False,
                f"{root_key} UUID {attr_key[0]} attribute {attr_key[1]} is not found in data pointers",
            )
        # validate data under frames intervals with data pointers intervals

        for start, end in attr_intervals:
            if any(
                frame_interval[0] <= start <= frame_interval[1]
                and frame_interval[0] <= end <= frame_interval[1]
                for frame_interval in data_pointer_frame_intervals
            ):
                continue
            return (
                False,
                f"{root_key} UUID {attr_key[0]} with data pointer {attr_key[1]} frame interval(s) error,"
                + f" current data pointer interval {start,end} doesn't match with {root_key}"
                + f" interval {data_pointer_frame_intervals}",
            )

    return True, ""


def validate_dynamic_attrs_data_pointer_semantic_values(
    dynamic_attrs: Dict[Tuple[str, str], Dict],
    tags_count: int,
    img_area: int = -1,
) -> Tuple[bool, str]:
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

    msg: str = ""
    for frame_data in dynamic_attrs.values():
        for frame_num, attr_info in frame_data.items():
            mask_rle: str = ""
            if attr_info["type"] == "binary":
                mask_rle = attr_info["val"]

            if not mask_rle:
                msg += f"frame {frame_num} doesn't contains mask RLE information\n"
                continue

            # retrieve classes from #pixelnumVclass
            # TODO: move this to visionai-data-format
            pixel_list: List[str] = [data for data in mask_rle.split("#") if data]
            pixel_total: int = 0
            cls_list: List[int] = []

            for data in pixel_list:
                pixel_count, cls_idx = data.rsplit("V", 1)
                pixel_total += int(pixel_count)
                cls_list.append(int(cls_idx))

            if tags_count == 0 and cls_list:
                msg += "Can't declare RLE if the tag classes is missing\n"

            # validate whether annotation class indices are lower or higher than allowed
            if max(cls_list) >= tags_count or min(cls_list) < 0:
                msg += (
                    f"Frame {frame_num} with class indices {cls_list} contains disallowed index "
                    + f"while only allowed from 0 to {tags_count-1} classes\n"
                )

            # TODO: retrieve saved image area
            if img_area != -1 and pixel_total != img_area:
                msg += (
                    f"Frame {frame_num} RLE pixel count {pixel_total}"
                    + f" doesn't match with image with area {img_area}\n"
                )

    if msg:
        return False, msg
    return True, ""


def validate_visionai_data(
    data_under_vai: Dict,
    frames: Dict[str, Dict],
    root_key: str = "contexts",
    sub_root_key: str = "context_data",
    pointer_type: str = "context_data_pointers",
    tags_count: int = -1,
) -> Tuple[bool, str]:

    parsed_data_pointers: Tuple[
        Dict[Tuple[str, str], Dict], Dict[str, List]
    ] = parse_data_pointers(
        data_under_vai,
        pointer_type,
    )
    data_pointers, data_obj_under_vai_intervals = parsed_data_pointers

    static_attrs: Dict[Tuple[str, str], Dict] = parse_static_attrs(
        data_under_vai,
        sub_root_key,
    )

    dynamic_attrs: Dict[Tuple[str, str], Dict] = parse_dynamic_attrs(
        frames,
        root_key,
        sub_root_key,
        data_pointers,
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

    logger.debug(f"static_attrs_keys : {static_attrs_keys}")
    logger.debug(f"data_pointers_keys : {data_pointers_keys}")

    dynamic_attrs_uuid: Set[str] = (
        set() if not dynamic_attrs else {key[0] for key in dynamic_attrs.keys()}
    )
    static_attrs_uuid: Set[str] = (
        set() if not static_attrs else {key[0] for key in static_attrs.keys()}
    )

    # validate if static attributes declared on dynamic attributes
    intersection_attrs_uuid = dynamic_attrs_uuid & static_attrs_uuid
    if root_key == "context" and (intersection_attrs_uuid):
        return (
            False,
            f"UUID {intersection_attrs_uuid} duplicated in static and dynamic attributes",
        )

    # validate if combinations of static and dynamic equals to data pointers
    combination_attrs = static_attrs_keys | dynamic_attrs_keys
    if combination_attrs ^ data_pointers_keys:

        extra_attributes_name: Set[str] = combination_attrs - data_pointers_keys
        missing_attributes_name: Set[str] = data_pointers_keys - combination_attrs
        msg = ""
        if extra_attributes_name:
            msg += f"Extra attributes from data pointers : {extra_attributes_name} \n"

        if missing_attributes_name:
            msg += (
                f"Missing attributes from data pointers : {missing_attributes_name} \n"
            )
        return False, msg

    # retrieve frame numbers
    frame_numbers = [int(frame_num) for frame_num in frames.keys()]

    # create frame intervals from frame numbers in case the frames is not continuous
    visionai_frame_intervals: List[Tuple[int, int]] = gen_intervals(frame_numbers)

    # validate data under vai intervals with the frame intervals
    status, err = validate_vai_data_frame_intervals(
        root_key=root_key,
        data_obj_under_vai_intervals=data_obj_under_vai_intervals,
        visionai_frame_intervals=visionai_frame_intervals,
    )
    if not status:
        return status, err

    # validate data under vai intervals with data pointers intervals
    status, return_content = vai_data_data_pointers_intervals(
        root_key=root_key,
        data_pointers=data_pointers,
        data_obj_under_vai_intervals=data_obj_under_vai_intervals,
    )
    if not status:
        return status, return_content

    # retrieve data pointers frame intervals
    data_pointers_frames_intervals: Dict[
        Tuple[str, str], List[Tuple[int, int]]
    ] = return_content

    # validate dynamic attributes frames intervals with data pointer intervals
    status, err = validate_dynamic_attrs_data_pointer_intervals(
        root_key=root_key,
        dynamic_attrs=dynamic_attrs,
        data_pointers_frames_intervals=data_pointers_frames_intervals,
    )

    if not status:
        return status, err

    # validate if current image_type is semantic_segmentation
    if tags_count != -1 and root_key == "objects":
        status, err = validate_dynamic_attrs_data_pointer_semantic_values(
            dynamic_attrs=dynamic_attrs,
            tags_count=tags_count,
        )
        if not status:
            return status, err
    return True, ""


def validate_streams_obj(
    streams_data: Dict[str, Stream], project_sensors: Dict[str, str]
) -> bool:
    if not streams_data:
        return False
    for stream_name, stream_obj in streams_data.items():
        stream_obj_type = stream_obj.get("type", "")
        if (
            stream_name not in project_sensors
            or stream_obj_type != project_sensors[stream_name]
        ):
            return False
    return True


def validate_coor_system_obj(
    coord_systems_data: Dict[str, CoordinateSystem], project_sensors_name_set: Set[str]
) -> bool:

    if not coord_systems_data:
        return False
    data_sensors = {
        sensor_name
        for sensor_name, sensor_info in coord_systems_data.items()
        if sensor_info["type"] != "local_cs"
    }
    extra_sensors = data_sensors - project_sensors_name_set
    if len(extra_sensors) != 0:
        return False
    return True


def validate_attribute(
    label_attributes: Dict, attributes: Dict, excluded_attributes: Optional[Set] = None
) -> Tuple[bool, Union[Tuple, None]]:

    for label_class, label_attrs in label_attributes.items():
        # already valid the class in previous step
        ontology_attr_name_type_dict: Dict[str, Set] = attributes.get(label_class, {})
        ontology_attr_name_type_set: Set[str] = set(ontology_attr_name_type_dict.keys())
        label_name_type_set: Set[str] = set(label_attrs.keys())

        extra_attr = label_name_type_set - ontology_attr_name_type_set
        if extra_attr:
            raise ValueError(
                f"\nContain extra attributes {extra_attr} from"
                + f" ontology class {label_class} attributes : {ontology_attr_name_type_set}"
            )

        for label_attr_name_type, label_attr_options in label_attrs.items():
            # `label_attr_name_type` is combination of attribute name with its type
            #  i.e : `STREAM:text`
            label_attr_name, label_attr_type = label_attr_name_type.split(":")

            # Check whether attribute name in the excluded attributes set
            if (
                excluded_attributes and label_attr_name.lower() in excluded_attributes
            ) or label_attr_type.lower() != "option":
                continue

            ontology_attr_options: Set[str] = ontology_attr_name_type_dict.get(
                label_attr_name_type, {}
            )

            # Change all attribute values to upper strings
            processed_options = (
                set()
                if not label_attr_options
                else {str(opt).upper() for opt in label_attr_options}
            )
            if not ontology_attr_options or (processed_options - ontology_attr_options):
                return False, (label_class, label_attr_name_type)
    return True, None


def validate_frame_object_data(
    data_root_key: str,
    data_child_key: str,
    frames: Dict,
    has_lidar_sensor: bool,
    has_multi_sensor: bool,
    sensor_name_set: Set[str],
    required_data_type: Optional[List[str]] = None,
) -> Optional[str]:

    for frame_id, frame_obj in frames.items():
        cur_obj_data_type = set()
        cur_obj_stream_sensor = set()
        cur_obj_coor_sensor = set()
        if data_root_key not in frame_obj:
            return f"Current frame {frame_id} doesn't have `{data_root_key}` key"
        for obj in frame_obj[data_root_key].values():
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
            return f"frame stream sensor(s) {extra} are not in project sensor {sensor_name_set}"
        extra = cur_obj_coor_sensor - sensor_name_set
        if has_lidar_sensor and extra:
            return f"current frame coordinate system sensor(s) {extra} are not in project sensor {sensor_name_set}"

        extra = cur_obj_data_type - set(required_data_type)
        if required_data_type and (extra):
            return (
                f"current project required object data type : {set(required_data_type)},"
                + f" data type {extra} are not found in current frame"
            )


def get_frame_object_attr_type(
    frame_objects: Dict[str, Frame],
    all_objects: Dict[str, Object],
    exist_attributes: Dict,
    subroot_key: str,
) -> None:
    """get frame object/context attributes and convert to upper case to compare"""
    if not frame_objects:
        return
    if subroot_key not in {"object_data", "context_data"}:
        raise ValueError(f"Current subroot_key ({subroot_key}) is invalid.")

    obj_data_ele_set = {"bbox", "poly2d", "point2d", "binary"}

    for obj_id, obj_data in frame_objects.items():
        global_object = all_objects.get(obj_id)
        if not global_object:
            continue

        data = obj_data.get(subroot_key)
        if not data:
            continue

        obj_class = global_object["type"]
        obj_attributes = exist_attributes[obj_class]

        if subroot_key == "object_data":
            for ele_type in obj_data_ele_set:
                ele_obj = data.get(ele_type)
                if not ele_obj:
                    continue
                for ele in ele_obj:
                    get_all_attrs_type(ele.get("attributes"), obj_attributes)
        else:
            get_all_attrs_type(data, obj_attributes)


def parse_visionai_frames_content(
    frames: Dict[str, Frame],
    objects: Dict[str, Union[Object, Context]],
    label_attributes: Dict,
    root_key: str,
) -> None:
    """get vision ai frames and convert object/context type and attribute to upper case to compare"""
    if not frames:
        return
    subroot_key = "object_data" if root_key == "objects" else "context_data"
    for data in frames.values():
        obj = data.get(root_key, None)
        if not obj:
            continue
        get_frame_object_attr_type(obj, objects, label_attributes, subroot_key)


def validate_data_pointers(
    attribute_data: Dict[str, Dict],
    data_under_vai: Dict[str, Dict],
    pointer_type: str = "context_data_pointers",
    only_tags: bool = True,
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
    only_tags : bool, optional
        flag to retrieve only tags related data, by default True

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
        # skip the type besides '*tagging' when only_tags is true
        if only_tags and data_info["type"] != "*tagging":
            continue

        # skip the type of '*tagging' when only_tags is false
        if not only_tags and data_info["type"] == "*tagging":
            continue

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
