from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from visionai_data_format.schemas.ontology import Ontology
from visionai_data_format.utils.validator import (
    parse_visionai_frames_content,
    validate_attribute,
    validate_classes,
    validate_coor_system_obj,
    validate_data_pointers,
    validate_frame_object_data,
    validate_streams_obj,
    validate_tags_classes,
    validate_visionai_data,
)


class VisionAIValidator:
    def __init__(self, ontology: Ontology):
        self._ontology = ontology
        self._build_ontology_attr_map()

    def _build_ontology_attr_map(self):
        attributes = defaultdict(lambda: defaultdict(set))
        ontology_keys = {"objects", "contexts", "tags"}
        for ontology_root in ontology_keys:
            ontology_info = self._ontology.get(ontology_root)
            if not ontology_info:
                continue
            for _class_name, _class_data in ontology_info.items():
                attributes[_class_name] = defaultdict(set)
                if not _class_data:
                    continue
                for attr_name, attr_info in _class_data.get("attributes", {}).items():
                    # validate class and attribute name with uppercase
                    key = f"{attr_name.upper()}:{attr_info['type']}"
                    options = {}
                    if attr_info.get("value"):
                        options = {
                            val.upper() if isinstance(val, str) else val
                            for val in attr_info["value"]
                        }
                    attributes[_class_name][key].update(options)
        self._attributes = attributes

    def _validate_tags(
        self, visionai: Dict, tags: Dict, *args, **kwargs
    ) -> Tuple[str, int]:

        ontology_classes: Set[str] = set(tags.keys())

        return validate_tags_classes(
            visionai=visionai, ontology_classes=ontology_classes
        )

    def _validate_visionai_pointers(
        self,
        visionai: Dict,
        data_info: Dict,
        root_key: str,
        data_key_map: Dict,
        sensor_info: Dict,
        has_lidar_sensor: bool,
        has_multi_sensor: bool,
        tags_count: int = -1,
        *args,
        **kwargs,
    ) -> Optional[str]:
        ontology_classes = set(data_info.keys())
        visionai_frames = visionai.get("frames", {})
        visionai_objects = visionai.get(root_key, {})

        label_attributes = defaultdict(dict)
        extra_classes: set = validate_classes(
            visionai=visionai,
            ontology_classes=ontology_classes,
            label_attributes=label_attributes,
            root_key=root_key,
            sub_root_key=data_key_map["sub_root_key"],
            is_semantic=kwargs.get("is_semantic", False),
        )

        if extra_classes:
            return f"Label with classes {extra_classes} doesn't accepted"

        valid_attr, valid_attr_data = validate_attribute(
            label_attributes, self._attributes
        )
        if not valid_attr:
            obj_cls, name_type = valid_attr_data
            return f"Attribute Object error : class [{obj_cls}] attribute error [{name_type}]"

        sensor_name_set = set(sensor_info.keys())
        error_msg = validate_frame_object_data(
            data_root_key=root_key,
            data_child_key=data_key_map["sub_root_key"],
            frames=visionai_frames,
            has_lidar_sensor=has_lidar_sensor,
            has_multi_sensor=has_multi_sensor,
            sensor_name_set=sensor_name_set,
        )

        if error_msg:
            return error_msg

        label_attributes.clear()
        parse_visionai_frames_content(
            visionai_frames, visionai_objects, label_attributes, root_key
        )
        valid_attr, valid_attr_data = validate_attribute(
            label_attributes, self._attributes
        )
        if not valid_attr:
            obj_cls, name_type = valid_attr_data
            return f"Attribute Object error : class [{obj_cls}] attribute error [{name_type}]"

        status, err = validate_visionai_data(
            data_under_vai=visionai_objects,
            frames=visionai_frames,
            root_key=root_key,
            sub_root_key=data_key_map["sub_root_key"],
            pointer_type=data_key_map["pointer_type"],
            tags_count=tags_count,
        )

        if not status:
            return f"validate {root_key} error: {err}"

    def _validate_contexts(
        self,
        visionai: Dict,
        data_info: Dict,
        has_lidar_sensor: bool,
        has_multi_sensor: bool,
        sensor_info: Dict,
        tags_count: int = -1,
        *args,
        **kwargs,
    ):
        root_key = "contexts"
        data_key_map: Dict[str, str] = {
            "sub_root_key": "context_data",
            "pointer_type": "context_data_pointers",
        }

        return self._validate_visionai_pointers(
            visionai=visionai,
            data_info=data_info,
            root_key=root_key,
            data_key_map=data_key_map,
            sensor_info=sensor_info,
            has_lidar_sensor=has_lidar_sensor,
            has_multi_sensor=has_multi_sensor,
            tags_count=tags_count,
        )

    def _validate_objects(
        self,
        visionai: Dict,
        data_info: Dict,
        tags: Dict,
        has_lidar_sensor: bool,
        has_multi_sensor: bool,
        sensor_info: Dict,
        *args,
        **kwargs,
    ):
        root_key = "objects"
        data_key_map: Dict[str, str] = {
            "sub_root_key": "object_data",
            "pointer_type": "object_data_pointers",
        }

        tags_count = -1
        if tags:
            error_msg, tags_count = self._validate_tags(visionai=visionai, tags=tags)
            if error_msg:
                return error_msg

        return self._validate_visionai_pointers(
            visionai=visionai,
            data_info=data_info,
            root_key=root_key,
            data_key_map=data_key_map,
            sensor_info=sensor_info,
            has_lidar_sensor=has_lidar_sensor,
            has_multi_sensor=has_multi_sensor,
            tags_count=tags_count,
        )

    def _validate_taggings(
        self, visionai: Dict, data_info: Dict, *args, **kwargs
    ) -> Optional[str]:

        # validate current project tags with context
        visionai_context = visionai.get("contexts", {})
        visionai_frames = visionai.get("frames", {})
        status, err = validate_data_pointers(
            data_info, visionai_context, "context_data_pointers", only_tags=True
        )
        if not status:
            raise ValueError(
                f"validate project tags context_data_pointers error: {err}"
            )
        # validate defined tags from project with current vai context data
        status, err = validate_visionai_data(
            data_under_vai=visionai_context,
            frames=visionai_frames,
            root_key="contexts",
            sub_root_key="context_data",
            pointer_type="context_data_pointers",
        )

        if not status:
            raise ValueError(f"validate project tags error: {err}")

    def _validate_streams(
        self,
        visionai: Dict,
        sensor_info: Dict[str, str],
        has_multi_sensor: bool,
        has_lidar_sensor: bool,
        *args,
        **kwargs,
    ) -> Optional[str]:

        # verify the streams based on sensors
        if not validate_streams_obj(
            streams_data=visionai.get("streams"),
            project_sensors=sensor_info,
        ):
            return "streams error"
        project_sensors_name_set = set(sensor_info.keys())
        if has_multi_sensor and has_lidar_sensor:
            error = validate_coor_system_obj(
                coord_systems_data=visionai.get("coordinate_systems"),
                project_sensors_name_set=project_sensors_name_set,
            )
            if error:
                return f"coordinate systems error : {error}"

    @classmethod
    def validate(cls, ontology: Ontology, visionai: Dict) -> List[str]:
        """
        visionai validator

        Parameters
        ----------
        ontology: Ontology
            ontology data
        visionai : Dict
            visionai data
        Returns
        -------
        List[str]
            list of validation error message
        """
        ontology = cls(ontology=ontology)

        validator_map = {
            "streams": ontology._validate_streams,
            "contexts": ontology._validate_contexts,
            "objects": ontology._validate_objects,
            "taggings": ontology._validate_taggings,
        }

        errors: List[str] = []

        tags = ontology._ontology.get("tags", {})

        is_semantic = True if tags else False

        visionai = visionai["visionai"]
        if not visionai.get("streams"):
            raise ValueError("VisionAI missing streams data")

        streams_data = ontology._ontology["streams"]
        has_multi_sensor: bool = len(streams_data) > 1

        sensor_info: Set[str, str] = {
            sensor_name: sensor_obj["type"]
            for sensor_name, sensor_obj in streams_data.items()
        }
        has_lidar_sensor: bool = any(
            sensor_type == "lidar" for sensor_type in sensor_info.values()
        )

        for ontology_type, ontology_data in ontology._ontology.items():
            if not ontology_data or ontology_type not in validator_map:
                continue
            err = validator_map[ontology_type](
                visionai=visionai,
                data_info=ontology_data,
                is_semantic=is_semantic,
                tags=tags,
                sensor_info=sensor_info,
                has_multi_sensor=has_multi_sensor,
                has_lidar_sensor=has_lidar_sensor,
            )
            if err:
                errors.append(err)

        return errors