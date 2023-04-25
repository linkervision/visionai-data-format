from visionai_data_format.schemas.ontology import Ontology
from visionai_data_format.validators.visionai import VisionAIValidator


def test_ontology(
    fake_project_ontology, fake_visionai_ontology, fake_objects_data_single_lidar
):

    project_ontology = fake_project_ontology["ontology"]
    project_tagging = fake_project_ontology["project_tag"]
    project_sensors = fake_project_ontology["sensors"]

    # convert project ontology into visionai ontology
    project_ontology_classes = project_ontology["classes"]
    image_type = project_ontology["image_type"]

    has_lidar_sensor = any(
        sensor_info["type"] == "lidar" for sensor_info in project_sensors
    )

    root_key = "objects" if image_type != "classification" else "context"

    dataset_type_label_map = {
        "point": [("point2d_shape", "point2d")],
        "polyline": [("poly2d_shape", "poly2d")],
        "polygon": [("poly2d_shape", "poly2d")],
        "2d_bounding_box": [("bbox_shape", "bbox")],
        "semantic_segmentation": [("semantic_mask", "binary")],
        "classification": [],
    }

    additional_attributes = dataset_type_label_map.get(image_type, []) + (
        [("cuboid_shape", "cuboid")] if has_lidar_sensor else []
    )

    visionai_ontology = {}
    mapping_attribute_type = {"number": "num", "option": "vec"}

    for project_ontology_info in project_ontology_classes:
        cls_name = project_ontology_info["name"]
        ontology_attributes = project_ontology_info.get("attributes", [])
        attributes = {
            attr_name: {"type": attr_type}
            for attr_name, attr_type in additional_attributes
        }
        for _attr in ontology_attributes:

            attr_name = _attr["name"]
            attr_type = mapping_attribute_type.get(_attr["type"], _attr["type"])

            attr_value = []
            if attr_type == "vec" and _attr.get("options"):
                options = _attr.get("options")
                for opt in options:
                    attr_value.append(opt["value"])

            attributes.update(**{attr_name: {"type": attr_type, "value": attr_value}})

        visionai_ontology.update({cls_name: {"attributes": attributes}})

    # convert project sensors into visionai streams
    visionai_streams = {
        sensor["name"]: {"type": sensor["type"]} for sensor in project_sensors
    }

    visionai_taggings = {}
    for tagging in project_tagging["attributes"]:
        tagging_name = tagging["name"]
        tagging_type = mapping_attribute_type.get(tagging["type"], tagging["type"])
        tagging_options = [opt["value"] for opt in tagging["options"]]
        tagging_data = {tagging_name: {"type": tagging_type, "value": tagging_options}}
        visionai_taggings.update(tagging_data)

    ontology = {
        root_key: visionai_ontology,
        "streams": visionai_streams,
        "taggings": visionai_taggings,
    }
    ontology = Ontology(**ontology).dict()
    assert ontology == fake_visionai_ontology

    errors = VisionAIValidator.validate(
        ontology=ontology,
        visionai=fake_objects_data_single_lidar,
    )

    assert errors == []
