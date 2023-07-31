import pytest

from visionai_data_format.schemas.ontology import Ontology
from visionai_data_format.schemas.visionai_schema import VisionAIModel


def test_validate_bbox(fake_visionai_ontology, fake_objects_data_single_lidar):
    ontology = Ontology(**fake_visionai_ontology).dict(exclude_unset=True)

    errors = VisionAIModel(**fake_objects_data_single_lidar).validate_with_ontology(
        ontology=ontology,
    )

    assert errors == []


def test_validate_bbox_wrong_frame_properties_sensor_name(
    fake_visionai_ontology, fake_objects_data_wrong_frame_properties_sensor
):
    ontology = Ontology(**fake_visionai_ontology).dict(exclude_unset=True)

    errors = VisionAIModel(
        **fake_objects_data_wrong_frame_properties_sensor
    ).validate_with_ontology(
        ontology=ontology,
    )

    assert errors == ["Frame properties contains extra sensor(s) : {'camera2'}"]


def test_validate_bbox_wrong_streams_under_visionai(
    fake_visionai_ontology, fake_objects_data_wrong_frame_properties_sensor
):
    ontology = Ontology(**fake_visionai_ontology).dict(exclude_unset=True)

    errors = VisionAIModel(
        **fake_objects_data_wrong_frame_properties_sensor
    ).validate_with_ontology(
        ontology=ontology,
    )

    assert errors == ["Frame properties contains extra sensor(s) : {'camera2'}"]


def test_validate_bbox_wrong_class_under_visionai(
    fake_visionai_ontology, fake_objects_data_single_lidar_wrong_class
):
    ontology = Ontology(**fake_visionai_ontology).dict(exclude_unset=True)

    errors = VisionAIModel(
        **fake_objects_data_single_lidar_wrong_class
    ).validate_with_ontology(
        ontology=ontology,
    )

    assert errors == ["Attribute objects with classes {'children'} doesn't accepted"]


def test_validate_semantic_segmentation(
    fake_visionai_semantic_ontology, fake_objects_semantic_segmentation
):
    ontology = Ontology(**fake_visionai_semantic_ontology).dict(exclude_unset=True)

    errors = VisionAIModel(**fake_objects_semantic_segmentation).validate_with_ontology(
        ontology=ontology,
    )

    assert errors == []


def test_validate_semantic_segmentation_visionai_without_tags(
    fake_visionai_semantic_ontology, fake_objects_semantic_segmentation_without_tags
):
    ontology = Ontology(**fake_visionai_semantic_ontology).dict(exclude_unset=True)
    with pytest.raises(Exception):
        errors = VisionAIModel(
            **fake_objects_semantic_segmentation_without_tags
        ).validate_with_ontology(
            ontology=ontology,
        )
        assert errors == []


def test_validate_semantic_segmentation_visionai_wrong_tags_classes(
    fake_visionai_semantic_ontology,
    fake_objects_semantic_segmentation_wrong_tags_classes,
):
    ontology = Ontology(**fake_visionai_semantic_ontology).dict(exclude_unset=True)
    errors = VisionAIModel(
        **fake_objects_semantic_segmentation_wrong_tags_classes
    ).validate_with_ontology(
        ontology=ontology,
    )
    assert errors == ["Tag label with classes {'road'} doesn't accepted"]


def test_validate_classification(
    fake_visionai_classification_ontology, fake_contexts_data
):
    ontology = Ontology(**fake_visionai_classification_ontology).dict(
        exclude_unset=True
    )
    errors = VisionAIModel(**fake_contexts_data).validate_with_ontology(
        ontology=ontology,
    )
    assert errors == []


def test_validate_wrong_visionai_frame_intervals(
    fake_visionai_ontology,
    fake_objects_data_single_lidar_wrong_visionai_frame_intervals,
):
    ontology = Ontology(**fake_visionai_ontology).dict(exclude_unset=True)

    errors = VisionAIModel(
        **fake_objects_data_single_lidar_wrong_visionai_frame_intervals
    ).validate_with_ontology(
        ontology=ontology,
    )

    assert errors == [
        "Extra frames from `frame_intervals` : {1, 2}\n",
        "validate objects error: objects UUID 893ac389-7782-4bc3-8f61-09a8e48c819f with data pointer"
        + " bbox_shape frame interval(s) error, current data pointer interval (0, 2) "
        + "doesn't match with objects interval [(0, 0)]",
    ]


def test_validate_wrong_object_frame_intervals(
    fake_visionai_ontology,
    fake_objects_data_single_lidar_wrong_objects_frame_intervals,
):
    ontology = Ontology(**fake_visionai_ontology).dict(exclude_unset=True)

    errors = VisionAIModel(
        **fake_objects_data_single_lidar_wrong_objects_frame_intervals
    ).validate_with_ontology(
        ontology=ontology,
    )

    assert errors == [
        "validate objects error: objects UUID 893ac389-7782-4bc3-8f61-09a8e48c819f with data pointer"
        + " bbox_shape frame interval(s) error, current data pointer interval (0, 2) "
        + "doesn't match with objects interval [(0, 0)]"
    ]
