import re

import pytest

from visionai_data_format.exceptions import VisionAIException
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

    with pytest.raises(
        VisionAIException,
        match="Contains extra stream sensors {'camera2'} with type camera or lidar",
    ):
        errors = VisionAIModel(
            **fake_objects_data_wrong_frame_properties_sensor
        ).validate_with_ontology(
            ontology=ontology,
        )
        raise errors[0]


def test_validate_bbox_wrong_streams_under_visionai(
    fake_visionai_ontology, fake_objects_data_wrong_frame_properties_sensor
):
    ontology = Ontology(**fake_visionai_ontology).dict(exclude_unset=True)

    with pytest.raises(
        VisionAIException,
        match="Contains extra stream sensors {'camera2'} with type camera or lidar",
    ):
        errors = VisionAIModel(
            **fake_objects_data_wrong_frame_properties_sensor
        ).validate_with_ontology(
            ontology=ontology,
        )
        raise errors[0]


def test_validate_bbox_wrong_class_under_visionai(
    fake_visionai_ontology, fake_objects_data_single_lidar_wrong_class
):
    ontology = Ontology(**fake_visionai_ontology).dict(exclude_unset=True)

    with pytest.raises(VisionAIException, match="Contains extra classes {'children'}"):
        errors = VisionAIModel(
            **fake_objects_data_single_lidar_wrong_class
        ).validate_with_ontology(
            ontology=ontology,
        )
        raise errors[0]


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

    assert len(errors) == 2
    with pytest.raises(
        expected_exception=VisionAIException, match="Contains extra classes {'road'}"
    ):
        raise errors[0]

    with pytest.raises(expected_exception=VisionAIException, match="Invalid key tags"):
        raise errors[1]


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

    assert len(errors) == 3

    with pytest.raises(
        expected_exception=VisionAIException,
        match="Extra frame from frame_intervals : {1, 2}",
    ):
        raise errors[0]

    message = (
        "objects 893ac389-7782-4bc3-8f61-09a8e48c819f with data pointer bbox_shape frame interval(s) error"
        + ", current interval [0,2] doesn't match with frames objects intervals [(0, 0)]"
    )
    with pytest.raises(expected_exception=VisionAIException, match=re.escape(message)):
        raise errors[1]

    message = (
        "objects 893ac389-7782-4bc3-8f61-09a8e48c819f with data pointer cuboid_shape frame interval(s) "
        + "error, current interval [0,2] doesn't match with frames objects intervals [(0, 0)]"
    )
    with pytest.raises(expected_exception=VisionAIException, match=re.escape(message)):
        raise errors[2]


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

    assert len(errors) == 2

    message = (
        "objects 893ac389-7782-4bc3-8f61-09a8e48c819f with data pointer bbox_shape frame interval(s)"
        + " error, current interval [0,2] doesn't match with frames objects intervals [(0, 0)]"
    )
    with pytest.raises(expected_exception=VisionAIException, match=re.escape(message)):
        raise errors[0]

    message = (
        "objects 893ac389-7782-4bc3-8f61-09a8e48c819f with data pointer cuboid_shape frame"
        + " interval(s) error, current interval [0,2] doesn't match with frames objects intervals [(0, 0)]"
    )
    with pytest.raises(expected_exception=VisionAIException, match=re.escape(message)):
        raise errors[1]


def test_validate_wrong_context_vector_attribute(
    fake_visionai_classification_ontology, fake_contexts_data_wrong_vector_value
):
    ontology = Ontology(**fake_visionai_classification_ontology).dict(
        exclude_unset=True
    )

    errors = VisionAIModel(
        **fake_contexts_data_wrong_vector_value
    ).validate_with_ontology(
        ontology=ontology,
    )
    assert len(errors) == 1

    message = (
        "Contains extra attributes TIMEOFDAY:vec from ontology class *tagging "
        + "attributes : {'THIS_IS_THE_WRONG_VALUE'}"
    )
    with pytest.raises(expected_exception=VisionAIException, match=re.escape(message)):
        raise errors[0]


def test_validate_wrong_context_vector_attribute_classification(
    fake_visionai_classification_ontology,
    fake_contexts_classification_wrong_vector_value,
):
    ontology = Ontology(**fake_visionai_classification_ontology).dict(
        exclude_unset=True
    )

    errors = VisionAIModel(
        **fake_contexts_classification_wrong_vector_value
    ).validate_with_ontology(
        ontology=ontology,
    )

    message = "Contains extra attributes TIMEOFDAY:vec from ontology class *tagging attributes : {'ASDFLL'}"
    with pytest.raises(expected_exception=VisionAIException, match=re.escape(message)):
        raise errors[0]
