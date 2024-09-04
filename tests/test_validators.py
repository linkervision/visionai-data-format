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
        match="Additional stream sensors {'camera2'} with type camera or lidar"
        + " are present that are not required by the visionai format.",
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
        match="Additional stream sensors {'camera2'} with type camera or lidar"
        + " are present that are not required by the visionai format.",
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

    with pytest.raises(
        VisionAIException,
        match="The data contains additional classes {'children'} that are not expected.",
    ):
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


def test_validate_instance_segmentation(
    fake_visionai_instance_segmentation_ontology, fake_objects_instance_segmentation
):
    ontology = Ontology(**fake_visionai_instance_segmentation_ontology).dict(
        exclude_unset=True
    )

    errors = VisionAIModel(**fake_objects_instance_segmentation).validate_with_ontology(
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
        expected_exception=VisionAIException,
        match="The data contains additional classes {'road'} that are not expected.",
    ):
        raise errors[0]

    with pytest.raises(
        expected_exception=VisionAIException,
        match="The key tags is invalid or not recognized.",
    ):
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
        match="An extra frame was detected beyond the defined frame intervals: {1, 2}.",
    ):
        raise errors[0]

    message = (
        "For objects 893ac389-7782-4bc3-8f61-09a8e48c819f with data pointer bbox_shape, "
        + "the current interval [0,2] does not match with frames objects intervals [(0, 0)]."
    )
    with pytest.raises(expected_exception=VisionAIException, match=re.escape(message)):
        raise errors[1]

    message = (
        "For objects 893ac389-7782-4bc3-8f61-09a8e48c819f with data pointer cuboid_shape,"
        + " the current interval [0,2] does not match with frames objects intervals [(0, 0)]."
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
        "For objects 893ac389-7782-4bc3-8f61-09a8e48c819f with data pointer bbox_shape,"
        + " the current interval [0,2] does not match with frames objects intervals [(0, 0)]."
    )
    with pytest.raises(expected_exception=VisionAIException, match=re.escape(message)):
        raise errors[0]

    message = (
        "For objects 893ac389-7782-4bc3-8f61-09a8e48c819f with data pointer cuboid_shape,"
        + " the current interval [0,2] does not match with frames objects intervals [(0, 0)]."
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
        "Extra attributes timeofday:vec:{'this_is_the_wrong_value1'} "
        + "are present that are not defined in the ontology class *tagging."
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

    message = (
        "Extra attributes timeofday:vec:{'Night'} "
        + "are present that are not defined in the ontology class *tagging."
    )
    with pytest.raises(expected_exception=VisionAIException, match=re.escape(message)):
        raise errors[0]


def test_validate_single_lidar_without_camera_intrinsics_pinhole(
    fake_visionai_ontology,
    fake_objects_data_single_lidar_without_camera_intrinsics_pinhole,
):
    ontology = Ontology(**fake_visionai_ontology).dict(exclude_unset=True)

    with pytest.raises(
        VisionAIException,
        match="Missing field intrinsics_pinhole with value at camera stream_properties "
        + "when sensors contain at least one lidar",
    ):
        errors = VisionAIModel(
            **fake_objects_data_single_lidar_without_camera_intrinsics_pinhole
        ).validate_with_ontology(
            ontology=ontology,
        )
        raise errors[0]
