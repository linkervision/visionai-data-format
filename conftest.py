import json
import os

import pytest

TEST_DATA_FOLDER = "tests/test_data"


@pytest.fixture(scope="session")
def fake_raw_visionai_data():
    file_name = "fake_raw_data.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_generated_raw_visionai_data():
    file_name = "generated_raw_data.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_objects_visionai_data():
    file_name = "fake_objects_data.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_generated_objects_visionai_data():
    file_name = "generated_objects_data.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_visionai_ontology():
    file_name = "fake_visionai_ontology.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_visionai_semantic_ontology():
    file_name = "fake_visionai_semantic_ontology.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_visionai_classification_ontology():
    file_name = "fake_visionai_classification_ontology.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_objects_data_single_lidar():
    file_name = "fake_objects_data_single_lidar.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_objects_data_single_lidar_wrong_class():
    file_name = "fake_objects_data_single_lidar_wrong_class.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_objects_data_wrong_frame_properties_sensor():
    file_name = "fake_objects_data_wrong_frame_properties_sensor.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_objects_semantic_segmentation_wrong_visionai_streams():
    file_name = "fake_objects_data_wrong_visionai_streams_sensor.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_objects_semantic_segmentation():
    file_name = "fake_objects_semantic_segmentation.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_objects_semantic_segmentation_without_tags():
    file_name = "fake_objects_semantic_segmentation_without_tags.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_objects_semantic_segmentation_wrong_tags_classes():
    file_name = "fake_objects_semantic_segmentation_wrong_tags_classes.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_contexts_data():
    file_name = "fake_contexts_data.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_objects_data_single_lidar_wrong_visionai_frame_intervals():
    file_name = "fake_objects_data_single_lidar_wrong_visionai_frame_intervals.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))


@pytest.fixture(scope="session")
def fake_objects_data_single_lidar_wrong_objects_frame_intervals():
    file_name = "fake_objects_data_single_lidar_wrong_objects_frame_intervals.json"
    return json.load(open(os.path.join(TEST_DATA_FOLDER, file_name)))
