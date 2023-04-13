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
