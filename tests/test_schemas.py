import pytest

from visionai_data_format.schemas.bdd_schema import BDDSchema
from visionai_data_format.schemas.coco_schema import COCO
from visionai_data_format.schemas.visionai_schema import VisionAIModel


def test_coco():
    input_data = {
        "info": {
            "year": "",
            "version": "",
            "description": "",
            "contributor": "",
            "url": "",
            "date_created": "",
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    generated_data = {
        "info": {
            "year": "",
            "version": "",
            "description": "",
            "contributor": "",
            "url": "",
            "date_created": "",
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    assert COCO(**input_data).dict() == generated_data


def test_visionai_model():
    input_data = {"visionai": {}}
    generated_data = {
        "visionai": {
            "contexts": {},
            "frame_intervals": [],
            "frames": {},
            "objects": {},
            "coordinate_systems": {},
            "streams": {},
            "tags": {},
            "metadata": {"schema_version": "1.0.0"},
        }
    }
    with pytest.raises(Exception):
        assert VisionAIModel(**input_data).dict() == generated_data


def test_visionai(
    fake_raw_visionai_data,
    fake_generated_raw_visionai_data,
    fake_objects_visionai_data,
    fake_generated_objects_visionai_data,
):
    assert (
        VisionAIModel(**fake_raw_visionai_data).dict(exclude_unset=True)
        == fake_generated_raw_visionai_data
    )

    assert (
        VisionAIModel(**fake_objects_visionai_data).dict(exclude_unset=True)
        == fake_generated_objects_visionai_data
    )


def test_bdd():
    input_data = {"frame_list": []}
    generated_data = {
        "bdd_version": "1.1.4",
        "company_code": None,
        "inference_object": "detection",
        "meta_ds": {},
        "meta_se": {},
        "frame_list": [],
    }
    assert BDDSchema(**input_data).dict() == generated_data
