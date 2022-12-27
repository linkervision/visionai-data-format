from visionai_data_format.schemas.bdd_schema import BDDSchema
from visionai_data_format.schemas.coco_schema import Coco
from visionai_data_format.schemas.visionai_schema import VisionAI, VisionAIModel


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

    assert Coco(**input_data).dict() == generated_data


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
            "metadata": {"schema_version": "1.0.0"},
        }
    }

    assert VisionAIModel(**input_data).dict() == generated_data


def test_visionai():
    input_data = {
        "contexts": {},
        "frame_intervals": [],
        "frames": {},
        "objects": {},
        "coordinate_systems": {},
        "streams": {},
    }
    generated_data = {
        "contexts": {},
        "frame_intervals": [],
        "frames": {},
        "objects": {},
        "coordinate_systems": {},
        "streams": {},
        "metadata": {"schema_version": "1.0.0"},
    }

    assert VisionAI(**input_data).dict() == generated_data


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
