import json
import logging
import os
from typing import Optional, Union

from schemas.bdd_schema import BDDSchema
from schemas.visionai_schema import VisionAIModel

logger = logging.getLogger(__name__)


def validate_vai(data: dict) -> Union[VisionAIModel, None]:
    try:
        vai = VisionAIModel(**data)
        logger.info("[validated_vai] Validate success")
        return vai
    except Exception as e:
        logger.error("[validated_vai] Validate failed : " + str(e))
        return None


def validate_bdd(data: dict) -> Union[BDDSchema, None]:
    try:
        bdd = BDDSchema(**data)
        logger.info("[validate_bdd] Validation success")
        return bdd
    except Exception as e:
        logger.error("[validate_bdd] Validation failed : " + str(e))
        return None


def attribute_generator(
    category: str, attribute: dict, ontology_class_attrs: dict
) -> dict:

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


def save_as_json(data: dict, file_name: str, folder_name: Optional[str] = "") -> None:
    try:
        if folder_name:
            os.makedirs(folder_name, exist_ok=True)
        file = open(os.path.join(folder_name, file_name), "w")
        logger.info(
            f"[save_as_json] Save file to {os.path.join(folder_name,file_name)} started "
        )
        json.dump(data, file, indent=4)
        logger.info(
            f"[save_as_json] Save file to {os.path.join(folder_name,file_name)} success"
        )

    except Exception as e:
        logger.error("[save_as_json] Save file failed : " + str(e))
