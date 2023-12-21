import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

from ..schemas.common import (
    AnnotationFormat,
    DatasetType,
    OntologyImageType,
    OntologyPcdType,
)

logger = logging.getLogger(__name__)


class BaseFormatChecker(ABC):
    def __init__(
        self,
        dataset_type: DatasetType,
        sequential: bool,
        has_attribute: bool,
        camera_sensors: Optional[list[str]] = None,
        lidar_sensors: Optional[list[str]] = None,
        image_type: Optional[OntologyImageType] = None,
        pcd_type: Optional[OntologyPcdType] = None,
    ) -> None:
        """
        Can be used to check if current combination of data is valid or not step by step.
        Just instantiate the checker and get the result by checker.valid

        Please follow the decision tree for checkers:
        https://drive.google.com/file/d/13limq9gQnmpy2bJoe6fGTDboOuIKWjrL/view?usp=sharing

        Parameters
        ----------
        dataset_type : DatasetType
        sequential : bool
            Current data is sequential or not
        has_attribute : bool
            Current data has attribute or not.
        camera_sensors : Optional[list[str]], optional
            The camera-type sensor names of current data.
        lidar_sensors : Optional[list[str]], optional
            The lidar-type sensor names of current data.
        image_type : Optional[OntologyImageType], optional
            The image type of current data ontology.
        pcd_type : Optional[OntologyPcdType], optional
            The pcd type of current data ontology.
        """
        self.data = {
            "dataset_type": dataset_type,
            "sequential": sequential,
            "has_attribute": has_attribute,
            "camera_sensors": camera_sensors if camera_sensors else [],
            "lidar_sensors": lidar_sensors if lidar_sensors else [],
            "image_type": image_type,
            "pcd_type": pcd_type,
        }

    @property
    def valid(self) -> bool:
        try:
            # If any validation functions return False then it's not valid.
            # IMPORTANT: The order matters.
            for index, valid_func in enumerate(
                (
                    self.verify_dataset_type,
                    self.verify_sequence,
                    self.verify_sensors,
                    self.verify_ontology_shapes,
                    self.verify_attribute,
                ),
                1,
            ):
                if not valid_func(**self.data):
                    logger.warning(
                        f"Failed to pass the No.{index} validation function ({valid_func.__name__})."
                    )
                    return False
        except Exception as e:
            logger.error(f"Unexpected exception raised: {e}")
            return False
        return True

    @staticmethod
    @abstractmethod
    def verify_dataset_type(**kwargs) -> bool:
        return NotImplemented

    @staticmethod
    @abstractmethod
    def verify_sequence(**kwargs) -> bool:
        return NotImplemented

    @staticmethod
    @abstractmethod
    def verify_sensors(**kwargs) -> bool:
        return NotImplemented

    @staticmethod
    @abstractmethod
    def verify_ontology_shapes(**kwargs) -> bool:
        return NotImplemented

    @staticmethod
    @abstractmethod
    def verify_attribute(**kwargs) -> bool:
        return NotImplemented


class KittiFormatChecker(BaseFormatChecker):
    @staticmethod
    def verify_dataset_type(**kwargs) -> bool:
        dataset_type: DatasetType = kwargs["dataset_type"]
        if dataset_type in {DatasetType.RAW_DATA, DatasetType.ANNOTATED_DATA}:
            return True
        return False

    @staticmethod
    def verify_sequence(**kwargs) -> bool:
        sequential: bool = kwargs["sequential"]
        if not sequential:
            return True
        return False

    @staticmethod
    def verify_sensors(**kwargs) -> bool:
        camera_sensor_count: int = len(kwargs["camera_sensors"])
        lidar_sensor_count: int = len(kwargs["lidar_sensors"])

        if camera_sensor_count == 1 and lidar_sensor_count == 1:
            return True
        elif camera_sensor_count == 0 and lidar_sensor_count == 1:
            return True
        elif camera_sensor_count == 1 and lidar_sensor_count == 0:
            return True
        return False

    @staticmethod
    def verify_ontology_shapes(**kwargs) -> bool:
        dataset_type: DatasetType = kwargs["dataset_type"]
        camera_sensor_count: int = len(kwargs["camera_sensors"])
        lidar_sensor_count: int = len(kwargs["lidar_sensors"])
        image_type: Optional[OntologyImageType] = kwargs["image_type"]
        pcd_type: Optional[OntologyPcdType] = kwargs["pcd_type"]
        if dataset_type == DatasetType.RAW_DATA:
            return True
        elif dataset_type == DatasetType.ANNOTATED_DATA:
            if (
                camera_sensor_count == 1
                and lidar_sensor_count == 1
                and image_type == OntologyImageType._2D_BOUNDING_BOX
                and pcd_type == OntologyPcdType.CUBOID
            ):
                return True
            elif (
                camera_sensor_count == 0
                and lidar_sensor_count == 1
                and pcd_type == OntologyPcdType.CUBOID
            ):
                return True
            elif (
                camera_sensor_count == 1
                and lidar_sensor_count == 0
                and image_type == OntologyImageType._2D_BOUNDING_BOX
            ):
                return True
        return False

    @staticmethod
    def verify_attribute(**kwargs) -> bool:
        dataset_type: DatasetType = kwargs["dataset_type"]
        has_attribute: bool = kwargs["has_attributes"]
        if dataset_type == DatasetType.RAW_DATA:
            return True
        elif dataset_type == DatasetType.ANNOTATED_DATA:
            if not has_attribute:
                return True
        return False


class CocoFormatChecker(BaseFormatChecker):
    @staticmethod
    def verify_dataset_type(**kwargs) -> bool:
        dataset_type: DatasetType = kwargs["dataset_type"]
        if dataset_type == DatasetType.ANNOTATED_DATA:
            return True
        return False

    @staticmethod
    def verify_sequence(**kwargs) -> bool:
        sequential: bool = kwargs["sequential"]
        if not sequential:
            return True
        return False

    @staticmethod
    def verify_sensors(**kwargs) -> bool:
        camera_sensor_count: int = len(kwargs["camera_sensors"])
        lidar_sensor_count: int = len(kwargs["lidar_sensors"])
        if camera_sensor_count == 1 and lidar_sensor_count == 0:
            return True
        return False

    @staticmethod
    def verify_ontology_shapes(**kwargs) -> bool:
        image_type: Optional[OntologyImageType] = kwargs["image_type"]
        if image_type == OntologyImageType._2D_BOUNDING_BOX:
            return True
        return False

    @staticmethod
    def verify_attribute(**kwargs) -> bool:
        has_attribute: bool = kwargs["has_attributes"]
        if not has_attribute:
            return True
        return False


class ImageFormatChecker(BaseFormatChecker):
    @staticmethod
    def verify_dataset_type(**kwargs) -> bool:
        dataset_type: DatasetType = kwargs["dataset_type"]
        if dataset_type == DatasetType.RAW_DATA:
            return True
        return False

    @staticmethod
    def verify_sequence(**kwargs) -> bool:
        sequential: bool = kwargs["sequential"]
        if not sequential:
            return True
        return False

    @staticmethod
    def verify_sensors(**kwargs) -> bool:
        camera_sensor_count: int = len(kwargs["camera_sensors"])
        lidar_sensor_count: int = len(kwargs["lidar_sensors"])
        if camera_sensor_count == 1 and lidar_sensor_count == 0:
            return True
        return False

    @staticmethod
    def verify_ontology_shapes(**kwargs) -> bool:
        return True

    @staticmethod
    def verify_attribute(**kwargs) -> bool:
        return True


class BddPlusFormatChecker(BaseFormatChecker):
    @staticmethod
    def verify_dataset_type(**kwargs) -> bool:
        dataset_type: DatasetType = kwargs["dataset_type"]
        if dataset_type == DatasetType.ANNOTATED_DATA:
            return True
        return False

    @staticmethod
    def verify_sequence(**kwargs) -> bool:
        sequential: bool = kwargs["sequential"]
        if not sequential:
            return True
        return False

    @staticmethod
    def verify_sensors(**kwargs) -> bool:
        camera_sensor_count: int = len(kwargs["camera_sensors"])
        lidar_sensor_count: int = len(kwargs["lidar_sensors"])

        if camera_sensor_count == 1 and lidar_sensor_count == 0:
            return True
        return False

    @staticmethod
    def verify_ontology_shapes(**kwargs) -> bool:
        image_type: Optional[OntologyImageType] = kwargs["image_type"]
        if image_type in {
            OntologyImageType._2D_BOUNDING_BOX,
            OntologyImageType.POLYGON,
            OntologyImageType.POLYLINE,
            OntologyImageType.POINT,
        }:
            return True
        return False

    @staticmethod
    def verify_attribute(**kwargs) -> bool:
        return True


class VisionAIFormatChecker(BaseFormatChecker):
    @staticmethod
    def verify_dataset_type(**kwargs) -> bool:
        return True

    @staticmethod
    def verify_sequence(**kwargs) -> bool:
        return True

    @staticmethod
    def verify_sensors(**kwargs) -> bool:
        return True

    @staticmethod
    def verify_ontology_shapes(**kwargs) -> bool:
        return True

    @staticmethod
    def verify_attribute(**kwargs) -> bool:
        return True


def get_format_checker(
    annotation_format: Union[str, AnnotationFormat]
) -> type[BaseFormatChecker]:
    if annotation_format not in AnnotationFormat:
        raise ValueError(f"The {annotation_format} format is not supported.")

    map_ = {
        AnnotationFormat.KITTI: KittiFormatChecker,
        AnnotationFormat.COCO: CocoFormatChecker,
        AnnotationFormat.IMAGE: ImageFormatChecker,
        AnnotationFormat.BDDP: BddPlusFormatChecker,
        AnnotationFormat.VISION_AI: VisionAIFormatChecker,
    }

    try:
        return map_[annotation_format]
    except KeyError:
        logger.error("The map of format checker is incorrect.")
        raise
