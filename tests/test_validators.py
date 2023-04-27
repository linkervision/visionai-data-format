from visionai_data_format.schemas.ontology import Ontology
from visionai_data_format.schemas.visionai_schema import VisionAIModel


def test_ontology(fake_visionai_ontology, fake_objects_data_single_lidar):

    ontology = Ontology(**fake_visionai_ontology).dict(exclude_unset=True)

    errors = VisionAIModel(**fake_objects_data_single_lidar).validate_with_ontology(
        ontology=ontology,
    )

    assert errors == []
