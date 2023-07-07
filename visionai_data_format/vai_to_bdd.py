import argparse
import logging

from visionai_data_format.utils.converter import convert_vai_to_bdd
from visionai_data_format.utils.validator import save_as_json, validate_bdd

logger = logging.getLogger(__name__)


def vai_to_bdd(
    vai_src_folder: str,
    bdd_dest_file: str,
    company_code: int,
    storage_name: str,
    container_name: str,
    annotation_name:str
) -> None:
    try:
        bdd_data = convert_vai_to_bdd(
            folder_name=vai_src_folder,
            company_code=company_code,
            storage_name=storage_name,
            container_name=container_name,
            annotation_name=annotation_name
        )
        bdd = validate_bdd(data=bdd_data)
        save_as_json(bdd.dict(), file_name=bdd_dest_file)
    except Exception as e:
        logger.error("Convert vai to bdd format failed : " + str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-vai_src_folder",
        type=str,
        required=True,
        help="VisionAI format source folder path",
    )
    parser.add_argument(
        "-bdd_dest_file",
        type=str,
        required=True,
        help="BDD+ format destination file name (i.e : bdd_dest.json)",
    )
    parser.add_argument(
        "-company_code",
        type=int,
        required=True,
        help="Company code information for BDD+",
    )
    parser.add_argument(
        "-storage_name",
        type=str,
        required=True,
        help="Storage name information for BDD+",
    )
    parser.add_argument(
        "-container_name",
        type=str,
        required=True,
        help="Container name information for BDD+",
    )
    parser.add_argument(
        "-annotation_name",
        type=str,
        required=True,
        default="groundtruth",
        help="annotation folder name in VAI",
    )

    FORMAT = "%(asctime)s[%(process)d][%(levelname)s] %(name)-16s : %(message)s"
    DATEFMT = "[%d-%m-%Y %H:%M:%S]"

    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG,
        datefmt=DATEFMT,
    )

    args = parser.parse_args()

    vai_to_bdd(
        args.vai_src_folder,
        args.bdd_dest_file,
        args.company_code,
        args.storage_name,
        args.container_name,
        args.annotation_name,
    )
