import argparse
import logging
import os

from utils.common import LOGGING_DATEFMT, LOGGING_FORMAT
from utils.converter import convert_vai_to_bdd
from utils.validator import save_as_json, validate_bdd

logger = logging.getLogger(__name__)

logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.DEBUG,
    datefmt=LOGGING_DATEFMT,
)


def vai_to_bdd(
    vai_src_folder: str,
    bdd_dest: str,
    company_code: int,
    storage_name: str,
    container_name: str,
    copy_image: bool,
) -> None:

    try:
        os.makedirs(bdd_dest, exist_ok=True)
        bdd_data = convert_vai_to_bdd(
            root_folder=vai_src_folder,
            company_code=company_code,
            storage_name=storage_name,
            container_name=container_name,
            bdd_dest=bdd_dest if copy_image else None,
        )
        bdd = validate_bdd(data=bdd_data)
        bdd_json_file = os.path.join(bdd_dest, "labels.json")
        save_as_json(bdd.dict(), file_name=bdd_json_file)

    except Exception as e:
        logger.error("Convert vai to bdd format failed : " + str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--src",
        type=str,
        required=True,
        help="Path of vision_ai dataset containing 'data' and 'annotation' subfolder, i.e : ~/vision_ai/train/",
    )

    parser.add_argument(
        "-d",
        "--dst",
        type=str,
        required=True,
        help="BDD+ format destination folder",
    )
    parser.add_argument(
        "--company-code",
        type=int,
        required=True,
        help="Company code information for BDD+",
    )
    parser.add_argument(
        "--storage-name",
        type=str,
        required=True,
        help="Storage name information for BDD+",
    )
    parser.add_argument(
        "--container-name",
        type=str,
        required=True,
        help="Container name information for BDD+",
    )

    parser.add_argument("--copy-image", action="store_true")

    args = parser.parse_args()

    vai_to_bdd(
        args.src,
        args.dst,
        args.company_code,
        args.storage_name,
        args.container_name,
        args.copy_image,
    )
