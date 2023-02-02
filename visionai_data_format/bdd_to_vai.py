import argparse
import json
import logging
import os

from utils.common import LOGGING_DATEFMT, LOGGING_FORMAT
from utils.converter import convert_bdd_to_vai
from utils.validator import validate_bdd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.DEBUG,
    datefmt=LOGGING_DATEFMT,
)


def bdd_to_vai(
    bdd_src_folder: str, vai_dest_folder: str, sensor_name: str, copy_image: bool
) -> None:
    try:

        label_file = os.path.join(bdd_src_folder, "labels.json")
        raw_data = json.load(open(label_file))

        bdd_data = validate_bdd(raw_data).dict()
        convert_bdd_to_vai(
            bdd_src_folder, bdd_data, vai_dest_folder, sensor_name, copy_image
        )

    except Exception as e:
        logger.error("Convert bdd to vai format failed : " + str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--src",
        type=str,
        required=True,
        help="BDD+ format source file name (i.e : bdd_dest.json)",
    )
    parser.add_argument(
        "-d",
        "--dst",
        type=str,
        required=True,
        help="VisionAI format destination folder path",
    )
    parser.add_argument(
        "--sensor", type=str, help="Sensor name, i.e : `camera1`", default="camera1"
    )
    parser.add_argument("--copy-image", action="store_true")
    args = parser.parse_args()

    bdd_to_vai(args.src, args.dst, args.sensor, args.copy_image)
