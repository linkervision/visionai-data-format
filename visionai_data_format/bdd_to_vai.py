import argparse
import json
import logging

from utils.converter import convert_bdd_to_vai
from utils.validator import validate_bdd

logger = logging.getLogger(__name__)


def bdd_to_vai(bdd_src_file: str, vai_dest_folder: str, sensor_name: str) -> None:
    try:
        raw_data = json.load(open(bdd_src_file))
        bdd_data = validate_bdd(raw_data).dict()
        convert_bdd_to_vai(bdd_data, vai_dest_folder, sensor_name)
    except Exception as e:
        logger.error("Convert bdd to vai format failed : " + str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-vai_dest_folder",
        type=str,
        required=True,
        help="VisionAI format destination folder path",
    )
    parser.add_argument(
        "-bdd_src_file",
        type=str,
        required=True,
        help="BDD+ format source file name (i.e : bdd_dest.json)",
    )
    parser.add_argument(
        "--sensor", type=str, help="Sensor name, i.e : `camera1`", default="camera1"
    )

    FORMAT = "%(asctime)s[%(process)d][%(levelname)s] %(name)-16s : %(message)s"
    DATEFMT = "[%d-%m-%Y %H:%M:%S]"

    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG,
        datefmt=DATEFMT,
    )

    args = parser.parse_args()

    bdd_to_vai(args.bdd_src_file, args.vai_dest_folder, args.sensor)
