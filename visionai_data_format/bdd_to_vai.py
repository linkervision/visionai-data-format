import argparse
import json
import logging
from collections import defaultdict

from visionai_data_format.schemas.bdd_schema import BDDSchema
from visionai_data_format.utils.converter import convert_bdd_to_vai
from visionai_data_format.utils.validator import validate_bdd

logger = logging.getLogger(__name__)


def bdd_to_vai(
    bdd_src_file: str,
    vai_dest_folder: str,
    uri_root: str,
    sensor_name: str,
    annotation_name: str = "groundtruth",
    img_extention: str = ".jpg",
) -> None:
    try:
        raw_data = json.load(open(bdd_src_file))
        bdd_data = validate_bdd(raw_data).dict()
        sequence_frames = defaultdict(list)
        for frame in bdd_data["frame_list"]:
            sequence_frames[frame["sequence"]].append(frame)
        # one bdd file might contain mutiple sequqences
        # we record the mapping here and convert one sequence at one time
        sequence_name_map = {}
        i = 0
        for sequence_key, frame_list in sequence_frames.items():
            sequence_bdd_data = BDDSchema(frame_list=frame_list).dict()
            sequence_name = f"{i:012d}"
            convert_bdd_to_vai(
                bdd_data=sequence_bdd_data,
                vai_dest_folder=vai_dest_folder,
                sensor_name=sensor_name,
                sequence_name=sequence_name,
                uri_root=uri_root,
                annotation_name=annotation_name,
                img_extention=img_extention,
            )
            sequence_name_map[sequence_key] = sequence_name
            i += 1
        # User might need the mapping for moving image files
        logger.info(f"mapping of sequence name: {sequence_name_map}")
    except Exception as e:
        logger.error("Convert bdd to vai format failed : " + str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bdd_src_file",
        type=str,
        required=True,
        help="BDD+ format source file name (i.e : bdd_dest.json)",
    )
    parser.add_argument(
        "-vai_dest_folder",
        type=str,
        required=True,
        help="VisionAI format destination folder path",
    )
    parser.add_argument(
        "-uri_root",
        type=str,
        help="uri root for storage i.e: https://azuresorate/container1",
    )
    parser.add_argument(
        "-sensor", type=str, help="Sensor name, i.e : `camera1`", default="camera1"
    )
    parser.add_argument(
        "-annotation_name",
        type=str,
        default="groundtruth",
        help=" annotation folder name (default: 'groundtruth')",
    )
    parser.add_argument(
        "-img_extention",
        type=str,
        default=".jpg",
        help="image extention (default: .jpg)",
    )

    FORMAT = "%(asctime)s[%(process)d][%(levelname)s] %(name)-16s : %(message)s"
    DATEFMT = "[%d-%m-%Y %H:%M:%S]"

    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG,
        datefmt=DATEFMT,
    )

    args = parser.parse_args()

    bdd_to_vai(
        args.bdd_src_file,
        args.vai_dest_folder,
        args.uri_root,
        args.sensor,
        args.annotation_name,
        args.img_extention,
    )
