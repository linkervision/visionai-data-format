# visionai-data-format

`VisionAI` format is [Dataverse](https://www.linkervision.com/visionai-platform-dataverse) standardized annotation format to label objects and sequences in the context of Autonomous Driving System(ADS). `VisionAI` provides consistent and effective driving environment description and categorization in the real-world case.

This tool provides validator of `VisionAI` format schema. Currently, the library supports:
  - Validate created `VisionAI` data format
  - Validate `VisionAI` data attributes with given `Ontology` information.

For a more in-depth understanding of the VisionAI format, please visit this URL: https://linkervision.gitbook.io/dataverse/visionai-format/visionai-data-format.


[Package (PyPi)](https://pypi.org/project/visionai-data-format/)    |   [Source code](https://github.com/linkernetworks/visionai-data-format)


## Getting started

(WIP)

### Install the package

```
pip3 install visionai-data-format
```

**Prerequisites**: You must have [Python 3.7](https://www.python.org/downloads/) and above to use this package.


## Example
The following sections provide examples for the following:

* [Validate VisionAI schema](###validate-visionai-schema)
* [Validate VisionAI data with given Ontology](###validate-visionai-data-with-given-ontology)

### Validate VisionAI schema

#### Example

To validate `VisionAI` data structure, could follow the example below:

```Python
from visionai_data_format.schemas.visionai_schema import VisionAIModel

# your custom visionai data
custom_visionai_data = {
    "visionai": {
        "frame_intervals": [
            {
                "frame_start": 0,
                "frame_end": 0
            }
        ],
        "frames": {
            "000000000000": {
                "objects": {
                    "893ac389-7782-4bc3-8f61-09a8e48c819f": {
                        "object_data": {
                            "bbox": [
                                {
                                    "name": "bbox_shape",
                                    "stream":"camera1",
                                    "val": [761.565,225.46,98.33000000000004, 164.92000000000002]
                                }
                            ],
                            "cuboid": [
                                {
                                    "name": "cuboid_shape",
                                    "stream": "lidar1",
                                    "val": [
                                        8.727633224700037,-1.8557590122690717,-0.6544039394148177, 0.0,
                                        0.0,-1.5807963267948966,1.2,0.48,1.89
                                    ]
                                }
                            ]
                        }
                    }
                },
                "frame_properties": {
                    "streams": {
                        "camera1": {
                            "uri": "https://helenmlopsstorageqatest.blob.core.windows.net/vainewformat/kitti/kitti_small/data/000000000000/data/camera1/000000000000.png"
                        },
                        "lidar1": {
                            "uri": "https://helenmlopsstorageqatest.blob.core.windows.net/vainewformat/kitti/kitti_small/data/000000000000/data/lidar1/000000000000.pcd"
                        }
                    }
                }
            }
        },
        "objects": {
            "893ac389-7782-4bc3-8f61-09a8e48c819f": {
                "frame_intervals": [
                    {
                        "frame_start": 0,
                        "frame_end": 0
                    }
                ],
                "name": "pedestrian",
                "object_data_pointers": {
                    "bbox_shape": {
                        "frame_intervals": [
                            {
                                "frame_start": 0,
                                "frame_end": 0
                            }
                        ],
                        "type": "bbox"
                    },
                    "cuboid_shape": {
                        "frame_intervals": [
                            {
                                "frame_start": 0,
                                "frame_end": 0
                            }
                        ],
                        "type": "cuboid"
                    }
                },
                "type": "pedestrian"
            }
        },
        "coordinate_systems": {
            "lidar1": {
                "type": "sensor_cs",
                "parent": "",
                "children": [
                    "camera1"
                ]
            },
            "camera1": {
                "type": "sensor_cs",
                "parent": "lidar1",
                "children": [],
                "pose_wrt_parent": {
                    "matrix4x4": [
                        -0.00159609942076306,
                        -0.005270645688933059,
                        0.999984790046273,
                        0.3321936949138632,
                        -0.9999162467477257,
                        0.012848695454066989,
                        -0.0015282672486530082,
                        -0.022106263278130818,
                        -0.012840436309973332,
                        -0.9999035522454274,
                        -0.0052907123281999745,
                        -0.06171977032225582,
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    ]
                }
            }
        },
        "streams": {
            "camera1": {
                "type": "camera",
                "uri": "https://helenmlopsstorageqatest.blob.core.windows.net/vainewformat/kitti/kitti_small/data/000000000000/data/camera1/000000000000.png",
                "description": "Frontal camera",
                "stream_properties": {
                    "intrinsics_pinhole": {
                        "camera_matrix_3x4": [
                            -1.1285209781809271,
                            -706.9900823216068,
                            -181.46849639413674,
                            0.2499212908887926,
                            -3.726606344908137,
                            9.084661126711246,
                            -1.8645282480709864,
                            -0.31027342289053916,
                            707.0385458128643,
                            -1.0805602883730354,
                            603.7910589125847,
                            45.42556655376811
                        ],
                        "height_px": 370,
                        "width_px": 1224
                    }
                }
            },
            "lidar1": {
                "type": "lidar",
                "uri": "https://helenmlopsstorageqatest.blob.core.windows.net/vainewformat/kitti/kitti_small/data/000000000000/data/lidar1/000000000000.pcd",
                "description": "Central lidar"
            }
        },
        "metadata": {
            "schema_version": "1.0.0"
        }
    }
}

# validate custom data
# If the data structure doesn't meets the VisionAI requirements, it would raise BaseModel error message
# otherwise, it will returns dictionary of validated visionai data
validated_visionai = VisionAIModel(**custom_visionai_data).dict()

```

#### Explanation
To begin, we define our custom `VisionAI` data. Subsequently, we employ the `VisionAI(**custom_visionai_data).dict()` to ensure the conformity of our custom data with the `VisionAI` schema. If there are any missing required fields or if the value types deviate from the defined data types, an error will be raised (prompting a list of `VisionAIException` exceptions). On the other hand, if the data passes validation, the function will yield a dictionary containing the validated `VisionAI` data.

### Validate VisionAI data with given Ontology

#### Ontology Schema
Before uploading a dataset to the `Dataverse` platform, it's advisable to validate VisionAI annotations using the `Ontology` schema. The `Ontology` schema serves as a predefined structure for Project `Ontology` data in `Dataverse`."

1. `contexts` :
    fill this section only if the project ontology is of the `classification` type.
2. `objects` : fill this section for project ontologies other than `classification`, such as `bounding_box` or `semantic_segmentation`.
3. `streams` :
    this section is mandatory as it contains project sensor-related information.
4. `tags` :
    complete this section for `semantic_segmentation` project ontologies.

#### Example

Here is an example of the `Ontology` Schema and how to validate `VisionAI` data using it:

```Python

from visionai_data_format.schemas.ontology import Ontology

custom_ontology = {
    "objects": {
        "pedestrian": {
            "attributes": {
                "bbox_shape": {
                    "type": "bbox",
                    "value": None
                },
                "cuboid_shape": {
                    "type": "cuboid",
                    "value": None
                },
                "activity": {
                    "type": "text",
                    "value": []
                }
            }
        },
        "truck": {
            "attributes": {
                "bbox_shape": {
                    "type": "bbox",
                    "value": None
                },
                "cuboid_shape": {
                    "type": "cuboid",
                    "value": None
                },
                "color": {
                    "type": "text",
                    "value": []
                },
                "new": {
                    "type": "boolean",
                    "value": []
                },
                "year": {
                    "type": "num",
                    "value": []
                },
                "status": {
                    "type": "vec",
                    "value": [
                        "stop",
                        "run",
                        "small",
                        "large"
                    ]
                }
            }
        },
        "car": {
            "attributes": {
                "bbox_shape": {
                    "type": "bbox",
                    "value": None
                },
                "cuboid_shape": {
                    "type": "cuboid",
                    "value": None
                },
                "color": {
                    "type": "text",
                    "value": []
                },
                "new": {
                    "type": "boolean",
                    "value": []
                },
                "year": {
                    "type": "num",
                    "value": []
                },
                "status": {
                    "type": "vec",
                    "value": [
                        "stop",
                        "run",
                        "small",
                        "large"
                    ]
                }
            }
        },
        "cyclist": {
            "attributes": {
                "bbox_shape": {
                    "type": "bbox",
                    "value": None
                },
                "cuboid_shape": {
                    "type": "cuboid",
                    "value": None
                }
            }
        },
        "dontcare": {
            "attributes": {
                "bbox_shape": {
                    "type": "bbox",
                    "value": None
                },
                "cuboid_shape": {
                    "type": "cuboid",
                    "value": None
                }
            }
        },
        "misc": {
            "attributes": {
                "bbox_shape": {
                    "type": "bbox",
                    "value": None
                },
                "cuboid_shape": {
                    "type": "cuboid",
                    "value": None
                },
                "color": {
                    "type": "text",
                    "value": []
                },
                "info": {
                    "type": "vec",
                    "value": [
                        "toyota",
                        "new"
                    ]
                }
            }
        },
        "van": {
            "attributes": {
                "bbox_shape": {
                    "type": "bbox",
                    "value": None
                },
                "cuboid_shape": {
                    "type": "cuboid",
                    "value": None
                }
            }
        },
        "tram": {
            "attributes": {
                "bbox_shape": {
                    "type": "bbox",
                    "value": None
                },
                "cuboid_shape": {
                    "type": "cuboid",
                    "value": None
                }
            }
        },
        "person_sitting": {
            "attributes": {
                "bbox_shape": {
                    "type": "bbox",
                    "value": None
                },
                "cuboid_shape": {
                    "type": "cuboid",
                    "value": None
                }
            }
        }
    },
    "contexts":{
        "*tagging": {
            "attributes":{
                "profession": {
                    "type": "text",
                    "value": []
                },
                "roadname": {
                    "type": "text",
                    "value": []
                },
                "name": {
                    "type": "text",
                    "value": []
                },
                "unknown_object": {
                    "type": "vec",
                    "value": [
                        "sky",
                        "leaves",
                        "wheel_vehicle",
                        "fire",
                        "water"
                    ]
                },
                "static_status": {
                    "type": "boolean",
                    "value": [
                        "true",
                        "false"
                    ]
                },
                "year": {
                    "type": "num",
                    "value": []
                },
                "weather": {
                    "type": "text",
                    "value": []
                }
            }
        }
    },
    "streams": {
        "camera1": {
            "type": "camera"
        },
        "lidar1": {
            "type": "lidar"
        }
    },
    "tags": None
}

# Validate your custom ontology
validated_ontology = Ontology(**custom_ontology).dict()

# Validate VisionAI data with our ontology, custom_visionai_data is the custom data from upper example
errors = VisionAIModel(**custom_visionai_data).validate_with_ontology(ontology=validated_ontology)

# Shows the errors
# If there is any error occurred, it will returns list of `VisionAIException` exceptions
# Otherwise, it will return empty list
# example of errors :
# >[visionai_data_format.exceptions.visionai.VisionAIException("frame stream sensors {'lidar2'} doesn't match with visionai streams sensor {'camera1', 'lidar1'}.")]
print(errors)

```
#### Explanation
Begin by creating a new `Ontology` that includes the project ontology. Subsequently, use the `validate_with_ontology(ontology=validated_ontology)` function to check if the current `VisionAI` data aligns with the information in the `Ontology`. The function will return a list of `VisionAIException` if any issues are detected; otherwise, it returns an empty list.

## Converter tools

### Convert `BDD+` format data to `VisionAI` format
(Only support box2D and camera sensor data only for now)

```
python3 visionai_data_format/convert_dataset.py -input_format bddp -output_format vision_ai -image_annotation_type 2d_bounding_box -input_annotation_path ./bdd_test.json -source_data_root ./data_root -output_dest_folder ~/visionai_output_dir -uri_root http://storage_test -n_frame 5 -sequence_idx_start 0 -camera_sensor_name camera1 -annotation_name groundtruth -img_extension .jpg --copy_sensor_data
```

Arguments :
- `-input_format`  : input format (use bddp for BDD+)
- `-output_format`  : output format (vision_ai)
- `-image_annotation_type`  : label annotation type for image (`2d_bounding_box` for box2D)
- `-input_annotation_path`  : source annotation path (BDD+ format json file)
- `-source_data_root`  : source data root for sensor data and calibration data (will find and copy image from this root)
- `-output_dest_folder` : output root folder (VisionAI local root folder)
- `-uri_root` : uri root for target upload VAI storage i.e: https://azuresorate/vai_dataset
- `-n_frame`  : number of frame to be converted (-1 means all), by default -1
- `-sequence_idx_start `  : sequence start id, by default 0
- `-camera_sensor_name`  : camera sensor name (default: "", specified it if need to convert camera data)
- `-lidar_sensor_name`  : lidar sensor name (default: "", specified it if need to convert lidar data)
- `-annotation_name` : annotation folder name (default: "groundtruth")
- `-img_extension` : image file extension (default: ".jpg")
- `--copy_sensor_data` :enable to copy image/lidar data




### Convert `VisionAI` format data to `BDD+` format
(Only support box2D for now)
The script below could help convert `VisionAI` annotation data to `BDD+` json file

```
python3 visionai_data_format/vai_to_bdd.py -vai_src_folder /path_for_visionai_root_folder -bdd_dest_file /dest_path/bdd.json -company_code 99 -storage_name storage1 -container_name dataset1 -annotation_name groundtruth
```

Arguments :
- `-vai_src_folder` : VAI root folder contains VAI format json file
- `-bdd_dest_file`  : BDD+ format file save destination
- `-company_code`  : company code
- `-storage_name`  : storage name
- `-container_name`  : container name (dataset name)
- `-annotation_name` : annotation folder name (default: "groundtruth")



### Convert `Kitti` format data to `VisionAI` format

```(Only support KITTI with one camera and one lidar sensor)```

Important:
- image type is not restricted, could be ".jpg" or ".png", but we will convert it into ".jpg" in `VisionAI` format
- only support for `P2` projection matrix calibration information

Currently,only support `KITTI` dataset with structure folder :
```bash
.kitti_folder
├── calib
│   ├── 000000.txt
│   ├── 000001.txt
│   ├── 000002.txt
│   ├── 000003.txt
│   └── 000004.txt
├── data
│   ├── 000000.png
│   ├── 000001.png
│   ├── 000002.png
│   ├── 000003.png
│   └── 000004.png
├── labels
│   ├── 000000.txt
│   ├── 000001.txt
│   ├── 000002.txt
│   ├── 000003.txt
│   └── 000004.txt
└── pcd
    ├── 000000.pcd
    ├── 000001.pcd
    ├── 000002.pcd
    ├── 000003.pcd
    └── 000004.pcd
```

Command :

```
python3 visionai_data_format/convert_dataset.py -input_format kitti -output_format vision_ai -image_annotation_type 2d_bounding_box -source_data_root ./data_root -output_dest_folder ~/visionai_output_dir -uri_root http://storage_test -n_frame 5 -sequence_idx_start 0 -camera_sensor_name camera1 -lidar_sensor_name lidar1 -annotation_name groundtruth -img_extension .jpg --copy_sensor_data
```

Arguments :
- `-input_format`  : input format (use kitti for KITTI)
- `-output_format`  : output format (vision_ai)
- `-image_annotation_type`  : label annotation type for image (2d_bounding_box for box2D)
- `-source_data_root`  : source data root for sensor data and calibration data (will find and copy image from this root)
- `-output_dest_folder` : output root folder (VisionAI local root folder)
- `-uri_root` : uri root for target upload VAI storage i.e: https://azuresorate/vai_dataset
- `-n_frame`  : number of frame to be converted (-1 means all), by default -1
- `-sequence_idx_start `  : sequence start id, by default 0
- `-camera_sensor_name`  : camera sensor name (default: "", specified it if need to convert camera data)
- `-lidar_sensor_name`  : lidar sensor name (default: "", specified it if need to convert lidar data)
- `-annotation_name` : annotation folder name (default: "groundtruth")
- `-img_extension` : image file extension (default: ".jpg")
- `--copy_sensor_data` : enable to copy image/lidar data


### Convert `COCO` format data to `VisionAI` format

```
python3 visionai_data_format/convert_dataset.py -input_format coco -output_format vision_ai -image_annotation_type 2d_bounding_box -input_annotation_path ./coco_instance.json -source_data_root ./coco_images/ -output_dest_folder ~/visionai_output_dir -uri_root http://storage_test -n_frame 5 -sequence_idx_start 0 -camera_sensor_name camera1 -annotation_name groundtruth -img_extension .jpg --copy_sensor_data
```

Arguments :
- `-input_format`  : input format (use coco for COCO format)
- `-output_format`  : output format (vision_ai)
- `-image_annotation_type`  : label annotation type for image (2d_bounding_box for box2D)
- `-input_annotation_path` : input annotation path for coco-label.json file
- `-source_data_root`  : image data folder
- `-output_dest_folder` : output root folder (VisionAI local root folder)
- `-uri_root` : uri root for target upload VisionAI storage i.e: https://azuresorate/vai_dataset
- `-n_frame`  : number of frame to be converted (-1 means all), by default -1
- `-sequence_idx_start `  : sequence start id, by default 0
- `-camera_sensor_name`  : camera sensor name (default: "", specified it if need to convert camera data)
- `-annotation_name` : VisionAI annotation folder name (default: "groundtruth")
- `-img_extension` : image file extension (default: ".jpg")
- `--copy_sensor_data` : enable to copy image data


### Convert `VisionAI` format data to `COCO` format

```
python3 visionai_data_format/convert_dataset.py -input_format vision_ai -output_format coco -image_annotation_type 2d_bounding_box -source_data_root ./visionai_data_root -output_dest_folder ~/coco_output_dir -uri_root http://storage_test -n_frame 5 -camera_sensor_name camera1 -annotation_name groundtruth -img_extension .jpg --copy_sensor_data
```
Arguments :
- `-input_format`  : input format (vision_ai)
- `-output_format`  : output format (use coco for COCO format)
- `-image_annotation_type`  : label annotation type for image (2d_bounding_box for box2D)
- `-source_data_root`  : visionai local data root folder
- `-output_dest_folder` : output root folder (COCO local root folder)
- `-uri_root` : uri root for target upload for coco i.e: https://azuresorate/coco_dataset
- `-n_frame`  : number of frame to be converted (-1 means all), by default -1
- `-camera_sensor_name`  : camera sensor name (required for getting the target camera sensor data)
- `-annotation_name` : VisionAI annotation folder name (default: "groundtruth")
- `-img_extension` : image file extension (default: ".jpg")
- `--copy_sensor_data` : enable to copy image data

### Convert `YOLO` format data to `VisionAI` format

```
python3 visionai_data_format/convert_dataset.py -input_format yolo -output_format vision_ai -image_annotation_type 2d_bounding_box -source_data_root ./path_to_yolo_format_dir -output_dest_folder ./output_visionai_dir -n_frame -1 -sequence_idx_start 0 -uri_root http://storage_test -camera_sensor_name camera1 -annotation_name groundtruth -img_extension .jpg  --copy_sensor_data -classes_file category.txt
```

Arguments :
- `-input_format`  : input format (use yolo for YOLO format)
- `-output_format`  : output format (vision_ai)
- `-image_annotation_type`  : label annotation type for image (2d_bounding_box for box2D)
- `-source_data_root`  : data root folder of yolo format
- `-output_dest_folder` : output root folder (VisionAI local root folder)
- `-uri_root` : uri root for target upload VisionAI storage i.e: https://azuresorate/vai_dataset
- `-n_frame`  : number of frame to be converted (-1 means all), by default -1
- `-sequence_idx_start `  : sequence start id, by default 0
- `-camera_sensor_name`  : camera sensor name (default: "", specified it if need to convert camera data)
- `-annotation_name` : VisionAI annotation folder name (default: "groundtruth")
- `-img_extension` : image file extension (default: ".jpg")
- `--copy_sensor_data` : enable to copy image data
- `-classes_file` : txt file contain category names in each line, by default "classes.txt"
- `-img_height` : image height for all images (default: None, which will read the image and get the size)
- `-img_width` : image width for all images (default: None, which will read the image and get the size)


* The `YOLO` dataset should follow the data structure as below:
```bash
.yolo-format-root_folder
├── classes.txt
├── images
│   ├── 000000.png
│   ├── 000001.png
│   ├── 000002.png
│   └── 000003.png
├── labels
│   ├── 000000.txt
│   ├── 000001.txt
│   ├── 000002.txt
│   ├── 000003.txt
```


## Troubleshooting

(WIP)

## Next steps

(WIP)

## Contributing

(WIP)

## Links to language repos

(WIP)

[Python Readme](https://github.com/linkervision/visionai-data-format/blob/master/README.md)
