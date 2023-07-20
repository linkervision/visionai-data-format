# visionai-data-format

`VisionAI` format is Dataverse["url"] standardized annotation format to label objects and sequences in the context of Autonomous Driving System(ADS). `VisionAI` provides consistent and effective driving environment description and categorization in the real-world case.

This tool provides validator of `VisionAI` format schema. Currently, the library supports:
  - Validate created `VisionAI` data format
  - Validate `VisionAI` data attributes with given `Ontology` information.

[Package (PyPi)](https://pypi.org/project/visionai-data-format/)    |   [Source code](https://github.com/linkernetworks/visionai-data-format)


## Getting started

(WIP)

### Install the package

```
pip install visionai-data-format
```

**Prerequisites**: You must have [Python 3.7](https://www.python.org/downloads/) and above to use this package.

## Example
The following sections provide examples for the following:

* [Validate VisionAI schema](###validate-visionai-schema)
* [Validate VisionAI data with given Ontology](###validate-visionai-data-with-given-ontology)

### Validate VisionAI schema

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

First, we declare our custom `VisionAI` data, then call `VisionAI(**custom_visionai_data).dict()` to validate our custom data visionai schema. It will raise error if any of required fields is missing or the value type doesn't meet with defined data type ( `BaseModel` error message). Otherwise, it will return dictionary of validated `VisionAI` data

### Validate VisionAI data with given Ontology

Before upload dataset into `Dataverse` platform, we could try to validate a `VisionAI` annotation with `Ontology` schema. `Ontology` schema works as a predefined `Project Ontology` data in `Dataverse`.

`Ontology` contains `contexts`, `objects`, `streams`, and `tags` four main elements similar to `VisioniAI` schema. The difference is that `Ontology` is the union of all categories and attributes that will be compared with a `VisionAI` data.

1. `contexts`
    need to be filled if only the project ontology is `classification` type.
2. `objects`
    need to be filled for other project ontologies instead of `classification`, such as `bounding_box` or `semantic_segmentation`, etc.
3. `streams`
    required to be filled, since it is the project sensor related information.
4. `tags`
    need to be filled in case of `semantic_segmentation` project ontology.

Following is the example of `Ontology` Schema and how to validate `VisionAI` data with it:

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
# If there is any error occurred, it will returns list of error messages
# Otherwise, it will return empty list
# example of errors :
# >["validate objects error: Missing attributes from data pointers : {('893ac389-7782-4bc3-8f61-09a8e48c819f', 'bbox_shape'), ('893ac389-7782-4bc3-8f61-09a8e48c819f', 'cuboid_shape')} \n"]
print(errors)

```

First, create a new `Ontology` that contains the project ontology. Then, call `validate_with_ontology(ontology=validated_ontology)` to validate whether current `VisionAI` data meets the `Ontology` data information. It will returns list of error messages if any error occured, otherwise it returns empty list.


## Tools
### Convert `VisionAI` format data to `BDD+` format
#### (Only support box2D for now)
The script below could help convert `VisionAI` annotation data to `BDD+` json file
```
python visionai_data_format/vai_to_bdd.py -vai_src_folder /path_for_visionai_root_folder -bdd_dest_file /dest_path/bdd.json -company_code 99 -storage_name storge1 -container_name dataset1 -annotation_name groundtruth
```
Arguments :
- `-vai_src_folder` : VAI root folder contains VAI format json file
- `-bdd_dest_file`  : BDD+ format file save destination
- `-company_code`  : company code
- `-storage_name`  : storage name
- `-container_name`  : container name (dataset name)
- `-annotation_name` : annotation folder name (default: "groundtruth")


### Convert `BDD+` format data to `VisionAI` format
#### (Only support box2D for now)
```
 python visionai_data_format/bdd_to_vai.py -bdd_src_file /path/bdd.json -vai_dest_folder /path_vai_root/vai_dataset -uri_root https://azure_storage/vai_dataset -sensor camera1 -annotation_name groundtruth -img_extention .jpg

```
Arguments :
- `-bdd_src_file`  : source BDD+ format file
- `-vai_dest_folder` : VAI root folder (would put VAI format json files here)
- `-uri_root` : uri root for VAI storage i.e: https://azuresorate/vai_dataset
- `-sensor`  : sensor name
- `-annotation_name` : annotation folder name (default: "groundtruth")


## Troubleshooting

(WIP)

## Next steps

(WIP)

## Contributing

(WIP)

## Links to language repos

(WIP)

[Python Readme](https://github.com/linkernetworks/visionai-data-format/tree/develop/README.md)
