# visionai-data-format

`VisionAI` format is Dataverse["url"] standardized annotation format to label objects and sequences in the context of Autonomous Driving System(ADS). `VisionAI` provides consistent and effective driving environment description and categorization in the real-world case.

This tool provides validator of `VisionAI` format schema. Currently, the library supports:
  - Validate created `VisionAI` data format
  - Validate `VisionAI` data attributes with given `Ontology` information.

[Package (PyPi)](https://pypi.org/project/visionai-data-format/)    |   [Source code](https://github.com/linkernetworks/visionai-data-format)


## Getting started

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

To validated `VisionAI` format schema, following is the example :

```Python
from visionai_data_format.schemas.visionai_schema import VisionAIModel
from dataverse_sdk.connections import get_connection

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
validated_visionai = VisionAIModel(**custom_visionai_data).dict()
```

### Validate VisionAI data with given Ontology

Before upload dataset into `DataVerse` platform, we could try to validate a `VisionAI` annotation with `Ontology` schema. `Ontology` schema works as a predefined `Project Ontology` data in `DataVerse`.

`Ontology` contains `contexts`, `objects`, `streams`, and `tags` five main elements similar to `VisioniAI` schema. The difference is that `Ontology` is the union of all categories and attributes that will be compared with a `VisionAI` data.

Following is the example of `Ontology` Schema and how to validate `VisionAI` data:


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

# Validate our ontology
validated_ontology = Ontology(**custom_ontology).dict()

# Validate VisionAI data with our ontology, custom_visionai_data is the custom data from upper example
errors = VisionAIModel(**custom_visionai_data).validate_with_ontology(ontology=validated_ontology)

# shows errors
print(errors)

```

## Troubleshooting


## Next steps


## Contributing



## Links to language repos

[Python Readme](https://github.com/linkernetworks/visionai-data-format/tree/develop/python/README.md)
