# CARLA_CLASSES = [
#     "None",
#     "Building",
#     "Fences",
#     "Other",
#     "Pedestrian",
#     "Poles",
#     "RoadLine",
#     "Road",
#     "Sidewalk",
#     "TrafficSign",
#     "Vegetation",
#     "Vehicle",
#     "Wall",
#     "Sky",
#     "Ground",
#     "Bridge",
#     "RailTrack",
#     "GuardRail",
#     "TrafficLight",
#     "Static",
#     "Dynamic",
#     "Water",
#     "Terrain",
# ]


def gen_ontology_classes_dict(ontology_classes):
    # string to list
    ontology_classes = ontology_classes.split(",")
    # Mapping ontology classes to (name, id)
    return {
        oc_name.strip(): oc_id
        for oc_name, oc_id in zip(ontology_classes, range(len(ontology_classes)))
    }
