from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset
from .semantickitti import SemanticKITTIDataset

dataset_factory = {
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset,
    "SemanticKITTI": SemanticKITTIDataset
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
