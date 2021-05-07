from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset
from .maxus import MaxusDataset

dataset_factory = {
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset,
    "MAXUS": MaxusDataset
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
