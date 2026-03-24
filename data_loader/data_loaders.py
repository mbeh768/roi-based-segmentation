from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.datasets import VentricleDataset


class VentricleDataLoader(BaseDataLoader):
    """
    DataLoader for the ventricle segmentation task.

    Config usage:
        "data_loader": {
            "type": "VentricleDataLoader",
            "args": {
                "data_dir": "data/",
                "batch_size": 4,
                "shuffle": true,
                "validation_split": 0.1,
                "num_workers": 2
            }
        }
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = VentricleDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
