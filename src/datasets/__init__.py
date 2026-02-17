"""
Dataset loaders for medical image classification
"""

from .brain_tumor import BrainTumorDataset
from .chest_xray import ChestXRayDataset
from .colorectal import ColorectalDataset
from .download_datasets import DatasetDownloader

__all__ = [
    'BrainTumorDataset',
    'ChestXRayDataset',
    'ColorectalDataset',
    'DatasetDownloader'
]

def get_dataset(dataset_name: str, root_dir: str, split: str = 'train', transform=None):
    """
    Factory function to get the appropriate dataset.

    Args:
        dataset_name: Name of the dataset ('brain_tumor', 'chest_xray', 'colorectal')
        root_dir: Root directory containing the dataset
        split: Dataset split ('train', 'val', 'test')
        transform: Torchvision transforms to apply

    Returns:
        Dataset object
    """
    datasets = {
        'brain_tumor': BrainTumorDataset,
        'chest_xray': ChestXRayDataset,
        'colorectal': ColorectalDataset
    }

    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return datasets[dataset_name](root_dir, split, transform)