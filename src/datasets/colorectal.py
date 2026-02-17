"""
Colorectal Cancer Histopathology Dataset Loader

Handles loading and preprocessing of colorectal tissue microscopy images.
Dataset: Kather Texture 2016 (5,000 images, 8 tissue types)
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Callable, Dict, List


class ColorectalDataset(Dataset):
    """
    Dataset class for Colorectal Cancer Histopathology Classification.

    8 tissue types (Kather et al. 2016):
    - TUMOR: Colorectal adenocarcinoma epithelium
    - STROMA: Cancer-associated stroma
    - COMPLEX: Complex stroma (mixed tissue)
    - LYMPHO: Immune cell clusters (lymphocytes)
    - DEBRIS: Necrosis and debris
    - MUCOSA: Normal colon mucosa
    - ADIPOSE: Adipose (fatty) tissue
    - EMPTY: Background / empty regions
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Class names matching Kather 2016 dataset
        self.classes = ['TUMOR', 'STROMA', 'COMPLEX', 'LYMPHO', 'DEBRIS', 'MUCOSA', 'ADIPOSE', 'EMPTY']
        self.class_descriptions = {
            'TUMOR': 'Colorectal adenocarcinoma epithelium',
            'STROMA': 'Cancer-associated stroma',
            'COMPLEX': 'Complex stroma (mixed tissue)',
            'LYMPHO': 'Immune cell clusters (lymphocytes)',
            'DEBRIS': 'Necrosis and debris',
            'MUCOSA': 'Normal colon mucosa',
            'ADIPOSE': 'Adipose (fatty) tissue',
            'EMPTY': 'Background / empty regions'
        }

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Load image paths and labels
        self.image_paths, self.labels = self._load_dataset()

        print(f"Loaded {len(self.image_paths)} colorectal tissue images for {split} split")

    def _load_dataset(self) -> Tuple[List[Path], List[int]]:
        image_paths = []
        labels = []

        split_dir = self.root_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for class_name in self.classes:
            class_dir = split_dir / class_name

            if class_dir.exists():
                # Kather 2016 images are .tif format
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                    for img_path in class_dir.glob(ext):
                        image_paths.append(img_path)
                        labels.append(self.class_to_idx[class_name])

        return image_paths, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        distribution = {cls: 0 for cls in self.classes}
        for label in self.labels:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1
        return distribution

    def get_sample_weights(self) -> torch.Tensor:
        class_counts = torch.zeros(len(self.classes))
        for label in self.labels:
            class_counts[label] += 1

        class_weights = 1.0 / class_counts
        sample_weights = torch.zeros(len(self.labels))
        for idx, label in enumerate(self.labels):
            sample_weights[idx] = class_weights[label]

        return sample_weights

    def get_cancer_vs_normal_distribution(self) -> Dict[str, int]:
        cancer_classes = ['TUMOR', 'STROMA', 'COMPLEX']
        normal_classes = ['MUCOSA', 'ADIPOSE']
        immune_classes = ['LYMPHO']
        other_classes = ['DEBRIS', 'EMPTY']

        distribution = {
            'cancer_related': 0,
            'normal': 0,
            'immune': 0,
            'other': 0
        }

        for label in self.labels:
            class_name = self.idx_to_class[label]
            if class_name in cancer_classes:
                distribution['cancer_related'] += 1
            elif class_name in normal_classes:
                distribution['normal'] += 1
            elif class_name in immune_classes:
                distribution['immune'] += 1
            else:
                distribution['other'] += 1

        return distribution


def get_colorectal_transforms(
    image_size: Tuple[int, int] = (224, 224),
    split: str = 'train',
    augmentation_config: Optional[Dict] = None
) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train' and augmentation_config:
        transform_list = [
            transforms.Resize((image_size[0] + 20, image_size[1] + 20)),
            transforms.RandomCrop(image_size)
        ]

        if augmentation_config.get('random_horizontal_flip', {}).get('enabled', True):
            transform_list.append(
                transforms.RandomHorizontalFlip(
                    p=augmentation_config['random_horizontal_flip'].get('p', 0.5)
                )
            )

        if augmentation_config.get('random_vertical_flip', {}).get('enabled', True):
            transform_list.append(
                transforms.RandomVerticalFlip(
                    p=augmentation_config['random_vertical_flip'].get('p', 0.5)
                )
            )

        if augmentation_config.get('random_rotation', {}).get('enabled', True):
            transform_list.append(
                transforms.RandomRotation(
                    degrees=augmentation_config['random_rotation'].get('degrees', 90)
                )
            )

        if augmentation_config.get('color_jitter', {}).get('enabled', True):
            config = augmentation_config['color_jitter']
            transform_list.append(
                transforms.ColorJitter(
                    brightness=config.get('brightness', 0.3),
                    contrast=config.get('contrast', 0.3),
                    saturation=config.get('saturation', 0.3),
                    hue=config.get('hue', 0.15)
                )
            )

        if augmentation_config.get('random_affine', {}).get('enabled', True):
            config = augmentation_config['random_affine']
            transform_list.append(
                transforms.RandomAffine(
                    degrees=config.get('degrees', 15),
                    translate=config.get('translate', (0.1, 0.1)),
                    scale=config.get('scale', (0.8, 1.2)),
                    shear=config.get('shear', 10)
                )
            )

        if augmentation_config.get('gaussian_blur', {}).get('enabled', True):
            transform_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(
                        kernel_size=augmentation_config['gaussian_blur'].get('kernel_size', 5)
                    )],
                    p=augmentation_config['gaussian_blur'].get('p', 0.2)
                )
            )

        transform_list.extend([
            transforms.ToTensor(),
            normalize
        ])

        if augmentation_config.get('random_erasing', {}).get('enabled', True):
            config = augmentation_config['random_erasing']
            transform_list.append(
                transforms.RandomErasing(
                    p=config.get('p', 0.2),
                    scale=config.get('scale', (0.02, 0.2)),
                    ratio=config.get('ratio', (0.3, 3.3))
                )
            )

    else:
        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize
        ]

    return transforms.Compose(transform_list)


def create_colorectal_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[Dict] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = get_colorectal_transforms(
        image_size, 'train', augmentation_config
    )
    val_transform = get_colorectal_transforms(image_size, 'val')
    test_transform = get_colorectal_transforms(image_size, 'test')

    train_dataset = ColorectalDataset(data_dir, 'train', train_transform)
    val_dataset = ColorectalDataset(data_dir, 'val', val_transform)
    test_dataset = ColorectalDataset(data_dir, 'test', test_transform)

    sample_weights = train_dataset.get_sample_weights()
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("\nColorectal Histopathology Dataset Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print("\nClass distribution (train):")
    for cls, count in train_dataset.get_class_distribution().items():
        desc = train_dataset.class_descriptions[cls]
        print(f"  {cls:8s} ({desc:40s}): {count:5d}")

    print("\nTissue type distribution (train):")
    tissue_dist = train_dataset.get_cancer_vs_normal_distribution()
    for tissue_type, count in tissue_dist.items():
        print(f"  {tissue_type:15s}: {count:5d}")

    return train_loader, val_loader, test_loader
