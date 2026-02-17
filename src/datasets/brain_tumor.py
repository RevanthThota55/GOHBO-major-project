"""
Brain Tumor MRI Dataset Loader

Handles loading and preprocessing of brain tumor MRI images.
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Callable, Dict, List


class BrainTumorDataset(Dataset):
    """
    Dataset class for Brain Tumor MRI Classification.

    Supports four classes:
    - glioma_tumor
    - meningioma_tumor
    - no_tumor
    - pituitary_tumor
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Initialize Brain Tumor Dataset.

        Args:
            root_dir: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to apply to images
            target_transform: Optional transform to apply to labels
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Class names and mappings
        self.classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Load image paths and labels
        self.image_paths, self.labels = self._load_dataset()

        print(f"Loaded {len(self.image_paths)} images for {split} split")

    def _load_dataset(self) -> Tuple[List[Path], List[int]]:
        """
        Load dataset image paths and labels.

        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []

        split_dir = self.root_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for class_name in self.classes:
            class_dir = split_dir / class_name

            if class_dir.exists():
                # Get all image files
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for img_path in class_dir.glob(ext):
                        image_paths.append(img_path)
                        labels.append(self.class_to_idx[class_name])

        return image_paths, labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.

        Returns:
            Dictionary mapping class names to counts
        """
        distribution = {cls: 0 for cls in self.classes}

        for label in self.labels:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1

        return distribution

    def get_sample_weights(self) -> torch.Tensor:
        """
        Calculate sample weights for balanced sampling.

        Returns:
            Tensor of sample weights
        """
        class_counts = torch.zeros(len(self.classes))

        for label in self.labels:
            class_counts[label] += 1

        # Calculate weights (inverse of class frequency)
        class_weights = 1.0 / class_counts
        sample_weights = torch.zeros(len(self.labels))

        for idx, label in enumerate(self.labels):
            sample_weights[idx] = class_weights[label]

        return sample_weights


def get_brain_tumor_transforms(
    image_size: Tuple[int, int] = (224, 224),
    split: str = 'train',
    augmentation_config: Optional[Dict] = None
) -> transforms.Compose:
    """
    Get appropriate transforms for brain tumor dataset.

    Args:
        image_size: Target image size
        split: Dataset split ('train', 'val', 'test')
        augmentation_config: Augmentation configuration dictionary

    Returns:
        Composition of transforms
    """
    # ImageNet normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train' and augmentation_config:
        # Training transforms with augmentation
        transform_list = [
            transforms.Resize((image_size[0] + 20, image_size[1] + 20)),
            transforms.RandomCrop(image_size)
        ]

        # Add augmentations based on config
        if augmentation_config.get('random_horizontal_flip', {}).get('enabled', True):
            transform_list.append(
                transforms.RandomHorizontalFlip(
                    p=augmentation_config['random_horizontal_flip'].get('p', 0.5)
                )
            )

        if augmentation_config.get('random_rotation', {}).get('enabled', True):
            transform_list.append(
                transforms.RandomRotation(
                    degrees=augmentation_config['random_rotation'].get('degrees', 15)
                )
            )

        if augmentation_config.get('color_jitter', {}).get('enabled', True):
            config = augmentation_config['color_jitter']
            transform_list.append(
                transforms.ColorJitter(
                    brightness=config.get('brightness', 0.2),
                    contrast=config.get('contrast', 0.2),
                    saturation=config.get('saturation', 0.2),
                    hue=config.get('hue', 0.1)
                )
            )

        if augmentation_config.get('random_affine', {}).get('enabled', True):
            config = augmentation_config['random_affine']
            transform_list.append(
                transforms.RandomAffine(
                    degrees=config.get('degrees', 10),
                    translate=config.get('translate', (0.1, 0.1)),
                    scale=config.get('scale', (0.9, 1.1))
                )
            )

        if augmentation_config.get('gaussian_blur', {}).get('enabled', True):
            transform_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(
                        kernel_size=augmentation_config['gaussian_blur'].get('kernel_size', 3)
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
                    scale=config.get('scale', (0.02, 0.33)),
                    ratio=config.get('ratio', (0.3, 3.3))
                )
            )

    else:
        # Validation/Test transforms (no augmentation)
        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize
        ]

    return transforms.Compose(transform_list)


def create_brain_tumor_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[Dict] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for brain tumor dataset.

    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        image_size: Target image size
        augmentation_config: Augmentation configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create transforms
    train_transform = get_brain_tumor_transforms(
        image_size, 'train', augmentation_config
    )
    val_transform = get_brain_tumor_transforms(image_size, 'val')
    test_transform = get_brain_tumor_transforms(image_size, 'test')

    # Create datasets
    train_dataset = BrainTumorDataset(data_dir, 'train', train_transform)
    val_dataset = BrainTumorDataset(data_dir, 'val', val_transform)
    test_dataset = BrainTumorDataset(data_dir, 'test', test_transform)

    # Get sample weights for balanced sampling
    sample_weights = train_dataset.get_sample_weights()
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )

    # Create DataLoaders
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

    # Print dataset statistics
    print("\nBrain Tumor Dataset Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print("\nClass distribution (train):")
    for cls, count in train_dataset.get_class_distribution().items():
        print(f"  {cls}: {count}")

    return train_loader, val_loader, test_loader