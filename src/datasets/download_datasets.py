"""
Dataset downloader for public medical imaging datasets.

Downloads and organizes datasets from Kaggle and other sources.
"""

import os
import zipfile
import shutil
from pathlib import Path
import requests
from tqdm import tqdm
import json


class DatasetDownloader:
    """
    Handles downloading and organizing medical imaging datasets.
    """

    def __init__(self, data_dir: str):
        """
        Initialize dataset downloader.

        Args:
            data_dir: Directory to store datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Dataset information
        self.datasets_info = {
            'brain_tumor': {
                'kaggle_dataset': 'sartajbhuvaji/brain-tumor-classification-mri',
                'description': 'Brain Tumor MRI Classification Dataset',
                'classes': ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'],
                'structure': 'folder_per_class'
            },
            'chest_xray': {
                'kaggle_dataset': 'paultimothymooney/chest-xray-pneumonia',
                'description': 'Chest X-Ray Images (Pneumonia)',
                'classes': ['NORMAL', 'PNEUMONIA'],
                'structure': 'train_test_split'
            },
            'colorectal': {
                'kaggle_dataset': 'kmader/colorectal-histology-mnist',
                'description': 'Colorectal Histopathology Images',
                'classes': ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'],
                'structure': 'folder_per_class'
            }
        }

    def download_all(self):
        """Download all datasets."""
        print("=" * 50)
        print("Medical Imaging Dataset Downloader")
        print("=" * 50)

        for dataset_name in self.datasets_info:
            self.download_dataset(dataset_name)

    def download_dataset(self, dataset_name: str):
        """
        Download a specific dataset.

        Args:
            dataset_name: Name of the dataset to download
        """
        if dataset_name not in self.datasets_info:
            print(f"Unknown dataset: {dataset_name}")
            return

        dataset_path = self.data_dir / dataset_name
        if dataset_path.exists() and any(dataset_path.iterdir()):
            print(f"\n✓ {dataset_name} dataset already exists at {dataset_path}")
            return

        info = self.datasets_info[dataset_name]
        print(f"\n📥 Downloading {info['description']}...")
        print(f"   Kaggle dataset: {info['kaggle_dataset']}")

        # Check if Kaggle API is configured
        kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
        if not kaggle_config.exists():
            print("\n⚠️  Kaggle API not configured!")
            print("Please follow these steps:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Click 'Create New API Token' to download kaggle.json")
            print("3. Place kaggle.json in ~/.kaggle/ directory")
            print("4. Run this script again")
            print("\nAlternatively, you can manually download datasets from:")
            print(f"   https://www.kaggle.com/datasets/{info['kaggle_dataset']}")
            print(f"   and extract them to: {dataset_path}")
            return

        try:
            # Use Kaggle API to download
            import kaggle

            # Create dataset directory
            dataset_path.mkdir(parents=True, exist_ok=True)

            # Download dataset
            print(f"   Downloading to {dataset_path}...")
            kaggle.api.dataset_download_files(
                info['kaggle_dataset'],
                path=str(dataset_path),
                unzip=True
            )

            # Organize dataset structure if needed
            self._organize_dataset(dataset_name, dataset_path)

            print(f"✓ {dataset_name} dataset downloaded successfully!")

        except ImportError:
            print("⚠️  Kaggle package not installed. Install with: pip install kaggle")
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {str(e)}")
            print("\nYou can manually download the dataset from:")
            print(f"   https://www.kaggle.com/datasets/{info['kaggle_dataset']}")
            print(f"   and extract it to: {dataset_path}")

    def _organize_dataset(self, dataset_name: str, dataset_path: Path):
        """
        Organize dataset into train/val/test splits.

        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to the dataset
        """
        info = self.datasets_info[dataset_name]

        if dataset_name == 'brain_tumor':
            self._organize_brain_tumor(dataset_path, info['classes'])
        elif dataset_name == 'chest_xray':
            self._organize_chest_xray(dataset_path)
        elif dataset_name == 'colorectal':
            self._organize_colorectal(dataset_path, info['classes'])

    def _organize_brain_tumor(self, dataset_path: Path, classes: list):
        """Organize brain tumor dataset."""
        # Check if 'Training' folder exists (common structure)
        training_path = dataset_path / 'Training'
        testing_path = dataset_path / 'Testing'

        if training_path.exists():
            # Reorganize into train/val/test
            self._create_splits_from_folders(
                training_path,
                testing_path,
                dataset_path,
                classes,
                train_ratio=0.7,
                val_ratio=0.15
            )
            # Remove original folders
            shutil.rmtree(training_path, ignore_errors=True)
            shutil.rmtree(testing_path, ignore_errors=True)

    def _organize_chest_xray(self, dataset_path: Path):
        """Organize chest X-ray dataset."""
        # Dataset usually comes with train/test structure
        # Create validation set from training data
        train_path = dataset_path / 'train'
        val_path = dataset_path / 'val'
        test_path = dataset_path / 'test'

        if train_path.exists() and not val_path.exists():
            val_path.mkdir(exist_ok=True)

            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_train_path = train_path / class_name
                class_val_path = val_path / class_name
                class_val_path.mkdir(exist_ok=True)

                if class_train_path.exists():
                    # Move 15% of training images to validation
                    images = list(class_train_path.glob('*.jpeg')) + list(class_train_path.glob('*.jpg'))
                    num_val = int(len(images) * 0.15)

                    for img in images[:num_val]:
                        shutil.move(str(img), str(class_val_path / img.name))

    def _organize_colorectal(self, dataset_path: Path, classes: list):
        """Organize colorectal dataset."""
        # Check if dataset needs reorganization
        if (dataset_path / 'Kather_texture_2016_image_tiles_5000').exists():
            source_path = dataset_path / 'Kather_texture_2016_image_tiles_5000'

            # Create train/val/test splits
            for split in ['train', 'val', 'test']:
                split_path = dataset_path / split
                split_path.mkdir(exist_ok=True)

                for class_name in classes:
                    class_path = split_path / class_name
                    class_path.mkdir(exist_ok=True)

            # Distribute images
            for class_name in classes:
                class_source = source_path / class_name
                if class_source.exists():
                    images = list(class_source.glob('*.tif'))

                    # Split ratios: 70% train, 15% val, 15% test
                    num_train = int(len(images) * 0.7)
                    num_val = int(len(images) * 0.15)

                    # Move images
                    for img in images[:num_train]:
                        shutil.copy2(str(img), str(dataset_path / 'train' / class_name / img.name))

                    for img in images[num_train:num_train + num_val]:
                        shutil.copy2(str(img), str(dataset_path / 'val' / class_name / img.name))

                    for img in images[num_train + num_val:]:
                        shutil.copy2(str(img), str(dataset_path / 'test' / class_name / img.name))

            # Remove original folder
            shutil.rmtree(source_path, ignore_errors=True)

    def _create_splits_from_folders(
        self,
        training_path: Path,
        testing_path: Path,
        dataset_path: Path,
        classes: list,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        """
        Create train/val/test splits from existing folders.

        Args:
            training_path: Path to training folder
            testing_path: Path to testing folder
            dataset_path: Root dataset path
            classes: List of class names
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
        """
        # Create split directories
        for split in ['train', 'val', 'test']:
            split_path = dataset_path / split
            split_path.mkdir(exist_ok=True)

            for class_name in classes:
                class_path = split_path / class_name
                class_path.mkdir(exist_ok=True)

        # Process training data
        if training_path.exists():
            for class_name in classes:
                class_source = training_path / class_name
                if class_source.exists():
                    images = list(class_source.glob('*.jpg')) + list(class_source.glob('*.jpeg'))

                    # Split into train and val
                    num_train = int(len(images) * (train_ratio / (train_ratio + val_ratio)))

                    for img in images[:num_train]:
                        shutil.copy2(str(img), str(dataset_path / 'train' / class_name / img.name))

                    for img in images[num_train:]:
                        shutil.copy2(str(img), str(dataset_path / 'val' / class_name / img.name))

        # Process testing data
        if testing_path.exists():
            for class_name in classes:
                class_source = testing_path / class_name
                if class_source.exists():
                    images = list(class_source.glob('*.jpg')) + list(class_source.glob('*.jpeg'))

                    for img in images:
                        shutil.copy2(str(img), str(dataset_path / 'test' / class_name / img.name))

    def check_dataset_integrity(self, dataset_name: str) -> bool:
        """
        Check if dataset is properly downloaded and organized.

        Args:
            dataset_name: Name of the dataset to check

        Returns:
            True if dataset is ready, False otherwise
        """
        dataset_path = self.data_dir / dataset_name

        if not dataset_path.exists():
            print(f"❌ {dataset_name} dataset not found")
            return False

        # Check for train/val/test splits
        splits = ['train', 'val', 'test']
        for split in splits:
            split_path = dataset_path / split
            if not split_path.exists():
                print(f"❌ {dataset_name}/{split} not found")
                return False

            # Check for class folders
            classes = self.datasets_info[dataset_name]['classes']
            for class_name in classes:
                class_path = split_path / class_name
                if not class_path.exists():
                    print(f"❌ {dataset_name}/{split}/{class_name} not found")
                    return False

                # Check for images
                images = list(class_path.glob('*.jpg')) + \
                        list(class_path.glob('*.jpeg')) + \
                        list(class_path.glob('*.png')) + \
                        list(class_path.glob('*.tif'))

                if len(images) == 0:
                    print(f"⚠️  No images in {dataset_name}/{split}/{class_name}")

        print(f"✓ {dataset_name} dataset is ready")
        return True

    def get_dataset_stats(self, dataset_name: str):
        """
        Print statistics about a dataset.

        Args:
            dataset_name: Name of the dataset
        """
        dataset_path = self.data_dir / dataset_name

        if not dataset_path.exists():
            print(f"Dataset {dataset_name} not found")
            return

        print(f"\nDataset: {self.datasets_info[dataset_name]['description']}")
        print("-" * 50)

        total_images = 0
        for split in ['train', 'val', 'test']:
            split_path = dataset_path / split
            if split_path.exists():
                split_total = 0
                print(f"\n{split.upper()}:")

                for class_name in self.datasets_info[dataset_name]['classes']:
                    class_path = split_path / class_name
                    if class_path.exists():
                        images = list(class_path.glob('*.jpg')) + \
                                list(class_path.glob('*.jpeg')) + \
                                list(class_path.glob('*.png')) + \
                                list(class_path.glob('*.tif'))
                        count = len(images)
                        split_total += count
                        print(f"  {class_name:20s}: {count:5d} images")

                print(f"  {'Total':20s}: {split_total:5d} images")
                total_images += split_total

        print(f"\n{'TOTAL DATASET':20s}: {total_images:5d} images")


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import DATA_DIR

    downloader = DatasetDownloader(DATA_DIR)

    # Download all datasets
    # downloader.download_all()

    # Or download specific dataset
    # downloader.download_dataset('brain_tumor')

    # Check dataset integrity
    for dataset in ['brain_tumor', 'chest_xray', 'colorectal']:
        downloader.check_dataset_integrity(dataset)
        downloader.get_dataset_stats(dataset)