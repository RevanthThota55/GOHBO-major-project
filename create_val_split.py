"""
Create validation split from training data

This script takes 20% of the training images and moves them to a validation set.
We need a validation set to monitor the model during training and prevent overfitting.
"""

import os
import shutil
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)

# Paths
data_dir = Path("data/brain_tumor/archive")
train_dir = data_dir / "train"
val_dir = data_dir / "val"

# Create val directory if it doesn't exist
val_dir.mkdir(exist_ok=True)

# Get all class names (subdirectories in train/)
classes = [d for d in train_dir.iterdir() if d.is_dir()]

print("Creating validation split (20% of training data)...")
print()

total_moved = 0

for class_dir in classes:
    class_name = class_dir.name
    print(f"Processing class: {class_name}")

    # Create val class directory
    val_class_dir = val_dir / class_name
    val_class_dir.mkdir(exist_ok=True)

    # Get all images in this class
    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

    # Calculate 20% for validation
    num_val = int(len(images) * 0.2)

    # Randomly select images for validation
    val_images = random.sample(images, num_val)

    # Move images to validation directory
    for img in val_images:
        dest = val_class_dir / img.name
        shutil.move(str(img), str(dest))

    total_moved += num_val
    print(f"  Moved {num_val} images to validation set")
    print(f"  Remaining in training: {len(images) - num_val}")

print()
print(f"[OK] Validation split created!")
print(f"Total images moved to validation: {total_moved}")
print()
print("Dataset structure:")
print(f"  train: {train_dir}")
print(f"  val: {val_dir}")
print(f"  test: {data_dir / 'test'}")
