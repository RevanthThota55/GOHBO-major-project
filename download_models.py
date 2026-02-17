"""
Download pre-trained model weights from GitHub Releases.
Run this after cloning the repository.

Usage:
    python download_models.py
"""

import os
import sys
import urllib.request
from pathlib import Path

REPO = "RevanthThota55/GOHBO-major-project"
TAG = "v1.0-models"

MODELS = {
    "models/checkpoints/best_model.pth": {
        "description": "Brain Tumor MRI Classification (95.42% accuracy)",
        "size_mb": 43,
    },
    "models/checkpoints/chest_xray_resnet18.pth": {
        "description": "Chest X-Ray Pneumonia Detection (97.03% accuracy)",
        "size_mb": 133,
    },
    "models/checkpoints/colorectal_resnet18.pth": {
        "description": "Colorectal Cancer Histopathology (94.56% accuracy)",
        "size_mb": 133,
    },
}

BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"


def download_file(url, dest_path, description, size_mb):
    """Download a file with progress indicator."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists")
        return True

    print(f"  Downloading {dest.name} ({size_mb} MB) ...")
    print(f"  {description}")

    try:
        urllib.request.urlretrieve(url, str(dest), _progress_hook)
        print(f"\n  [OK] Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"\n  [ERROR] Failed to download {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        sys.stdout.write(f"\r  [{bar}] {pct}%")
        sys.stdout.flush()


def main():
    print("=" * 60)
    print("ExplainableMed-GOHBO - Model Weight Downloader")
    print("=" * 60)

    total = len(MODELS)
    success = 0

    for path, info in MODELS.items():
        filename = os.path.basename(path)
        url = f"{BASE_URL}/{filename}"
        print(f"\n[{success + 1}/{total}] {info['description']}")

        if download_file(url, path, info["description"], info["size_mb"]):
            success += 1

    print(f"\n{'=' * 60}")
    print(f"Downloaded: {success}/{total} models")

    if success == total:
        print("\nAll models ready! Run the webapp:")
        print("  cd webapp")
        print("  python app.py")
    else:
        print("\nSome models failed to download.")
        print("You can download them manually from:")
        print(f"  https://github.com/{REPO}/releases/tag/{TAG}")

    print("=" * 60)


if __name__ == "__main__":
    main()
