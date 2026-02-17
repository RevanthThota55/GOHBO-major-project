"""
Baseline Evaluation Script

This script establishes a baseline for model performance BEFORE optimization.

What is a baseline?
-------------------
A baseline is the model's performance BEFORE we make any improvements.
Think of it like measuring your running speed before starting a training program.
We need this so we can later say "we improved accuracy from X% to Y%".

This script will:
1. Check if a trained model already exists
2. If yes, load it and evaluate it
3. If no, train a quick baseline model with default settings
4. Save all the results so we can compare later
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, PHASE1_OUTPUT, MODELS_DIR
from src.models.resnet18_medical import MedicalResNet18
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator


def find_existing_model(models_dir: Path, dataset_name: str) -> Path:
    """
    Look for an existing trained model file.

    What this does:
    - Searches the models directory for .pth files (PyTorch model files)
    - Tries to find a model specifically for the dataset we're using
    - Returns the path if found, None if not found

    Args:
        models_dir: Directory where models are stored
        dataset_name: Name of the dataset (e.g., 'brain_tumor')

    Returns:
        Path to model file if found, None otherwise
    """
    # First, try to find a model specifically named for this dataset
    expected_name = f"{dataset_name}_resnet18.pth"
    expected_path = models_dir / expected_name

    if expected_path.exists():
        print(f"[OK] Found existing model: {expected_path}")
        return expected_path

    # If not found, look for any .pth file in the models directory
    pth_files = list(models_dir.glob("*.pth"))
    if pth_files:
        # Use the most recently modified model file
        most_recent = max(pth_files, key=lambda p: p.stat().st_mtime)
        print(f"[OK] Found existing model: {most_recent}")
        return most_recent

    # No model found
    print("[X] No existing trained model found")
    return None


def train_baseline_model(config: dict, dataset_name: str, device: str) -> nn.Module:
    """
    Train a quick baseline model with default settings.

    What this does:
    - Creates a fresh ResNet-18 model
    - Trains it for just 10 epochs (quick training)
    - Uses default hyperparameters (not optimized)
    - Saves the model for future use

    This is ONLY called if no trained model exists yet.

    Args:
        config: Configuration dictionary
        dataset_name: Name of dataset to train on
        device: Device to use (cuda or cpu)

    Returns:
        Trained baseline model
    """
    print("\n" + "="*70)
    print("TRAINING BASELINE MODEL")
    print("="*70)
    print("No existing model found, so we'll train a quick baseline.")
    print("This uses default settings (not optimized) for comparison.")
    print(f"Training for 10 epochs on {device}...")
    print()

    # Import dataset loaders
    from src.datasets.brain_tumor import create_brain_tumor_dataloaders

    # Get data loaders
    # These load the images from disk in batches so we can train the model
    train_loader, val_loader, test_loader = create_brain_tumor_dataloaders(
        data_dir=config['dataset']['data_path'],
        batch_size=32,  # Process 32 images at a time
        num_workers=2
    )

    # Create model
    # ResNet-18 is a neural network architecture designed for image classification
    model = MedicalResNet18(
        num_classes=config['dataset']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    )
    model = model.to(device)

    # Create trainer
    # The Trainer handles all the details of training (feeding data, updating weights, etc.)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Train for 10 epochs (quick baseline)
    # An epoch = one complete pass through all training images
    print("Starting training...")
    trainer.train(num_epochs=10)

    # Save the baseline model
    baseline_model_path = MODELS_DIR / f"{dataset_name}_baseline.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'dataset': dataset_name,
        'epochs_trained': 10,
        'timestamp': datetime.now().isoformat()
    }, baseline_model_path)

    print(f"\n[OK] Baseline model saved to: {baseline_model_path}")

    return model


def evaluate_baseline(model: nn.Module, config: dict, dataset_name: str, device: str) -> dict:
    """
    Evaluate the model on the test set and record all metrics.

    What this does:
    - Runs the model on all test images (images it has never seen before)
    - Calculates important metrics:
      * Accuracy: How many predictions were correct overall
      * Precision: Of all images predicted as a tumor, how many actually were
      * Recall: Of all actual tumor images, how many did we find
      * F1-score: Balance between precision and recall
      * ROC-AUC: How well the model separates different classes

    Args:
        model: Model to evaluate
        config: Configuration dictionary
        dataset_name: Name of dataset
        device: Device to use

    Returns:
        Dictionary containing all metrics
    """
    print("\n" + "="*70)
    print("EVALUATING BASELINE PERFORMANCE")
    print("="*70)

    # Import dataset loader
    from src.datasets.brain_tumor import get_brain_tumor_loaders

    # Get test loader
    # The test set contains images the model has NEVER seen during training
    # This gives us an honest measure of how well it can generalize
    _, _, test_loader = get_brain_tumor_loaders(
        data_dir=config['dataset']['data_path'],
        batch_size=32,
        num_workers=2,
        augment=False  # No augmentation for testing (we want real images)
    )

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        class_names=config['dataset']['classes'],
        config=config,
        device=device
    )

    # Run evaluation
    print("Running model on test set...")
    metrics = evaluator.evaluate()

    return metrics


def save_baseline_metrics(metrics: dict, output_path: Path):
    """
    Save baseline metrics to a JSON file.

    What is JSON?
    A simple text format for storing data that both humans and computers can read.
    We save metrics in JSON so we can load them later for comparison.

    Args:
        metrics: Dictionary of metric values
        output_path: Where to save the JSON file
    """
    # Make sure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save metrics to JSON file
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)  # indent=2 makes it readable

    print(f"\n[OK] Baseline metrics saved to: {output_path}")


def print_baseline_summary(metrics: dict):
    """
    Print a clear, easy-to-read summary of baseline performance.

    Args:
        metrics: Dictionary of metric values
    """
    print("\n" + "="*70)
    print("BASELINE PERFORMANCE SUMMARY")
    print("="*70)
    print()
    print("This is how well the model performs BEFORE optimization.")
    print("We'll use these numbers to measure improvement later.")
    print()
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%  <- Overall correctness")
    print(f"  Precision: {metrics['precision']*100:.2f}%  <- How reliable are positive predictions")
    print(f"  Recall:    {metrics['recall']*100:.2f}%  <- How many actual cases did we find")
    print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%  <- Balance of precision and recall")

    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']*100:.2f}%  <- How well model separates classes")

    print()
    print("="*70)
    print()

    # Interpret the results
    accuracy_pct = metrics['accuracy'] * 100
    if accuracy_pct >= 95:
        print("🎉 Great! Baseline already meets the 95% target.")
    elif accuracy_pct >= 90:
        print("👍 Good baseline. We'll work on getting it above 95%.")
    elif accuracy_pct >= 80:
        print("📈 Moderate baseline. Optimization should help significantly.")
    else:
        print("⚠️  Low baseline. Model may need more training or better data.")


def main():
    """
    Main function that runs the baseline evaluation.
    """
    # Parse command line arguments
    # Arguments let you customize how the script runs without editing code
    parser = argparse.ArgumentParser(
        description='Establish baseline model performance before optimization'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='brain_tumor',
        choices=['brain_tumor', 'chest_xray', 'colorectal'],
        help='Which dataset to evaluate on'
    )
    parser.add_argument(
        '--force_retrain',
        action='store_true',
        help='Train a new baseline even if a model exists'
    )

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.dataset)

    # Determine device (GPU if available, otherwise CPU)
    # GPU = Graphics Processing Unit, much faster for training neural networks
    # CPU = Central Processing Unit, slower but always available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("Note: Training on CPU will be slower. GPU recommended but not required.")

    # Check if dataset exists
    if not config['dataset']['data_path'].exists():
        print(f"\n❌ ERROR: Dataset not found at {config['dataset']['data_path']}")
        print("Please download the dataset first using src/datasets/download_datasets.py")
        sys.exit(1)

    # Try to find existing model
    model_path = None if args.force_retrain else find_existing_model(MODELS_DIR, args.dataset)

    if model_path is not None:
        # Load existing model
        print(f"\nLoading existing model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)

        model = MedicalResNet18(
            num_classes=config['dataset']['num_classes'],
            dropout_rate=config['model']['dropout_rate']
        )

        # Handle different checkpoint formats
        # Some checkpoints have the state_dict wrapped in a dictionary with 'model_state_dict' key
        # Others have the state_dict directly
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume checkpoint is the state_dict directly
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()  # Set to evaluation mode (turns off dropout, etc.)

        print("[OK] Model loaded successfully")
    else:
        # Train a new baseline model
        model = train_baseline_model(config, args.dataset, device)

    # Evaluate the model
    metrics = evaluate_baseline(model, config, args.dataset, device)

    # Save metrics
    output_path = PHASE1_OUTPUT / 'baseline_metrics.json'
    save_baseline_metrics(metrics, output_path)

    # Print summary
    print_baseline_summary(metrics)

    print("\n[OK] Baseline evaluation complete!")
    print(f"Results saved to: {output_path}")
    print("\nNext step: Run optimize_hyperparams.py to find better settings")


if __name__ == '__main__':
    main()
