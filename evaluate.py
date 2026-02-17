"""
Evaluation script for trained medical image classification models.

Performs comprehensive evaluation on the test set and generates visualizations.
"""

import argparse
import sys
from pathlib import Path
import torch
import json
import pathlib
import platform

# Fix for loading models trained on Linux/Colab (PosixPath) on Windows (WindowsPath)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config import get_config
from models.resnet18_medical import MedicalResNet18
from datasets.brain_tumor import create_brain_tumor_dataloaders
from datasets.chest_xray import create_chest_xray_dataloaders
from datasets.colorectal import create_colorectal_dataloaders
from training.evaluator import Evaluator


def get_dataloaders(dataset_name: str, config: dict):
    """Get appropriate dataloaders for the specified dataset."""
    dataset_path = config['dataset']['data_path']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    image_size = config['dataset']['image_size']
    augmentation_config = None  # No augmentation for evaluation

    if dataset_name == 'brain_tumor':
        return create_brain_tumor_dataloaders(
            dataset_path,
            batch_size,
            num_workers,
            image_size,
            augmentation_config
        )
    elif dataset_name == 'chest_xray':
        return create_chest_xray_dataloaders(
            dataset_path,
            batch_size,
            num_workers,
            image_size,
            augmentation_config
        )
    elif dataset_name == 'colorectal':
        return create_colorectal_dataloaders(
            dataset_path,
            batch_size,
            num_workers,
            image_size,
            augmentation_config
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Medical Image Classification Model')
    parser.add_argument('--dataset', type=str, default='brain_tumor',
                       choices=['brain_tumor', 'chest_xray', 'colorectal'],
                       help='Dataset to evaluate on')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--baseline_path', type=str, default=None,
                       help='Path to baseline metrics JSON for comparison (D-9)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: results/phase1/)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for evaluation')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for evaluation (overrides config)')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Disable visualization generation')

    args = parser.parse_args()

    # Get configuration
    config = get_config(args.dataset)

    # Set output directory (D-8: visualizations go to phase1)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config['paths']['results'] / 'phase1'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.no_visualize:
        config['evaluation']['visualization'] = {
            'plot_confusion_matrix': False,
            'plot_roc_curves': False,
            'plot_class_distribution': False,
            'save_misclassified': False
        }

    print("\n" + "=" * 70)
    print(f"MODEL EVALUATION (Phase 1)")
    print("=" * 70)
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Classes: {config['dataset']['num_classes']}")
    print(f"Model Path: {args.model_path}")
    print(f"Output Directory: {output_dir}")
    if args.baseline_path:
        print(f"Baseline Comparison: {args.baseline_path}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Create dataloaders
    print("\n Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(args.dataset, config)

    # Create model
    print("\n Creating model...")
    model = MedicalResNet18(
        num_classes=config['dataset']['num_classes'],
        input_channels=config['dataset']['channels'],
        pretrained=False,  # We're loading trained weights
        freeze_backbone=False,
        dropout_rate=config['model']['dropout_rate'],
        hidden_units=config['model']['hidden_units']
    )

    # Load trained weights
    print(f" Loading model weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  [OK] Model loaded from epoch {checkpoint.get('epoch', 'unknown') + 1}")
        if 'val_acc' in checkpoint:
            print(f"  [OK] Validation accuracy at checkpoint: {checkpoint['val_acc']:.2f}%")
    else:
        # Assume it's just the state dict
        model.load_state_dict(checkpoint)
        print(f"  [OK] Model weights loaded")

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        class_names=config['dataset']['classes'],
        config=config,
        device=args.device
    )

    # Perform evaluation
    print("\n Evaluating model on test set...")
    metrics = evaluator.evaluate()

    # Save comprehensive results
    results = {
        'dataset': args.dataset,
        'model_path': str(args.model_path),
        'test_metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'roc_auc': float(metrics['roc_auc'])
        },
        'per_class_metrics': metrics['per_class_metrics'],
        'confusion_matrix': metrics['confusion_matrix'],
        'num_misclassified': metrics['num_misclassified'],
        'total_samples': metrics['total_samples']
    }

    # Save results
    results_path = config['evaluation']['output_dir'] / f'evaluation_results_{args.dataset}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Detailed results saved to {config['evaluation']['output_dir']}")
    print(f"[OK] Evaluation complete!")

    return metrics


if __name__ == '__main__':
    metrics = main()