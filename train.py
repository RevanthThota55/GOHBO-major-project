"""
Main training script for medical image classification with GOHBO-optimized ResNet-18.
"""

import argparse
import sys
from pathlib import Path
import torch
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config import get_config
from src.models.resnet18_medical import MedicalResNet18
from src.datasets.brain_tumor import create_brain_tumor_dataloaders
from src.datasets.chest_xray import create_chest_xray_dataloaders
from src.datasets.colorectal import create_colorectal_dataloaders
from src.training.trainer import Trainer


def get_dataloaders(dataset_name: str, config: dict):
    """
    Get appropriate dataloaders for the specified dataset.

    Args:
        dataset_name: Name of the dataset
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_path = config['dataset']['data_path']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    image_size = config['dataset']['image_size']
    augmentation_config = config['augmentation']['train']

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
    parser = argparse.ArgumentParser(description='Train Medical Image Classification Model')
    parser.add_argument('--dataset', type=str, default='brain_tumor',
                       choices=['brain_tumor', 'chest_xray', 'colorectal'],
                       help='Dataset to train on')
    parser.add_argument('--use_optimized', action='store_true',
                       help='Use optimized hyperparameters from grid search (recommended!)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (manual override)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (manual override)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (manual override)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--no_tensorboard', action='store_true',
                       help='Disable TensorBoard logging')

    args = parser.parse_args()

    # Get configuration
    config = get_config(args.dataset)

    learning_rate = None

    # T-4: Load optimized hyperparameters if requested
    if args.use_optimized:
        # Try to load from Phase 1 optimization results
        optimization_results = config['paths']['results'] / 'phase1' / 'best_hyperparameters.json'
        if optimization_results.exists():
            with open(optimization_results, 'r') as f:
                opt_results = json.load(f)
                opt_params = opt_results['best_hyperparameters']

                # Load all three optimized hyperparameters
                learning_rate = opt_params['learning_rate']
                config['training']['batch_size'] = opt_params['batch_size']
                config['training']['epochs'] = opt_params['epochs']

                print("\n" + "="*70)
                print(" USING OPTIMIZED HYPERPARAMETERS")
                print("="*70)
                print(f"Loaded from: {optimization_results}")
                print(f"  Learning Rate: {learning_rate:.6f}")
                print(f"  Batch Size:    {opt_params['batch_size']}")
                print(f"  Epochs:        {opt_params['epochs']}")
                if 'validation_results' in opt_results:
                    print(f"  Validation Accuracy Achieved: {opt_results['validation_results']['accuracy']:.2f}%")
                print("="*70 + "\n")
        else:
            print("\n[WARNING]  Warning: No optimization results found at {optimization_results}")
            print("   Run optimize_hyperparams.py first, or provide manual hyperparameters.")
            print("   Using config defaults for now.\n")

    # Manual overrides (these override even optimized values if provided)
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
        print(f"Manual override: Learning rate = {learning_rate:.6f}")

    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
        print(f"Manual override: Epochs = {args.epochs}")

    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        print(f"Manual override: Batch size = {args.batch_size}")

    if args.no_tensorboard:
        config['logging']['tensorboard']['enabled'] = False

    print("\n" + "=" * 70)
    print(f"MEDICAL IMAGE CLASSIFICATION TRAINING")
    print("=" * 70)
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Classes: {config['dataset']['num_classes']}")
    print(f"Device: {args.device}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    if learning_rate:
        print(f"Learning Rate: {learning_rate:.6f}")

    # Store learning rate in config for Trainer to use
    if learning_rate:
        config['optimized_learning_rate'] = learning_rate

    # Create dataloaders
    print("\n Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(args.dataset, config)

    # Create model
    print("\n Creating model...")
    model = MedicalResNet18(
        num_classes=config['dataset']['num_classes'],
        input_channels=config['dataset']['channels'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone'],
        dropout_rate=config['model']['dropout_rate'],
        hidden_units=config['model']['hidden_units']
    )

    print(f"Model parameters: {model.get_num_trainable_params():,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train model
    print("\n Starting training...")
    print("=" * 60)

    history = trainer.train(
        num_epochs=config['training']['epochs'],
        learning_rate=learning_rate
    )

    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")

    # Save configuration with results
    results = {
        'dataset': args.dataset,
        'best_val_acc': trainer.best_val_acc,
        'best_val_loss': trainer.best_val_loss,
        'final_train_acc': history['train_acc'][-1],
        'final_train_loss': history['train_loss'][-1],
        'learning_rate': learning_rate if learning_rate else config['training']['optimizer'].get('lr', 1e-3),
        'epochs_trained': len(history['train_loss']),
        'config': config
    }

    results_path = config['paths']['results'] / f'training_results_{args.dataset}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to {results_path}")
    print(f"[OK] Model checkpoints saved to {config['paths']['models'] / 'checkpoints'}")

    return trainer, history


if __name__ == '__main__':
    trainer, history = main()