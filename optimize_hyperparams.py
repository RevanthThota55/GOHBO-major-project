"""
Hyperparameter Optimization Script using Smart Grid Search

What is hyperparameter optimization?
-------------------------------------
Hyperparameters are settings that control how the model learns. Think of them like
settings on a kitchen mixer:
- Learning rate = speed setting (how fast to mix)
- Batch size = how much batter to mix at once
- Epochs = how many times to mix the whole bowl

This script tries different combinations of these settings to find which ones
make the model perform best.

What is grid search?
--------------------
Grid search is like testing different combinations systematically. Instead of
guessing randomly, we test specific values we think might work well.

For example:
- Learning rate: Try 0.0001, 0.0005, 0.001, 0.005, 0.01
- Batch size: Try 16, 32, 64
- Epochs: Try 30, 50, 80, 100

That gives us 5  3  4 = 60 possible combinations. We'll test up to 50 of them,
stopping early if we find a great combination (95% accuracy or higher).
"""

import argparse
import sys
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Tuple

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, PHASE1_OUTPUT, MODELS_DIR
from src.models.resnet18_medical import MedicalResNet18


def get_dataloaders(dataset_name: str, config: dict, batch_size: int):
    """
    Get data loaders for the specified dataset with a specific batch size.

    What are data loaders?
    ----------------------
    Data loaders are like conveyor belts that feed images to the model in batches.
    Instead of loading all images at once (which would fill up memory), they load
    small batches one at a time.

    Args:
        dataset_name: Which dataset to use ('brain_tumor', 'chest_xray', or 'colorectal')
        config: Configuration dictionary
        batch_size: How many images per batch (16, 32, or 64)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Import the appropriate dataset loader
    if dataset_name == 'brain_tumor':
        from src.datasets.brain_tumor import create_brain_tumor_dataloaders
        return create_brain_tumor_dataloaders(
            data_dir=config['dataset']['data_path'],
            batch_size=batch_size,
            num_workers=2
        )
    elif dataset_name == 'chest_xray':
        from src.datasets.chest_xray import get_chest_xray_loaders
        return get_chest_xray_loaders(
            data_dir=config['dataset']['data_path'],
            batch_size=batch_size,
            num_workers=2,
            augment=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_subset_loader(loader: DataLoader, fraction: float) -> DataLoader:
    """
    Create a smaller version of a data loader for faster testing.

    Why use a subset?
    -----------------
    During optimization, we test many combinations. If we used ALL the training
    data for each test, it would take forever! Instead, we use a small subset
    (like 20% of the data) to get a quick estimate of performance.

    Think of it like taste-testing: you don't need to eat the whole cake to know
    if the recipe is good - a small bite is enough.

    Args:
        loader: Original data loader
        fraction: What fraction to keep (0.2 = 20%)

    Returns:
        New data loader with only the subset of data
    """
    dataset = loader.dataset
    num_samples = len(dataset)
    num_subset = int(num_samples * fraction)

    # Pick random indices for the subset
    indices = np.random.choice(num_samples, num_subset, replace=False)
    subset = Subset(dataset, indices)

    # Create new loader with the subset
    subset_loader = DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 workers for speed during optimization
        pin_memory=loader.pin_memory
    )

    return subset_loader


def evaluate_hyperparameters(
    learning_rate: float,
    batch_size: int,
    epochs: int,
    config: dict,
    dataset_name: str,
    device: torch.device
) -> Dict:
    """
    Test a specific combination of hyperparameters.

    What this does:
    ---------------
    1. Creates a fresh model (no previous training)
    2. Trains it with the given hyperparameters for 5 epochs (quick test)
    3. Evaluates it on validation data
    4. Returns the results (accuracy and loss)

    Args:
        learning_rate: How fast the model learns (e.g., 0.001)
        batch_size: How many images per update (16, 32, or 64)
        epochs: How many epochs we WOULD use for full training (we only do 5 here)
        config: Configuration dictionary
        dataset_name: Which dataset we're using
        device: CPU or GPU

    Returns:
        Dictionary with results: val_loss, val_acc, train_loss
    """
    print(f"\n{'='*70}")
    print(f"Testing: LR={learning_rate:.6f}, Batch={batch_size}, Epochs={epochs}")
    print(f"{'='*70}")

    # Get data loaders with the specific batch size we're testing
    train_loader, val_loader, _ = get_dataloaders(dataset_name, config, batch_size)

    # Create subset loaders for faster evaluation
    # We use 20% of training data and 30% of validation data
    subset_train = create_subset_loader(train_loader, 0.2)
    subset_val = create_subset_loader(val_loader, 0.3)

    # Create a brand new model
    # "Brand new" means it hasn't learned anything yet - random weights
    model = MedicalResNet18(
        num_classes=config['dataset']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)

    # Create optimizer with the learning rate we're testing
    # The optimizer is what actually updates the model's weights
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,  # THIS is what we're testing!
        weight_decay=config['training']['optimizer']['weight_decay']
    )

    # Loss function measures how wrong the model's predictions are
    # CrossEntropyLoss is standard for classification problems
    criterion = nn.CrossEntropyLoss()

    # Quick training for 5 epochs
    # We only do 5 epochs here to save time. The "epochs" parameter above
    # is what we WOULD use if this combination wins.
    num_eval_epochs = 5
    model.train()  # Set model to training mode

    print(f"Training for {num_eval_epochs} epochs (quick evaluation)...")

    for epoch in range(num_eval_epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Loop through batches of training data
        for images, labels in subset_train:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: model makes predictions
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass: calculate how to improve
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights

            # Track statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Calculate average training loss and accuracy
        train_loss /= len(subset_train)
        train_acc = 100.0 * train_correct / train_total

        print(f"  Epoch {epoch+1}/{num_eval_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")

    # Now evaluate on validation data (data the model hasn't seen during training)
    # This tells us how well the model generalizes to new data
    print("\nEvaluating on validation set...")
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # torch.no_grad() means "don't calculate gradients" - saves memory
    with torch.no_grad():
        for images, labels in subset_val:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(subset_val)
    val_acc = 100.0 * val_correct / val_total

    print(f"\n[CHART] Results: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    # Return all the important metrics
    return {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'train_loss': train_loss,
        'train_acc': train_acc
    }


def smart_grid_search(
    config: dict,
    dataset_name: str,
    device: torch.device,
    max_iterations: int = 50
) -> Tuple[Dict, List[Dict]]:
    """
    Search for the best hyperparameters using smart grid search.

    How smart grid search works:
    ----------------------------
    Instead of testing ALL possible combinations (60 total), we:
    1. Start with some promising combinations
    2. Test them one by one
    3. If we find one with 95%+ accuracy, STOP (we found a winner!)
    4. Otherwise, keep testing up to max_iterations combinations
    5. Return the best one we found

    Why "smart"?
    ------------
    We shuffle the combinations randomly, so we're not testing in a boring order.
    This increases our chances of finding a good combination early.

    Args:
        config: Configuration dictionary
        dataset_name: Which dataset to optimize for
        device: CPU or GPU
        max_iterations: Maximum combinations to test (default 50)

    Returns:
        Tuple of (best_hyperparameters_dict, all_results_list)
    """
    print("\n" + "="*70)
    print("SMART GRID SEARCH HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    print("\nWhat we're optimizing:")
    print("  1. Learning Rate: How fast the model learns")
    print("  2. Batch Size: How many images per update")
    print("  3. Epochs: How many times to go through all training data")
    print()
    print("Target: Find settings that give >=95% validation accuracy")
    print("="*70)

    # Define the search space
    # These are the specific values we'll test for each hyperparameter
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    batch_sizes = [16, 32, 64]
    epoch_options = [30, 50, 80, 100]

    # Generate all possible combinations
    combinations = []
    for lr in learning_rates:
        for bs in batch_sizes:
            for ep in epoch_options:
                combinations.append({
                    'learning_rate': lr,
                    'batch_size': bs,
                    'epochs': ep
                })

    # Shuffle combinations so we try them in random order
    # This increases chance of finding a good one early
    random.shuffle(combinations)

    print(f"\nTotal possible combinations: {len(combinations)}")
    print(f"Will test up to: {min(max_iterations, len(combinations))} combinations")
    print(f"Early stopping: If accuracy >= {config['gohbo']['target_accuracy']*100}%, we'll stop\n")

    # Track results
    all_results = []
    best_result = None
    best_acc = 0.0
    target_acc = config['gohbo']['target_accuracy'] * 100  # Convert to percentage

    # Test combinations one by one
    for i, combo in enumerate(combinations[:max_iterations], 1):
        print(f"\n Testing combination {i}/{min(max_iterations, len(combinations))}")

        # Evaluate this combination
        result = evaluate_hyperparameters(
            learning_rate=combo['learning_rate'],
            batch_size=combo['batch_size'],
            epochs=combo['epochs'],
            config=config,
            dataset_name=dataset_name,
            device=device
        )

        all_results.append(result)

        # Check if this is the best so far
        if result['val_acc'] > best_acc:
            best_acc = result['val_acc']
            best_result = result
            print(f"\n NEW BEST! Accuracy: {best_acc:.2f}%")

        # Early stopping: if we hit target accuracy, stop searching
        if result['val_acc'] >= target_acc:
            print(f"\n{'='*70}")
            print(f"[DONE] TARGET ACHIEVED! Found settings with {result['val_acc']:.2f}% accuracy!")
            print(f"Stopping early (tested {i} out of {max_iterations} combinations)")
            print(f"{'='*70}")
            break

    return best_result, all_results


def save_best_hyperparameters(best_result: Dict, all_results: List[Dict], output_dir: Path):
    """
    Save the best hyperparameters in TWO formats: JSON and Python.

    Why two formats?
    ----------------
    - JSON: Easy for programs to read, and humans can read it too
    - Python: Can be imported directly into other Python scripts

    Args:
        best_result: Dictionary with the best hyperparameters and their results
        all_results: List of all combinations tested
        output_dir: Where to save the files (results/phase1/)
    """
    # Make sure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Format 1: JSON
    # JSON is a standard format that both humans and computers can read easily
    json_path = output_dir / 'best_hyperparameters.json'

    json_data = {
        'best_hyperparameters': {
            'learning_rate': best_result['learning_rate'],
            'batch_size': best_result['batch_size'],
            'epochs': best_result['epochs']
        },
        'validation_results': {
            'accuracy': best_result['val_acc'],
            'loss': best_result['val_loss']
        },
        'training_results': {
            'accuracy': best_result.get('train_acc', 0.0),
            'loss': best_result['train_loss']
        },
        'optimization_summary': {
            'num_combinations_tested': len(all_results),
            'target_accuracy': 95.0,
            'target_met': best_result['val_acc'] >= 95.0,
            'timestamp': datetime.now().isoformat()
        },
        'all_tested_combinations': all_results[:10]  # Save top 10 to keep file small
    }

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\n Saved JSON format to: {json_path}")

    # Format 2: Python
    # Python format can be imported directly: from best_hyperparameters import BEST_LEARNING_RATE
    py_path = output_dir / 'best_hyperparameters.py'

    py_content = f'''"""
Best Hyperparameters Found by Grid Search Optimization

These are the hyperparameters that gave the best validation accuracy.
You can import these values directly into other Python scripts.

Example:
    from results.phase1.best_hyperparameters import BEST_LEARNING_RATE

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Best hyperparameters
# These gave {best_result['val_acc']:.2f}% validation accuracy

# Learning Rate: How fast the model learns
# Higher = learns faster but might overshoot the optimal solution
# Lower = learns slower but more precisely
BEST_LEARNING_RATE = {best_result['learning_rate']}

# Batch Size: How many images the model looks at before updating weights
# Larger = more stable updates but uses more memory
# Smaller = more frequent updates but noisier
BEST_BATCH_SIZE = {best_result['batch_size']}

# Epochs: How many times to go through ALL training data
# More epochs = more learning but risk of overfitting
BEST_EPOCHS = {best_result['epochs']}

# Results achieved with these hyperparameters
VALIDATION_ACCURACY = {best_result['val_acc']:.2f}  # percentage
VALIDATION_LOSS = {best_result['val_loss']:.4f}
TRAINING_LOSS = {best_result['train_loss']:.4f}

# Target status
TARGET_ACCURACY = 95.0  # We aimed for 95% or higher
TARGET_MET = {str(best_result['val_acc'] >= 95.0)}  # Did we achieve it?
'''

    with open(py_path, 'w') as f:
        f.write(py_content)

    print(f" Saved Python format to: {py_path}")


def main():
    """
    Main function that runs the optimization.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Optimize hyperparameters using smart grid search'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='brain_tumor',
        choices=['brain_tumor', 'chest_xray', 'colorectal'],
        help='Dataset to optimize for'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=50,
        help='Maximum number of combinations to test (default: 50)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (cuda for GPU, cpu for CPU)'
    )

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.dataset)

    # Update max_iterations if provided
    if args.iterations:
        config['gohbo']['max_iterations'] = args.iterations

    # Determine device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING]  Warning: CUDA not available, falling back to CPU")
        device = torch.device('cpu')

    print(f"\n  Using device: {device}")
    if device.type == 'cpu':
        print("   Note: CPU training is slower. GPU recommended but not required.")

    # Check if dataset exists
    if not config['dataset']['data_path'].exists():
        print(f"\n ERROR: Dataset not found at {config['dataset']['data_path']}")
        print("Please download the dataset first.")
        sys.exit(1)

    # Run optimization
    print("\n[START] Starting hyperparameter optimization...")
    best_result, all_results = smart_grid_search(
        config=config,
        dataset_name=args.dataset,
        device=device,
        max_iterations=config['gohbo']['max_iterations']
    )

    # Print final results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print("\n Best Hyperparameters Found:")
    print(f"   Learning Rate: {best_result['learning_rate']:.6f}")
    print(f"   Batch Size:    {best_result['batch_size']}")
    print(f"   Epochs:        {best_result['epochs']}")
    print(f"\n[CHART] Performance:")
    print(f"   Validation Accuracy: {best_result['val_acc']:.2f}%")
    print(f"   Validation Loss:     {best_result['val_loss']:.4f}")

    if best_result['val_acc'] >= 95.0:
        print(f"\n SUCCESS! Achieved target accuracy of >=95%")
    else:
        print(f"\n[WARNING]  Target of 95% not reached, but found best possible settings")
        print(f"   (May achieve 95% with full training on complete dataset)")

    # Save results
    save_best_hyperparameters(best_result, all_results, PHASE1_OUTPUT)

    # Print next steps
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Train the model with optimized hyperparameters:")
    print(f"   python train.py --dataset {args.dataset} --use_optimized")
    print()
    print("2. The training script will automatically load the best hyperparameters from:")
    print(f"   {PHASE1_OUTPUT / 'best_hyperparameters.json'}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
