"""
Visualization utilities for medical image classification.

Provides functions for visualizing training progress, model predictions, and results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import cv2


def plot_training_history(
    history: Dict[str, List],
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot training history including loss and accuracy curves.

    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot training and validation loss
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot training and validation accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot learning rate schedule
    axes[1, 0].plot(epochs, history['learning_rates'], 'g-')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot validation metrics over time
    val_loss_smooth = smooth_curve(history['val_loss'], window=5)
    val_acc_smooth = smooth_curve(history['val_acc'], window=5)

    axes[1, 1].plot(epochs, history['val_loss'], 'r-', alpha=0.3, label='Val Loss (Raw)')
    axes[1, 1].plot(epochs, val_loss_smooth, 'r-', label='Val Loss (Smooth)')
    ax2 = axes[1, 1].twinx()
    ax2.plot(epochs, history['val_acc'], 'b-', alpha=0.3, label='Val Acc (Raw)')
    ax2.plot(epochs, val_acc_smooth, 'b-', label='Val Acc (Smooth)')

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss', color='r')
    ax2.set_ylabel('Accuracy (%)', color='b')
    axes[1, 1].set_title('Smoothed Validation Metrics')
    axes[1, 1].tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='b')
    axes[1, 1].grid(True, alpha=0.3)

    # Add legend
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.suptitle('Training History', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def smooth_curve(values: List[float], window: int = 5) -> List[float]:
    """
    Smooth a curve using moving average.

    Args:
        values: List of values to smooth
        window: Window size for moving average

    Returns:
        Smoothed values
    """
    if len(values) < window:
        return values

    smoothed = []
    for i in range(len(values)):
        start_idx = max(0, i - window // 2)
        end_idx = min(len(values), i + window // 2 + 1)
        smoothed.append(np.mean(values[start_idx:end_idx]))

    return smoothed


def visualize_batch(
    dataloader: DataLoader,
    class_names: List[str],
    num_images: int = 16,
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Visualize a batch of images from a dataloader.

    Args:
        dataloader: DataLoader to get images from
        class_names: List of class names
        num_images: Number of images to display
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    # Get a batch
    images, labels = next(iter(dataloader))
    num_images = min(num_images, len(images))

    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_images)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_images):
        img = images[i].cpu().numpy()
        label = labels[i].item()

        # Denormalize if needed (assuming ImageNet normalization)
        img = img.transpose(1, 2, 0)  # CHW to HWC
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f'{class_names[label]}', fontsize=10)
        axes[i].axis('off')

    # Hide remaining subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Sample Images from Dataset', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Batch visualization saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_class_activation_maps(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    device: str = 'cuda',
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot Class Activation Maps (CAM) for model predictions.

    Args:
        model: Trained model with get_cam method
        images: Batch of images
        labels: True labels
        class_names: List of class names
        device: Device to use
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    model.eval()
    model.to(device)
    images = images.to(device)

    num_images = min(8, len(images))
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        # Get predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Get CAMs
        cams = model.get_cam(images[:num_images])

    for i in range(num_images):
        # Original image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original\nTrue: {class_names[labels[i]]}')
        axes[i, 0].axis('off')

        # CAM
        cam = cams[i].cpu().numpy()
        axes[i, 1].imshow(cam, cmap='jet')
        axes[i, 1].set_title(f'CAM\nPred: {class_names[preds[i]]}')
        axes[i, 1].axis('off')

        # Overlay
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        cam_colored = plt.cm.jet(cam_resized)[:, :, :3]
        overlay = 0.6 * img + 0.4 * cam_colored

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')

    plt.suptitle('Class Activation Maps', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ CAM visualization saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_gohbo_convergence(
    convergence_history: List[float],
    diversity_history: List[float],
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot GOHBO optimization convergence.

    Args:
        convergence_history: List of best fitness values over iterations
        diversity_history: List of population diversity over iterations
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iterations = range(1, len(convergence_history) + 1)

    # Plot convergence
    axes[0].plot(iterations, convergence_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Best Fitness (Validation Loss)')
    axes[0].set_title('GOHBO Convergence')
    axes[0].grid(True, alpha=0.3)

    # Add improvement annotation
    initial = convergence_history[0]
    final = convergence_history[-1]
    improvement = ((initial - final) / initial) * 100
    axes[0].annotate(
        f'Improvement: {improvement:.1f}%',
        xy=(len(convergence_history), final),
        xytext=(len(convergence_history) * 0.7, (initial + final) / 2),
        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
        fontsize=12,
        color='red'
    )

    # Plot diversity
    axes[1].plot(iterations, diversity_history, 'g-', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Population Diversity')
    axes[1].set_title('Population Diversity Over Time')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('GOHBO Optimization Progress', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ GOHBO convergence plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_comparison_plot(
    results_dict: Dict[str, Dict],
    metric: str = 'accuracy',
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Create comparison plot for different models/datasets.

    Args:
        results_dict: Dictionary of results {name: metrics_dict}
        metric: Metric to compare
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    names = list(results_dict.keys())
    values = [results_dict[name][metric] for name in names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(names))))

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom')

    plt.xlabel('Model/Dataset')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()