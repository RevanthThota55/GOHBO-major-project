"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for medical image classification.

Provides visual explanations of CNN predictions by highlighting important regions
in the input image that contribute to the model's decision.

Reference:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for generating visual explanations.

    This class hooks into the last convolutional layer of a CNN to capture
    feature maps and gradients, then generates a heatmap showing which
    regions of the input image were most important for the prediction.

    Attributes:
        model: The neural network model
        target_layer: The convolutional layer to hook into
        gradients: Stored gradients from backward pass
        activations: Stored activations from forward pass

    Example:
        >>> model = MedicalResNet18(num_classes=4)
        >>> gradcam = GradCAM(model, target_layer='layer4')
        >>> heatmap = gradcam.generate_heatmap(image, class_idx=1)
        >>> overlay = gradcam.overlay_heatmap(image, heatmap)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str = 'layer4',
        device: str = 'cuda'
    ):
        """
        Initialize Grad-CAM.

        Args:
            model: PyTorch model (should be in eval mode)
            target_layer: Name of the target convolutional layer
                         For ResNet-18: 'layer4' (last conv block)
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.eval()

        # Storage for gradients and activations
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Get target layer
        self.target_layer = self._get_target_layer(target_layer)

        # Register hooks
        self._register_hooks()

    def _get_target_layer(self, layer_name: str) -> nn.Module:
        """
        Get the target layer from the model.

        Args:
            layer_name: Name of the layer

        Returns:
            The target layer module
        """
        # For MedicalResNet18, access through backbone
        if hasattr(self.model, 'backbone'):
            if hasattr(self.model.backbone, layer_name):
                return getattr(self.model.backbone, layer_name)

        # Fallback: search in model
        for name, module in self.model.named_modules():
            if name.endswith(layer_name):
                return module

        raise ValueError(f"Layer {layer_name} not found in model")

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            """Store activations from forward pass."""
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            """Store gradients from backward pass."""
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_image: torch.Tensor,
        class_idx: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a given input image.

        Args:
            input_image: Input tensor of shape (1, C, H, W) or (C, H, W)
            class_idx: Target class index. If None, uses predicted class
            normalize: Whether to normalize heatmap to [0, 1]

        Returns:
            Heatmap as numpy array of shape (H, W)

        Example:
            >>> image = torch.randn(1, 3, 224, 224)
            >>> heatmap = gradcam.generate_heatmap(image, class_idx=0)
        """
        # Ensure input is 4D
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)

        input_image = input_image.to(self.device)
        input_image.requires_grad = True

        # Forward pass
        output = self.model(input_image)

        # Get target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Calculate weights using global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Convert to numpy
        cam = cam.cpu().numpy()

        # Normalize if requested
        if normalize:
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            image: Original image as numpy array (H, W, C) in range [0, 255] or [0, 1]
            heatmap: Grad-CAM heatmap of shape (H, W) in range [0, 1]
            alpha: Transparency of heatmap overlay (0 = transparent, 1 = opaque)
            colormap: OpenCV colormap to use for heatmap

        Returns:
            Overlaid image as numpy array (H, W, C) in range [0, 255]

        Example:
            >>> image = np.random.rand(224, 224, 3)  # Original image
            >>> heatmap = gradcam.generate_heatmap(torch_image)
            >>> overlay = gradcam.overlay_heatmap(image, heatmap)
        """
        # Normalize image to [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Convert heatmap to colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            colormap
        )

        # Convert BGR to RGB (OpenCV uses BGR)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Ensure image is RGB
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Overlay heatmap on image
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

        return overlaid

    def generate_multiple_heatmaps(
        self,
        images: torch.Tensor,
        class_indices: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Generate heatmaps for multiple images.

        Args:
            images: Batch of images (B, C, H, W)
            class_indices: List of target class indices. If None, uses predicted classes

        Returns:
            List of heatmaps
        """
        heatmaps = []

        for i in range(images.size(0)):
            class_idx = class_indices[i] if class_indices else None
            heatmap = self.generate_heatmap(images[i:i+1], class_idx=class_idx)
            heatmaps.append(heatmap)

        return heatmaps


def generate_gradcam_visualization(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    class_names: List[str],
    true_label: int,
    device: str = 'cuda',
    save_path: Optional[Path] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate a complete Grad-CAM visualization with predictions.

    This is a convenience function that creates a comprehensive visualization
    showing the original image, heatmap, overlay, and prediction information.

    Args:
        model: Trained model
        image: Preprocessed image tensor (1, C, H, W)
        original_image: Original image as numpy array (H, W, C)
        class_names: List of class names
        true_label: Ground truth label
        device: Device to run on
        save_path: Optional path to save the visualization

    Returns:
        Tuple of (overlaid image, info dictionary)

    Example:
        >>> overlay, info = generate_gradcam_visualization(
        ...     model, image_tensor, original_image,
        ...     class_names=['Normal', 'Tumor'],
        ...     true_label=1
        ... )
    """
    # Initialize Grad-CAM
    gradcam = GradCAM(model, device=device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image.to(device))
        probabilities = F.softmax(output, dim=1)
        pred_prob, pred_class = torch.max(probabilities, 1)

    # Generate heatmap for predicted class
    heatmap = gradcam.generate_heatmap(image, class_idx=pred_class.item())

    # Create overlay
    overlay = gradcam.overlay_heatmap(original_image, heatmap)

    # Prepare info dictionary
    info = {
        'predicted_class': class_names[pred_class.item()],
        'predicted_idx': pred_class.item(),
        'confidence': pred_prob.item(),
        'true_class': class_names[true_label],
        'true_idx': true_label,
        'is_correct': pred_class.item() == true_label,
        'all_probabilities': {
            class_names[i]: probabilities[0, i].item()
            for i in range(len(class_names))
        }
    }

    # Create visualization if save path provided
    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title(f'Original\nTrue: {info["true_class"]}')
        axes[0].axis('off')

        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(overlay)
        title = f'Prediction: {info["predicted_class"]}\nConfidence: {info["confidence"]:.2%}'
        if info['is_correct']:
            axes[2].set_title(title, color='green')
        else:
            axes[2].set_title(title, color='red')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return overlay, info


def create_gradcam_grid(
    model: nn.Module,
    images: torch.Tensor,
    original_images: List[np.ndarray],
    class_names: List[str],
    true_labels: List[int],
    device: str = 'cuda',
    grid_size: Tuple[int, int] = (4, 4),
    save_path: Optional[Path] = None
) -> None:
    """
    Create a grid visualization of multiple Grad-CAM results.

    Args:
        model: Trained model
        images: Batch of preprocessed images (B, C, H, W)
        original_images: List of original images
        class_names: List of class names
        true_labels: List of ground truth labels
        device: Device to run on
        grid_size: Number of rows and columns in grid
        save_path: Path to save the grid visualization

    Example:
        >>> create_gradcam_grid(
        ...     model, images, originals,
        ...     class_names=['Normal', 'Tumor'],
        ...     true_labels=[0, 1, 1, 0],
        ...     grid_size=(2, 2)
        ... )
    """
    num_images = min(grid_size[0] * grid_size[1], len(original_images))

    # Initialize Grad-CAM
    gradcam = GradCAM(model, device=device)

    # Create figure
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1]*4, grid_size[0]*4))
    axes = axes.flatten() if num_images > 1 else [axes]

    for idx in range(num_images):
        # Get prediction
        model.eval()
        with torch.no_grad():
            output = model(images[idx:idx+1].to(device))
            probabilities = F.softmax(output, dim=1)
            pred_prob, pred_class = torch.max(probabilities, 1)

        # Generate heatmap and overlay
        heatmap = gradcam.generate_heatmap(images[idx:idx+1], class_idx=pred_class.item())
        overlay = gradcam.overlay_heatmap(original_images[idx], heatmap)

        # Plot
        axes[idx].imshow(overlay)
        title = f'True: {class_names[true_labels[idx]]}\n'
        title += f'Pred: {class_names[pred_class.item()]} ({pred_prob.item():.1%})'

        color = 'green' if pred_class.item() == true_labels[idx] else 'red'
        axes[idx].set_title(title, color=color, fontsize=10)
        axes[idx].axis('off')

    # Hide remaining subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Grad-CAM Visualizations', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Grad-CAM grid saved to {save_path}")

    plt.show()
    plt.close()