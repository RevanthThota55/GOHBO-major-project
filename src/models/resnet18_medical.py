"""
ResNet-18 Model adapted for Medical Image Classification with MC Dropout support

This module implements a customized ResNet-18 architecture specifically
designed for medical image classification tasks with Monte Carlo Dropout
support for uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict, Any


class MedicalResNet18(nn.Module):
    """
    ResNet-18 architecture adapted for medical image classification.

    Features:
    - Pre-trained ImageNet weights as initialization
    - Custom classification head for medical imaging
    - Support for both grayscale and RGB images
    - Dropout and batch normalization for regularization
    - Attention mechanism for important feature focus
    - Monte Carlo Dropout for uncertainty quantification
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        hidden_units: list = [512, 256],
        use_attention: bool = True,
        enable_mc_dropout: bool = False
    ):
        """
        Initialize MedicalResNet18.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            pretrained: Whether to use ImageNet pre-trained weights
            freeze_backbone: Whether to freeze backbone during training
            dropout_rate: Dropout rate for regularization
            hidden_units: Hidden layer sizes in classifier head
            use_attention: Whether to use attention mechanism
            enable_mc_dropout: Enable Monte Carlo Dropout for uncertainty estimation
        """
        super(MedicalResNet18, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        self.mc_dropout_enabled = enable_mc_dropout

        # Load pre-trained ResNet-18
        self.backbone = models.resnet18(pretrained=pretrained)

        # Modify first conv layer if input is not 3 channels
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features

        # Remove the original fc layer
        self.backbone.fc = nn.Identity()

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Add attention mechanism if specified
        if use_attention:
            self.attention = SpatialAttention(num_features)

        # Custom classification head
        self.classifier = self._build_classifier(
            num_features,
            hidden_units,
            num_classes,
            dropout_rate
        )

        # Initialize weights for new layers
        self._initialize_weights()

    def _build_classifier(
        self,
        input_features: int,
        hidden_units: list,
        num_classes: int,
        dropout_rate: float
    ) -> nn.Sequential:
        """
        Build the classification head.

        Args:
            input_features: Number of input features
            hidden_units: List of hidden layer sizes
            num_classes: Number of output classes
            dropout_rate: Dropout probability

        Returns:
            Sequential model for classification
        """
        layers = []
        in_features = input_features

        # Add hidden layers
        for hidden_size in hidden_units:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_size

        # Add output layer
        layers.append(nn.Linear(in_features, num_classes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights for new layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_mc_dropout_mode(self, enabled: bool):
        """
        Enable or disable Monte Carlo Dropout mode.

        When MC Dropout is enabled, dropout layers remain active during inference
        for uncertainty estimation.

        Args:
            enabled: Whether to enable MC Dropout mode

        Example:
            >>> model.set_mc_dropout_mode(True)  # Enable for uncertainty estimation
            >>> model.set_mc_dropout_mode(False)  # Disable for standard inference
        """
        self.mc_dropout_enabled = enabled

        if enabled:
            # Enable dropout during inference
            for module in self.classifier.modules():
                if isinstance(module, nn.Dropout):
                    module.train()  # Keep dropout active
        else:
            # Disable dropout during inference (standard behavior)
            for module in self.classifier.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features using backbone
        features = self.extract_features(x)

        # Apply global average pooling
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        pooled = pooled.view(pooled.size(0), -1)

        # Apply attention if specified
        if self.use_attention:
            attention_weights = self.attention(features)
            attended = (features * attention_weights).sum(dim=[2, 3])
            pooled = pooled + attended

        # Enable/disable dropout based on MC mode
        if self.mc_dropout_enabled and not self.training:
            # Keep dropout active for MC dropout
            for module in self.classifier.modules():
                if isinstance(module, nn.Dropout):
                    module.train()

        # Classification
        output = self.classifier(pooled)

        return output

    def forward_with_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dropout enabled (for MC Dropout).

        This is an alias for forward() with MC dropout mode temporarily enabled.

        Args:
            x: Input tensor

        Returns:
            Output logits with dropout active
        """
        original_mode = self.mc_dropout_enabled
        self.set_mc_dropout_mode(True)
        output = self.forward(x)
        self.set_mc_dropout_mode(original_mode)
        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone.

        Args:
            x: Input tensor

        Returns:
            Feature tensor
        """
        # ResNet-18 forward pass until the last conv layer
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x

    def get_cam(self, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        """
        Generate Class Activation Map (CAM) for visualization.

        Args:
            x: Input tensor
            class_idx: Target class index (if None, use predicted class)

        Returns:
            CAM heatmap
        """
        features = self.extract_features(x)
        batch_size, num_channels, h, w = features.shape

        # Get classifier weights
        classifier_weights = list(self.classifier.parameters())[0]

        # Forward pass to get predictions
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).view(batch_size, -1)
        logits = self.classifier(pooled)

        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        # Generate CAM
        cam = torch.zeros(batch_size, h, w).to(x.device)

        for i in range(batch_size):
            weights = classifier_weights[class_idx[i]][:num_channels]
            cam[i] = (weights.view(-1, 1, 1) * features[i]).sum(dim=0)

        # Normalize CAM
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam

    def freeze_backbone(self):
        """Freeze the backbone for feature extraction."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for focusing on important regions.
    """

    def __init__(self, num_channels: int):
        """
        Initialize spatial attention.

        Args:
            num_channels: Number of input channels
        """
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(num_channels // 8, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.

        Args:
            x: Input feature tensor

        Returns:
            Attention weights
        """
        # Generate attention map
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = torch.sigmoid(attention)

        return attention


def create_model(
    dataset_name: str,
    config: Dict[str, Any],
    device: str = 'cuda',
    enable_mc_dropout: bool = False
) -> MedicalResNet18:
    """
    Create a MedicalResNet18 model for a specific dataset.

    Args:
        dataset_name: Name of the dataset ('brain_tumor', 'chest_xray', 'colorectal')
        config: Configuration dictionary
        device: Device to place model on
        enable_mc_dropout: Enable Monte Carlo Dropout

    Returns:
        Configured MedicalResNet18 model
    """
    dataset_config = config['dataset']
    model_config = config['model']

    # Create model
    model = MedicalResNet18(
        num_classes=dataset_config['num_classes'],
        input_channels=dataset_config['channels'],
        pretrained=model_config['pretrained'],
        freeze_backbone=model_config['freeze_backbone'],
        dropout_rate=model_config['dropout_rate'],
        hidden_units=model_config['hidden_units'],
        use_attention=True,
        enable_mc_dropout=enable_mc_dropout
    )

    # Move to device
    model = model.to(device)

    # Print model summary
    print(f"\nModel created for {dataset_config['name']}")
    print(f"Number of classes: {dataset_config['num_classes']}")
    print(f"Input channels: {dataset_config['channels']}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    if enable_mc_dropout:
        print("MC Dropout: ENABLED for uncertainty estimation")

    return model


class ModelCheckpoint:
    """
    Utility class for saving and loading model checkpoints.
    """

    def __init__(self, filepath: str, monitor: str = 'val_loss', mode: str = 'min'):
        """
        Initialize model checkpoint.

        Args:
            filepath: Path to save checkpoint
            monitor: Metric to monitor
            mode: 'min' for loss, 'max' for accuracy
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')

    def save_if_best(
        self,
        model: nn.Module,
        metric: float,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        additional_info: Optional[Dict] = None
    ) -> bool:
        """
        Save model if metric improved.

        Args:
            model: Model to save
            metric: Current metric value
            epoch: Current epoch
            optimizer: Optimizer state to save
            additional_info: Additional information to save

        Returns:
            True if model was saved
        """
        is_best = False

        if self.mode == 'min' and metric < self.best:
            is_best = True
            self.best = metric
        elif self.mode == 'max' and metric > self.best:
            is_best = True
            self.best = metric

        if is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_metric': self.best,
                'monitor': self.monitor
            }

            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            if additional_info is not None:
                checkpoint.update(additional_info)

            torch.save(checkpoint, self.filepath)
            print(f"Checkpoint saved: {self.monitor} = {metric:.4f}")

        return is_best

    @staticmethod
    def load_checkpoint(filepath: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint
            model: Model to load weights into
            optimizer: Optimizer to load state into

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint