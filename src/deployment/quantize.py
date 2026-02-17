"""
Post-training quantization for model compression and acceleration.

Implements INT8 quantization to reduce model size by ~4x and improve
inference speed on CPU devices while maintaining accuracy.

Supports:
- Static quantization (requires calibration data)
- Dynamic quantization (weights only)
- Accuracy validation
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path
import copy
from tqdm import tqdm
import time


class ModelQuantizer:
    """
    Post-training quantization for PyTorch models.

    Reduces model size and improves CPU inference speed through INT8 quantization.
    Includes calibration and validation to ensure acceptable accuracy loss.

    Example:
        >>> quantizer = ModelQuantizer(model, calibration_loader)
        >>> quantized_model = quantizer.quantize_static()
        >>> accuracy_loss = quantizer.validate_quantized_model(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_loader: Optional[DataLoader] = None,
        device: str = 'cpu'
    ):
        """
        Initialize model quantizer.

        Args:
            model: PyTorch model to quantize (should be on CPU)
            calibration_loader: DataLoader for calibration (required for static quantization)
            device: Device for validation ('cpu' recommended for quantized models)

        Note:
            Model will be moved to CPU for quantization as quantization
            is primarily designed for CPU deployment.
        """
        self.original_model = model.cpu()  # Quantization requires CPU
        self.calibration_loader = calibration_loader
        self.device = 'cpu'  # Force CPU for quantized models
        self.quantized_model = None

    def prepare_calibration_data(
        self,
        full_loader: DataLoader,
        num_samples: int = 100
    ) -> DataLoader:
        """
        Prepare calibration dataset from full dataset.

        Args:
            full_loader: Full training DataLoader
            num_samples: Number of samples for calibration (default: 100)

        Returns:
            Calibration DataLoader

        Example:
            >>> calib_loader = quantizer.prepare_calibration_data(train_loader, 100)
        """
        dataset = full_loader.dataset

        # Randomly sample indices
        total_samples = len(dataset)
        indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)

        # Create subset
        calib_dataset = Subset(dataset, indices)

        # Create calibration loader
        calib_loader = DataLoader(
            calib_dataset,
            batch_size=full_loader.batch_size,
            shuffle=False,
            num_workers=0  # Use single worker for calibration
        )

        print(f"✓ Calibration dataset created: {len(calib_dataset)} samples")
        return calib_loader

    def quantize_dynamic(self) -> nn.Module:
        """
        Apply dynamic quantization (weights only, activations at runtime).

        Faster to apply but less optimized than static quantization.
        Good for models with dynamic input shapes.

        Returns:
            Dynamically quantized model

        Example:
            >>> quantized_model = quantizer.quantize_dynamic()
        """
        print("\nApplying dynamic quantization...")

        # Create a copy of the model
        model_to_quantize = copy.deepcopy(self.original_model)
        model_to_quantize.eval()

        # Apply dynamic quantization to linear layers
        quantized_model = quant.quantize_dynamic(
            model_to_quantize,
            {nn.Linear, nn.Conv2d},  # Quantize these layer types
            dtype=torch.qint8
        )

        self.quantized_model = quantized_model
        print("✓ Dynamic quantization complete")

        return quantized_model

    def quantize_static(
        self,
        backend: str = 'fbgemm'
    ) -> nn.Module:
        """
        Apply static quantization (weights and activations).

        Provides best optimization but requires calibration data.
        Recommended for production deployment.

        Args:
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)

        Returns:
            Statically quantized model

        Example:
            >>> quantized_model = quantizer.quantize_static(backend='fbgemm')

        Note:
            Requires calibration_loader to be provided during initialization.
        """
        if self.calibration_loader is None:
            raise ValueError("Calibration loader required for static quantization. "
                           "Use prepare_calibration_data() first.")

        print(f"\nApplying static quantization (backend: {backend})...")

        # Create a copy of the model
        model_to_quantize = copy.deepcopy(self.original_model)
        model_to_quantize.eval()

        # Set quantization backend
        torch.backends.quantized.engine = backend

        # Specify quantization config
        qconfig = quant.get_default_qconfig(backend)
        model_to_quantize.qconfig = qconfig

        # Prepare model for quantization
        print("  Preparing model...")
        model_prepared = quant.prepare(model_to_quantize, inplace=False)

        # Calibration phase
        print(f"  Calibrating with {len(self.calibration_loader)} batches...")
        model_prepared.eval()

        with torch.no_grad():
            for images, _ in tqdm(self.calibration_loader, desc="  Calibrating"):
                images = images.to(self.device)
                _ = model_prepared(images)

        # Convert to quantized model
        print("  Converting to quantized model...")
        quantized_model = quant.convert(model_prepared, inplace=False)

        self.quantized_model = quantized_model
        print("✓ Static quantization complete")

        return quantized_model

    def validate_quantized_model(
        self,
        test_loader: DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Validate quantized model and compare with original.

        Args:
            test_loader: Test DataLoader
            criterion: Loss criterion (default: CrossEntropyLoss)

        Returns:
            Dictionary containing validation metrics for both models

        Example:
            >>> results = quantizer.validate_quantized_model(test_loader)
            >>> print(f"Accuracy loss: {results['accuracy_loss']:.2%}")
        """
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Run quantize_static() or quantize_dynamic() first.")

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        print("\nValidating quantized model...")

        # Validate original model
        print("  Evaluating original model...")
        original_results = self._evaluate_model(self.original_model, test_loader, criterion)

        # Validate quantized model
        print("  Evaluating quantized model...")
        quantized_results = self._evaluate_model(self.quantized_model, test_loader, criterion)

        # Calculate differences
        accuracy_loss = original_results['accuracy'] - quantized_results['accuracy']
        speedup = original_results['inference_time'] / quantized_results['inference_time']

        # Get model sizes
        original_size = self._get_model_size(self.original_model)
        quantized_size = self._get_model_size(self.quantized_model)
        size_reduction = 1 - (quantized_size / original_size)

        comparison = {
            'original': original_results,
            'quantized': quantized_results,
            'accuracy_loss': accuracy_loss,
            'accuracy_loss_pct': accuracy_loss * 100,
            'speedup': speedup,
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'size_reduction': size_reduction,
            'size_reduction_pct': size_reduction * 100
        }

        # Print summary
        self._print_validation_summary(comparison)

        return comparison

    def _evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        model.eval()
        model = model.to(self.device)

        total_loss = 0.0
        correct = 0
        total = 0
        inference_times = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="    Evaluating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Measure inference time
                start_time = time.time()
                outputs = model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # Calculate loss and accuracy
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        avg_inference_time = np.mean(inference_times)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'inference_time': avg_inference_time,
            'throughput': len(dataloader.dataset) / sum(inference_times)  # images/sec
        }

    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        # Save model to temporary buffer
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_mb = buffer.tell() / (1024 * 1024)
        return size_mb

    def _print_validation_summary(self, comparison: Dict[str, Any]):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("QUANTIZATION VALIDATION RESULTS")
        print("=" * 60)

        print("\nOriginal Model:")
        print(f"  Accuracy: {comparison['original']['accuracy']:.2%}")
        print(f"  Loss: {comparison['original']['loss']:.4f}")
        print(f"  Inference Time: {comparison['original']['inference_time']*1000:.2f} ms/batch")
        print(f"  Throughput: {comparison['original']['throughput']:.1f} images/sec")
        print(f"  Model Size: {comparison['original_size_mb']:.2f} MB")

        print("\nQuantized Model:")
        print(f"  Accuracy: {comparison['quantized']['accuracy']:.2%}")
        print(f"  Loss: {comparison['quantized']['loss']:.4f}")
        print(f"  Inference Time: {comparison['quantized']['inference_time']*1000:.2f} ms/batch")
        print(f"  Throughput: {comparison['quantized']['throughput']:.1f} images/sec")
        print(f"  Model Size: {comparison['quantized_size_mb']:.2f} MB")

        print("\nComparison:")
        print(f"  Accuracy Loss: {comparison['accuracy_loss_pct']:.2f}%", end="")
        if comparison['accuracy_loss_pct'] < 2.0:
            print(" ✓ (Acceptable)")
        else:
            print(" ⚠ (High - consider recalibration)")

        print(f"  Speedup: {comparison['speedup']:.2f}x")
        print(f"  Size Reduction: {comparison['size_reduction_pct']:.1f}%")

        print("=" * 60)

    def save_quantized_model(self, save_path: Path):
        """
        Save quantized model to file.

        Args:
            save_path: Path to save the quantized model

        Example:
            >>> quantizer.save_quantized_model(Path('models/quantized_model.pth'))
        """
        if self.quantized_model is None:
            raise ValueError("No quantized model to save. Run quantization first.")

        torch.save(self.quantized_model.state_dict(), save_path)
        print(f"✓ Quantized model saved to {save_path}")

        # Also save entire model (for easier loading)
        save_path_full = save_path.parent / (save_path.stem + '_full.pth')
        torch.save(self.quantized_model, save_path_full)
        print(f"✓ Full quantized model saved to {save_path_full}")

    @staticmethod
    def load_quantized_model(model_path: Path) -> nn.Module:
        """
        Load a quantized model.

        Args:
            model_path: Path to the quantized model

        Returns:
            Loaded quantized model

        Example:
            >>> model = ModelQuantizer.load_quantized_model(Path('models/quantized_model.pth'))
        """
        model = torch.load(model_path)
        model.eval()
        print(f"✓ Quantized model loaded from {model_path}")
        return model


def quantize_model(
    model: nn.Module,
    calibration_loader: DataLoader,
    test_loader: DataLoader,
    save_path: Optional[Path] = None,
    quantization_type: str = 'static',
    backend: str = 'fbgemm'
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function to quantize a model with validation.

    Args:
        model: Model to quantize
        calibration_loader: Calibration data (for static quantization)
        test_loader: Test data for validation
        save_path: Optional path to save quantized model
        quantization_type: 'static' or 'dynamic'
        backend: Quantization backend ('fbgemm' or 'qnnpack')

    Returns:
        Tuple of (quantized_model, validation_results)

    Example:
        >>> quantized_model, results = quantize_model(
        ...     model, calib_loader, test_loader,
        ...     save_path=Path('models/quantized.pth')
        ... )
    """
    # Initialize quantizer
    quantizer = ModelQuantizer(model, calibration_loader)

    # Quantize
    if quantization_type == 'static':
        quantized_model = quantizer.quantize_static(backend=backend)
    elif quantization_type == 'dynamic':
        quantized_model = quantizer.quantize_dynamic()
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")

    # Validate
    results = quantizer.validate_quantized_model(test_loader)

    # Save if path provided
    if save_path:
        quantizer.save_quantized_model(save_path)

    return quantized_model, results