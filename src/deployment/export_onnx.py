"""
ONNX Export and Verification for cross-platform deployment.

Exports PyTorch models to ONNX format for deployment on various platforms
including mobile, edge devices, and different deep learning frameworks.

Supports:
- ONNX export with dynamic batch size
- Verification of exported model
- ONNX Runtime inference
- Cross-platform compatibility
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import time


class ONNXExporter:
    """
    ONNX exporter for PyTorch models.

    Handles export, verification, and provides inference utilities for ONNX models.

    Example:
        >>> exporter = ONNXExporter(model)
        >>> exporter.export(save_path='model.onnx')
        >>> verified = exporter.verify_export(test_input)
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        device: str = 'cpu'
    ):
        """
        Initialize ONNX exporter.

        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape (B, C, H, W)
            device: Device to run model on

        Note:
            Model should be in eval mode and on CPU for ONNX export.
        """
        self.model = model.cpu().eval()  # ONNX export requires CPU
        self.input_shape = input_shape
        self.device = 'cpu'
        self.onnx_model_path = None
        self.ort_session = None

    def export(
        self,
        save_path: Path,
        opset_version: int = 14,
        dynamic_axes: Optional[Dict] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Path:
        """
        Export PyTorch model to ONNX format.

        Args:
            save_path: Path to save the ONNX model
            opset_version: ONNX opset version (default: 14)
            dynamic_axes: Dictionary specifying dynamic axes
            input_names: List of input names
            output_names: List of output names
            verbose: Whether to print verbose export information

        Returns:
            Path to saved ONNX model

        Example:
            >>> exporter.export(
            ...     save_path=Path('model.onnx'),
            ...     dynamic_axes={'input': {0: 'batch_size'}}
            ... )
        """
        print(f"\nExporting model to ONNX format...")
        print(f"  Opset version: {opset_version}")
        print(f"  Input shape: {self.input_shape}")

        # Create dummy input
        dummy_input = torch.randn(*self.input_shape, device=self.device)

        # Default names
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']

        # Default dynamic axes (allow variable batch size)
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }

        # Export to ONNX
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                str(save_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=verbose
            )

            self.onnx_model_path = save_path
            print(f"✓ Model exported successfully to {save_path}")

            # Verify ONNX model
            self._check_onnx_model(save_path)

            # Get model size
            model_size = save_path.stat().st_size / (1024 * 1024)
            print(f"✓ ONNX model size: {model_size:.2f} MB")

            return save_path

        except Exception as e:
            print(f"✗ Export failed: {str(e)}")
            raise

    def _check_onnx_model(self, model_path: Path):
        """
        Check ONNX model for correctness.

        Args:
            model_path: Path to ONNX model
        """
        try:
            # Load and check the model
            onnx_model = onnx.load(str(model_path))
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model check passed")

        except Exception as e:
            print(f"⚠ ONNX model check failed: {str(e)}")

    def verify_export(
        self,
        test_input: Optional[torch.Tensor] = None,
        tolerance: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Verify that ONNX model produces same output as PyTorch model.

        Args:
            test_input: Test input tensor. If None, uses random tensor
            tolerance: Maximum allowed difference between outputs

        Returns:
            Dictionary with verification results

        Example:
            >>> results = exporter.verify_export(test_input)
            >>> if results['verified']:
            ...     print("Export verified successfully!")
        """
        if self.onnx_model_path is None:
            raise ValueError("No ONNX model exported. Call export() first.")

        print("\nVerifying ONNX export...")

        # Create test input if not provided
        if test_input is None:
            test_input = torch.randn(*self.input_shape, device=self.device)

        # Get PyTorch output
        self.model.eval()
        with torch.no_grad():
            pytorch_output = self.model(test_input).cpu().numpy()

        # Get ONNX Runtime output
        onnx_output = self._run_onnx_inference(test_input.cpu().numpy())

        # Compare outputs
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        verified = max_diff < tolerance

        results = {
            'verified': verified,
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'tolerance': tolerance,
            'pytorch_output_shape': pytorch_output.shape,
            'onnx_output_shape': onnx_output.shape
        }

        # Print results
        if verified:
            print(f"✓ Verification passed")
            print(f"  Max difference: {max_diff:.2e}")
            print(f"  Mean difference: {mean_diff:.2e}")
        else:
            print(f"✗ Verification failed")
            print(f"  Max difference: {max_diff:.2e} (tolerance: {tolerance:.2e})")

        return results

    def _run_onnx_inference(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference using ONNX Runtime.

        Args:
            input_data: Input as numpy array

        Returns:
            Output as numpy array
        """
        # Create ONNX Runtime session if not exists
        if self.ort_session is None:
            self.ort_session = ort.InferenceSession(str(self.onnx_model_path))

        # Get input name
        input_name = self.ort_session.get_inputs()[0].name

        # Run inference
        outputs = self.ort_session.run(None, {input_name: input_data})

        return outputs[0]

    def benchmark_onnx(
        self,
        test_inputs: List[np.ndarray],
        warmup_runs: int = 10,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark ONNX Runtime inference.

        Args:
            test_inputs: List of test input arrays
            warmup_runs: Number of warmup runs
            num_runs: Number of benchmark runs

        Returns:
            Dictionary with benchmark results

        Example:
            >>> test_data = [np.random.randn(1, 3, 224, 224) for _ in range(100)]
            >>> benchmark = exporter.benchmark_onnx(test_data)
            >>> print(f"Average latency: {benchmark['mean_latency_ms']:.2f} ms")
        """
        if self.ort_session is None:
            self.ort_session = ort.InferenceSession(str(self.onnx_model_path))

        input_name = self.ort_session.get_inputs()[0].name

        # Warmup
        for _ in range(warmup_runs):
            _ = self.ort_session.run(None, {input_name: test_inputs[0]})

        # Benchmark
        latencies = []
        for i in range(num_runs):
            input_data = test_inputs[i % len(test_inputs)]

            start_time = time.time()
            _ = self.ort_session.run(None, {input_name: input_data})
            latency = (time.time() - start_time) * 1000  # Convert to ms

            latencies.append(latency)

        # Calculate statistics
        latencies = np.array(latencies)
        results = {
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'throughput_fps': 1000.0 / float(np.mean(latencies))
        }

        return results

    def generate_inference_code(
        self,
        save_path: Optional[Path] = None,
        language: str = 'python'
    ) -> str:
        """
        Generate example inference code for ONNX model.

        Args:
            save_path: Optional path to save the code
            language: Programming language ('python' or 'cpp')

        Returns:
            Generated code as string

        Example:
            >>> code = exporter.generate_inference_code(language='python')
            >>> print(code)
        """
        if language == 'python':
            code = f"""
# ONNX Runtime Inference Example (Python)
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load ONNX model
session = ort.InferenceSession('{self.onnx_model_path.name}')

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize({self.input_shape[2:]}),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load and preprocess image
image = Image.open('your_image.jpg')
input_tensor = preprocess(image).unsqueeze(0).numpy()

# Run inference
input_name = session.get_inputs()[0].name
output = session.run(None, {{input_name: input_tensor}})

# Get prediction
probabilities = output[0]
predicted_class = np.argmax(probabilities)
confidence = probabilities[0][predicted_class]

print(f"Predicted class: {{predicted_class}}")
print(f"Confidence: {{confidence:.2%}}")
"""
        elif language == 'cpp':
            code = f"""
// ONNX Runtime Inference Example (C++)
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>

int main() {{
    // Create ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "medical_classification");
    Ort::SessionOptions session_options;

    // Load model
    Ort::Session session(env, "{self.onnx_model_path.name}", session_options);

    // Load and preprocess image
    cv::Mat image = cv::imread("your_image.jpg");
    cv::resize(image, image, cv::Size({self.input_shape[3]}, {self.input_shape[2]}));

    // Convert to tensor (simplified - needs normalization)
    std::vector<float> input_tensor_values;
    // ... populate input_tensor_values ...

    // Create input tensor
    std::vector<int64_t> input_shape = {{1, {self.input_shape[1]}, {self.input_shape[2]}, {self.input_shape[3]}}};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Run inference
    const char* input_names[] = {{"input"}};
    const char* output_names[] = {{"output"}};
    auto output_tensors = session.Run(
        Ort::RunOptions{{nullptr}},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    // Get output
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    // ... process output ...

    return 0;
}}
"""
        else:
            raise ValueError(f"Unsupported language: {language}")

        if save_path:
            with open(save_path, 'w') as f:
                f.write(code)
            print(f"✓ Inference code saved to {save_path}")

        return code


def export_and_verify(
    model: nn.Module,
    save_path: Path,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 14,
    run_benchmark: bool = False
) -> Tuple[Path, Dict[str, Any]]:
    """
    Convenience function to export and verify ONNX model.

    Args:
        model: PyTorch model to export
        save_path: Path to save ONNX model
        input_shape: Input tensor shape
        opset_version: ONNX opset version
        run_benchmark: Whether to run benchmark

    Returns:
        Tuple of (onnx_path, verification_results)

    Example:
        >>> onnx_path, results = export_and_verify(
        ...     model,
        ...     Path('models/model.onnx')
        ... )
    """
    # Initialize exporter
    exporter = ONNXExporter(model, input_shape=input_shape)

    # Export
    onnx_path = exporter.export(save_path, opset_version=opset_version)

    # Verify
    verification_results = exporter.verify_export()

    # Benchmark if requested
    if run_benchmark:
        print("\nRunning benchmark...")
        test_inputs = [np.random.randn(*input_shape).astype(np.float32) for _ in range(100)]
        benchmark_results = exporter.benchmark_onnx(test_inputs)

        print(f"\nONNX Runtime Benchmark:")
        print(f"  Mean latency: {benchmark_results['mean_latency_ms']:.2f} ms")
        print(f"  P95 latency: {benchmark_results['p95_latency_ms']:.2f} ms")
        print(f"  Throughput: {benchmark_results['throughput_fps']:.1f} FPS")

        verification_results['benchmark'] = benchmark_results

    # Generate example code
    code_path = save_path.parent / (save_path.stem + '_inference.py')
    exporter.generate_inference_code(save_path=code_path, language='python')

    return onnx_path, verification_results