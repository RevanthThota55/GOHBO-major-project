"""
Deployment module for model optimization and export.

Provides tools for deploying models to edge devices:
- Post-training quantization (INT8) for size reduction
- ONNX export for cross-platform compatibility
- Model optimization and benchmarking
"""

from .quantize import ModelQuantizer, quantize_model
from .export_onnx import ONNXExporter, export_and_verify

__all__ = [
    'ModelQuantizer',
    'quantize_model',
    'ONNXExporter',
    'export_and_verify'
]