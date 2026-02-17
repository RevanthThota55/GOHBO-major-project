"""
Explainability module for medical image classification.

Provides tools for model interpretability and uncertainty quantification:
- Grad-CAM: Visual explanations of model decisions
- Monte Carlo Dropout: Uncertainty estimation
"""

from .gradcam import GradCAM, generate_gradcam_visualization
from .uncertainty import MCDropoutPredictor, UncertaintyAnalyzer

__all__ = [
    'GradCAM',
    'generate_gradcam_visualization',
    'MCDropoutPredictor',
    'UncertaintyAnalyzer'
]