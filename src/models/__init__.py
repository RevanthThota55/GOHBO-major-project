"""
Model architectures for medical image classification
"""

from .resnet18_medical import MedicalResNet18, create_model

__all__ = [
    'MedicalResNet18',
    'create_model'
]