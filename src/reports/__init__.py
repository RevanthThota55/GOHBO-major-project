"""
Clinical report generation module for medical image classification.

Generates professional PDF reports with:
- Patient information
- Diagnosis with confidence
- Grad-CAM visualization
- AI reasoning explanation
- Recommended actions
"""

from .report_generator import ClinicalReportGenerator

__all__ = ['ClinicalReportGenerator']