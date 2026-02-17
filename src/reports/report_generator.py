"""
Clinical Report Generator for Brain Tumor Classification

Generates professional PDF reports with diagnosis, Grad-CAM visualizations,
and AI reasoning explanations for medical professionals.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from datetime import datetime
import io
import base64
from PIL import Image
from typing import Dict, Any, Optional
import uuid


class ClinicalReportGenerator:
    """
    Generates clinical PDF reports for brain tumor classification results.

    Features:
    - Professional medical report layout
    - Embedded Grad-CAM visualizations
    - AI reasoning explanation
    - Recommended actions
    - Disclaimer for research use

    Example:
        >>> generator = ClinicalReportGenerator()
        >>> pdf_buffer = generator.generate_report(
        ...     prediction='Glioma Tumor',
        ...     confidence='95.30',
        ...     probabilities={...},
        ...     explanation={...},
        ...     heatmap_base64='data:image/png;base64,...'
        ... )
    """

    def __init__(self):
        """Initialize report generator with styles."""
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Create custom paragraph styles for the report."""
        # Only add styles if they don't already exist (prevents conflicts)

        # Title style
        if 'CustomTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#2563EB'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))

        # Section header style
        if 'SectionHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#1F2937'),
                spaceBefore=20,
                spaceAfter=12,
                fontName='Helvetica-Bold',
                borderWidth=0,
                borderColor=colors.HexColor('#2563EB'),
                borderPadding=5,
                leftIndent=0
            ))

        # Body text style (use existing 'Normal' style or create 'MedicalBody')
        if 'MedicalBody' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='MedicalBody',
                parent=self.styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#374151'),
                spaceAfter=12,
                alignment=TA_JUSTIFY,
                leading=16
            ))

        # Disclaimer style
        if 'MedicalDisclaimer' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='MedicalDisclaimer',
                parent=self.styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#EF4444'),
                spaceAfter=6,
                alignment=TA_JUSTIFY,
                leading=12,
                fontName='Helvetica-Oblique'
            ))

    def generate_report(
        self,
        prediction: str,
        confidence: str,
        probabilities: Dict[str, float],
        explanation: Dict[str, Any],
        heatmap_base64: str,
        uncertainty: Optional[Dict[str, float]] = None,
        patient_id: Optional[str] = None
    ) -> io.BytesIO:
        """
        Generate clinical PDF report.

        Args:
            prediction: Predicted class name
            confidence: Confidence percentage as string
            probabilities: Dictionary of class probabilities
            explanation: Explanation dictionary with diagnosis, assessment, findings
            heatmap_base64: Base64-encoded Grad-CAM overlay image
            uncertainty: Optional uncertainty metrics from MC Dropout
            patient_id: Optional patient ID (auto-generated if not provided)

        Returns:
            BytesIO buffer containing the PDF

        Example:
            >>> pdf = generator.generate_report(
            ...     prediction='Glioma Tumor',
            ...     confidence='95.30',
            ...     probabilities={'Glioma': 0.953, ...},
            ...     explanation={'diagnosis': '...', ...},
            ...     heatmap_base64='data:image/png;base64,...'
            ... )
        """
        # Create PDF buffer
        buffer = io.BytesIO()

        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Container for elements
        elements = []

        # Generate patient ID if not provided
        if not patient_id:
            patient_id = f"P-{uuid.uuid4().hex[:8].upper()}"

        # Add header
        elements.extend(self._create_header(patient_id))

        # Add scan information
        elements.extend(self._create_scan_info())

        # Add diagnosis section
        elements.extend(self._create_diagnosis_section(prediction, confidence))

        # Add Grad-CAM visualization
        if heatmap_base64:
            elements.extend(self._create_visualization_section(heatmap_base64))

        # Add probabilities table
        elements.extend(self._create_probabilities_table(probabilities))

        # Add AI analysis section
        elements.extend(self._create_analysis_section(explanation))

        # Add uncertainty section if available
        if uncertainty:
            elements.extend(self._create_uncertainty_section(uncertainty))

        # Add recommendations
        elements.extend(self._create_recommendations(prediction))

        # Add model information
        elements.extend(self._create_model_info())

        # Add disclaimer
        elements.extend(self._create_disclaimer())

        # Build PDF
        doc.build(elements)

        # Reset buffer position
        buffer.seek(0)

        return buffer

    def _create_header(self, patient_id: str) -> list:
        """Create report header"""
        elements = []

        # Title
        title = Paragraph(
            "BRAIN TUMOR CLASSIFICATION REPORT",
            self.styles['CustomTitle']
        )
        elements.append(title)

        # Subtitle
        subtitle = Paragraph(
            "<b>Generated by ExplainableMed-GOHBO AI System</b>",
            self.styles['MedicalBody']
        )
        subtitle.alignment = TA_CENTER
        elements.append(subtitle)

        elements.append(Spacer(1, 0.3 * inch))

        # Report info table
        report_data = [
            ['Report ID:', f"RPT-{uuid.uuid4().hex[:12].upper()}"],
            ['Patient ID:', patient_id],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Report Type:', 'Brain MRI Analysis']
        ]

        report_table = Table(report_data, colWidths=[2*inch, 4*inch])
        report_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6B7280')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#1F2937')),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))

        elements.append(report_table)
        elements.append(Spacer(1, 0.4 * inch))

        return elements

    def _create_scan_info(self) -> list:
        """Create scan information section"""
        elements = []

        section_header = Paragraph("SCAN INFORMATION", self.styles['SectionHeader'])
        elements.append(section_header)

        scan_info = Paragraph(
            "<b>Scan Type:</b> Brain MRI (T1-weighted)<br/>"
            "<b>Image Dimensions:</b> 224 x 224 pixels<br/>"
            "<b>Processing:</b> Pre-processed with ImageNet normalization<br/>"
            "<b>Analysis Method:</b> Deep Learning with Grad-CAM Explainability",
            self.styles['MedicalBody']
        )
        elements.append(scan_info)
        elements.append(Spacer(1, 0.2 * inch))

        return elements

    def _create_diagnosis_section(self, prediction: str, confidence: str) -> list:
        """Create diagnosis section"""
        elements = []

        section_header = Paragraph("DIAGNOSIS", self.styles['SectionHeader'])
        elements.append(section_header)

        # Diagnosis table
        diagnosis_data = [
            ['Classification:', prediction],
            ['Confidence Level:', f"{confidence}%"],
            ['Status:', self._get_diagnosis_status(prediction)]
        ]

        diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
        diagnosis_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6B7280')),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor('#2563EB')),
            ('FONTSIZE', (1, 0), (1, 0), 14),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))

        elements.append(diagnosis_table)
        elements.append(Spacer(1, 0.2 * inch))

        return elements

    def _get_diagnosis_status(self, prediction: str) -> str:
        """Get diagnosis status"""
        if 'No Tumor' in prediction:
            return 'Normal - No abnormalities detected'
        else:
            return 'Abnormality Detected - Further evaluation recommended'

    def _create_visualization_section(self, heatmap_base64: str) -> list:
        """Create Grad-CAM visualization section"""
        elements = []

        section_header = Paragraph("GRAD-CAM VISUALIZATION", self.styles['SectionHeader'])
        elements.append(section_header)

        description = Paragraph(
            "The heatmap below shows the regions of the brain scan that the AI model focused on "
            "to make its classification decision. Red/yellow areas indicate high importance regions, "
            "while blue areas had less influence on the diagnosis.",
            self.styles['MedicalBody']
        )
        elements.append(description)
        elements.append(Spacer(1, 0.2 * inch))

        # Convert base64 to image
        try:
            # Remove data URL prefix if present
            if 'base64,' in heatmap_base64:
                heatmap_base64 = heatmap_base64.split('base64,')[1]

            # Decode base64
            image_data = base64.b64decode(heatmap_base64)
            image_buffer = io.BytesIO(image_data)

            # Add image to PDF
            img = RLImage(image_buffer, width=4*inch, height=4*inch)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 0.2 * inch))

        except Exception as e:
            error_text = Paragraph(
                f"<i>Visualization unavailable: {str(e)}</i>",
                self.styles['MedicalBody']
            )
            elements.append(error_text)

        return elements

    def _create_probabilities_table(self, probabilities: Dict[str, float]) -> list:
        """Create probabilities table"""
        elements = []

        section_header = Paragraph("CLASS PROBABILITIES", self.styles['SectionHeader'])
        elements.append(section_header)

        # Prepare table data
        table_data = [['Class Name', 'Probability', 'Percentage']]

        for class_name, prob in probabilities.items():
            percentage = f"{prob * 100:.2f}%"
            table_data.append([class_name, f"{prob:.4f}", percentage])

        # Create table
        prob_table = Table(table_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        prob_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#374151')),

            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
        ]))

        elements.append(prob_table)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_analysis_section(self, explanation: Dict[str, Any]) -> list:
        """Create AI analysis section"""
        elements = []

        section_header = Paragraph("AI ANALYSIS & REASONING", self.styles['SectionHeader'])
        elements.append(section_header)

        # Diagnosis explanation
        diagnosis_text = Paragraph(
            f"<b>Diagnosis Explanation:</b><br/>{explanation.get('diagnosis', 'N/A')}",
            self.styles['MedicalBody']
        )
        elements.append(diagnosis_text)

        # Confidence assessment
        confidence_text = Paragraph(
            f"<b>Confidence Assessment:</b><br/>{explanation.get('confidence_assessment', 'N/A')}",
            self.styles['MedicalBody']
        )
        elements.append(confidence_text)

        # Uncertainty assessment
        uncertainty_text = Paragraph(
            f"<b>Uncertainty Analysis:</b><br/>{explanation.get('uncertainty_assessment', 'N/A')}",
            self.styles['MedicalBody']
        )
        elements.append(uncertainty_text)

        # Key findings
        findings_header = Paragraph("<b>Key Findings:</b>", self.styles['MedicalBody'])
        elements.append(findings_header)

        for finding in explanation.get('key_findings', []):
            finding_text = Paragraph(f"• {finding}", self.styles['MedicalBody'])
            elements.append(finding_text)

        elements.append(Spacer(1, 0.2 * inch))

        return elements

    def _create_uncertainty_section(self, uncertainty: Dict[str, float]) -> list:
        """Create uncertainty quantification section"""
        elements = []

        section_header = Paragraph("UNCERTAINTY QUANTIFICATION (MC DROPOUT)", self.styles['SectionHeader'])
        elements.append(section_header)

        uncertainty_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Confidence Score', f"{uncertainty.get('confidence', 0):.4f}", self._interpret_confidence(uncertainty.get('confidence', 0))],
            ['Predictive Entropy', f"{uncertainty.get('entropy', 0):.4f}", 'Lower is better'],
            ['Mean Std Deviation', f"{uncertainty.get('mean_std', 0):.4f}", 'Lower indicates consistency']
        ]

        uncertainty_table = Table(uncertainty_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        uncertainty_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8B5CF6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))

        elements.append(uncertainty_table)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence score"""
        if confidence >= 0.9:
            return 'High - Reliable'
        elif confidence >= 0.75:
            return 'Good - Acceptable'
        else:
            return 'Low - Review recommended'

    def _create_recommendations(self, prediction: str) -> list:
        """Create recommendations section"""
        elements = []

        section_header = Paragraph("RECOMMENDED ACTIONS", self.styles['SectionHeader'])
        elements.append(section_header)

        recommendations = {
            'Glioma Tumor': [
                'Immediate consultation with neuro-oncologist recommended',
                'Additional imaging (contrast MRI) may be beneficial',
                'Biopsy for tissue diagnosis and grading',
                'Multidisciplinary team evaluation'
            ],
            'Meningioma Tumor': [
                'Consultation with neurosurgeon for evaluation',
                'Monitor growth with follow-up scans (6-12 months)',
                'Assess for symptoms (headaches, vision changes)',
                'Consider treatment options if symptomatic'
            ],
            'No Tumor': [
                'No immediate action required based on this scan',
                'Continue routine health monitoring',
                'Consult physician if symptoms develop',
                'Routine follow-up as per physician guidance'
            ],
            'Pituitary Tumor': [
                'Endocrinologist consultation recommended',
                'Hormone level testing (prolactin, growth hormone, etc.)',
                'Visual field examination',
                'Follow-up imaging in 6-12 months'
            ]
        }

        rec_list = recommendations.get(prediction, ['Consult with medical professional for guidance'])

        for i, rec in enumerate(rec_list, 1):
            rec_text = Paragraph(f"{i}. {rec}", self.styles['MedicalBody'])
            elements.append(rec_text)

        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_model_info(self) -> list:
        """Create model information section"""
        elements = []

        section_header = Paragraph("MODEL INFORMATION", self.styles['SectionHeader'])
        elements.append(section_header)

        model_data = [
            ['Model Architecture:', 'ResNet-18 with Spatial Attention'],
            ['Optimization Method:', 'GOHBO (Hybrid Meta-heuristic)'],
            ['Training Accuracy:', '95.2%'],
            ['Validation Accuracy:', '94.8%'],
            ['Dataset Size:', '~3,000 MRI scans'],
            ['Model Version:', 'v1.0']
        ]

        model_table = Table(model_data, colWidths=[2.5*inch, 3*inch])
        model_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6B7280')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#1F2937')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))

        elements.append(model_table)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_disclaimer(self) -> list:
        """Create disclaimer section"""
        elements = []

        section_header = Paragraph("IMPORTANT DISCLAIMER", self.styles['SectionHeader'])
        elements.append(section_header)

        disclaimer_text = Paragraph(
            "<b>FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY</b><br/><br/>"
            "This report is generated by an artificial intelligence system for research and "
            "educational purposes only. It is NOT intended for clinical diagnosis, treatment decisions, "
            "or patient care. All results must be verified and interpreted by qualified medical "
            "professionals. This AI system should be used as a supplementary tool only, and not as "
            "a replacement for professional medical judgment.<br/><br/>"
            "The accuracy and reliability of this AI system have been evaluated on research datasets "
            "and may not reflect performance in real clinical settings. Clinical validation is required "
            "before any potential use in patient care.<br/><br/>"
            "Users assume all risks associated with the use of this system. The developers and "
            "associated institutions accept no liability for any consequences arising from the use "
            "of this AI system or this report.",
            self.styles['MedicalDisclaimer']
        )
        elements.append(disclaimer_text)

        return elements

    def _base64_to_image(self, base64_string: str) -> Optional[RLImage]:
        """Convert base64 string to ReportLab Image"""
        try:
            if 'base64,' in base64_string:
                base64_string = base64_string.split('base64,')[1]

            image_data = base64.b64decode(base64_string)
            image_buffer = io.BytesIO(image_data)

            return RLImage(image_buffer)

        except Exception as e:
            print(f"Error converting base64 to image: {e}")
            return None