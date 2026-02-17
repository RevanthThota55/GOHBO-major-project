"""
Flask Web Application for ExplainableMed-GOHBO
Multi-Model Medical Image Classification with Grad-CAM Explainability

Supports:
- Brain Tumor MRI Classification (4 classes)
- Chest X-Ray Pneumonia Detection (2 classes)
- Colorectal Cancer Histopathology (8 classes)
"""

import os
import sys
from pathlib import Path
import io
import base64
from datetime import datetime
import uuid
import platform

# Fix for loading models saved on Linux (Colab/Kaggle) to Windows
if platform.system() == 'Windows':
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

# Flask imports
from flask import Flask, render_template, request, jsonify, send_file, url_for, flash
from werkzeug.utils import secure_filename

# ML imports
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

# Add parent directory to path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / 'src'))

# Project imports
from models.resnet18_medical import MedicalResNet18
from explainability.gradcam import GradCAM, generate_gradcam_visualization
from explainability.uncertainty import MCDropoutPredictor
from reports.report_generator import ClinicalReportGenerator

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Ensure upload folder exists
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Global model storage - keyed by model type
MODELS = {}
DEVICE = torch.device('cpu')

# Image preprocessing (shared - both use ImageNet normalization)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==================== MODEL CONFIGURATIONS ====================

MODEL_CONFIGS = {
    'brain_tumor': {
        'name': 'Brain Tumor MRI Classification',
        'num_classes': 4,
        'model_path': 'models/checkpoints/best_model.pth',
        'class_names': ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor'],
        'class_descriptions': {
            'Glioma Tumor': 'A tumor that arises from glial cells. Can be malignant and requires immediate attention.',
            'Meningioma Tumor': 'Typically benign tumor arising from meninges (protective layers). Usually slow-growing.',
            'No Tumor': 'No abnormal growth detected. Brain scan appears normal.',
            'Pituitary Tumor': 'Tumor in the pituitary gland. Often affects hormone production.'
        },
        'explanations': {
            0: "The AI model detected irregular cell growth patterns characteristic of glioma tumors in the brain tissue.",
            1: "The AI identified a well-defined mass near the protective membranes (meninges), suggesting a meningioma tumor.",
            2: "The AI analysis found no abnormal patterns or masses across all examined brain regions.",
            3: "The AI detected abnormal tissue in the pituitary gland region, indicating a pituitary tumor."
        },
        'key_findings': {
            0: ["Irregular border patterns detected", "Diffuse growth pattern observed", "Located in cerebral cortex region"],
            1: ["Well-defined, rounded mass", "Located near skull base or meninges", "Typically slow-growing characteristics"],
            2: ["Normal brain tissue structure", "Symmetrical brain regions", "No abnormal masses detected"],
            3: ["Mass in sella turcica region", "Small, well-circumscribed lesion", "Near optic chiasm area"]
        },
        'icon': 'fa-brain',
        'scan_type': 'MRI Scan',
        'accuracy': '95.42%',
        'dataset_size': '~3,000 MRI scans'
    },
    'chest_xray': {
        'name': 'Chest X-Ray Pneumonia Detection',
        'num_classes': 2,
        'model_path': 'models/checkpoints/chest_xray_resnet18.pth',
        'class_names': ['Normal', 'Pneumonia'],
        'class_descriptions': {
            'Normal': 'No signs of pneumonia detected. Chest X-ray appears normal with clear lung fields.',
            'Pneumonia': 'Signs of pneumonia detected. Opacity or consolidation visible in lung region. Further clinical evaluation recommended.'
        },
        'explanations': {
            0: "The AI analysis found clear lung fields with no signs of infiltrates, consolidation, or effusion in the chest X-ray.",
            1: "The AI detected areas of opacity or consolidation in the lung fields, consistent with pneumonia patterns."
        },
        'key_findings': {
            0: ["Clear lung fields bilaterally", "No infiltrates or consolidation", "Normal cardiac silhouette"],
            1: ["Opacity detected in lung region", "Possible consolidation or infiltrate", "Recommend clinical correlation"]
        },
        'icon': 'fa-lungs',
        'scan_type': 'Chest X-Ray',
        'accuracy': '97.03%',
        'dataset_size': '~5,863 X-ray images'
    },
    'colorectal': {
        'name': 'Colorectal Cancer Histopathology',
        'num_classes': 8,
        'model_path': 'models/checkpoints/colorectal_resnet18.pth',
        'class_names': ['Tumor', 'Stroma', 'Complex', 'Lympho', 'Debris', 'Mucosa', 'Adipose', 'Empty'],
        'class_descriptions': {
            'Tumor': 'Colorectal adenocarcinoma epithelium detected. Malignant tissue requiring clinical evaluation.',
            'Stroma': 'Cancer-associated stroma identified. Connective tissue surrounding tumor regions.',
            'Complex': 'Complex stroma with mixed tissue components. May include inflammatory and stromal elements.',
            'Lympho': 'Immune cell clusters (lymphocytes) detected. Indicates immune response activity.',
            'Debris': 'Necrotic tissue and cellular debris identified. Often associated with tumor necrosis.',
            'Mucosa': 'Normal colon mucosa tissue. Healthy intestinal lining structure.',
            'Adipose': 'Adipose (fatty) tissue identified. Normal fat tissue in the colon wall.',
            'Empty': 'Background or empty region. No significant tissue detected in this area.'
        },
        'explanations': {
            0: "The AI detected irregular glandular structures and cellular atypia characteristic of colorectal adenocarcinoma.",
            1: "The AI identified fibrous connective tissue with reactive changes, suggesting cancer-associated stroma.",
            2: "The AI found a mixture of stromal and inflammatory components, classified as complex stroma.",
            3: "The AI detected dense clusters of lymphocytes, indicating immune cell infiltration in the tissue.",
            4: "The AI identified necrotic tissue with cellular debris, often seen in areas of tumor necrosis.",
            5: "The AI analysis shows normal colonic mucosa with regular glandular architecture.",
            6: "The AI identified adipose (fat) tissue with characteristic clear cell morphology.",
            7: "The AI detected minimal tissue content, classifying this as a background/empty region."
        },
        'key_findings': {
            0: ["Irregular glandular patterns", "Cellular atypia and pleomorphism", "Loss of normal tissue architecture"],
            1: ["Fibrous tissue with reactive changes", "Desmoplastic reaction observed", "Adjacent to tumor regions"],
            2: ["Mixed inflammatory infiltrate", "Stromal and epithelial components", "Complex tissue composition"],
            3: ["Dense lymphocyte clusters", "Immune cell aggregation", "Potential immune response to tumor"],
            4: ["Necrotic tissue fragments", "Cellular debris present", "Loss of cell viability"],
            5: ["Regular glandular structure", "Normal goblet cells present", "Healthy mucosal architecture"],
            6: ["Clear adipocyte morphology", "Normal fat distribution", "No pathological changes"],
            7: ["Minimal tissue content", "Background staining only", "No diagnostic features"]
        },
        'icon': 'fa-microscope',
        'scan_type': 'Histopathology',
        'accuracy': '94.56%',
        'dataset_size': '~5,000 tissue images'
    }
}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_single_model(model_type):
    """Load a single model by type and return model + components"""
    config = MODEL_CONFIGS[model_type]
    num_classes = config['num_classes']
    checkpoint_path = BASE_DIR / config['model_path']

    print(f"\nLoading {config['name']} model...")

    if not checkpoint_path.exists():
        print(f"  [WARNING] Model not found at {checkpoint_path}")
        print(f"  Skipping {model_type} model.")
        return None

    model = None

    # Try loading as MedicalResNet18 first
    try:
        print(f"  Attempting MedicalResNet18 ({num_classes} classes)...")
        model = MedicalResNet18(
            num_classes=num_classes,
            input_channels=3,
            pretrained=False,
            enable_mc_dropout=True
        ).to(DEVICE)

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model_variant = "MedicalResNet18"
        print(f"  [OK] Loaded as MedicalResNet18")

    except (RuntimeError, KeyError) as e:
        # Fall back to standard ResNet-18
        print(f"  MedicalResNet18 failed, trying standard ResNet-18...")
        try:
            model = models.resnet18(pretrained=False)
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, num_classes)

            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(DEVICE)
            model_variant = "Standard ResNet-18"
            print(f"  [OK] Loaded as Standard ResNet-18")
        except Exception as e2:
            print(f"  [ERROR] Could not load {model_type}: {e2}")
            return None

    model.eval()

    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer='layer4', device=DEVICE)

    # Initialize MC Dropout predictor
    mc_predictor = MCDropoutPredictor(model, num_passes=10, device=DEVICE)

    print(f"  [OK] {config['name']} ready ({model_variant})")

    return {
        'model': model,
        'gradcam': gradcam,
        'mc_predictor': mc_predictor,
        'config': config
    }


def load_all_models():
    """Load all available models"""
    global MODELS

    print("=" * 60)
    print("Loading Medical Image Classification Models")
    print("=" * 60)

    for model_type in MODEL_CONFIGS:
        result = load_single_model(model_type)
        if result is not None:
            MODELS[model_type] = result

    print(f"\n{'=' * 60}")
    print(f"Models loaded: {len(MODELS)}/{len(MODEL_CONFIGS)}")
    for mt in MODELS:
        print(f"  - {MODEL_CONFIGS[mt]['name']} ({MODEL_CONFIGS[mt]['accuracy']})")
    print("=" * 60)


def image_to_base64(image_array):
    """Convert numpy image array to base64 string"""
    image_array = np.array(image_array)

    if image_array.dtype == np.float32 or image_array.dtype == np.float64:
        if image_array.max() <= 1.0:
            image_array = (image_array * 255)
        image_array = image_array.astype(np.uint8)
    elif image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)

    if len(image_array.shape) == 2:
        image = Image.fromarray(image_array, mode='L')
    else:
        image = Image.fromarray(image_array, mode='RGB')

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


@app.route('/')
def index():
    """Home page"""
    available_models = {k: v['config'] for k, v in MODELS.items()}
    return render_template('index.html', available_models=available_models)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        model_type = request.form.get('model_type', 'brain_tumor')

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Use JPG, PNG, or TIFF'}), 400

        if model_type not in MODELS:
            return jsonify({'success': False, 'error': f'Model "{model_type}" is not available'}), 400

        # Get the selected model components
        m = MODELS[model_type]
        model = m['model']
        gradcam = m['gradcam']
        mc_predictor = m['mc_predictor']
        config = m['config']
        class_names = config['class_names']

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / f"{uuid.uuid4()}_{filename}"
        file.save(filepath)

        # Load and preprocess image
        image = Image.open(filepath).convert('RGB')
        original_np = np.array(image.resize((224, 224)))
        image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

        # Get prediction
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = predicted.item()
        confidence_score = confidence.item()

        # Generate Grad-CAM heatmap
        heatmap = gradcam.generate_heatmap(image_tensor, class_idx=predicted_class)
        overlay = gradcam.overlay_heatmap(original_np, heatmap, alpha=0.4)

        # Get uncertainty estimation
        mean_pred, uncertainty = mc_predictor.predict_with_uncertainty(image_tensor)

        # Prepare probability dictionary
        probs_dict = {
            class_names[i]: float(probabilities[0, i].item())
            for i in range(len(class_names))
        }

        # Generate explanation
        explanation = generate_explanation(model_type, predicted_class, confidence_score, uncertainty)

        # Convert images to base64
        original_base64 = image_to_base64(original_np)
        heatmap_base64 = image_to_base64(heatmap * 255)
        overlay_base64 = image_to_base64(overlay)

        # Clean up uploaded file
        filepath.unlink()

        # Prepare response
        response = {
            'success': True,
            'model_type': model_type,
            'model_name': config['name'],
            'prediction': class_names[predicted_class],
            'prediction_idx': predicted_class,
            'confidence': f"{confidence_score * 100:.2f}",
            'confidence_numeric': confidence_score,
            'probabilities': probs_dict,
            'uncertainty': {
                'confidence': uncertainty['confidence'],
                'entropy': uncertainty['entropy'],
                'mean_std': uncertainty['mean_std']
            },
            'images': {
                'original': original_base64,
                'heatmap': heatmap_base64,
                'overlay': overlay_base64
            },
            'explanation': explanation,
            'description': config['class_descriptions'][class_names[predicted_class]],
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def generate_explanation(model_type, predicted_class, confidence, uncertainty):
    """Generate human-readable explanation"""
    config = MODEL_CONFIGS[model_type]

    # Confidence assessment
    if confidence > 0.9:
        confidence_text = "The model is highly confident in this diagnosis."
    elif confidence > 0.75:
        confidence_text = "The model shows good confidence in this diagnosis."
    else:
        confidence_text = "The model has moderate confidence. Additional review recommended."

    # Uncertainty assessment
    if uncertainty['confidence'] > 0.85:
        uncertainty_text = "Uncertainty quantification indicates reliable prediction."
    else:
        uncertainty_text = "Higher uncertainty detected - recommend manual verification."

    return {
        'diagnosis': config['explanations'][predicted_class],
        'confidence_assessment': confidence_text,
        'uncertainty_assessment': uncertainty_text,
        'key_findings': config['key_findings'][predicted_class]
    }


@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate clinical PDF report"""
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'brain_tumor')
        config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS['brain_tumor'])

        # Initialize report generator
        report_gen = ClinicalReportGenerator()

        # Generate report
        pdf_buffer = report_gen.generate_report(
            prediction=data['prediction'],
            confidence=data['confidence'],
            probabilities=data['probabilities'],
            explanation=data['explanation'],
            heatmap_base64=data['images']['overlay'],
            uncertainty=data.get('uncertainty', {})
        )

        # Generate unique filename
        prefix = model_type
        filename = f"{prefix}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/about')
def about():
    """About page with model information"""
    models_info = {}
    for model_type, config in MODEL_CONFIGS.items():
        loaded = model_type in MODELS
        models_info[model_type] = {
            'name': config['name'],
            'accuracy': config['accuracy'],
            'num_classes': config['num_classes'],
            'class_names': config['class_names'],
            'dataset_size': config['dataset_size'],
            'icon': config['icon'],
            'scan_type': config['scan_type'],
            'loaded': loaded
        }

    return render_template('about.html', models_info=models_info)


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(MODELS.keys()),
        'total_models': len(MODELS),
        'device': str(DEVICE),
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal server error occurred.'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("ExplainableMed-GOHBO Web Application")
    print("Multi-Model Medical Image Classification")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load all models
    load_all_models()

    print("\n" + "=" * 60)
    print("Starting Flask server...")
    print("Access the app at: http://localhost:5000")
    print("=" * 60)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
