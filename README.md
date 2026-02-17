# ExplainableMed-GOHBO

> Multi-Model Medical Image Classification with GOHBO Optimization, Grad-CAM Explainability, and Uncertainty Quantification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)

---

## Model Performance

| Model | Modality | Classes | Accuracy |
|-------|----------|---------|----------|
| Brain Tumor | MRI Scan | 4 (Glioma, Meningioma, No Tumor, Pituitary) | **95.42%** |
| Chest X-Ray | X-Ray | 2 (Normal, Pneumonia) | **97.03%** |
| Colorectal Cancer | Histopathology | 8 tissue types | **94.56%** |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/RevanthThota55/GOHBO-major-project.git
cd GOHBO-major-project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download model weights (~309 MB)

```bash
python download_models.py
```

This downloads the 3 pre-trained `.pth` files from GitHub Releases into `models/checkpoints/`.

**Manual download:** If the script fails, download from the [Releases page](https://github.com/RevanthThota55/GOHBO-major-project/releases/tag/v1.0-models) and place the files in `models/checkpoints/`.

### 3. Run the webapp

```bash
cd webapp
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Features

- **3 Specialized Models** — Brain tumor MRI, chest X-ray pneumonia, colorectal histopathology
- **GOHBO Optimization** — Hybrid meta-heuristic (Grey Wolf + Heap-Based + Orthogonal Learning) for hyperparameter tuning
- **Grad-CAM Heatmaps** — Visual explanations showing which regions influenced the AI's decision
- **MC Dropout Uncertainty** — Monte Carlo Dropout quantifies prediction confidence
- **Clinical PDF Reports** — Downloadable reports with diagnosis, heatmaps, and confidence scores
- **Dark Premium UI** — Modern medical-themed web interface

---

## Project Structure

```
GOHBO-major-project/
├── webapp/                     # Flask web application
│   ├── app.py                  # Main Flask app (routes, model loading, inference)
│   ├── templates/              # HTML templates (index, about, base)
│   └── static/                 # CSS, JS assets
├── src/                        # Core ML source code
│   ├── algorithms/             # GOHBO optimization (gwo.py, hbo.py, orthogonal.py, gohbo.py)
│   ├── models/                 # ResNet-18 with MC Dropout (resnet18_medical.py)
│   ├── explainability/         # Grad-CAM and uncertainty quantification
│   ├── reports/                # Clinical PDF report generator
│   ├── training/               # Training pipeline (trainer, evaluator, optimizer)
│   ├── datasets/               # Dataset loaders (brain_tumor, chest_xray, colorectal)
│   └── deployment/             # ONNX export and INT8 quantization
├── models/checkpoints/         # Model weights (download via download_models.py)
├── colab_chest_xray/           # Google Colab training notebook for chest X-ray
├── colab_colorectal/           # Google Colab training notebook for colorectal
├── notebooks/                  # Jupyter exploration notebooks
├── results/                    # Training metrics and evaluation plots
├── config.py                   # Project configuration
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── download_models.py          # Model weight downloader
└── requirements.txt            # Python dependencies
```

---

## Training Your Own Models

Pre-trained weights are provided, but you can retrain using Google Colab:

- **Brain Tumor**: `train_brain_tumor.py` (local) or original Colab notebook
- **Chest X-Ray**: `colab_chest_xray/COLAB_TRAINING_CHEST_XRAY.ipynb`
- **Colorectal**: `colab_colorectal/COLAB_TRAINING_COLORECTAL.ipynb`

---

## Tech Stack

- **PyTorch** — Deep learning framework
- **Flask** — Web application
- **OpenCV** — Image processing
- **Matplotlib** — Visualization
- **NumPy / SciPy** — Scientific computing

---

## Medical Disclaimer

This system is for **research and educational purposes only**. It is NOT approved for clinical diagnosis. All predictions must be reviewed by qualified medical professionals.

---

## Author

**Revanth Thota** — [GitHub](https://github.com/RevanthThota55)

## License

MIT License
