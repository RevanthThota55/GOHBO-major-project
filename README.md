<div align="center">

# ExplainableMed-GOHBO

### Multi-Model Medical Image Classification with Explainable AI

*GOHBO-optimized ResNet-18 | Grad-CAM Visual Explanations | MC Dropout Uncertainty*

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

---

## About

A production-ready deep learning system that classifies medical images across **three imaging modalities** using GOHBO-optimized ResNet-18 architectures. Unlike black-box models, this system provides **visual explanations** through Grad-CAM heatmaps and **uncertainty quantification** via Monte Carlo Dropout, giving clinicians transparency and confidence in AI-assisted diagnosis.

---

## Model Performance

<div align="center">

| Model | Imaging Modality | Classes | Accuracy | Dataset |
|:------|:----------------|:--------|:---------|:--------|
| **Brain Tumor** | MRI Scan | 4 (Glioma, Meningioma, No Tumor, Pituitary) | **95.42%** | ~3,000 MRI scans |
| **Chest X-Ray** | X-Ray | 2 (Normal, Pneumonia) | **97.03%** | ~5,863 X-ray images |
| **Colorectal Cancer** | Histopathology | 8 tissue types | **94.56%** | ~5,000 tissue images |

</div>

---

## Key Features

<table>
<tr>
<td width="50%">

**GOHBO Optimization**
Hybrid meta-heuristic combining Grey Wolf Optimizer, Heap-Based Optimization, and Orthogonal Learning for superior hyperparameter tuning.

</td>
<td width="50%">

**Grad-CAM Explainability**
Visual heatmaps reveal exactly which image regions influenced the AI's diagnostic decision, making predictions transparent and verifiable.

</td>
</tr>
<tr>
<td width="50%">

**MC Dropout Uncertainty**
Monte Carlo Dropout provides calibrated confidence scores, flagging uncertain predictions that need human review.

</td>
<td width="50%">

**Clinical PDF Reports**
Downloadable reports with diagnosis, confidence scores, Grad-CAM heatmaps, and probability distributions for clinical documentation.

</td>
</tr>
</table>

---

## Quick Start

> **3 steps** to get the webapp running on your machine.

### Step 1 &mdash; Clone & Install

```bash
git clone https://github.com/RevanthThota55/GOHBO-major-project.git
cd GOHBO-major-project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2 &mdash; Download Model Weights (~309 MB)

```bash
python download_models.py
```

Downloads the 3 pre-trained `.pth` files from [GitHub Releases](https://github.com/RevanthThota55/GOHBO-major-project/releases/tag/v1.0-models) into `models/checkpoints/`.

<details>
<summary><strong>Manual download (if script fails)</strong></summary>

Download these files from the [Releases page](https://github.com/RevanthThota55/GOHBO-major-project/releases/tag/v1.0-models) and place them in `models/checkpoints/`:

| File | Size |
|------|------|
| `best_model.pth` | 43 MB |
| `chest_xray_resnet18.pth` | 133 MB |
| `colorectal_resnet18.pth` | 133 MB |

</details>

### Step 3 &mdash; Run

```bash
cd webapp
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## How It Works

```
Medical Image Input (MRI / X-Ray / Histopathology)
        |
        v
 +-----------------+     +------------------+     +-------------------+
 |  Preprocessing  | --> | GOHBO-Optimized  | --> |   Classification  |
 |  (224x224, Norm)|     | ResNet-18 Model  |     |   + Confidence %  |
 +-----------------+     +------------------+     +-------------------+
                                |                          |
                                v                          v
                      +------------------+     +-------------------+
                      |   Grad-CAM       |     |   MC Dropout      |
                      |   Heatmap        |     |   Uncertainty     |
                      +------------------+     +-------------------+
                                |                          |
                                +----------+---------------+
                                           |
                                           v
                                  +------------------+
                                  |  Clinical Report |
                                  |  (PDF Download)  |
                                  +------------------+
```

---

## The GOHBO Algorithm

GOHBO is a **hybrid meta-heuristic optimizer** that finds optimal hyperparameters faster than traditional grid search by combining three algorithms:

| Component | Inspiration | Role |
|:----------|:-----------|:-----|
| **Grey Wolf Optimizer** | Wolf pack hunting hierarchy | Intelligent global search with alpha/beta/delta guidance |
| **Heap-Based Optimizer** | Heap data structures | Efficient solution ranking and rapid convergence |
| **Orthogonal Learning** | Orthogonal experimental design | Prevents premature convergence through diversity |

**Result:** Optimal learning rates found in 30-50 iterations vs. 100+ trials with grid search.

---

## Project Structure

```
GOHBO-major-project/
|
+-- webapp/                         Flask web application
|   +-- app.py                      Routes, model loading, inference
|   +-- templates/                  HTML (index, about, base)
|   +-- static/                     CSS, JavaScript
|
+-- src/                            Core ML source code
|   +-- algorithms/                 GOHBO (gwo.py, hbo.py, orthogonal.py, gohbo.py)
|   +-- models/                     ResNet-18 with MC Dropout
|   +-- explainability/             Grad-CAM + uncertainty quantification
|   +-- reports/                    Clinical PDF report generator
|   +-- training/                   Trainer, evaluator, optimizer
|   +-- datasets/                   Brain tumor, chest X-ray, colorectal loaders
|   +-- deployment/                 ONNX export, INT8 quantization
|
+-- models/checkpoints/             Pre-trained weights (via download_models.py)
+-- colab_chest_xray/               Colab training notebook
+-- colab_colorectal/               Colab training notebook
+-- notebooks/                      Jupyter exploration notebooks
+-- results/                        Training metrics & evaluation plots
|
+-- config.py                       Configuration
+-- train.py                        Training script
+-- evaluate.py                     Evaluation script
+-- download_models.py              Model weight downloader
+-- requirements.txt                Dependencies
```

---

## Training Your Own Models

Pre-trained weights are provided. To retrain from scratch:

| Model | Local Script | Cloud Notebook |
|:------|:------------|:---------------|
| Brain Tumor | `python train_brain_tumor.py` | &mdash; |
| Chest X-Ray | `python train_chest_xray.py` | [`colab_chest_xray/`](colab_chest_xray/COLAB_TRAINING_CHEST_XRAY.ipynb) |
| Colorectal | &mdash; | [`colab_colorectal/`](colab_colorectal/COLAB_TRAINING_COLORECTAL.ipynb) |

> Free GPU available via [Google Colab](https://colab.research.google.com/) or [Kaggle Notebooks](https://www.kaggle.com/code).

---

## Tech Stack

<div align="center">

| | Technology | Purpose |
|:--|:----------|:--------|
| <img src="https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" /> | PyTorch | Deep learning framework |
| <img src="https://img.shields.io/badge/-Flask-000000?style=flat-square&logo=flask&logoColor=white" /> | Flask | Web application server |
| <img src="https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" /> | OpenCV | Image processing |
| <img src="https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white" /> | NumPy / SciPy | Scientific computing |
| <img src="https://img.shields.io/badge/-Matplotlib-11557c?style=flat-square" /> | Matplotlib | Visualization |
| <img src="https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" /> | scikit-learn | Metrics & evaluation |

</div>

---

## Disclaimer

> **This system is for research and educational purposes only.** It is NOT approved for clinical diagnosis or treatment decisions. All predictions must be reviewed and validated by qualified medical professionals. The model's output should be considered supplementary information, never a replacement for expert clinical judgment.

---

<div align="center">

## Author

**Revanth Thota**

[![GitHub](https://img.shields.io/badge/GitHub-RevanthThota55-181717?style=for-the-badge&logo=github)](https://github.com/RevanthThota55)

---

**MIT License** &copy; 2025

*Built with PyTorch, Flask, and Advanced Machine Learning*

</div>
