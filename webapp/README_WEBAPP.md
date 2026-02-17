# 🌐 ExplainableMed-GOHBO Web Application

## Professional Flask Web Interface for Brain Tumor Classification

---

## 📋 Overview

This web application provides an intuitive interface for brain tumor classification using the GOHBO-optimized ResNet-18 model with Grad-CAM explainability and MC Dropout uncertainty quantification.

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install Flask>=2.3.0 Flask-CORS>=4.0.0 reportlab>=4.0.0
```

### 2. Run the Application
```bash
cd "D:/Major Project/medical-image-classification"
python webapp/app.py
```

### 3. Access the Interface
Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🎨 Features

### 1. **Interactive Upload Interface**
- Drag-and-drop file upload
- Click to browse files
- Image preview before analysis
- Support for JPG, PNG, TIFF formats
- Maximum file size: 16MB

### 2. **Real-Time Classification**
- Instant prediction with GOHBO-optimized model
- Class probabilities for all 4 tumor types
- Confidence scores
- Processing time: 2-3 seconds

### 3. **Grad-CAM Visual Explainability**
- Heatmap overlay showing AI focus areas
- Toggle between original and overlay views
- Adjustable transparency slider
- Color-coded importance (Red=High, Blue=Low)

### 4. **Uncertainty Quantification**
- Monte Carlo Dropout (10 forward passes)
- Confidence score (0-1)
- Entropy measurement
- Flag uncertain predictions (<85% confidence)

### 5. **Clinical PDF Reports**
- Professional medical report format
- Embedded Grad-CAM visualizations
- AI reasoning explanation
- Recommended actions
- Model metadata
- Proper research disclaimers

### 6. **About Page**
- Model architecture details
- Performance metrics
- GOHBO algorithm explanation
- Classification classes
- Important disclaimers

---

## 📂 File Structure

```
webapp/
├── app.py                     # Main Flask application
├── templates/                 # HTML templates
│   ├── base.html             # Base template with navigation
│   ├── index.html            # Home page with upload
│   └── about.html            # About page
├── static/                    # Static assets
│   ├── css/
│   │   └── styles.css        # Professional medical styling
│   ├── js/
│   │   ├── main.js           # Core functionality
│   │   └── visualization.js  # Visualization utilities
│   ├── images/               # Logo and assets
│   └── samples/              # Sample MRI scans for demo
└── uploads/                   # Temporary upload directory
```

---

## 🛠️ API Endpoints

### GET /
**Description**: Home page with upload interface
**Response**: HTML page

### POST /predict
**Description**: Upload image and get prediction
**Input**: Multipart form data with 'file' field
**Response**: JSON
```json
{
  "success": true,
  "prediction": "Glioma Tumor",
  "confidence": "95.30",
  "probabilities": {
    "Glioma Tumor": 0.953,
    "Meningioma Tumor": 0.032,
    "No Tumor": 0.012,
    "Pituitary Tumor": 0.003
  },
  "uncertainty": {
    "confidence": 0.89,
    "entropy": 0.234,
    "mean_std": 0.045
  },
  "images": {
    "original": "data:image/png;base64,...",
    "heatmap": "data:image/png;base64,...",
    "overlay": "data:image/png;base64,..."
  },
  "explanation": {
    "diagnosis": "...",
    "confidence_assessment": "...",
    "uncertainty_assessment": "...",
    "key_findings": ["...", "..."]
  }
}
```

### POST /generate_report
**Description**: Generate clinical PDF report
**Input**: JSON with prediction data
**Response**: PDF file download

### GET /about
**Description**: About page with model information
**Response**: HTML page

### GET /api/health
**Description**: Health check endpoint
**Response**: JSON
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": "2024-..."
}
```

---

## 🎨 UI/UX Features

### Design Elements:
- **Clean Medical Theme**: Blue and white color scheme
- **Responsive Layout**: Works on desktop, tablet, mobile
- **Smooth Animations**: Fade-ins, slide-ins, progress bars
- **Interactive Controls**: Toggles, sliders, buttons
- **Loading States**: Spinners with informative messages
- **Error Handling**: User-friendly error messages

### Accessibility:
- High contrast text
- Clear button labels
- Keyboard navigation support
- Screen reader compatible
- Alt text for images

---

## 🔧 Configuration

### Customize in `app.py`:

```python
# File upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Server settings
app.run(
    host='0.0.0.0',  # Accept external connections
    port=5000,       # Port number
    debug=True       # Enable debug mode (disable in production)
)

# MC Dropout settings
MC_PREDICTOR = MCDropoutPredictor(MODEL, num_passes=10)  # 10 for speed, 20 for accuracy
```

---

## 📊 Performance

### Response Times:
- **Upload**: < 1 second
- **Prediction**: 2-3 seconds (GPU), 5-8 seconds (CPU)
- **Grad-CAM**: +500ms
- **MC Dropout**: +2-3 seconds (10 passes)
- **PDF Generation**: 1-2 seconds

### Resource Usage:
- **Memory**: ~2-3 GB (with model loaded)
- **GPU VRAM**: ~1.5 GB
- **Disk**: Minimal (temporary uploads auto-deleted)

---

## 🚀 Deployment Options

### Local Development:
```bash
python webapp/app.py
```

### Production Deployment:

#### Option 1: Gunicorn (Linux/Mac)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 webapp.app:app
```

#### Option 2: Waitress (Windows)
```bash
pip install waitress
waitress-serve --port=5000 webapp.app:app
```

#### Option 3: Docker
```dockerfile
FROM python:3.8
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "webapp/app.py"]
```

---

## 🔒 Security Considerations

### For Academic Demo (Current):
✅ Suitable for localhost demonstration
✅ Basic file validation
✅ Size limits enforced

### For Production (If Deploying):
⚠️ Add user authentication
⚠️ Implement rate limiting
⚠️ Use HTTPS/SSL
⚠️ Sanitize file uploads
⚠️ Add CSRF protection
⚠️ Set up proper error logging

---

## 🎓 Educational Features

### For Students/Researchers:
- **Interactive Learning**: See how AI makes decisions
- **Visual Feedback**: Understand Grad-CAM heatmaps
- **Uncertainty Awareness**: Learn about AI confidence
- **Clinical Context**: Understand tumor types

### For Instructors:
- Easy to demonstrate in lectures
- Live predictions maintain engagement
- Visual explanations aid understanding
- Open-source for modification

---

## 📈 Future Enhancements (Optional)

Potential additions for extended projects:
- [ ] Multi-image comparison (temporal tracking)
- [ ] Batch processing interface
- [ ] User accounts and history
- [ ] Database integration for reports
- [ ] Real-time collaboration features
- [ ] Mobile app version
- [ ] Integration with PACS systems

---

## 🐛 Troubleshooting

### Common Issues:

**1. Port already in use**
```bash
# Change port in app.py or use different port:
python webapp/app.py --port 8080
```

**2. Model not found**
- App will show warning
- Demo will work with random weights
- Download or train model first

**3. Grad-CAM error**
- Check model has 'layer4' layer
- Verify model is in eval mode

**4. PDF generation fails**
```bash
pip install reportlab --upgrade
```

**5. Static files not loading**
- Clear browser cache (Ctrl+Shift+R)
- Check Flask static folder path

---

## 📞 Support

For issues or questions:
- Check inline code documentation
- Review DEMO_GUIDE.md for presentation tips
- Open issue on GitHub
- See IMPLEMENTATION_SUMMARY.md for technical details

---

## ✅ Checklist Before Demo

- [ ] Flask app starts without errors
- [ ] Can upload and get prediction
- [ ] Grad-CAM heatmap displays
- [ ] PDF report downloads
- [ ] All pages load correctly
- [ ] Have 3-5 sample MRI scans ready
- [ ] Tested on presentation computer
- [ ] Backup plan prepared

---

**Your professional web interface is ready for your academic demo!** 🎉

Built with Flask, PyTorch, and modern web technologies for an impressive presentation. 🚀