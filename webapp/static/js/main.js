/**
 * MedVision AI - Main JavaScript
 * Handles model selection, file upload, prediction, and UI interactions
 */

// Global variables
let selectedFile = null;
let currentPredictionData = null;
let selectedModelType = null;

// Model display config
const MODEL_UI = {
    brain_tumor: {
        icon: 'fa-brain',
        uploadTitle: 'Upload Brain MRI Scan',
        uploadSubtitle: 'Upload a brain MRI image for tumor classification',
        uploadIcon: 'fa-brain',
        uploadText: 'Drag & drop your MRI scan here',
        loadingText: 'Analyzing brain MRI scan...',
        loadingSubtext: 'Running GOHBO-optimized ResNet-18 for tumor detection',
        heatmapTitle: 'TUMOR DETECTION HEATMAP',
        originalLabel: 'Original MRI Scan',
        resultBadgeHtml: '<i class="fas fa-brain"></i> Brain Tumor Classification',
        reportPrefix: 'brain_tumor'
    },
    chest_xray: {
        icon: 'fa-lungs',
        uploadTitle: 'Upload Chest X-Ray',
        uploadSubtitle: 'Upload a chest X-ray image for pneumonia detection',
        uploadIcon: 'fa-lungs',
        uploadText: 'Drag & drop your chest X-ray here',
        loadingText: 'Analyzing chest X-ray...',
        loadingSubtext: 'Running GOHBO-optimized ResNet-18 for pneumonia detection',
        heatmapTitle: 'PNEUMONIA DETECTION HEATMAP',
        originalLabel: 'Original Chest X-Ray',
        resultBadgeHtml: '<i class="fas fa-lungs"></i> Chest X-Ray Pneumonia Detection',
        reportPrefix: 'chest_xray'
    },
    colorectal: {
        icon: 'fa-microscope',
        uploadTitle: 'Upload Histopathology Image',
        uploadSubtitle: 'Upload a colorectal tissue histopathology image for classification',
        uploadIcon: 'fa-microscope',
        uploadText: 'Drag & drop your histopathology image here',
        loadingText: 'Analyzing tissue sample...',
        loadingSubtext: 'Running GOHBO-optimized ResNet-18 for tissue classification',
        heatmapTitle: 'TISSUE CLASSIFICATION HEATMAP',
        originalLabel: 'Original Histopathology Image',
        resultBadgeHtml: '<i class="fas fa-microscope"></i> Colorectal Cancer Histopathology',
        reportPrefix: 'colorectal'
    }
};

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const analyzeBtn = document.getElementById('analyzeBtn');
const changeImageBtn = document.getElementById('changeImageBtn');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const uploadSection = document.getElementById('uploadSection');
const modelSelector = document.getElementById('modelSelector');

// Initialize event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeModelSelector();
    initializeUploadZone();
    initializeButtons();
    initializeScrollReveal();
    checkServerHealth();
});

/**
 * Initialize model selection cards
 */
function initializeModelSelector() {
    if (!modelSelector) return;

    const modelCards = modelSelector.querySelectorAll('.model-card');
    modelCards.forEach(card => {
        card.addEventListener('click', function() {
            const modelType = this.dataset.model;
            selectModel(modelType);
        });
    });
}

/**
 * Select a model and show upload section
 */
function selectModel(modelType) {
    selectedModelType = modelType;
    const ui = MODEL_UI[modelType];
    if (!ui) return;

    // Highlight selected card
    const allCards = modelSelector.querySelectorAll('.model-card');
    allCards.forEach(c => c.classList.remove('selected'));
    const selectedCard = document.getElementById('modelCard_' + modelType);
    if (selectedCard) selectedCard.classList.add('selected');

    // Update upload section UI
    const uploadTitle = document.getElementById('uploadTitle');
    const uploadSubtitle = document.getElementById('uploadSubtitle');
    const uploadIcon = document.getElementById('uploadIcon');
    const uploadText = document.getElementById('uploadText');

    if (uploadTitle) uploadTitle.innerHTML = '<i class="fas ' + ui.icon + '"></i> ' + ui.uploadTitle;
    if (uploadSubtitle) uploadSubtitle.textContent = ui.uploadSubtitle;
    if (uploadIcon) {
        uploadIcon.className = 'fas ' + ui.uploadIcon + ' upload-icon';
    }
    if (uploadText) uploadText.textContent = ui.uploadText;

    // Show upload section with animation
    if (uploadSection) {
        uploadSection.style.display = 'block';
        uploadSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // Reset any previous upload state
    resetUpload();
}

/**
 * Check if server is healthy and models are loaded
 */
async function checkServerHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();

        if (data.status === 'healthy' && data.total_models > 0) {
            console.log('Server healthy. Models loaded:', data.models_loaded);
        } else {
            console.warn('Server responded but no models loaded');
        }
    } catch (error) {
        console.error('Cannot connect to server:', error);
        showServerWarning();
    }
}

/**
 * Show server connection warning
 */
function showServerWarning() {
    const warning = document.createElement('div');
    warning.className = 'server-warning';
    warning.innerHTML = `
        <div style="background: rgba(245, 158, 11, 0.1); border: 2px solid rgba(245, 158, 11, 0.5); border-radius: 12px; padding: 16px; margin: 16px; text-align: center;">
            <i class="fas fa-exclamation-triangle" style="color: #F59E0B; font-size: 1.5rem; margin-bottom: 8px;"></i>
            <p style="color: #FFFFFF; margin: 0;">Cannot connect to server. Make sure the Flask application is running.</p>
        </div>
    `;

    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(warning, container.firstChild);
    }
}

/**
 * Initialize upload zone with drag & drop functionality
 */
function initializeUploadZone() {
    if (!uploadZone) return;

    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', handleFileSelect);

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

/**
 * Initialize button event listeners
 */
function initializeButtons() {
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeScan);
    }

    if (changeImageBtn) {
        changeImageBtn.addEventListener('click', resetUpload);
    }

    const downloadReportBtn = document.getElementById('downloadReportBtn');
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', downloadReport);
    }

    const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
    if (analyzeAnotherBtn) {
        analyzeAnotherBtn.addEventListener('click', resetAll);
    }

    const backToModelsBtn = document.getElementById('backToModelsBtn');
    if (backToModelsBtn) {
        backToModelsBtn.addEventListener('click', backToModelSelection);
    }

    const toggleOverlayBtn = document.getElementById('toggleOverlay');
    if (toggleOverlayBtn) {
        toggleOverlayBtn.addEventListener('click', toggleHeatmapOverlay);
    }

    const opacitySlider = document.getElementById('opacitySlider');
    if (opacitySlider) {
        opacitySlider.addEventListener('input', updateOpacity);
    }
}

/**
 * Initialize scroll-triggered reveal animations
 */
function initializeScrollReveal() {
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const revealElements = document.querySelectorAll('.reveal, .reveal-stagger');
    if (revealElements.length === 0) return;

    if (prefersReduced) {
        revealElements.forEach(el => el.classList.add('revealed'));
        return;
    }

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.05,
        rootMargin: '0px 0px -20px 0px'
    });

    revealElements.forEach(function(el) {
        observer.observe(el);
    });
}

/**
 * Go back to model selection
 */
function backToModelSelection() {
    resetUpload();
    if (uploadSection) uploadSection.style.display = 'none';
    if (resultsSection) resultsSection.style.display = 'none';
    selectedModelType = null;

    // Remove selected state from cards
    const allCards = modelSelector.querySelectorAll('.model-card');
    allCards.forEach(c => c.classList.remove('selected'));

    // Scroll to model selector
    const modelSection = document.querySelector('.model-selection-section');
    if (modelSection) {
        modelSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

/**
 * Handle file selection from input
 */
function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

/**
 * Handle file upload and preview
 */
function handleFile(file) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff'];
    if (!allowedTypes.includes(file.type)) {
        alert('Please upload a valid image file (JPG, PNG, or TIFF)');
        return;
    }

    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        alert('File too large. Maximum size is 16MB.');
        return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        uploadZone.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.style.display = 'inline-flex';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

/**
 * Reset upload UI
 */
function resetUpload() {
    selectedFile = null;
    if (fileInput) fileInput.value = '';
    if (uploadZone) uploadZone.style.display = 'block';
    if (imagePreview) imagePreview.style.display = 'none';
    if (analyzeBtn) analyzeBtn.style.display = 'none';
}

/**
 * Reset all and start over
 */
function resetAll() {
    resetUpload();
    if (resultsSection) resultsSection.style.display = 'none';

    // Go back to model selection
    backToModelSelection();
}

/**
 * Analyze the uploaded scan
 */
async function analyzeScan() {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }

    if (!selectedModelType) {
        alert('Please select a scan type first');
        return;
    }

    const ui = MODEL_UI[selectedModelType];

    // Show loading state
    imagePreview.style.display = 'none';
    analyzeBtn.style.display = 'none';
    const backBtn = document.getElementById('backToModelsBtn');
    if (backBtn) backBtn.style.display = 'none';
    loadingState.style.display = 'block';

    // Update loading text
    const loadingText = document.getElementById('loadingText');
    const loadingSubtext = document.getElementById('loadingSubtext');
    if (loadingText) loadingText.textContent = ui.loadingText;
    if (loadingSubtext) loadingSubtext.textContent = ui.loadingSubtext;

    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model_type', selectedModelType);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();

        if (data.success) {
            currentPredictionData = data;
            loadingState.style.display = 'none';
            displayResults(data);
        } else {
            throw new Error(data.error || 'Prediction failed');
        }

    } catch (error) {
        console.error('Error:', error);

        let errorMessage = 'An error occurred while analyzing the scan.';

        if (error.message.includes('Failed to fetch') || error.message.includes('ERR_CONNECTION_REFUSED')) {
            errorMessage = 'Cannot connect to server. Please make sure the application is running.';
        } else if (error.message.includes('Server error')) {
            errorMessage = 'Server error occurred. Please try again.';
        } else {
            errorMessage = error.message;
        }

        alert(errorMessage);
        loadingState.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.style.display = 'inline-flex';
        if (backBtn) backBtn.style.display = 'block';
    }
}

/**
 * Display prediction results
 */
function displayResults(data) {
    const modelType = data.model_type || selectedModelType;
    const ui = MODEL_UI[modelType] || MODEL_UI['brain_tumor'];

    // Show results section
    resultsSection.style.display = 'block';
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Update model badge
    const badge = document.getElementById('modelResultBadge');
    if (badge) badge.innerHTML = ui.resultBadgeHtml;

    // Update heatmap title
    const heatmapTitle = document.getElementById('heatmapTitle');
    if (heatmapTitle) heatmapTitle.textContent = ui.heatmapTitle;

    // Update original label
    const originalLabel = document.getElementById('originalLabel');
    if (originalLabel) originalLabel.textContent = ui.originalLabel;

    // Update diagnosis
    document.getElementById('diagnosisLabel').textContent = data.prediction;
    document.getElementById('confidenceValue').textContent = data.confidence + '%';

    // Update confidence bar
    const confidenceBar = document.getElementById('confidenceBar');
    confidenceBar.style.width = data.confidence + '%';

    // Update description
    document.getElementById('descriptionText').textContent = data.description;

    // Update images
    document.getElementById('originalImage').src = data.images.original;
    document.getElementById('heatmapImage').src = data.images.overlay;

    // Update probabilities
    updateProbabilities(data.probabilities);

    // Update explanation
    updateExplanation(data.explanation);

    // Set diagnosis color based on result
    const diagnosisLabel = document.getElementById('diagnosisLabel');
    const prediction = data.prediction.toLowerCase();

    if (modelType === 'brain_tumor') {
        if (prediction.includes('no tumor')) {
            diagnosisLabel.style.color = 'var(--green-400)';
        } else {
            diagnosisLabel.style.color = 'var(--red-500)';
        }
    } else if (modelType === 'chest_xray') {
        if (prediction.includes('normal')) {
            diagnosisLabel.style.color = 'var(--green-400)';
        } else {
            diagnosisLabel.style.color = 'var(--red-500)';
        }
    } else if (modelType === 'colorectal') {
        if (prediction.includes('tumor') || prediction.includes('debris') || prediction.includes('stroma') || prediction.includes('complex')) {
            diagnosisLabel.style.color = 'var(--red-500)';
        } else if (prediction.includes('mucosa') || prediction.includes('adipose') || prediction.includes('empty')) {
            diagnosisLabel.style.color = 'var(--green-400)';
        } else {
            diagnosisLabel.style.color = 'var(--blue-400)';
        }
    }
}

/**
 * Update probability bars
 */
function updateProbabilities(probabilities) {
    const container = document.getElementById('probabilitiesContainer');
    container.innerHTML = '';

    for (const [className, probability] of Object.entries(probabilities)) {
        const probabilityPct = (probability * 100).toFixed(2);

        const item = document.createElement('div');
        item.className = 'probability-item';
        item.innerHTML = `
            <div class="probability-label">${className}</div>
            <div class="probability-bar-container">
                <div class="probability-bar" style="width: ${probabilityPct}%">
                    ${probabilityPct}%
                </div>
            </div>
            <div class="probability-value">${probabilityPct}%</div>
        `;

        container.appendChild(item);
    }
}

/**
 * Update explanation section
 */
function updateExplanation(explanation) {
    document.getElementById('diagnosisExplanation').textContent = explanation.diagnosis;
    document.getElementById('confidenceAssessment').textContent = explanation.confidence_assessment;
    document.getElementById('uncertaintyAssessment').textContent = explanation.uncertainty_assessment;

    const findingsList = document.getElementById('keyFindingsList');
    findingsList.innerHTML = '';

    explanation.key_findings.forEach(finding => {
        const li = document.createElement('li');
        li.textContent = finding;
        findingsList.appendChild(li);
    });
}

/**
 * Toggle heatmap overlay
 */
function toggleHeatmapOverlay() {
    const heatmapImg = document.getElementById('heatmapImage');

    if (currentPredictionData && currentPredictionData.images) {
        if (heatmapImg.src.includes('overlay')) {
            heatmapImg.src = currentPredictionData.images.heatmap;
        } else {
            heatmapImg.src = currentPredictionData.images.overlay;
        }
    }
}

/**
 * Update heatmap opacity
 */
function updateOpacity(event) {
    const opacity = event.target.value;
    document.getElementById('opacityValue').textContent = opacity + '%';

    const heatmapImg = document.getElementById('heatmapImage');
    heatmapImg.style.opacity = opacity / 100;
}

/**
 * Download clinical report
 */
async function downloadReport() {
    if (!currentPredictionData) {
        alert('No prediction data available');
        return;
    }

    try {
        const response = await fetch('/generate_report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentPredictionData)
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;

            const ui = MODEL_UI[selectedModelType] || MODEL_UI['brain_tumor'];
            a.download = `${ui.reportPrefix}_report_${new Date().getTime()}.pdf`;

            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            showNotification('Report downloaded successfully!', 'success');
        } else {
            throw new Error('Failed to generate report');
        }

    } catch (error) {
        console.error('Error downloading report:', error);
        alert('Error generating report: ' + error.message);
    }
}

/**
 * Show notification message
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-check-circle"></i>
        <span>${message}</span>
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, 3000);
}
