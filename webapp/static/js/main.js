/**
 * ExplainableMed-GOHBO - Main JavaScript
 * Handles model selection, file upload, prediction, and UI interactions
 */

// Global variables
let selectedFile = null;
let currentPredictionData = null;
let selectedModelType = null;
let previewObjectUrl = null;

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
const uploadStatusPill = document.getElementById('uploadStatusPill');
const previewSkeleton = document.getElementById('previewSkeleton');
const previewFallback = document.getElementById('previewFallback');
const previewFileName = document.getElementById('previewFileName');
const previewFileDetails = document.getElementById('previewFileDetails');
const previewTypeBadge = document.getElementById('previewTypeBadge');
const previewSizeBadge = document.getElementById('previewSizeBadge');
const previewHeading = document.getElementById('previewHeading');

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
    if (previewHeading) previewHeading.textContent = ui.uploadTitle + ' ready';
    setUploadStatus('Awaiting Scan', 'idle');

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
    if (document.querySelector('.server-warning')) return;

    const warning = document.createElement('div');
    warning.className = 'server-warning';
    warning.innerHTML = `
        <div class="server-warning-card">
            <i class="fas fa-exclamation-triangle"></i>
            <div>
                <strong>Server connection unavailable</strong>
                <p>Make sure the Flask application is running before starting an analysis.</p>
            </div>
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
async function handleFile(file) {
    if (!isAllowedImage(file)) {
        showNotification('Please upload a valid JPG, PNG, or TIFF scan.', 'error');
        return;
    }

    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showNotification('File too large. Maximum size is 16MB.', 'error');
        return;
    }

    selectedFile = file;
    clearPreviewState();
    populatePreviewMetadata(file);
    setUploadStatus('Preparing Preview', 'processing');

    uploadZone.style.display = 'none';
    imagePreview.style.display = 'block';
    analyzeBtn.style.display = 'inline-flex';
    analyzeBtn.disabled = true;
    changeImageBtn.disabled = true;

    showPreviewLoading();

    try {
        const preview = await buildPreviewSource(file);
        if (file !== selectedFile) return;

        renderPreview(preview);
        changeImageBtn.disabled = false;
        analyzeBtn.disabled = false;
        setUploadStatus('Ready For Analysis', 'ready');
    } catch (error) {
        console.error('Preview error:', error);
        if (file !== selectedFile) return;

        showPreviewFallback(
            'The scan is uploaded and ready for analysis, but the local preview could not be generated.'
        );
        changeImageBtn.disabled = false;
        analyzeBtn.disabled = false;
        setUploadStatus('Preview Limited', 'warning');
        showNotification('Preview could not be generated, but the scan can still be analyzed.', 'warning');
    }
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
    if (loadingState) loadingState.style.display = 'none';
    const backBtn = document.getElementById('backToModelsBtn');
    if (backBtn) backBtn.style.display = 'inline-flex';
    clearPreviewState();
    setUploadStatus('Awaiting Scan', 'idle');
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
        showNotification('Select a scan image before starting analysis.', 'warning');
        return;
    }

    if (!selectedModelType) {
        showNotification('Choose a scan type before starting analysis.', 'warning');
        return;
    }

    const ui = MODEL_UI[selectedModelType];

    // Show loading state
    imagePreview.style.display = 'none';
    analyzeBtn.style.display = 'none';
    const backBtn = document.getElementById('backToModelsBtn');
    if (backBtn) backBtn.style.display = 'none';
    loadingState.style.display = 'block';
    setUploadStatus('Analyzing', 'processing');

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

        loadingState.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.style.display = 'inline-flex';
        if (backBtn) backBtn.style.display = 'block';
        setUploadStatus('Ready For Analysis', 'ready');
        showNotification(errorMessage, 'error');
    }
}

/**
 * Display prediction results
 */
function displayResults(data) {
    const modelType = data.model_type || selectedModelType;
    const ui = MODEL_UI[modelType] || MODEL_UI['brain_tumor'];
    setUploadStatus('Analysis Complete', 'ready');

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
                <div class="probability-bar" style="width: ${probabilityPct}%"></div>
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
        showNotification('No prediction data available yet.', 'warning');
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
        showNotification('Error generating report: ' + error.message, 'error');
    }
}

/**
 * Show notification message
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    const icons = {
        success: 'fa-circle-check',
        error: 'fa-circle-xmark',
        warning: 'fa-triangle-exclamation',
        info: 'fa-circle-info'
    };
    notification.innerHTML = `
        <i class="fas ${icons[type] || icons.info}"></i>
        <span>${message}</span>
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, 3000);
}

function isAllowedImage(file) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff'];
    const extension = getFileExtension(file.name);
    return allowedTypes.includes(file.type) || ['jpg', 'jpeg', 'png', 'tif', 'tiff'].includes(extension);
}

function getFileExtension(filename) {
    return filename.includes('.') ? filename.split('.').pop().toLowerCase() : '';
}

function isTiffFile(file) {
    return ['tif', 'tiff'].includes(getFileExtension(file.name)) || file.type === 'image/tiff';
}

function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function setUploadStatus(label, state = 'idle') {
    if (!uploadStatusPill) return;

    uploadStatusPill.textContent = label;
    uploadStatusPill.dataset.state = state;
}

function populatePreviewMetadata(file, dimensionsText = 'Preview is ready for inspection.') {
    if (previewFileName) previewFileName.textContent = file.name;
    if (previewFileDetails) previewFileDetails.textContent = dimensionsText;
    if (previewTypeBadge) previewTypeBadge.textContent = getFileExtension(file.name).toUpperCase() || 'IMAGE';
    if (previewSizeBadge) previewSizeBadge.textContent = formatFileSize(file.size);
}

function showPreviewLoading() {
    if (previewSkeleton) previewSkeleton.style.display = 'flex';
    if (previewImg) {
        previewImg.style.display = 'none';
        previewImg.removeAttribute('src');
    }
    if (previewFallback) previewFallback.style.display = 'none';
}

function showPreviewFallback(message) {
    if (previewSkeleton) previewSkeleton.style.display = 'none';
    if (previewImg) {
        previewImg.style.display = 'none';
        previewImg.removeAttribute('src');
    }
    if (previewFallback) {
        const text = previewFallback.querySelector('p');
        if (text) text.textContent = message;
        previewFallback.style.display = 'flex';
    }
}

function clearPreviewState() {
    if (previewObjectUrl) {
        URL.revokeObjectURL(previewObjectUrl);
        previewObjectUrl = null;
    }

    if (previewSkeleton) previewSkeleton.style.display = 'none';
    if (previewFallback) previewFallback.style.display = 'none';
    if (previewImg) {
        previewImg.style.display = 'none';
        previewImg.removeAttribute('src');
    }
    if (previewFileName) previewFileName.textContent = 'No file selected';
    if (previewFileDetails) {
        previewFileDetails.textContent = 'Once loaded, the scan preview and file details will appear here.';
    }
    if (previewTypeBadge) previewTypeBadge.textContent = '-';
    if (previewSizeBadge) previewSizeBadge.textContent = '-';
}

async function buildPreviewSource(file) {
    if (isTiffFile(file)) {
        return requestServerPreview(file);
    }

    const objectUrl = URL.createObjectURL(file);

    try {
        const dimensions = await preloadImage(objectUrl);
        return {
            src: objectUrl,
            objectUrl,
            details: `${dimensions.width} × ${dimensions.height} px • Ready for analysis.`
        };
    } catch (error) {
        URL.revokeObjectURL(objectUrl);
        return requestServerPreview(file);
    }
}

function preloadImage(src) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => {
            resolve({
                width: image.naturalWidth,
                height: image.naturalHeight
            });
        };
        image.onerror = reject;
        image.src = src;
    });
}

async function requestServerPreview(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/preview_image', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error(`Preview request failed with ${response.status}`);
    }

    const data = await response.json();
    if (!data.success) {
        throw new Error(data.error || 'Preview generation failed');
    }

    return {
        src: data.preview,
        objectUrl: null,
        details: `${data.width} × ${data.height} px • Server-rendered preview for reliable viewing.`
    };
}

function renderPreview(preview) {
    if (preview.objectUrl) {
        previewObjectUrl = preview.objectUrl;
    }

    if (previewImg) {
        previewImg.src = preview.src;
        previewImg.style.display = 'block';
    }
    if (previewSkeleton) previewSkeleton.style.display = 'none';
    if (previewFallback) previewFallback.style.display = 'none';
    if (previewFileDetails) previewFileDetails.textContent = preview.details;
}
