// DOM Elements
const predictionForm = document.getElementById('prediction-form');
const imageUpload = document.getElementById('image-upload');
const imageUrl = document.getElementById('image-url');
const clearUploadBtn = document.getElementById('clear-upload');
const clearUrlBtn = document.getElementById('clear-url');
const uploadTabLink = document.getElementById('upload-tab-link');
const urlTabLink = document.getElementById('url-tab-link');
const imagePreviewContainer = document.getElementById('image-preview-container');
const imagePreview = document.getElementById('image-preview');
const resultsCard = document.getElementById('results-card');
const loader = document.getElementById('loader');
const resultsContent = document.getElementById('results-content');
const topPredictionLabel = document.getElementById('top-prediction-label');
const topPredictionProgress = document.getElementById('top-prediction-progress');
const topPredictionPercent = document.getElementById('top-prediction-percent');
const otherPredictionsBody = document.getElementById('other-predictions-body');
const errorMessage = document.getElementById('error-message');
const errorText = document.getElementById('error-text');
const batchPreview = document.getElementById('batch-preview');
const batchPreviewGrid = document.getElementById('batch-preview-grid');
const urlList = document.getElementById('url-list');
const urlPreviewGrid = document.getElementById('url-preview-grid');
const fileCount = document.getElementById('file-count');
const urlCount = document.getElementById('url-count');
const clearBatchBtn = document.getElementById('clear-batch');
const clearUrlsBtn = document.getElementById('clear-urls');
const rotateImageBtn = document.getElementById('rotate-image');
const autoEnhanceCheckbox = document.getElementById('auto-enhance');
const removeBackgroundCheckbox = document.getElementById('remove-background');
const batchProgress = document.getElementById('batch-progress');
const batchStatus = document.getElementById('batch-status');
const batchResultsContainer = document.getElementById('batch-results');
const batchResultsGrid = document.getElementById('batch-results-grid');
const exportCsvBtn = document.getElementById('export-csv');
const exportPdfBtn = document.getElementById('export-pdf');

// State
let currentRotation = 0;
let batchFiles = [];
let batchUrls = [];
let batchResultsData = [];

// Event Listeners
document.addEventListener('DOMContentLoaded', initApp);
predictionForm.addEventListener('submit', handleFormSubmit);
imageUpload.addEventListener('change', handleImageUpload);
imageUrl.addEventListener('input', handleImageUrl);
clearUploadBtn.addEventListener('click', clearUpload);
clearUrlBtn.addEventListener('click', clearUrl);
uploadTabLink.addEventListener('click', () => setActiveTab('upload'));
urlTabLink.addEventListener('click', () => setActiveTab('url'));
clearBatchBtn.addEventListener('click', clearBatch);
clearUrlsBtn.addEventListener('click', clearUrls);
rotateImageBtn.addEventListener('click', rotateImage);
exportCsvBtn.addEventListener('click', exportToCsv);
exportPdfBtn.addEventListener('click', exportToPdf);

// Initialize App
function initApp() {
    // Check for available models
    fetch('/available_models')
        .then(response => response.json())
        .then(data => {
            console.log('Available models:', data.models);
        })
        .catch(error => {
            console.error('Error fetching available models:', error);
        });
}

// Form Submission
function handleFormSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(predictionForm);
    const processingMode = formData.get('processing_mode');
    
    // Validate form data
    if (processingMode === 'single') {
        if (!validateSingleInput()) return;
    } else {
        if (!validateBatchInput()) return;
    }
    
    // Show loader
    hideError();
    showLoader();
    
    // Prepare form data based on mode
    if (processingMode === 'single') {
        handleSingleSubmission(formData);
    } else {
        handleBatchSubmission(formData);
    }
}

// Validation Functions
function validateSingleInput() {
    const hasImage = imageUpload.files.length > 0;
    const hasUrl = imageUrl.value.trim() !== '';
    
    if (!hasImage && !hasUrl) {
        showError('Please upload an image or provide an image URL');
        return false;
    }
    
    return true;
}

function validateBatchInput() {
    if (batchFiles.length === 0 && batchUrls.length === 0) {
        showError('Please add at least one image for batch processing');
        return false;
    }
    
    if (batchFiles.length > 10) {
        showError('Maximum 10 files allowed for batch processing');
        return false;
    }
    
    return true;
}

// Submission Handlers
function handleSingleSubmission(formData) {
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'An error occurred');
            });
        }
        return response.json();
    })
    .then(data => {
        hideLoader();
        displaySingleResults(data.predictions);
    })
    .catch(error => {
        hideLoader();
        showError(error.message || 'Failed to get prediction');
    });
}

function handleBatchSubmission(formData) {
    // Create a batch of promises for each image
    const promises = [];
    
    // Add file uploads
    batchFiles.forEach((file, index) => {
        const fileFormData = new FormData();
        fileFormData.append('model', formData.get('model'));
        fileFormData.append('image', file);
        fileFormData.append('auto_enhance', formData.get('auto_enhance'));
        fileFormData.append('remove_background', formData.get('remove_background'));
        
        promises.push(
            fetch('/predict', {
                method: 'POST',
                body: fileFormData
            }).then(response => response.json())
        );
    });
    
    // Add URL submissions
    batchUrls.forEach((url, index) => {
        const urlFormData = new FormData();
        urlFormData.append('model', formData.get('model'));
        urlFormData.append('url', url);
        urlFormData.append('auto_enhance', formData.get('auto_enhance'));
        urlFormData.append('remove_background', formData.get('remove_background'));
        
        promises.push(
            fetch('/predict', {
                method: 'POST',
                body: urlFormData
            }).then(response => response.json())
        );
    });
    
    // Process all promises
    Promise.all(promises)
        .then(results => {
            hideLoader();
            displayBatchResults(results);
        })
        .catch(error => {
            hideLoader();
            showError('Failed to process batch: ' + error.message);
        });
}

// Image Upload Handling
function handleImageUpload() {
    const files = Array.from(imageUpload.files);
    
    // Validate files
    files.forEach(file => {
        if (!file.type.startsWith('image/')) {
            showError('Please upload valid image files only');
            return;
        }
        
        if (file.size > 5 * 1024 * 1024) {
            showError('Image size should be less than 5MB');
            return;
        }
    });
    
    // Update batch files
    batchFiles = files;
    updateBatchPreview();
}

// Image URL Handling
function handleImageUrl() {
    const url = imageUrl.value.trim();
    
    if (url !== '') {
        // Validate URL
        try {
            new URL(url);
        } catch (e) {
            showError('Please enter a valid URL');
            return;
        }
        
        // Check if URL ends with an image extension
        const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp'];
        const hasImageExtension = imageExtensions.some(ext => url.toLowerCase().endsWith(ext));
        
        if (!hasImageExtension) {
            showError('Please provide a URL to an image file');
            return;
        }
        
        // Add URL to batch
        if (!batchUrls.includes(url)) {
            batchUrls.push(url);
            updateUrlPreview();
        }
    }
}

// Preview Functions
function updateBatchPreview() {
    batchPreviewGrid.innerHTML = '';
    fileCount.textContent = batchFiles.length;
    
    if (batchFiles.length > 0) {
        batchPreview.classList.remove('d-none');
        
        batchFiles.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = function(e) {
                const col = document.createElement('div');
                col.className = 'col-4 col-md-3';
                col.innerHTML = `
                    <div class="position-relative">
                        <img src="${e.target.result}" class="img-fluid rounded" alt="Preview ${index + 1}">
                        <button type="button" class="btn btn-sm btn-danger position-absolute top-0 end-0 m-1 rounded-circle" onclick="removeFile(${index})">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                `;
                batchPreviewGrid.appendChild(col);
            };
            reader.readAsDataURL(file);
        });
    } else {
        batchPreview.classList.add('d-none');
    }
}

function updateUrlPreview() {
    urlPreviewGrid.innerHTML = '';
    urlCount.textContent = batchUrls.length;
    
    if (batchUrls.length > 0) {
        urlList.classList.remove('d-none');
        
        batchUrls.forEach((url, index) => {
            const col = document.createElement('div');
            col.className = 'col-4 col-md-3';
            col.innerHTML = `
                <div class="position-relative">
                    <img src="${url}" class="img-fluid rounded" alt="Preview ${index + 1}">
                    <button type="button" class="btn btn-sm btn-danger position-absolute top-0 end-0 m-1 rounded-circle" onclick="removeUrl(${index})">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
            urlPreviewGrid.appendChild(col);
        });
    } else {
        urlList.classList.add('d-none');
    }
}

// Clear Functions
function clearUpload() {
    imageUpload.value = '';
    imagePreviewContainer.classList.add('d-none');
}

function clearUrl() {
    imageUrl.value = '';
    imagePreviewContainer.classList.add('d-none');
}

function clearBatch() {
    batchFiles = [];
    imageUpload.value = '';
    updateBatchPreview();
}

function clearUrls() {
    batchUrls = [];
    imageUrl.value = '';
    updateUrlPreview();
}

function removeFile(index) {
    batchFiles.splice(index, 1);
    updateBatchPreview();
}

function removeUrl(index) {
    batchUrls.splice(index, 1);
    updateUrlPreview();
}

// Image Manipulation
function rotateImage() {
    currentRotation = (currentRotation + 90) % 360;
    imagePreview.style.transform = `rotate(${currentRotation}deg)`;
}

// Results Display
function displaySingleResults(predictions) {
    if (!predictions || predictions.length === 0) {
        showError('No predictions received');
        return;
    }
    
    // Display top prediction
    const topPrediction = predictions[0];
    topPredictionLabel.textContent = topPrediction.class;
    topPredictionProgress.style.width = `${topPrediction.probability}%`;
    topPredictionPercent.textContent = `${topPrediction.probability.toFixed(2)}%`;
    
    // Display other predictions
    otherPredictionsBody.innerHTML = '';
    for (let i = 1; i < predictions.length; i++) {
        const prediction = predictions[i];
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${prediction.class}</td>
            <td>
                <div class="d-flex align-items-center">
                    <div class="progress flex-grow-1 me-2" style="height: 5px;">
                        <div class="progress-bar" role="progressbar" style="width: ${prediction.probability}%"></div>
                    </div>
                    <span>${prediction.probability.toFixed(2)}%</span>
                </div>
            </td>
        `;
        otherPredictionsBody.appendChild(row);
    }
    
    // Show results card
    resultsCard.classList.remove('d-none');
    batchResultsContainer.classList.add('d-none');
    document.getElementById('single-results').classList.remove('d-none');
}

function displayBatchResults(results) {
    batchResultsGrid.innerHTML = '';
    batchResultsContainer.classList.remove('d-none');
    document.getElementById('single-results').classList.add('d-none');
    
    results.forEach((result, index) => {
        const col = document.createElement('div');
        col.className = 'col-md-6 col-lg-4';
        
        const topPrediction = result.predictions[0];
        col.innerHTML = `
            <div class="card h-100">
                <div class="card-body">
                    <h6 class="card-title">${batchFiles[index]?.name || batchUrls[index]}</h6>
                    <div class="alert alert-success mb-2">
                        <strong>${topPrediction.class}</strong>
                        <div class="progress mt-2" style="height: 5px;">
                            <div class="progress-bar" role="progressbar" style="width: ${topPrediction.probability}%"></div>
                        </div>
                        <small>${topPrediction.probability.toFixed(2)}%</small>
                    </div>
                    <button class="btn btn-sm btn-outline-primary" onclick="showDetails(${index})">
                        View Details
                    </button>
                </div>
            </div>
        `;
        batchResultsGrid.appendChild(col);
    });
    
    resultsCard.classList.remove('d-none');
}

// Export Functions
function exportToCsv() {
    const csv = [
        ['Filename', 'Primary Diagnosis', 'Confidence', 'Alternative Diagnoses', 'Confidences'],
        ...batchResultsData.map((result, index) => {
            const filename = batchFiles[index]?.name || batchUrls[index];
            const predictions = result.predictions;
            const primary = predictions[0];
            const alternatives = predictions.slice(1);
            return [
                filename,
                primary.class,
                primary.probability.toFixed(2),
                alternatives.map(p => p.class).join('; '),
                alternatives.map(p => p.probability.toFixed(2)).join('; ')
            ];
        })
    ].map(row => row.join(',')).join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'crop_disease_predictions.csv';
    a.click();
    window.URL.revokeObjectURL(url);
}

function exportToPdf() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    doc.setFontSize(16);
    doc.text('Crop Disease Predictions', 20, 20);
    
    let y = 40;
    batchResultsData.forEach((result, index) => {
        const filename = batchFiles[index]?.name || batchUrls[index];
        const predictions = result.predictions;
        const primary = predictions[0];
        
        doc.setFontSize(12);
        doc.text(`File: ${filename}`, 20, y);
        y += 10;
        doc.text(`Primary Diagnosis: ${primary.class} (${primary.probability.toFixed(2)}%)`, 20, y);
        y += 10;
        
        if (y > 250) {
            doc.addPage();
            y = 20;
        }
    });
    
    doc.save('crop_disease_predictions.pdf');
}

// Utility Functions
function showLoader() {
    resultsCard.classList.remove('d-none');
    loader.classList.remove('d-none');
    resultsContent.style.display = 'none';
}

function hideLoader() {
    loader.classList.add('d-none');
    resultsContent.style.display = 'block';
}

function showError(message) {
    errorText.textContent = message;
    errorMessage.classList.remove('d-none');
    resultsCard.classList.add('d-none');
}

function hideError() {
    errorMessage.classList.add('d-none');
}

function setActiveTab(tab) {
    if (tab === 'upload') {
        imageUrl.value = '';
    } else if (tab === 'url') {
        imageUpload.value = '';
    }
    
    if (imageUpload.files.length === 0 && imageUrl.value.trim() === '') {
        imagePreviewContainer.classList.add('d-none');
    }
} 