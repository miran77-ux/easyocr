import { ModelLoader } from './utils/modelLoader.js';
import { ImageProcessor } from './utils/imageProcessor.js';
import { CameraHandler } from './utils/cameraHandler.js';
import { UIController } from './utils/uiController.js';

class YOLOOCRApp {
    constructor() {
        this.modelLoader = new ModelLoader();
        this.imageProcessor = new ImageProcessor();
        this.cameraHandler = new CameraHandler();
        this.uiController = new UIController();
        
        this.modelsLoaded = false;
        this.currentImage = null;
    }

    async init() {
        try {
            this.setupEventListeners();
            await this.loadModels();
            this.uiController.hideLoading();
            this.modelsLoaded = true;
        } catch (error) {
            console.error('Initialization failed:', error);
            this.uiController.showError('Failed to initialize application: ' + error.message);
        }
    }

    async loadModels() {
        this.uiController.showLoading('Loading AI models...');
        
        try {
            await this.modelLoader.loadYOLOModel();
            this.uiController.updateLoadingText('Loading OCR model...');
            
            await this.modelLoader.loadOCRModel();
            this.uiController.updateLoadingText('Models loaded successfully!');
            
            // Pass models to processors
            this.imageProcessor.setModels(
                this.modelLoader.yoloModel,
                this.modelLoader.ocrWorker
            );
            
            this.cameraHandler.setModels(
                this.modelLoader.yoloModel,
                this.modelLoader.ocrWorker
            );
            
        } catch (error) {
            throw new Error(`Model loading failed: ${error.message}`);
        }
    }

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // File upload
        const fileInput = document.getElementById('file-input');
        const uploadArea = document.getElementById('upload-area');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFile(files[0]);
            }
        });

        // Process button
        document.getElementById('process-btn').addEventListener('click', () => {
            this.processImage();
        });

        // Confidence slider
        const slider = document.getElementById('confidence-slider');
        const valueDisplay = document.getElementById('confidence-value');
        slider.addEventListener('input', (e) => {
            valueDisplay.textContent = e.target.value;
        });

        // Camera controls
        document.getElementById('start-camera').addEventListener('click', () => {
            this.startCamera();
        });
        
        document.getElementById('stop-camera').addEventListener('click', () => {
            this.stopCamera();
        });
        
        document.getElementById('capture-frame').addEventListener('click', () => {
            this.captureFrame();
        });

        // Download buttons
        document.getElementById('download-image').addEventListener('click', () => {
            this.downloadImage();
        });
        
        document.getElementById('download-text').addEventListener('click', () => {
            this.downloadText();
        });
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');

        // Stop camera if switching away from camera tab
        if (tabName !== 'camera') {
            this.stopCamera();
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    handleFile(file) {
        if (!file.type.startsWith('image/')) {
            this.uiController.showError('Please select a valid image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.currentImage = img;
                this.displayUploadedImage(img);
                document.getElementById('process-btn').disabled = false;
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    displayUploadedImage(img) {
        const uploadArea = document.getElementById('upload-area');
        uploadArea.innerHTML = `
            <img src="${img.src}" style="max-width: 100%; max-height: 300px; border-radius: 8px;">
            <p style="margin-top: 1rem; color: #666;">Image loaded successfully</p>
        `;
    }

    async processImage() {
        if (!this.currentImage || !this.modelsLoaded) return;

        this.uiController.showLoading('Processing image...');
        
        try {
            const settings = this.getOCRSettings();
            const results = await this.imageProcessor.processImage(this.currentImage, settings);
            
            this.displayResults(results);
            this.uiController.hideLoading();
            
        } catch (error) {
            console.error('Processing failed:', error);
            this.uiController.showError('Processing failed: ' + error.message);
            this.uiController.hideLoading();
        }
    }

    getOCRSettings() {
        return {
            enhanceOCR: document.getElementById('enhance-ocr').checked,
            confidenceThreshold: parseFloat(document.getElementById('confidence-slider').value)
        };
    }

    displayResults(results) {
        const resultsSection = document.getElementById('results-section');
        const resultCanvas = document.getElementById('result-canvas');
        const detectedText = document.getElementById('detected-text');
        const objectCount = document.getElementById('object-count');
        const textCount = document.getElementById('text-count');

        // Display processed image
        const ctx = resultCanvas.getContext('2d');
        resultCanvas.width = results.processedImage.width;
        resultCanvas.height = results.processedImage.height;
        ctx.drawImage(results.processedImage, 0, 0);

        // Display text and stats
        detectedText.value = results.detectedText;
        objectCount.textContent = results.objectCount;
        textCount.textContent = results.textCount;

        // Store results for download
        this.lastResults = results;

        resultsSection.style.display = 'block';
    }

    async startCamera() {
        if (!this.modelsLoaded) {
            this.uiController.showError('Models not loaded yet');
            return;
        }

        try {
            await this.cameraHandler.startCamera();
            this.updateCameraControls(true);
            this.startLiveProcessing();
        } catch (error) {
            this.uiController.showError('Camera access failed: ' + error.message);
        }
    }

    stopCamera() {
        this.cameraHandler.stopCamera();
        this.updateCameraControls(false);
        document.getElementById('live-results').style.display = 'none';
    }

    updateCameraControls(isActive) {
        document.getElementById('start-camera').disabled = isActive;
        document.getElementById('stop-camera').disabled = !isActive;
        document.getElementById('capture-frame').disabled = !isActive;
    }

    startLiveProcessing() {
        const liveResults = document.getElementById('live-results');
        liveResults.style.display = 'block';

        const settings = this.getOCRSettings();
        let lastProcessTime = 0;
        const processInterval = 1000; // Process every 1 second

        const processLoop = async () => {
            if (!this.cameraHandler.isActive) return;

            const now = Date.now();
            if (now - lastProcessTime > processInterval) {
                try {
                    const results = await this.cameraHandler.processCurrentFrame(settings);
                    if (results) {
                        this.updateLiveResults(results);
                        lastProcessTime = now;
                    }
                } catch (error) {
                    console.error('Live processing error:', error);
                }
            }

            requestAnimationFrame(processLoop);
        };

        processLoop();
    }

    updateLiveResults(results) {
        document.getElementById('live-object-count').textContent = results.objectCount;
        document.getElementById('live-text-count').textContent = results.textCount;
        document.getElementById('live-text').value = results.detectedText;
        
        // Update FPS (simplified)
        const fps = Math.round(1000 / 1000); // Based on process interval
        document.getElementById('fps-counter').textContent = fps;
    }

    captureFrame() {
        if (!this.cameraHandler.isActive) return;
        
        const canvas = this.cameraHandler.captureFrame();
        if (canvas) {
            // Convert to downloadable image
            canvas.toBlob((blob) => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `camera_capture_${Date.now()}.png`;
                a.click();
                URL.revokeObjectURL(url);
            });
            
            this.uiController.showSuccess('Frame captured and downloaded!');
        }
    }

    downloadImage() {
        if (!this.lastResults) return;
        
        const canvas = document.getElementById('result-canvas');
        canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `processed_image_${Date.now()}.png`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    downloadText() {
        if (!this.lastResults) return;
        
        const text = this.lastResults.detectedText;
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `detected_text_${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new YOLOOCRApp();
    app.init();
});