import { YOLODetector } from './utils/yoloDetector.js';
import { OCRProcessor } from './utils/ocrProcessor.js';
import { CameraManager } from './utils/cameraManager.js';

class YOLOOCRApp {
    constructor() {
        this.yoloDetector = new YOLODetector();
        this.ocrProcessor = new OCRProcessor();
        this.cameraManager = new CameraManager();
        
        this.modelsLoaded = false;
        this.currentImage = null;
        this.isProcessing = false;
    }

    async init() {
        this.setupEventListeners();
        await this.loadModels();
    }

    async loadModels() {
        const loading = document.getElementById('loading');
        const loadingText = document.getElementById('loading-text');
        
        loading.classList.remove('hidden');
        
        try {
            loadingText.textContent = 'Loading YOLO model...';
            await this.yoloDetector.loadModel();
            
            loadingText.textContent = 'Loading OCR model...';
            await this.ocrProcessor.loadModel();
            
            loadingText.textContent = 'Models loaded successfully!';
            setTimeout(() => {
                loading.classList.add('hidden');
                this.modelsLoaded = true;
            }, 1000);
            
        } catch (error) {
            loadingText.textContent = 'Error loading models: ' + error.message;
            console.error('Model loading failed:', error);
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
            uploadArea.style.background = '#e3f2fd';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = '';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = '';
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
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');

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
            alert('Please select a valid image file');
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
            <div style="margin-top: 1rem;">Image loaded successfully</div>
        `;
    }

    async processImage() {
        if (!this.currentImage || !this.modelsLoaded || this.isProcessing) return;

        this.isProcessing = true;
        const processBtn = document.getElementById('process-btn');
        processBtn.textContent = '⏳ Processing...';
        processBtn.disabled = true;
        
        try {
            const settings = {
                enhanceOCR: document.getElementById('enhance-ocr').checked,
                confidenceThreshold: parseFloat(document.getElementById('confidence-slider').value)
            };

            // Create canvas from image
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = this.currentImage.width;
            canvas.height = this.currentImage.height;
            ctx.drawImage(this.currentImage, 0, 0);

            // Run YOLO detection
            const objectDetections = await this.yoloDetector.detect(canvas);
            
            // Run OCR
            const ocrResults = await this.ocrProcessor.extractText(canvas, settings);
            
            // Draw results
            this.drawDetections(ctx, objectDetections, ocrResults, settings);
            
            // Display results
            this.displayResults(canvas, objectDetections, ocrResults, settings);
            
        } catch (error) {
            console.error('Processing failed:', error);
            alert('Processing failed: ' + error.message);
        } finally {
            this.isProcessing = false;
            processBtn.textContent = '🔍 Process Image';
            processBtn.disabled = false;
        }
    }

    drawDetections(ctx, objectDetections, ocrResults, settings) {
        // Draw object detections (red boxes)
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 3;
        ctx.font = '16px Arial';

        objectDetections.forEach(detection => {
            const { x, y, width, height, class: className, confidence } = detection;
            
            // Draw bounding box
            ctx.strokeRect(x, y, width, height);
            
            // Draw label
            const label = `${className} ${(confidence * 100).toFixed(1)}%`;
            const textWidth = ctx.measureText(label).width;
            
            ctx.fillStyle = '#ff0000';
            ctx.fillRect(x, y - 25, textWidth + 10, 25);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x + 5, y - 5);
        });

        // Draw OCR detections (green boxes)
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;

        ocrResults.words.forEach(word => {
            if (word.confidence > settings.confidenceThreshold * 100) {
                const { x0, y0, x1, y1 } = word.bbox;
                
                ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
                
                // Draw confidence
                ctx.fillStyle = '#00ff00';
                ctx.fillRect(x0, y0 - 20, 50, 20);
                ctx.fillStyle = '#ffffff';
                ctx.font = '12px Arial';
                ctx.fillText(`${word.confidence.toFixed(1)}%`, x0 + 2, y0 - 5);
            }
        });
    }

    displayResults(canvas, objectDetections, ocrResults, settings) {
        const resultsSection = document.getElementById('upload-results');
        const resultCanvas = document.getElementById('result-canvas');
        const detectedText = document.getElementById('detected-text');
        const objectCount = document.getElementById('object-count');
        const textCount = document.getElementById('text-count');

        // Display processed image
        resultCanvas.width = canvas.width;
        resultCanvas.height = canvas.height;
        const ctx = resultCanvas.getContext('2d');
        ctx.drawImage(canvas, 0, 0);

        // Extract and display text
        const extractedText = ocrResults.words
            .filter(word => word.confidence > settings.confidenceThreshold * 100)
            .map(word => `${word.text} [${word.confidence.toFixed(1)}%]`)
            .join('\n');

        detectedText.value = extractedText;
        objectCount.textContent = objectDetections.length;
        textCount.textContent = ocrResults.words.filter(word => 
            word.confidence > settings.confidenceThreshold * 100
        ).length;

        // Store for download
        this.lastResults = {
            canvas: resultCanvas,
            text: extractedText,
            objectDetections,
            ocrResults
        };

        resultsSection.style.display = 'grid';
    }

    async startCamera() {
        if (!this.modelsLoaded) {
            alert('Models not loaded yet');
            return;
        }

        try {
            await this.cameraManager.start();
            this.updateCameraControls(true);
            this.startLiveProcessing();
        } catch (error) {
            alert('Camera access failed: ' + error.message);
        }
    }

    stopCamera() {
        this.cameraManager.stop();
        this.updateCameraControls(false);
        document.getElementById('camera-results').style.display = 'none';
    }

    updateCameraControls(isActive) {
        document.getElementById('start-camera').disabled = isActive;
        document.getElementById('stop-camera').disabled = !isActive;
        document.getElementById('capture-frame').disabled = !isActive;
    }

    async startLiveProcessing() {
        const cameraResults = document.getElementById('camera-results');
        cameraResults.style.display = 'grid';

        let frameCount = 0;
        let lastTime = performance.now();

        const processLoop = async () => {
            if (!this.cameraManager.isActive) return;

            try {
                const canvas = this.cameraManager.getCurrentFrame();
                if (canvas) {
                    frameCount++;
                    
                    // Process every 30 frames (about 1 second at 30fps)
                    if (frameCount % 30 === 0) {
                        const settings = {
                            enhanceOCR: document.getElementById('enhance-ocr').checked,
                            confidenceThreshold: parseFloat(document.getElementById('confidence-slider').value)
                        };

                        const objectDetections = await this.yoloDetector.detect(canvas);
                        const ocrResults = await this.ocrProcessor.extractText(canvas, settings);
                        
                        this.updateLiveResults(objectDetections, ocrResults, settings);
                    }
                    
                    // Update FPS
                    const currentTime = performance.now();
                    if (currentTime - lastTime >= 1000) {
                        const fps = Math.round(frameCount * 1000 / (currentTime - lastTime));
                        document.getElementById('fps-counter').textContent = fps;
                        frameCount = 0;
                        lastTime = currentTime;
                    }
                }
            } catch (error) {
                console.error('Live processing error:', error);
            }

            requestAnimationFrame(processLoop);
        };

        processLoop();
    }

    updateLiveResults(objectDetections, ocrResults, settings) {
        const extractedText = ocrResults.words
            .filter(word => word.confidence > settings.confidenceThreshold * 100)
            .map(word => word.text)
            .join(' ');

        document.getElementById('live-object-count').textContent = objectDetections.length;
        document.getElementById('live-text-count').textContent = ocrResults.words.filter(word => 
            word.confidence > settings.confidenceThreshold * 100
        ).length;
        document.getElementById('live-text').value = extractedText;
    }

    captureFrame() {
        const canvas = this.cameraManager.captureFrame();
        if (canvas) {
            canvas.toBlob((blob) => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `capture_${Date.now()}.png`;
                a.click();
                URL.revokeObjectURL(url);
            });
        }
    }

    downloadImage() {
        if (!this.lastResults) return;
        
        this.lastResults.canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `processed_${Date.now()}.png`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    downloadText() {
        if (!this.lastResults) return;
        
        const blob = new Blob([this.lastResults.text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `text_${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    const app = new YOLOOCRApp();
    app.init();
});