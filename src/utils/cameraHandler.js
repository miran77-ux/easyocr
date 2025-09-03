export class CameraHandler {
    constructor() {
        this.stream = null;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.isActive = false;
        this.yoloModel = null;
        this.ocrWorker = null;
    }

    setModels(yoloModel, ocrWorker) {
        this.yoloModel = yoloModel;
        this.ocrWorker = ocrWorker;
    }

    async startCamera() {
        try {
            this.video = document.getElementById('camera-video');
            this.canvas = document.getElementById('camera-canvas');
            this.ctx = this.canvas.getContext('2d');

            // Request camera access
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment' // Use back camera on mobile
                }
            });

            this.video.srcObject = this.stream;
            this.isActive = true;

            // Wait for video to be ready
            await new Promise(resolve => {
                this.video.onloadedmetadata = resolve;
            });

            // Set canvas size to match video
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.canvas.style.display = 'block';

            console.log('Camera started successfully');

        } catch (error) {
            throw new Error(`Camera initialization failed: ${error.message}`);
        }
    }

    stopCamera() {
        this.isActive = false;
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.video) {
            this.video.srcObject = null;
        }
        
        if (this.canvas) {
            this.canvas.style.display = 'none';
        }

        console.log('Camera stopped');
    }

    async processCurrentFrame(settings) {
        if (!this.isActive || !this.video || !this.yoloModel || !this.ocrWorker) {
            return null;
        }

        try {
            // Capture current frame
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Run YOLO detection
            const objectDetections = await this.yoloModel.predict(this.canvas);
            
            // Run OCR on current frame
            const ocrResults = await this.runOCROnFrame(settings);
            
            // Draw detections on canvas overlay
            this.drawLiveDetections(objectDetections, ocrResults, settings);
            
            return {
                objectCount: objectDetections.length,
                textCount: ocrResults.words ? ocrResults.words.filter(word => 
                    word.confidence > settings.confidenceThreshold * 100
                ).length : 0,
                detectedText: this.extractTextFromOCR(ocrResults, settings.confidenceThreshold),
                objectDetections,
                ocrResults
            };
            
        } catch (error) {
            console.error('Frame processing error:', error);
            return null;
        }
    }

    async runOCROnFrame(settings) {
        try {
            // Create a temporary canvas for OCR
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            
            tempCanvas.width = this.canvas.width;
            tempCanvas.height = this.canvas.height;
            
            // Copy current frame
            tempCtx.drawImage(this.video, 0, 0);
            
            // Enhance if requested
            if (settings.enhanceOCR) {
                tempCtx.filter = 'contrast(1.3) brightness(1.1)';
                tempCtx.drawImage(tempCanvas, 0, 0);
            }
            
            const { data } = await this.ocrWorker.recognize(tempCanvas);
            return data;
            
        } catch (error) {
            console.error('OCR on frame failed:', error);
            return { words: [], text: '' };
        }
    }

    drawLiveDetections(objectDetections, ocrResults, settings) {
        // Clear previous drawings
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw current video frame
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Draw object detections
        this.ctx.strokeStyle = '#ff0000';
        this.ctx.lineWidth = 3;
        this.ctx.font = '16px Arial';

        objectDetections.forEach(detection => {
            const [x, y, w, h] = detection.bbox;
            
            // Draw bounding box
            this.ctx.strokeRect(x, y, w, h);
            
            // Draw label
            const label = `${detection.class} ${(detection.confidence * 100).toFixed(1)}%`;
            this.ctx.fillStyle = '#ff0000';
            this.ctx.fillRect(x, y - 25, this.ctx.measureText(label).width + 10, 25);
            this.ctx.fillStyle = '#ffffff';
            this.ctx.fillText(label, x + 5, y - 5);
        });

        // Draw OCR detections
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.lineWidth = 2;

        if (ocrResults.words) {
            ocrResults.words.forEach(word => {
                if (word.confidence > settings.confidenceThreshold * 100) {
                    const bbox = word.bbox;
                    const x = bbox.x0;
                    const y = bbox.y0;
                    const w = bbox.x1 - bbox.x0;
                    const h = bbox.y1 - bbox.y0;
                    
                    this.ctx.strokeRect(x, y, w, h);
                }
            });
        }
    }

    extractTextFromOCR(ocrResults, confidenceThreshold) {
        if (!ocrResults.words) return '';
        
        return ocrResults.words
            .filter(word => word.confidence > confidenceThreshold * 100)
            .map(word => word.text)
            .join(' ');
    }

    captureFrame() {
        if (!this.isActive || !this.canvas) return null;
        
        // Create a new canvas with current frame
        const captureCanvas = document.createElement('canvas');
        const captureCtx = captureCanvas.getContext('2d');
        
        captureCanvas.width = this.canvas.width;
        captureCanvas.height = this.canvas.height;
        
        // Copy current canvas content (with detections)
        captureCtx.drawImage(this.canvas, 0, 0);
        
        return captureCanvas;
    }
}