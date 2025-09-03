export class CameraManager {
    constructor() {
        this.stream = null;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.isActive = false;
    }

    async start() {
        try {
            this.video = document.getElementById('camera-video');
            this.canvas = document.getElementById('camera-canvas');
            this.ctx = this.canvas.getContext('2d');

            // Request camera access with high resolution
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280, min: 640 },
                    height: { ideal: 720, min: 480 },
                    facingMode: 'environment' // Prefer back camera
                }
            });

            this.video.srcObject = this.stream;
            this.isActive = true;

            // Wait for video metadata
            await new Promise(resolve => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    resolve();
                };
            });

            // Set canvas size to match video
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;

            // Start rendering loop
            this.startRenderLoop();

            console.log(`Camera started: ${this.video.videoWidth}x${this.video.videoHeight}`);

        } catch (error) {
            console.error('Camera start failed:', error);
            throw new Error(`Camera access failed: ${error.message}`);
        }
    }

    stop() {
        this.isActive = false;
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.video) {
            this.video.srcObject = null;
        }

        console.log('Camera stopped');
    }

    startRenderLoop() {
        const render = () => {
            if (!this.isActive || !this.video) return;

            // Draw current video frame to canvas
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            requestAnimationFrame(render);
        };
        
        render();
    }

    getCurrentFrame() {
        if (!this.isActive || !this.canvas) return null;
        
        // Return a copy of the current canvas
        const frameCanvas = document.createElement('canvas');
        const frameCtx = frameCanvas.getContext('2d');
        
        frameCanvas.width = this.canvas.width;
        frameCanvas.height = this.canvas.height;
        frameCtx.drawImage(this.canvas, 0, 0);
        
        return frameCanvas;
    }

    captureFrame() {
        if (!this.isActive) return null;
        
        // Create high-quality capture
        const captureCanvas = document.createElement('canvas');
        const captureCtx = captureCanvas.getContext('2d');
        
        captureCanvas.width = this.video.videoWidth;
        captureCanvas.height = this.video.videoHeight;
        
        // Draw current video frame
        captureCtx.drawImage(this.video, 0, 0);
        
        return captureCanvas;
    }

    drawDetections(objectDetections, ocrResults, settings) {
        if (!this.isActive || !this.ctx) return;

        // Clear and redraw video frame
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

        // Draw object detections (red)
        this.ctx.strokeStyle = '#ff0000';
        this.ctx.lineWidth = 3;
        this.ctx.font = '16px Arial';

        objectDetections.forEach(detection => {
            const { x, y, width, height, class: className, confidence } = detection;
            
            this.ctx.strokeRect(x, y, width, height);
            
            const label = `${className} ${(confidence * 100).toFixed(1)}%`;
            const textWidth = this.ctx.measureText(label).width;
            
            this.ctx.fillStyle = '#ff0000';
            this.ctx.fillRect(x, y - 25, textWidth + 10, 25);
            this.ctx.fillStyle = '#ffffff';
            this.ctx.fillText(label, x + 5, y - 5);
        });

        // Draw OCR detections (green)
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.lineWidth = 2;

        ocrResults.words.forEach(word => {
            if (word.confidence > settings.confidenceThreshold * 100) {
                const { x0, y0, x1, y1 } = word.bbox;
                
                this.ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
                
                // Draw text
                this.ctx.fillStyle = '#00ff00';
                this.ctx.fillRect(x0, y0 - 20, 60, 20);
                this.ctx.fillStyle = '#ffffff';
                this.ctx.font = '12px Arial';
                this.ctx.fillText(word.text, x0 + 2, y0 - 5);
            }
        });
    }
}