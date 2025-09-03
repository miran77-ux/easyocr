export class ImageProcessor {
    constructor() {
        this.yoloModel = null;
        this.ocrWorker = null;
    }

    setModels(yoloModel, ocrWorker) {
        this.yoloModel = yoloModel;
        this.ocrWorker = ocrWorker;
    }

    async processImage(image, settings) {
        if (!this.yoloModel || !this.ocrWorker) {
            throw new Error('Models not loaded');
        }

        // Create canvas for processing
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = image.width;
        canvas.height = image.height;
        ctx.drawImage(image, 0, 0);

        // Run YOLO detection
        const objectDetections = await this.yoloModel.predict(canvas);
        
        // Run OCR
        const ocrResults = await this.runOCR(canvas, settings);
        
        // Draw detections on canvas
        this.drawDetections(ctx, objectDetections, ocrResults, settings);
        
        // Create result image
        const processedImage = new Image();
        processedImage.src = canvas.toDataURL();
        
        await new Promise(resolve => {
            processedImage.onload = resolve;
        });

        return {
            processedImage,
            detectedText: this.extractText(ocrResults, settings.confidenceThreshold),
            objectCount: objectDetections.length,
            textCount: (ocrResults.data && ocrResults.data.words ? ocrResults.data.words : []).filter(word => 
                word.confidence > settings.confidenceThreshold * 100
            ).length,
            objectDetections,
            ocrResults
        };
    }

    async runOCR(canvas, settings) {
        try {
            // Enhance image for OCR if requested
            let processCanvas = canvas;
            if (settings.enhanceOCR) {
                processCanvas = this.enhanceImageForOCR(canvas);
            }

            const { data } = await this.ocrWorker.recognize(processCanvas);
            return data;
            
        } catch (error) {
            console.error('OCR failed:', error);
            return { words: [], text: '' };
        }
    }

    enhanceImageForOCR(canvas) {
        const enhancedCanvas = document.createElement('canvas');
        const ctx = enhancedCanvas.getContext('2d');
        
        enhancedCanvas.width = canvas.width;
        enhancedCanvas.height = canvas.height;
        
        // Apply contrast enhancement
        ctx.filter = 'contrast(1.5) brightness(1.1)';
        ctx.drawImage(canvas, 0, 0);
        
        return enhancedCanvas;
    }

    drawDetections(ctx, objectDetections, ocrResults, settings) {
        // Draw object detections (YOLO)
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 3;
        ctx.font = '16px Arial';
        ctx.fillStyle = '#ff0000';

        objectDetections.forEach(detection => {
            const [x, y, w, h] = detection.bbox;
            
            // Draw bounding box
            ctx.strokeRect(x, y, w, h);
            
            // Draw label
            const label = `${detection.class} ${(detection.confidence * 100).toFixed(1)}%`;
            const textWidth = ctx.measureText(label).width;
            
            ctx.fillStyle = '#ff0000';
            ctx.fillRect(x, y - 25, textWidth + 10, 25);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x + 5, y - 5);
        });

        // Draw OCR detections
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.fillStyle = '#00ff00';

        if (ocrResults.words) {
            ocrResults.words.forEach(word => {
                if (word.confidence > settings.confidenceThreshold * 100) {
                    const bbox = word.bbox;
                    const x = bbox.x0;
                    const y = bbox.y0;
                    const w = bbox.x1 - bbox.x0;
                    const h = bbox.y1 - bbox.y0;
                    
                    // Draw bounding box
                    ctx.strokeRect(x, y, w, h);
                    
                    // Draw confidence
                    const label = `${word.confidence.toFixed(1)}%`;
                    ctx.fillStyle = '#00ff00';
                    ctx.fillRect(x, y - 20, 40, 20);
                    ctx.fillStyle = '#ffffff';
                    ctx.font = '12px Arial';
                    ctx.fillText(label, x + 2, y - 5);
                }
            });
        }
    }

    extractText(ocrResults, confidenceThreshold) {
        if (!ocrResults.words) return '';
        
        return ocrResults.words
            .filter(word => word.confidence > confidenceThreshold * 100)
            .map(word => `${word.text} [${word.confidence.toFixed(1)}%]`)
            .join('\n');
    }
}