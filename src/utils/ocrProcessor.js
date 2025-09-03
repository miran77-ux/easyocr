import { createWorker } from 'tesseract.js';

export class OCRProcessor {
    constructor() {
        this.worker = null;
    }

    async loadModel() {
        try {
            this.worker = await createWorker('eng', 1, {
                logger: m => {
                    if (m.status === 'recognizing text') {
                        console.log(`OCR Progress: ${Math.round(m.progress * 100)}%`);
                    }
                }
            });

            // Configure for better accuracy
            await this.worker.setParameters({
                tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-:;',
                tessedit_pageseg_mode: '6', // Uniform block of text
                preserve_interword_spaces: '1'
            });

            console.log('OCR model loaded successfully');
            
        } catch (error) {
            console.error('OCR model loading failed:', error);
            throw error;
        }
    }

    async extractText(canvas, settings) {
        if (!this.worker) {
            throw new Error('OCR model not loaded');
        }

        try {
            let processCanvas = canvas;
            
            // Enhance image for better OCR if requested
            if (settings.enhanceOCR) {
                processCanvas = this.enhanceForOCR(canvas);
            }

            // Run OCR
            const { data } = await this.worker.recognize(processCanvas);
            
            // Process and clean results
            const processedResults = this.processOCRResults(data, settings);
            
            return processedResults;
            
        } catch (error) {
            console.error('OCR extraction failed:', error);
            return { words: [], text: '', confidence: 0 };
        }
    }

    enhanceForOCR(canvas) {
        const enhancedCanvas = document.createElement('canvas');
        const ctx = enhancedCanvas.getContext('2d');
        
        enhancedCanvas.width = canvas.width;
        enhancedCanvas.height = canvas.height;
        
        // Convert to grayscale and enhance contrast
        ctx.filter = 'grayscale(100%) contrast(150%) brightness(110%)';
        ctx.drawImage(canvas, 0, 0);
        
        // Additional processing for better text recognition
        const imageData = ctx.getImageData(0, 0, enhancedCanvas.width, enhancedCanvas.height);
        const data = imageData.data;
        
        // Apply threshold to make text more distinct
        for (let i = 0; i < data.length; i += 4) {
            const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
            const threshold = avg > 128 ? 255 : 0;
            data[i] = threshold;     // Red
            data[i + 1] = threshold; // Green
            data[i + 2] = threshold; // Blue
        }
        
        ctx.putImageData(imageData, 0, 0);
        
        return enhancedCanvas;
    }

    processOCRResults(data, settings) {
        const words = [];
        
        if (data.words) {
            data.words.forEach(word => {
                if (word.text.trim() && word.confidence > settings.confidenceThreshold * 100) {
                    // Clean up the text
                    const cleanText = this.cleanText(word.text);
                    
                    if (cleanText.length > 0) {
                        words.push({
                            text: cleanText,
                            confidence: word.confidence,
                            bbox: word.bbox
                        });
                    }
                }
            });
        }

        // Sort by reading order (top to bottom, left to right)
        words.sort((a, b) => {
            const yDiff = a.bbox.y0 - b.bbox.y0;
            if (Math.abs(yDiff) < 20) { // Same line
                return a.bbox.x0 - b.bbox.x0;
            }
            return yDiff;
        });

        const fullText = words.map(word => word.text).join(' ');

        return {
            words,
            text: fullText,
            confidence: words.length > 0 ? words.reduce((sum, word) => sum + word.confidence, 0) / words.length : 0
        };
    }

    cleanText(text) {
        // Remove extra whitespace and clean up common OCR errors
        let cleaned = text.trim();
        
        // Fix common OCR mistakes
        const corrections = {
            '0': 'O',
            '1': 'I',
            '5': 'S',
            '8': 'B',
            '6': 'G',
            '|': 'I',
            '!': 'I'
        };

        // Apply corrections only if the text looks like it might be words
        if (cleaned.length > 1 && /[a-zA-Z]/.test(cleaned)) {
            Object.entries(corrections).forEach(([wrong, correct]) => {
                // Only replace if surrounded by letters
                cleaned = cleaned.replace(new RegExp(`(?<=[a-zA-Z])${wrong}(?=[a-zA-Z])`, 'g'), correct);
            });
        }

        // Remove non-printable characters
        cleaned = cleaned.replace(/[^\x20-\x7E]/g, '');
        
        return cleaned;
    }

    terminate() {
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
        }
    }
}