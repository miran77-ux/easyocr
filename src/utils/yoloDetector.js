import * as tf from '@tensorflow/tfjs';

export class YOLODetector {
    constructor() {
        this.model = null;
        this.inputSize = 640;
        this.classNames = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ];
    }

    async loadModel() {
        try {
            // Load a real COCO-SSD model (TensorFlow.js compatible)
            this.model = await tf.loadGraphModel('https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1', {
                fromTFHub: true
            });
            
            console.log('YOLO model loaded successfully');
            
            // Warm up the model
            const dummyInput = tf.zeros([1, this.inputSize, this.inputSize, 3], 'int32');
            await this.model.executeAsync(dummyInput);
            dummyInput.dispose();
            
        } catch (error) {
            console.error('YOLO model loading failed:', error);
            throw error;
        }
    }

    async detect(canvas) {
        if (!this.model) {
            throw new Error('Model not loaded');
        }

        try {
            // Preprocess image
            const tensor = this.preprocessImage(canvas);
            
            // Run inference
            const predictions = await this.model.executeAsync(tensor);
            
            // Process predictions
            const detections = await this.processDetections(predictions, canvas.width, canvas.height);
            
            // Cleanup
            tensor.dispose();
            if (Array.isArray(predictions)) {
                predictions.forEach(p => p.dispose());
            } else {
                predictions.dispose();
            }
            
            return detections;
            
        } catch (error) {
            console.error('Detection failed:', error);
            return [];
        }
    }

    preprocessImage(canvas) {
        // Convert canvas to tensor
        const tensor = tf.browser.fromPixels(canvas)
            .resizeNearestNeighbor([this.inputSize, this.inputSize])
            .toInt()
            .expandDims(0);
        
        return tensor;
    }

    async processDetections(predictions, originalWidth, originalHeight) {
        try {
            // Handle different model output formats
            let boxes, scores, classes;
            
            if (Array.isArray(predictions)) {
                // SSD MobileNet format
                [boxes, scores, classes] = predictions;
            } else {
                // Single tensor output
                const predArray = await predictions.data();
                return this.parseSingleTensorOutput(predArray, originalWidth, originalHeight);
            }

            const boxesData = await boxes.data();
            const scoresData = await scores.data();
            const classesData = await classes.data();

            const detections = [];
            const numDetections = scoresData.length;

            for (let i = 0; i < numDetections; i++) {
                const score = scoresData[i];
                if (score > 0.5) { // Confidence threshold
                    const classId = Math.floor(classesData[i]);
                    const className = this.classNames[classId] || `class_${classId}`;
                    
                    // Convert normalized coordinates to pixel coordinates
                    const y1 = boxesData[i * 4] * originalHeight;
                    const x1 = boxesData[i * 4 + 1] * originalWidth;
                    const y2 = boxesData[i * 4 + 2] * originalHeight;
                    const x2 = boxesData[i * 4 + 3] * originalWidth;
                    
                    detections.push({
                        class: className,
                        confidence: score,
                        x: x1,
                        y: y1,
                        width: x2 - x1,
                        height: y2 - y1
                    });
                }
            }

            return detections;
            
        } catch (error) {
            console.error('Detection processing failed:', error);
            return [];
        }
    }

    parseSingleTensorOutput(predArray, originalWidth, originalHeight) {
        // Fallback parser for different model formats
        const detections = [];
        
        // This is a simplified parser - adjust based on your model's output format
        for (let i = 0; i < predArray.length; i += 6) {
            if (i + 5 < predArray.length) {
                const confidence = predArray[i + 4];
                if (confidence > 0.5) {
                    const classId = Math.floor(predArray[i + 5]);
                    const className = this.classNames[classId] || `object_${classId}`;
                    
                    detections.push({
                        class: className,
                        confidence: confidence,
                        x: predArray[i] * originalWidth,
                        y: predArray[i + 1] * originalHeight,
                        width: (predArray[i + 2] - predArray[i]) * originalWidth,
                        height: (predArray[i + 3] - predArray[i + 1]) * originalHeight
                    });
                }
            }
        }
        
        return detections;
    }
}