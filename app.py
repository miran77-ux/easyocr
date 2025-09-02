import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
import warnings
import urllib.request
from pathlib import Path
import tempfile
import os
import time
import io
import base64
from PIL import Image
import threading
import queue

# Monkey-patch for Pillow >= 10.0 (ANTIALIAS removed)
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# Suppress warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="YOLO + OCR Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-msg {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .error-msg {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    .camera-container {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 10px;
        background: #f8f9fa;
    }
    .tab-content {
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 YOLO + OCR Detection System</h1>
    <p>Upload images or use live camera for object detection and text recognition</p>
</div>
""", unsafe_allow_html=True)

def setup_pytorch_compatibility():
    """Setup PyTorch compatibility for YOLO"""
    try:
        from torch.serialization import add_safe_globals
        
        safe_classes = []
        
        # Add essential classes
        try:
            from ultralytics.nn.tasks import DetectionModel
            safe_classes.append(DetectionModel)
        except ImportError:
            pass
        
        try:
            from torch.nn.modules.container import Sequential
            safe_classes.append(Sequential)
        except ImportError:
            pass
        
        try:
            import ultralytics.nn.modules as ul_modules
            modules = ['Conv', 'C2f', 'SPPF', 'Bottleneck', 'DFL', 'Detect', 'Segment']
            for module_name in modules:
                if hasattr(ul_modules, module_name):
                    safe_classes.append(getattr(ul_modules, module_name))
        except ImportError:
            pass
        
        try:
            from collections import OrderedDict
            safe_classes.append(OrderedDict)
        except ImportError:
            pass
        
        if safe_classes:
            add_safe_globals(safe_classes)
            return True
            
    except ImportError:
        return True
    except Exception:
        return False
    
    return True

@st.cache_resource
def load_models():
    """Load YOLO and OCR models with caching"""
    
    # Setup compatibility
    setup_pytorch_compatibility()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Download YOLO model if needed
        status_text.text("📥 Checking YOLO model...")
        progress_bar.progress(20)
        
        model_path = "yolov8n.pt"
        if not Path(model_path).exists():
            status_text.text("📥 Downloading YOLO model...")
            progress_bar.progress(30)
            urllib.request.urlretrieve(
                "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                model_path
            )
        
        # Load YOLO model
        status_text.text("📄 Loading YOLO model...")
        progress_bar.progress(50)
        
        try:
            model = YOLO(model_path)
        except Exception as e:
            # Try monkey patching for PyTorch 2.6+
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
            try:
                model = YOLO(model_path)
            finally:
                torch.load = original_load
        
        # Load OCR model
        status_text.text("📄 Loading OCR model...")
        progress_bar.progress(80)
        
        reader = easyocr.Reader(['en'], verbose=False)
        
        progress_bar.progress(100)
        status_text.text("✅ All models loaded successfully!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return model, reader
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Failed to load models: {str(e)}")
        st.info("""
        **Troubleshooting:**
        1. Try refreshing the page
        2. Check internet connection
        3. Models will be downloaded automatically on first run
        """)
        st.stop()

def enhance_image_for_ocr(image, options):
    """Apply image preprocessing for better OCR"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    processed = gray.copy()
    
    if "Contrast Enhancement" in options:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed = clahe.apply(processed)
    
    if "Noise Reduction" in options:
        processed = cv2.medianBlur(processed, 3)
    
    if "Sharpening" in options:
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        processed = cv2.filter2D(processed, -1, kernel)
    
    if "Thresholding" in options:
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return processed

def clean_ocr_text(text):
    """Clean and correct OCR text"""
    if not text.strip():
        return text
    
    # Common OCR corrections
    corrections = {
        "0": "O", "1": "I", "3": "E", "5": "S", "8": "B", "6": "G"
    }
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if not line.strip():
            continue
        
        # Apply corrections
        corrected = line
        for old, new in corrections.items():
            corrected = corrected.replace(old, new)
        
        # Special cases
        if corrected.lower() in ["h3llo", "hel1o", "he11o"]:
            corrected = "HELLO"
        elif corrected.lower() in ["w0rld", "wor1d"]:
            corrected = "WORLD"
        
        cleaned_lines.append(corrected)
    
    return '\n'.join(cleaned_lines)

def process_image(image, model, reader, ocr_settings):
    """Process image with YOLO and OCR"""
    
    # Convert PIL to OpenCV
    if isinstance(image, Image.Image):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img_cv = image
    
    # YOLO Detection
    results = model(img_cv)
    annotated_frame = results[0].plot()
    
    # Prepare image for OCR
    ocr_image = img_cv.copy()
    if ocr_settings['enhance_ocr']:
        ocr_image = enhance_image_for_ocr(img_cv, ocr_settings['preprocessing'])
    
    # OCR Detection
    try:
        if ocr_settings['handwriting_mode']:
            ocr_results = reader.readtext(ocr_image, paragraph=False, width_ths=0.8)
        else:
            ocr_results = reader.readtext(ocr_image)
    except:
        ocr_results = reader.readtext(ocr_image)
    
    # Draw OCR results on image
    for detection in ocr_results:
        box = np.array(detection[0], dtype=np.int32)
        text = detection[1]
        conf = detection[2]
        
        if conf > ocr_settings['confidence_threshold']:
            # Draw bounding box
            cv2.polylines(annotated_frame, [box], True, (0, 255, 0), 2)
            
            # Add text label
            text_pos = (box[0][0], box[0][1] - 10)
            cv2.putText(annotated_frame, f"{text} ({conf:.2f})", text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Extract text
    detected_text = ""
    for detection in ocr_results:
        if detection[2] > ocr_settings['confidence_threshold']:
            detected_text += f"{detection[1]} [Conf: {detection[2]:.2f}]\n"
    
    # Clean text
    cleaned_text = clean_ocr_text(detected_text)
    
    # Convert back to PIL if needed
    if isinstance(image, Image.Image):
        result_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        return result_image, detected_text, cleaned_text, ocr_results
    else:
        return annotated_frame, detected_text, cleaned_text, ocr_results

class CameraHandler:
    """Handle camera operations for live detection"""
    
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        
    def initialize_camera(self, camera_index=0):
        """Initialize camera with given index"""
        try:
            if self.cap is not None:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                return False
                
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            return True
        except Exception as e:
            st.error(f"Camera initialization failed: {e}")
            return False
    
    def start_capture(self):
        """Start capturing frames"""
        self.is_running = True
        
        def capture_loop():
            while self.is_running and self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    # Add frame to queue (non-blocking)
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                time.sleep(0.033)  # ~30 FPS
        
        self.capture_thread = threading.Thread(target=capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def stop_capture(self):
        """Stop capturing frames"""
        self.is_running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_frame(self):
        """Get latest frame from queue"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def process_and_queue_result(self, frame, model, reader, ocr_settings):
        """Process frame and queue result"""
        try:
            result_frame, detected_text, cleaned_text, ocr_results = process_image(
                frame, model, reader, ocr_settings
            )
            
            if not self.result_queue.full():
                self.result_queue.put({
                    'frame': result_frame,
                    'text': detected_text,
                    'cleaned_text': cleaned_text,
                    'ocr_results': ocr_results,
                    'timestamp': time.time()
                })
        except Exception as e:
            print(f"Processing error: {e}")
    
    def get_result(self):
        """Get latest processing result"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

# Initialize camera handler
if 'camera_handler' not in st.session_state:
    st.session_state.camera_handler = CameraHandler()

# Sidebar Settings
st.sidebar.title("⚙️ Settings")

# Model loading
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

if not st.session_state.models_loaded:
    with st.sidebar:
        if st.button("🚀 Load Models", use_container_width=True):
            model, reader = load_models()
            st.session_state.yolo_model = model
            st.session_state.ocr_reader = reader
            st.session_state.models_loaded = True
            st.rerun()

if st.session_state.models_loaded:
    st.sidebar.success("✅ Models loaded successfully!")
    
    # OCR Settings
    st.sidebar.subheader("🔍 OCR Settings")
    
    ocr_settings = {
        'handwriting_mode': st.sidebar.checkbox("✍️ Handwriting Mode", False),
        'enhance_ocr': st.sidebar.checkbox("🔧 Enhance OCR", True),
        'confidence_threshold': st.sidebar.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    }
    
    if ocr_settings['enhance_ocr']:
        ocr_settings['preprocessing'] = st.sidebar.multiselect(
            "🛠️ Preprocessing Options",
            ["Contrast Enhancement", "Noise Reduction", "Sharpening", "Thresholding"],
            default=["Contrast Enhancement", "Noise Reduction"]
        )
    else:
        ocr_settings['preprocessing'] = []
    
    # Camera Settings
    st.sidebar.subheader("📹 Camera Settings")
    camera_index = st.sidebar.selectbox("Camera Index", [0, 1, 2], help="Try different values if camera doesn't work")
    
    # Main Content with Tabs
    tab1, tab2 = st.tabs(["📁 Upload Image", "📹 Live Camera"])
    
    # Tab 1: Upload Image Processing
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📁 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image for object detection and OCR"
            )
            
            if uploaded_file:
                # Display original image
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)
                
                # File info
                st.info(f"📊 **File Info:** {uploaded_file.name} | Size: {uploaded_file.size} bytes")
        
        with col2:
            st.subheader("🎯 Results")
            
            if uploaded_file:
                # Process button
                if st.button("🔍 Process Image", use_container_width=True, type="primary", key="process_upload"):
                    
                    # Process the image
                    result_image, detected_text, cleaned_text, ocr_results = process_image(
                        image, 
                        st.session_state.yolo_model, 
                        st.session_state.ocr_reader, 
                        ocr_settings
                    )
                    
                    # Store results in session state
                    st.session_state.upload_result_image = result_image
                    st.session_state.upload_detected_text = detected_text
                    st.session_state.upload_cleaned_text = cleaned_text
                    st.session_state.upload_ocr_results = ocr_results
        
        # Results Display for Upload
        if hasattr(st.session_state, 'upload_result_image'):
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Results in columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🖼️ Processed Image")
                st.image(st.session_state.upload_result_image, use_container_width=True)
                
                # Download button for processed image
                img_buffer = io.BytesIO()
                st.session_state.upload_result_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    "📥 Download Processed Image",
                    data=img_buffer.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )
            
            with col2:
                st.markdown("### 📝 Detected Text")
                
                # Statistics
                num_detections = len(st.session_state.upload_ocr_results)
                high_conf_detections = sum(1 for det in st.session_state.upload_ocr_results 
                                         if det[2] > ocr_settings['confidence_threshold'])
                
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Total Detections", num_detections)
                with col_stat2:
                    st.metric("High Confidence", high_conf_detections)
                
                # Raw OCR text
                if st.session_state.upload_detected_text.strip():
                    st.markdown("**Original OCR Output:**")
                    st.text_area("", st.session_state.upload_detected_text, height=150, key="upload_raw_text")
                    
                    # Cleaned text
                    if st.session_state.upload_cleaned_text.strip():
                        st.markdown("**Cleaned Text:**")
                        st.text_area("", st.session_state.upload_cleaned_text, height=100, key="upload_cleaned_text")
                    
                    # Download text
                    st.download_button(
                        "📥 Download Text",
                        data=st.session_state.upload_cleaned_text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("No text detected in the image.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Live Camera Processing
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Live Camera Feed")
            
            # Camera controls
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                if st.button("▶️ Start Camera", use_container_width=True, type="primary"):
                    if st.session_state.camera_handler.initialize_camera(camera_index):
                        st.session_state.camera_handler.start_capture()
                        st.session_state.camera_active = True
                        st.success("Camera started!")
                        st.rerun()
                    else:
                        st.error("Failed to initialize camera!")
            
            with col_btn2:
                if st.button("⏸️ Stop Camera", use_container_width=True):
                    st.session_state.camera_handler.stop_capture()
                    st.session_state.camera_active = False
                    st.success("Camera stopped!")
                    st.rerun()
            
            with col_btn3:
                save_frame = st.button("📷 Save Frame", use_container_width=True)
            
            # Camera display
            camera_placeholder = st.empty()
            
            # Initialize camera state
            if 'camera_active' not in st.session_state:
                st.session_state.camera_active = False
            
            # Live camera processing
            if st.session_state.camera_active:
                # Processing settings
                process_every_n_frames = st.slider("Process every N frames (for performance)", 1, 10, 3)
                frame_counter = 0
                
                while st.session_state.camera_active:
                    frame = st.session_state.camera_handler.get_frame()
                    
                    if frame is not None:
                        frame_counter += 1
                        
                        # Process frame periodically
                        if frame_counter % process_every_n_frames == 0:
                            st.session_state.camera_handler.process_and_queue_result(
                                frame, 
                                st.session_state.yolo_model, 
                                st.session_state.ocr_reader, 
                                ocr_settings
                            )
                        
                        # Get processed result
                        result = st.session_state.camera_handler.get_result()
                        if result is not None:
                            display_frame = result['frame']
                            st.session_state.current_camera_result = result
                        else:
                            # Show raw frame if no processed result available
                            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Convert to PIL for display
                        if isinstance(display_frame, np.ndarray):
                            display_image = Image.fromarray(display_frame)
                        else:
                            display_image = display_frame
                        
                        # Display in Streamlit
                        camera_placeholder.image(display_image, channels="RGB", use_container_width=True)
                        
                        # Save frame if requested
                        if save_frame and hasattr(st.session_state, 'current_camera_result'):
                            timestamp = int(time.time())
                            filename = f"camera_capture_{timestamp}.jpg"
                            
                            # Save the processed frame
                            if isinstance(st.session_state.current_camera_result['frame'], Image.Image):
                                st.session_state.current_camera_result['frame'].save(filename)
                            else:
                                cv2.imwrite(filename, st.session_state.current_camera_result['frame'])
                            
                            st.success(f"Saved frame as {filename}")
                        
                        time.sleep(0.1)  # Control frame rate
                    else:
                        time.sleep(0.1)
            
            else:
                camera_placeholder.markdown("""
                <div class="camera-container">
                    <div style="text-align: center; padding: 100px 0;">
                        <h3>📹 Camera Not Active</h3>
                        <p>Click "Start Camera" to begin live detection</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Live Detection Results")
            
            # Real-time stats
            if hasattr(st.session_state, 'current_camera_result'):
                result = st.session_state.current_camera_result
                
                # Stats
                num_detections = len(result['ocr_results'])
                high_conf_detections = sum(1 for det in result['ocr_results'] 
                                         if det[2] > ocr_settings['confidence_threshold'])
                
                st.metric("🎯 Total Detections", num_detections)
                st.metric("✅ High Confidence", high_conf_detections)
                st.metric("⏰ Last Update", f"{time.time() - result['timestamp']:.1f}s ago")
                
                # Detected text
                if result['text'].strip():
                    st.markdown("**📝 Detected Text:**")
                    st.text_area("Live OCR Results", result['text'], height=200, key="live_ocr_text")
                    
                    if result['cleaned_text'].strip():
                        st.markdown("**🧹 Cleaned Text:**")
                        st.text_area("Cleaned Results", result['cleaned_text'], height=100, key="live_cleaned_text")
                else:
                    st.info("No text detected in current frame")
                
                # Performance info
                st.markdown("---")
                st.markdown("**🚀 Performance:**")
                fps = 1.0 / max(0.1, time.time() - result['timestamp'])
                st.text(f"Processing FPS: ~{fps:.1f}")
                
            else:
                st.info("Start camera to see live detection results")
                
                # Instructions
                st.markdown("""
                **📋 Instructions:**
                1. Click "Start Camera" to begin
                2. Point camera at objects/text
                3. Adjust settings in sidebar
                4. Use "Save Frame" to capture results
                5. Click "Stop Camera" when done
                """)
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Landing page when models aren't loaded
    st.markdown("""
    ## 🚀 Welcome to YOLO + OCR Detection System
    
    This application combines:
    - **YOLOv8**: State-of-the-art object detection
    - **EasyOCR**: Accurate text recognition
    - **Smart Processing**: Automatic text cleaning and correction
    - **Live Camera**: Real-time detection from webcam
    
    ### Features:
    - 📁 Upload any image (PNG, JPG, JPEG)
    - 📹 Live camera detection with real-time processing
    - 🔍 Detect objects and text simultaneously
    - ✍️ Handwriting mode for better handwritten text recognition
    - 🔧 Advanced OCR preprocessing options
    - 📥 Download processed results
    - 📷 Save camera frames with detections
    
    ### Getting Started:
    1. Click "🚀 Load Models" in the sidebar
    2. Choose between "Upload Image" or "Live Camera" tabs
    3. Adjust settings as needed
    4. Start processing!
    
    **Click the "Load Models" button in the sidebar to begin!**
    """)
    
    # Demo image
    st.image("https://via.placeholder.com/800x400/667eea/ffffff?text=YOLO+%2B+OCR+Detection+System", 
             caption="Sample detection visualization")

# Cleanup on app exit
if hasattr(st.session_state, 'camera_handler'):
    # This will run when the session ends
    import atexit
    atexit.register(st.session_state.camera_handler.stop_capture)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    Made with ❤️ using Streamlit | YOLO + EasyOCR<br>
    <small>Perfect for testing object detection and text recognition on images and live camera</small>
</div>
""", unsafe_allow_html=True)