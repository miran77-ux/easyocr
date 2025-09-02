import streamlit as st
import warnings
import sys
import os

# Suppress warnings early
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Set environment variables for better compatibility
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['HF_HUB_CACHE'] = '/tmp/hf_cache'

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="YOLO + OCR Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import required libraries with error handling
try:
    import cv2
    cv2_available = True
except ImportError as e:
    st.error(f"OpenCV import failed: {e}")
    cv2_available = False

try:
    import numpy as np
    import easyocr
    from PIL import Image, ImageDraw, ImageFont
    import torch
    import urllib.request
    from pathlib import Path
    import tempfile
    import time
    import io
    import base64
    import threading
    import queue
    
    core_imports_available = True
except ImportError as e:
    st.error(f"Core imports failed: {e}")
    core_imports_available = False

# Try YOLO import separately
try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError as e:
    st.error(f"YOLO import failed: {e}")
    yolo_available = False

# Check if all required components are available
if not (cv2_available and core_imports_available and yolo_available):
    st.error("""
    ## ❌ Import Error
    
    Some required libraries failed to import. This is likely due to:
    
    1. **System dependencies missing** - The `packages.txt` file needs system packages
    2. **Library version conflicts** - PyTorch/OpenCV version mismatch
    3. **Streamlit Cloud limitations** - Some packages might not be fully supported
    
    ### 🛠️ Solutions:
    
    1. **Create/Update `packages.txt`** with system dependencies
    2. **Fix `requirements.txt`** with compatible versions
    3. **Restart the app** after making changes
    
    ### 📋 Required Files:
    - `requirements.txt` (with fixed versions)
    - `packages.txt` (with system dependencies)
    """)
    st.stop()

# Monkey-patch for Pillow >= 10.0 (ANTIALIAS removed)
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

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
    <p>Upload images for object detection and text recognition</p>
    <small>⚠️ Live camera disabled on Streamlit Cloud</small>
</div>
""", unsafe_allow_html=True)

def setup_pytorch_compatibility():
    """Setup PyTorch compatibility for YOLO with better error handling"""
    try:
        # Check PyTorch version
        torch_version = torch.__version__
        st.info(f"🔍 PyTorch version: {torch_version}")
        
        # For PyTorch 2.1+, try to set up safe globals
        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
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
                    st.success(f"✅ Added {len(safe_classes)} classes to safe globals")
                    
            except Exception as e:
                st.warning(f"⚠️ Safe globals setup warning: {e}")
        
        return True
        
    except Exception as e:
        st.error(f"❌ PyTorch compatibility setup failed: {e}")
        return False

@st.cache_resource
def load_models():
    """Load YOLO and OCR models with caching and better error handling"""
    
    # Setup compatibility
    if not setup_pytorch_compatibility():
        st.warning("⚠️ Compatibility setup had issues, but continuing...")
    
    # Progress tracking
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
            
            try:
                urllib.request.urlretrieve(
                    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                    model_path
                )
                st.success("✅ YOLO model downloaded")
            except Exception as download_error:
                st.error(f"❌ Failed to download YOLO model: {download_error}")
                raise download_error
        
        # Load YOLO model with multiple fallback methods
        status_text.text("📄 Loading YOLO model...")
        progress_bar.progress(50)
        
        model = None
        load_methods = [
            "standard",
            "weights_only_false",
            "fresh_download"
        ]
        
        for method in load_methods:
            try:
                if method == "standard":
                    model = YOLO(model_path)
                    break
                    
                elif method == "weights_only_false":
                    # Monkey patch for PyTorch 2.1+
                    original_load = torch.load
                    torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
                    try:
                        model = YOLO(model_path)
                        break
                    finally:
                        torch.load = original_load
                        
                elif method == "fresh_download":
                    # Download fresh model to temp location
                    temp_model = os.path.join(tempfile.gettempdir(), "yolov8n_fresh.pt")
                    urllib.request.urlretrieve(
                        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                        temp_model
                    )
                    
                    # Try loading with weights_only=False
                    original_load = torch.load
                    torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
                    try:
                        model = YOLO(temp_model)
                        break
                    finally:
                        torch.load = original_load
                        
            except Exception as method_error:
                st.warning(f"⚠️ Method '{method}' failed: {method_error}")
                continue
        
        if model is None:
            raise Exception("All YOLO loading methods failed")
            
        st.success("✅ YOLO model loaded successfully")
        
        # Load OCR model
        status_text.text("📄 Loading OCR model...")
        progress_bar.progress(80)
        
        try:
            reader = easyocr.Reader(['en'], verbose=False, gpu=False)  # Disable GPU for cloud
            st.success("✅ OCR model loaded successfully")
        except Exception as ocr_error:
            st.error(f"❌ Failed to load OCR model: {ocr_error}")
            raise ocr_error
        
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
        1. Check if all dependencies are installed correctly
        2. Try refreshing the page
        3. Make sure `requirements.txt` has compatible versions
        4. Check if `packages.txt` includes system dependencies
        """)
        raise e

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
        
        # Apply corrections (basic example)
        corrected = line
        for old, new in corrections.items():
            if old in corrected and not corrected.isdigit():  # Don't replace if it's clearly a number
                corrected = corrected.replace(old, new)
        
        cleaned_lines.append(corrected)
    
    return '\n'.join(cleaned_lines)

def process_image(image, model, reader, ocr_settings):
    """Process image with YOLO and OCR"""
    
    try:
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
        if ocr_settings.get('enhance_ocr', False):
            ocr_image = enhance_image_for_ocr(img_cv, ocr_settings.get('preprocessing', []))
        
        # OCR Detection
        try:
            ocr_results = reader.readtext(ocr_image, paragraph=False)
        except Exception as ocr_error:
            st.warning(f"OCR processing warning: {ocr_error}")
            ocr_results = []
        
        # Draw OCR results on image
        for detection in ocr_results:
            try:
                box = np.array(detection[0], dtype=np.int32)
                text = detection[1]
                conf = detection[2]
                
                if conf > ocr_settings.get('confidence_threshold', 0.5):
                    # Draw bounding box
                    cv2.polylines(annotated_frame, [box], True, (0, 255, 0), 2)
                    
                    # Add text label
                    text_pos = (box[0][0], box[0][1] - 10)
                    cv2.putText(annotated_frame, f"{text} ({conf:.2f})", text_pos,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as draw_error:
                continue  # Skip problematic detections
        
        # Extract text
        detected_text = ""
        for detection in ocr_results:
            if len(detection) >= 3 and detection[2] > ocr_settings.get('confidence_threshold', 0.5):
                detected_text += f"{detection[1]} [Conf: {detection[2]:.2f}]\n"
        
        # Clean text
        cleaned_text = clean_ocr_text(detected_text)
        
        # Convert back to PIL
        result_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        return result_image, detected_text, cleaned_text, ocr_results
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return image, "", "", []

# Sidebar Settings
st.sidebar.title("⚙️ Settings")

# System Info
with st.sidebar.expander("🔍 System Info"):
    st.write(f"Python: {sys.version.split()[0]}")
    if 'torch' in sys.modules:
        st.write(f"PyTorch: {torch.__version__}")
    if 'cv2' in sys.modules:
        st.write(f"OpenCV: {cv2.__version__}")
    st.write(f"Platform: Streamlit Cloud")

# Model loading
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

if not st.session_state.models_loaded:
    with st.sidebar:
        if st.button("🚀 Load Models", use_container_width=True):
            try:
                model, reader = load_models()
                st.session_state.yolo_model = model
                st.session_state.ocr_reader = reader
                st.session_state.models_loaded = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load models: {e}")

if st.session_state.models_loaded:
    st.sidebar.success("✅ Models loaded successfully!")
    
    # OCR Settings
    st.sidebar.subheader("🔍 OCR Settings")
    
    ocr_settings = {
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
    
    # Main Content - Image Upload Only (Camera disabled for cloud)
    st.subheader("📁 Upload Image for Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Image")
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
        st.markdown("### Results")
        
        if uploaded_file:
            # Process button
            if st.button("🔍 Process Image", use_container_width=True, type="primary"):
                
                with st.spinner("Processing image..."):
                    # Process the image
                    result_image, detected_text, cleaned_text, ocr_results = process_image(
                        image, 
                        st.session_state.yolo_model, 
                        st.session_state.ocr_reader, 
                        ocr_settings
                    )
                    
                    # Store results in session state
                    st.session_state.result_image = result_image
                    st.session_state.detected_text = detected_text
                    st.session_state.cleaned_text = cleaned_text
                    st.session_state.ocr_results = ocr_results
    
    # Results Display
    if hasattr(st.session_state, 'result_image'):
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Results in columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🖼️ Processed Image")
            st.image(st.session_state.result_image, use_container_width=True)
            
            # Download button for processed image
            img_buffer = io.BytesIO()
            st.session_state.result_image.save(img_buffer, format='PNG')
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
            num_detections = len(st.session_state.ocr_results)
            high_conf_detections = sum(1 for det in st.session_state.ocr_results 
                                     if len(det) >= 3 and det[2] > ocr_settings['confidence_threshold'])
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Total Detections", num_detections)
            with col_stat2:
                st.metric("High Confidence", high_conf_detections)
            
            # Raw OCR text
            if st.session_state.detected_text.strip():
                st.markdown("**Original OCR Output:**")
                st.text_area("", st.session_state.detected_text, height=150, key="raw_text")
                
                # Cleaned text
                if st.session_state.cleaned_text.strip():
                    st.markdown("**Cleaned Text:**")
                    st.text_area("", st.session_state.cleaned_text, height=100, key="cleaned_text")
                
                # Download text
                st.download_button(
                    "📥 Download Text",
                    data=st.session_state.cleaned_text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )
            else:
                st.info("No text detected in the image.")

else:
    # Landing page when models aren't loaded
    st.markdown("""
    ## 🚀 Welcome to YOLO + OCR Detection System
    
    This application combines:
    - **YOLOv8**: State-of-the-art object detection
    - **EasyOCR**: Accurate text recognition
    - **Smart Processing**: Automatic text cleaning and correction
    
    ### 🌟 Features:
    - 📁 Upload any image (PNG, JPG, JPEG)
    - 🔍 Detect objects and text simultaneously
    - 🔧 Advanced OCR preprocessing options
    - 📥 Download processed results
    
    ### 🚀 Getting Started:
    1. Click "🚀 Load Models" in the sidebar
    2. Upload an image using the file uploader
    3. Adjust settings as needed
    4. Click "🔍 Process Image"
    5. Download your results!
    
    ### ⚠️ Note:
    - Live camera is disabled on Streamlit Cloud
    - First model load may take a few minutes
    - Use high-quality images for best results
    
    **Click the "Load Models" button in the sidebar to begin!**
    """)
    
    # Show example
    st.image("https://via.placeholder.com/800x300/667eea/ffffff?text=YOLO+%2B+OCR+Detection+System", 
             caption="Sample detection visualization")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    Made with ❤️ using Streamlit | YOLO + EasyOCR<br>
    <small>✅ Optimized for Streamlit Cloud deployment</small>
</div>
""", unsafe_allow_html=True)
