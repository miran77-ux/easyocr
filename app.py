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
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU-only

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="YOLO + OCR Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import check and error handling
import_errors = []

try:
    import cv2
    cv2_available = True
except ImportError as e:
    cv2_available = False
    import_errors.append(f"OpenCV: {e}")

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
    core_imports_available = True
except ImportError as e:
    core_imports_available = False
    import_errors.append(f"Core libraries: {e}")

try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError as e:
    yolo_available = False
    import_errors.append(f"YOLO: {e}")

# Check if all required components are available
if not (cv2_available and core_imports_available and yolo_available):
    st.error("## Import Error")
    st.error("Some required libraries failed to import:")
    
    for error in import_errors:
        st.error(f"• {error}")
    
    st.markdown("""
    ### Solutions for Streamlit Cloud:
    
    1. **Update requirements.txt** with these exact versions:
    ```
    streamlit>=1.28.0
    torch==2.5.1
    torchvision==0.20.1
    ultralytics>=8.1.0
    opencv-python-headless==4.8.1.78
    ```
    
    2. **The issue**: PyTorch 2.1.0 is not available on Python 3.13
    
    3. **Solution**: Use PyTorch 2.5.1 which is available
    
    4. **Restart the app** after making changes
    """)
    
    st.info("Use the updated requirements.txt provided above.")
    st.stop()

# Monkey-patch for Pillow >= 10.0
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 YOLO + OCR Detection System</h1>
    <p>Optimized for Streamlit Cloud - Upload images for detection</p>
    <small>CPU-optimized for cloud deployment</small>
</div>
""", unsafe_allow_html=True)

def setup_pytorch_compatibility():
    """Enhanced PyTorch compatibility for newer versions"""
    try:
        torch_version = torch.__version__
        st.info(f"PyTorch version: {torch_version}")
        
        # For PyTorch 2.5+, the safe globals approach is different
        if hasattr(torch, 'serialization'):
            try:
                # Try to import and configure safe globals
                from torch.serialization import add_safe_globals
                
                safe_classes = []
                
                # Add essential classes
                try:
                    from ultralytics.nn.tasks import DetectionModel
                    safe_classes.append(DetectionModel)
                except ImportError:
                    pass
                
                try:
                    from torch.nn.modules.container import Sequential, ModuleList
                    safe_classes.extend([Sequential, ModuleList])
                except ImportError:
                    pass
                
                try:
                    from collections import OrderedDict
                    safe_classes.append(OrderedDict)
                except ImportError:
                    pass
                
                # Add common ultralytics classes
                try:
                    import ultralytics.nn.modules as ul_modules
                    module_names = ['Conv', 'C2f', 'SPPF', 'Bottleneck', 'DFL', 'Detect', 'Segment']
                    for name in module_names:
                        if hasattr(ul_modules, name):
                            safe_classes.append(getattr(ul_modules, name))
                except ImportError:
                    pass
                
                if safe_classes:
                    add_safe_globals(safe_classes)
                    st.success(f"Added {len(safe_classes)} classes to safe globals")
                    
            except Exception as e:
                st.warning(f"Safe globals setup: {e}")
        
        return True
        
    except Exception as e:
        st.error(f"PyTorch compatibility error: {e}")
        return False

@st.cache_resource
def load_models():
    """Load models with PyTorch 2.5+ compatibility"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Setup compatibility
        status_text.text("Setting up PyTorch compatibility...")
        progress_bar.progress(10)
        
        if not setup_pytorch_compatibility():
            st.warning("Compatibility issues detected, but continuing...")
        
        # Download YOLO model
        status_text.text("Checking YOLO model...")
        progress_bar.progress(30)
        
        model_path = "yolov8n.pt"
        if not Path(model_path).exists():
            status_text.text("Downloading YOLO model (this may take a moment)...")
            progress_bar.progress(40)
            
            try:
                urllib.request.urlretrieve(
                    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                    model_path
                )
                st.success("YOLO model downloaded successfully")
            except Exception as download_error:
                st.error(f"Model download failed: {download_error}")
                raise download_error
        
        # Load YOLO model with PyTorch 2.5+ approach
        status_text.text("Loading YOLO model...")
        progress_bar.progress(60)
        
        model = None
        
        # Strategy 1: Standard loading
        try:
            model = YOLO(model_path)
            st.info("Success with standard loading")
        except Exception as e:
            st.warning(f"Standard loading failed: {str(e)[:100]}...")
            
            # Strategy 2: Force weights_only=False for PyTorch 2.5+
            try:
                # Monkey patch approach for PyTorch 2.5+
                original_load = torch.load
                
                def patched_load(*args, **kwargs):
                    # Force weights_only=False for compatibility
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = patched_load
                try:
                    model = YOLO(model_path)
                    st.info("Success with weights_only bypass")
                finally:
                    torch.load = original_load
                    
            except Exception as strategy2_error:
                st.warning(f"Weights bypass failed: {str(strategy2_error)[:100]}...")
                
                # Strategy 3: Fresh download with bypass
                try:
                    temp_path = os.path.join(tempfile.gettempdir(), "yolov8n_fresh.pt")
                    urllib.request.urlretrieve(
                        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                        temp_path
                    )
                    
                    torch.load = patched_load
                    try:
                        model = YOLO(temp_path)
                        st.info("Success with fresh model")
                    finally:
                        torch.load = original_load
                        
                except Exception as strategy3_error:
                    st.error(f"All strategies failed: {strategy3_error}")
                    raise strategy3_error
        
        if model is None:
            raise Exception("Failed to load YOLO model with all strategies")
        
        # Load OCR model
        status_text.text("Loading OCR model...")
        progress_bar.progress(80)
        
        try:
            # Use CPU-only mode for cloud deployment
            reader = easyocr.Reader(['en'], verbose=False, gpu=False)
            st.success("OCR model loaded successfully")
        except Exception as ocr_error:
            st.error(f"OCR loading failed: {ocr_error}")
            raise ocr_error
        
        progress_bar.progress(100)
        status_text.text("All models loaded successfully!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return model, reader
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        
        error_msg = str(e)
        st.error(f"Model loading failed: {error_msg}")
        
        # Provide specific troubleshooting
        if "weights_only" in error_msg.lower() or "safe" in error_msg.lower():
            st.markdown("""
            ### PyTorch Loading Issue
            This is a PyTorch 2.5+ compatibility issue. Solutions:
            1. The app should handle this automatically
            2. If problems persist, try restarting the app
            3. Check that requirements.txt uses PyTorch 2.5.1
            """)
        elif "download" in error_msg.lower():
            st.markdown("""
            ### Model Download Issue
            1. Check internet connectivity
            2. Try refreshing the app
            3. Model downloads automatically on first run
            """)
        else:
            st.markdown("""
            ### General Troubleshooting
            1. Ensure requirements.txt uses compatible versions
            2. Try restarting the application
            3. Check Streamlit Cloud logs for details
            """)
        
        raise e

def process_image(image, model, reader, ocr_settings):
    """Process image with YOLO and OCR"""
    
    try:
        # Convert PIL to OpenCV
        if isinstance(image, Image.Image):
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_cv = image
        
        # YOLO Detection
        with st.spinner("Running object detection..."):
            results = model(img_cv)
            annotated_frame = results[0].plot()
        
        # OCR Detection
        with st.spinner("Extracting text..."):
            try:
                ocr_results = reader.readtext(img_cv, paragraph=False)
            except Exception as ocr_error:
                st.warning(f"OCR processing warning: {ocr_error}")
                ocr_results = []
        
        # Draw OCR results on image
        for detection in ocr_results:
            try:
                if len(detection) >= 3:
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
            except Exception:
                continue  # Skip problematic detections
        
        # Extract detected text
        detected_text = ""
        for detection in ocr_results:
            if len(detection) >= 3 and detection[2] > ocr_settings.get('confidence_threshold', 0.5):
                detected_text += f"{detection[1]} [Conf: {detection[2]:.2f}]\n"
        
        # Convert back to PIL
        result_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        return result_image, detected_text, ocr_results
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return image, "", []

# Sidebar
st.sidebar.title("Settings")

# System info
with st.sidebar.expander("System Info"):
    st.write(f"Python: {sys.version.split()[0]}")
    if 'torch' in sys.modules:
        st.write(f"PyTorch: {torch.__version__}")
    if 'cv2' in sys.modules:
        st.write(f"OpenCV: {cv2.__version__}")
    st.write("Platform: Streamlit Cloud")

# Model loading
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

if not st.session_state.models_loaded:
    with st.sidebar:
        if st.button("🚀 Load Models", use_container_width=True, type="primary"):
            try:
                model, reader = load_models()
                st.session_state.yolo_model = model
                st.session_state.ocr_reader = reader
                st.session_state.models_loaded = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load models: {e}")

if st.session_state.models_loaded:
    st.sidebar.success("Models loaded successfully!")
    
    # OCR Settings
    st.sidebar.subheader("OCR Settings")
    ocr_settings = {
        'confidence_threshold': st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    }
    
    # Main interface
    st.subheader("Upload Image for Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image for object detection and OCR"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            st.info(f"File: {uploaded_file.name} | Size: {uploaded_file.size} bytes")
    
    with col2:
        st.markdown("### Results")
        
        if uploaded_file and st.button("Process Image", use_container_width=True, type="primary"):
            # Process the image
            result_image, detected_text, ocr_results = process_image(
                image, 
                st.session_state.yolo_model, 
                st.session_state.ocr_reader, 
                ocr_settings
            )
            
            # Store results
            st.session_state.result_image = result_image
            st.session_state.detected_text = detected_text
            st.session_state.ocr_results = ocr_results
    
    # Results display
    if hasattr(st.session_state, 'result_image'):
        st.markdown("---")
        st.subheader("Analysis Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Processed Image")
            st.image(st.session_state.result_image, use_container_width=True)
            
            # Download processed image
            img_buffer = io.BytesIO()
            st.session_state.result_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                "Download Processed Image",
                data=img_buffer.getvalue(),
                file_name="processed_image.png",
                mime="image/png"
            )
        
        with col2:
            st.markdown("### Detected Text")
            
            # Statistics
            num_detections = len(st.session_state.ocr_results)
            high_conf = sum(1 for det in st.session_state.ocr_results 
                           if len(det) >= 3 and det[2] > ocr_settings['confidence_threshold'])
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Total Detections", num_detections)
            with col_stat2:
                st.metric("High Confidence", high_conf)
            
            # Display text
            if st.session_state.detected_text.strip():
                st.text_area("Detected Text", st.session_state.detected_text, height=200)
                
                st.download_button(
                    "Download Text",
                    data=st.session_state.detected_text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )
            else:
                st.info("No text detected in the image.")

else:
    # Welcome page
    st.markdown("""
    ## Welcome to YOLO + OCR Detection System
    
    **Optimized for Streamlit Cloud deployment**
    
    ### Features:
    - Upload images for analysis
    - Object detection with YOLOv8
    - Text recognition with EasyOCR
    - CPU-optimized for cloud deployment
    - Download processed results
    
    ### Getting Started:
    1. Click "🚀 Load Models" in the sidebar
    2. Upload an image using the file uploader
    3. Click "Process Image"
    4. Download your results!
    
    ### Cloud Deployment Notes:
    - First model load may take 2-3 minutes
    - Models are cached after initial load
    - Camera features disabled for cloud deployment
    - Optimized for CPU processing
    
    **Click "Load Models" in the sidebar to begin!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    Made with Streamlit | YOLO + EasyOCR<br>
    <small>Optimized for Streamlit Cloud deployment</small>
</div>
""", unsafe_allow_html=True)
