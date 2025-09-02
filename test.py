# Install required libraries first:
# pip install ultralytics opencv-python torch torchvision

import cv2
import torch
from ultralytics import YOLO
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

def setup_pytorch_compatibility():
    """Setup PyTorch 2.6+ compatibility for YOLO model loading"""
    try:
        from torch.serialization import add_safe_globals
        print("🔧 Setting up PyTorch 2.6+ compatibility...")
        
        # List of classes that need to be added to safe globals
        safe_classes = []
        
        # Try to import and add ultralytics classes
        try:
            from ultralytics.nn.tasks import DetectionModel
            safe_classes.append(DetectionModel)
            print("✅ Added DetectionModel to safe globals")
        except ImportError:
            print("⚠️ DetectionModel not found, skipping")
        
        try:
            from torch.nn.modules.container import Sequential
            safe_classes.append(Sequential)
            print("✅ Added Sequential to safe globals")
        except ImportError:
            print("⚠️ Sequential not found, skipping")
        
        # Try to import ultralytics modules dynamically
        ultralytics_modules = [
            'Conv', 'C2f', 'SPPF', 'Bottleneck', 'DFL', 
            'Detect', 'Segment', 'Concat', 'CBAM', 
            'ChannelAttention', 'SpatialAttention', 'Classify', 'Pose'
        ]
        
        try:
            import ultralytics.nn.modules as ul_modules
            for module_name in ultralytics_modules:
                if hasattr(ul_modules, module_name):
                    safe_classes.append(getattr(ul_modules, module_name))
                    print(f"✅ Added {module_name} to safe globals")
                else:
                    print(f"⚠️ {module_name} not found in ultralytics.nn.modules")
        except ImportError:
            print("⚠️ ultralytics.nn.modules not accessible")
        
        # Add collections.OrderedDict which is commonly needed
        try:
            from collections import OrderedDict
            safe_classes.append(OrderedDict)
            print("✅ Added OrderedDict to safe globals")
        except ImportError:
            pass
        
        # Apply safe globals if we have any classes to add
        if safe_classes:
            add_safe_globals(safe_classes)
            print(f"✅ Successfully added {len(safe_classes)} classes to PyTorch safe globals")
        else:
            print("⚠️ No classes found to add to safe globals")
            
    except ImportError:
        print("ℹ️ PyTorch version doesn't need safe_globals (likely < 2.6)")
        return True
    except Exception as e:
        print(f"❌ Error setting up compatibility: {e}")
        return False
    
    return True

def load_yolo_model_safely(model_path="yolov8n.pt"):
    """Safely load YOLO model with multiple fallback methods"""
    
    try:
        # Method 1: Try normal loading first
        print("🔄 Attempting standard YOLO model loading...")
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        error_str = str(e).lower()
        print(f"❌ Standard loading failed: {e}")
        
        # Check if it's a PyTorch 2.6+ weights_only issue
        if any(keyword in error_str for keyword in ['weightsunpickler', 'weights_only', 'global', 'safe_globals']):
            print("🔧 Detected PyTorch 2.6+ weights_only restriction...")
            
            # Method 2: Try monkey patching torch.load
            try:
                print("🔄 Trying torch.load monkey patch...")
                original_load = torch.load
                
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                # Apply patch
                torch.load = patched_load
                
                try:
                    model = YOLO(model_path)
                    print("✅ Model loaded with monkey patch!")
                    return model
                finally:
                    # Always restore original function
                    torch.load = original_load
                    
            except Exception as patch_error:
                print(f"❌ Monkey patch failed: {patch_error}")
                
                # Method 3: Try downloading fresh model
                try:
                    print("🔄 Downloading fresh model...")
                    import urllib.request
                    import tempfile
                    import os
                    
                    # Download to temp location
                    temp_model = os.path.join(tempfile.gettempdir(), "yolov8n_fresh.pt")
                    urllib.request.urlretrieve(
                        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                        temp_model
                    )
                    
                    # Try loading with patch
                    torch.load = patched_load
                    try:
                        model = YOLO(temp_model)
                        print("✅ Fresh model loaded successfully!")
                        return model
                    finally:
                        torch.load = original_load
                        
                except Exception as fresh_error:
                    print(f"❌ Fresh model download failed: {fresh_error}")
        
        # If all methods fail, provide helpful error message
        print("\n❌ All loading methods failed!")
        print("\n🛠️ **SOLUTION OPTIONS:**")
        print("1. Downgrade PyTorch:")
        print("   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1")
        print("\n2. Update Ultralytics:")
        print("   pip install --upgrade ultralytics")
        print("\n3. Complete reinstall:")
        print("   pip uninstall torch torchvision ultralytics -y")
        print("   pip install torch==2.5.1 torchvision==0.20.1 ultralytics")
        print("\n4. Delete existing model file:")
        print("   del yolov8n.pt  # (Windows)")
        print("   rm yolov8n.pt   # (Mac/Linux)")
        
        raise e

def main():
    """Main function to run object detection"""
    
    print("🚀 Starting YOLO Object Detection")
    print("=" * 50)
    
    # Setup PyTorch compatibility
    if not setup_pytorch_compatibility():
        print("⚠️ Compatibility setup had issues, but continuing...")
    
    print("\n" + "=" * 50)
    
    # Load YOLO model safely
    try:
        model = load_yolo_model_safely("yolov8n.pt")
    except Exception as e:
        print(f"\n❌ Failed to load YOLO model: {e}")
        print("Please try the solutions mentioned above.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("📹 Initializing camera...")
    
    # Open webcam (0 = default laptop camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        print("💡 Solutions:")
        print("- Check if camera is being used by another app")
        print("- Try different camera index (1, 2, etc.)")
        print("- Check camera permissions")
        sys.exit(1)
    
    # Optional: increase camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✅ Camera initialized successfully!")
    print("\n🎯 Starting object detection...")
    print("💡 Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to read frame from camera")
                break
            
            try:
                # Run object detection
                results = model(frame)
                
                # Draw detections on frame
                annotated_frame = results[0].plot()
                
                # Add frame counter
                cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow("YOLO Object Detection - Press 'q' to quit", annotated_frame)
                
                frame_count += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Quit requested by user")
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"detection_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"💾 Saved frame as {filename}")
                    
            except Exception as detection_error:
                print(f"❌ Error during detection: {detection_error}")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user (Ctrl+C)")
    
    finally:
        print("🧹 Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete!")

if __name__ == "__main__":
    main()