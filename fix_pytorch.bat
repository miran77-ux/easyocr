@echo off
echo ====================================
echo FIXING PYTORCH VERSION CONFLICTS
echo ====================================

echo.
echo 🔍 Current PyTorch setup has conflicts:
echo   - torch 2.6.0 (installed)
echo   - torchvision 0.23.0 (requires torch 2.8.0)
echo   - PyTorch 2.6.0 causes YOLO loading issues
echo.

echo 🔧 SOLUTION: Install compatible PyTorch 2.5.1 versions
echo.

echo Step 1: Uninstalling conflicting PyTorch packages...
pip uninstall torch torchvision torchaudio -y

echo.
echo Step 2: Installing compatible PyTorch 2.5.1 versions...
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --user

echo.
echo Step 3: Installing other required packages...
pip install ultralytics==8.0.196 --user
pip install streamlit opencv-python-headless easyocr --user

echo.
echo Step 4: Verifying installation...
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'✅ TorchVision: {torchvision.__version__}')"
python -c "from ultralytics import YOLO; print('✅ YOLO import successful')"
python -c "import streamlit; print('✅ Streamlit OK')"
python -c "import easyocr; print('✅ EasyOCR OK')"

echo.
echo ====================================
echo INSTALLATION COMPLETE!
echo ====================================
echo.
echo ✅ Now you can run: streamlit run app.py
echo.
pause