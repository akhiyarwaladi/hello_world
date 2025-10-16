@echo off
echo ========================================
echo Setup Conda Environment for Malaria Detection Pipeline
echo ========================================
echo.

echo [1/5] Activating conda base environment...
call conda activate base
if errorlevel 1 (
    echo ERROR: Failed to activate conda base environment
    exit /b 1
)

echo.
echo [2/5] Installing PyTorch with CUDA 12.8 support...
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    exit /b 1
)

echo.
echo [3/5] Installing Ultralytics (YOLO)...
pip install ultralytics==8.3.202
if errorlevel 1 (
    echo ERROR: Failed to install Ultralytics
    exit /b 1
)

echo.
echo [4/5] Installing remaining dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some packages failed to install, but continuing...
)

echo.
echo [5/5] Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo You can now run the main pipeline:
echo   python main_pipeline.py
echo.
echo For quick test:
echo   python main_pipeline.py --dataset iml_lifecycle --include yolo11 --classification-models densenet121 --epochs-det 5 --epochs-cls 5 --no-zip
echo.
pause
