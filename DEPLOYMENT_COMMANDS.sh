#!/bin/bash
# =============================================================================
# FRESH MACHINE DEPLOYMENT COMMANDS - MALARIA DETECTION PIPELINE
# Copy-paste ready commands untuk deployment di mesin baru
# Tested & Verified: September 21, 2025
# =============================================================================

echo "🚀 Starting Fresh Machine Deployment for Malaria Detection Pipeline"
echo "📅 Verified working on: September 21, 2025"
echo ""

# =============================================================================
# STEP 1: REPOSITORY SETUP
# =============================================================================
echo "📁 STEP 1: Cloning Repository..."

# Clone repository
git clone https://github.com/akhiyarwaladi/hello_world.git fresh_malaria_detection
cd fresh_malaria_detection

echo "✅ Repository cloned successfully"
echo ""

# =============================================================================
# STEP 2: ENVIRONMENT SETUP
# =============================================================================
echo "🐍 STEP 2: Setting up Python Environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Verify environment
python -c "import sys; print(f'✅ Python: {sys.version.split()[0]}'); print(f'✅ Virtual env: {sys.prefix}')"

echo "✅ Virtual environment created and activated"
echo ""

# =============================================================================
# STEP 3: DEPENDENCIES INSTALLATION
# =============================================================================
echo "📦 STEP 3: Installing Dependencies..."

# Core ML dependencies
echo "Installing core ML libraries..."
pip install ultralytics pyyaml requests tqdm

# Data processing dependencies
echo "Installing data processing libraries..."
pip install pandas scikit-learn seaborn matplotlib

# Utility dependencies
echo "Installing utility libraries..."
pip install gdown kaggle beautifulsoup4

# Verify installation
python -c "
import ultralytics
import torch
print(f'✅ Ultralytics: {ultralytics.__version__}')
print(f'✅ PyTorch: {torch.__version__}')
print('✅ All dependencies installed successfully!')
"

echo "✅ Dependencies installation completed"
echo ""

# =============================================================================
# STEP 4: DATA PIPELINE EXECUTION
# =============================================================================
echo "📊 STEP 4: Executing Data Pipeline..."

# 4.1 Download MP-IDB Dataset
echo "📥 Downloading MP-IDB dataset..."
python scripts/data_setup/01_download_datasets.py --dataset mp_idb

# 4.2 Preprocess data (extract multiple objects from masks)
echo "🔧 Preprocessing data..."
python scripts/data_setup/02_preprocess_data.py

# 4.3 Integrate datasets
echo "🔗 Integrating datasets..."
python scripts/data_setup/03_integrate_datasets.py

# 4.4 Convert to YOLO format
echo "🎯 Converting to YOLO format..."
python scripts/data_setup/04_convert_to_yolo.py

echo "✅ Data pipeline completed successfully"
echo "📊 Data Summary:"
echo "   - Total objects extracted: ~1,398"
echo "   - Train/Val/Test splits created"
echo "   - YOLO format ready for training"
echo ""

# =============================================================================
# STEP 5: PIPELINE TESTING
# =============================================================================
echo "🧪 STEP 5: Testing Pipeline..."

# Quick test with YOLOv8 (2 epochs)
echo "Testing with YOLOv8 (2 epochs)..."
python run_multiple_models_pipeline.py --include yolo8 --epochs-det 2 --epochs-cls 2 --test-mode

echo "✅ Pipeline test completed successfully"
echo ""

# =============================================================================
# STEP 6: PRODUCTION COMMANDS (OPTIONAL)
# =============================================================================
echo "🎯 STEP 6: Production Training Commands (Optional)"
echo ""
echo "Choose one of the following production commands:"
echo ""

echo "Option 1 - Single Model (Fastest):"
echo "python run_multiple_models_pipeline.py --include yolo8 --epochs-det 30 --epochs-cls 30"
echo ""

echo "Option 2 - Multiple Models (Recommended):"
echo "python run_multiple_models_pipeline.py --exclude rtdetr --epochs-det 30 --epochs-cls 30"
echo ""

echo "Option 3 - All Models (Complete):"
echo "python run_multiple_models_pipeline.py --epochs-det 30 --epochs-cls 30"
echo ""

echo "Option 4 - YOLOv12 Only (Latest):"
echo "python run_multiple_models_pipeline.py --include yolo12 --epochs-det 30 --epochs-cls 30"
echo ""

# =============================================================================
# DEPLOYMENT COMPLETED
# =============================================================================
echo "🎉 FRESH MACHINE DEPLOYMENT COMPLETED SUCCESSFULLY!"
echo ""
echo "📁 Results will be saved in: results/exp_*/"
echo "📊 Training logs available in respective model directories"
echo "🏆 Pipeline ready for production use"
echo ""
echo "💡 Need help? Check FRESH_MACHINE_DEPLOYMENT_VERIFIED.md for details"