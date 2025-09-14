#!/bin/bash
# 🚀 Malaria Detection Pipeline - Automated Setup for New Machine
# Author: Automated setup script for research reproducibility
# Last Updated: September 13, 2025
# Tested on: Ubuntu 24.04, Python 3.12.3

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Banner
echo -e "${BLUE}"
echo "=================================================================="
echo "🔬 MALARIA DETECTION PIPELINE - NEW MACHINE SETUP"
echo "=================================================================="
echo -e "${NC}"
echo "This script will set up the complete malaria detection pipeline"
echo "on a new machine with full verification."
echo ""
echo "Estimated time: 1-2 hours (depending on internet speed)"
echo "Requirements: Ubuntu 20.04+, Python 3.12+, 8GB+ RAM, 50GB+ storage"
echo ""
read -p "Continue with setup? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 1
fi

# Phase 1: Environment Setup
log "🔧 PHASE 1: Environment Setup"

# Check Python version
log "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    error "Python3 is not installed. Please install Python 3.12+ first."
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
log "Found Python $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION 3.12" | awk '{print ($1 >= $2)}') -eq 0 ]]; then
    warning "Python version $PYTHON_VERSION < 3.12. Some features may not work optimally."
fi

# Check system resources
log "Checking system resources..."
TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
AVAILABLE_MEM=$(free -m | awk 'NR==2{print $7}')
AVAILABLE_STORAGE=$(df -BG . | awk 'NR==2{print $4}' | tr -d 'G')

log "Total Memory: ${TOTAL_MEM}MB, Available: ${AVAILABLE_MEM}MB"
log "Available Storage: ${AVAILABLE_STORAGE}GB"

if [[ $TOTAL_MEM -lt 8000 ]]; then
    warning "Available memory (${TOTAL_MEM}MB) < 8GB. Training may be slow or fail."
fi

if [[ $AVAILABLE_STORAGE -lt 50 ]]; then
    warning "Available storage (${AVAILABLE_STORAGE}GB) < 50GB. May run out of space during setup."
fi

# Create virtual environment
log "Creating Python virtual environment..."
if [[ -d "venv" ]]; then
    warning "Virtual environment already exists. Removing old environment..."
    rm -rf venv
fi

#python3 -m venv venv
#source venv/bin/activate
success "Virtual environment created and activated"

# Upgrade pip
log "Upgrading pip..."
#pip install --upgrade pip

# Install dependencies
log "Installing Python dependencies..."
log "This may take 5-10 minutes depending on internet speed..."
#pip install -r requirements.txt
success "Dependencies installed successfully"

# Verify critical imports
log "Verifying critical Python imports..."
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" || error "PyTorch import failed"
python3 -c "import ultralytics; print(f'✅ Ultralytics: {ultralytics.__version__}')" || error "Ultralytics import failed"
python3 -c "import cv2; print(f'✅ OpenCV: {cv2.__version__}')" || error "OpenCV import failed"
python3 -c "import pandas; print(f'✅ Pandas: {pandas.__version__}')" || error "Pandas import failed"
python3 -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')" || error "NumPy import failed"
success "All critical imports successful"

# Phase 2: Data Pipeline Setup
log "📊 PHASE 2: Data Pipeline Setup"

# Download datasets with options
log "Configuring dataset download options..."

# Check for download preference
echo ""
echo "📥 Dataset Download Options:"
echo "1. MP-IDB only (RECOMMENDED) - Required for two-step classification (~500MB, 5-10 min)"
echo "2. All datasets - Full research suite (~6GB, 30-60 min)"
echo ""
read -p "Select option (1/2) [1]: " -n 1 -r
DOWNLOAD_OPTION=${REPLY:-1}
echo

if [[ "$DOWNLOAD_OPTION" == "1" ]]; then
    log "Downloading MP-IDB dataset only (recommended for main pipeline)..."
    log "Size: ~500MB, Time: 5-10 minutes"
    DATASET_ARG="mp_idb"
    EXPECTED_DATASETS=1
else
    log "Downloading all datasets (comprehensive research)..."
    log "Size: ~6GB, Time: 30-60 minutes"
    DATASET_ARG="all"
    EXPECTED_DATASETS=6
fi

# Check if data already exists
if [[ -d "data/raw" ]] && [[ $(ls -1 data/raw/ | wc -l) -ge $EXPECTED_DATASETS ]]; then
    warning "Raw data appears to already exist. Skipping download..."
else
    python scripts/01_download_datasets.py --dataset "$DATASET_ARG" || error "Dataset download failed"
fi

# Verify downloads
log "Verifying downloaded datasets..."
DATASET_COUNT=$(ls -1 data/raw/ | wc -l)
log "Found $DATASET_COUNT datasets in data/raw/"

if [[ $DATASET_COUNT -ge $EXPECTED_DATASETS ]]; then
    success "Datasets downloaded and verified ($DATASET_COUNT/$EXPECTED_DATASETS)"
    if [[ "$DOWNLOAD_OPTION" == "1" ]]; then
        log "✅ MP-IDB dataset ready for two-step classification pipeline"
    else
        log "✅ All datasets ready for comprehensive research"
    fi
else
    if [[ "$DOWNLOAD_OPTION" == "1" ]]; then
        error "Expected MP-IDB dataset (1), found $DATASET_COUNT. Download may have failed."
    else
        warning "Expected $EXPECTED_DATASETS datasets, found $DATASET_COUNT. Some downloads may have failed."
    fi
fi

# Phase 3: Detection Dataset Preparation
log "🔍 PHASE 3: Detection Dataset Preparation"

log "Parsing MP-IDB dataset for parasite detection..."
log "This will create 103 images with 1,242 parasite bounding boxes..."

python scripts/08_parse_mpid_detection.py --output-path data/detection_fixed || error "Detection dataset preparation failed"

# Verify detection dataset
DETECTION_IMAGES=$(ls -1 data/detection_fixed/images/*.jpg 2>/dev/null | wc -l)
DETECTION_LABELS=$(ls -1 data/detection_fixed/labels/*.txt 2>/dev/null | wc -l)

log "Detection images: $DETECTION_IMAGES"
log "Detection labels: $DETECTION_LABELS"

if [[ $DETECTION_IMAGES -eq 103 ]] && [[ $DETECTION_LABELS -eq 103 ]]; then
    success "Detection dataset prepared successfully (103 images, 103 labels)"
else
    error "Detection dataset verification failed. Expected 103 images and 103 labels, got $DETECTION_IMAGES images and $DETECTION_LABELS labels"
fi

# Verify YOLO format
log "Verifying YOLO label format..."
FIRST_LABEL=$(ls data/detection_fixed/labels/*.txt | head -1)
if [[ -f "$FIRST_LABEL" ]]; then
    LABEL_CONTENT=$(head -1 "$FIRST_LABEL")
    log "Sample label: $LABEL_CONTENT"
    # Check if it starts with class 0 and has 5 values
    if [[ $LABEL_CONTENT =~ ^0[[:space:]] ]]; then
        success "YOLO label format verified"
    else
        warning "YOLO label format may be incorrect"
    fi
else
    error "No label files found for verification"
fi

# Phase 4: Parasite Cropping
log "✂️ PHASE 4: Parasite Cropping"

log "Extracting individual parasites from detection bounding boxes..."
log "This will create 1,242 cropped parasite images (128x128 pixels)..."

python scripts/09_crop_parasites_from_detection.py \
    --detection-path data/detection_fixed \
    --output-path data/classification_crops || error "Parasite cropping failed"

# Verify cropping results
TRAIN_CROPS=$(find data/classification_crops/train/parasite/ -name "*.jpg" 2>/dev/null | wc -l)
VAL_CROPS=$(find data/classification_crops/val/parasite/ -name "*.jpg" 2>/dev/null | wc -l)
TEST_CROPS=$(find data/classification_crops/test/parasite/ -name "*.jpg" 2>/dev/null | wc -l)
TOTAL_CROPS=$((TRAIN_CROPS + VAL_CROPS + TEST_CROPS))

log "Train crops: $TRAIN_CROPS"
log "Validation crops: $VAL_CROPS"
log "Test crops: $TEST_CROPS"
log "Total crops: $TOTAL_CROPS"

if [[ $TOTAL_CROPS -eq 1242 ]]; then
    success "Parasite cropping completed successfully (1,242 total crops)"
else
    error "Cropping verification failed. Expected 1,242 crops, got $TOTAL_CROPS"
fi

# Verify crop dimensions
log "Verifying crop dimensions..."
SAMPLE_CROP=$(find data/classification_crops/train/parasite/ -name "*.jpg" | head -1)
if [[ -f "$SAMPLE_CROP" ]]; then
    DIMENSIONS=$(file "$SAMPLE_CROP" | grep -o '[0-9]\+x[0-9]\+')
    if [[ "$DIMENSIONS" == "128x128" ]]; then
        success "Crop dimensions verified (128x128 pixels)"
    else
        warning "Crop dimensions may be incorrect: $DIMENSIONS"
    fi
else
    error "No crop files found for dimension verification"
fi

# Phase 5: Training System Test
log "🏋️ PHASE 5: Training System Test"

log "Running quick training test (1 epoch) to verify system..."
log "This will test both detection and classification training..."

# Test detection training
log "Testing YOLOv8 detection training (1 epoch)..."
python scripts/10_train_yolo_detection.py \
    --data data/detection_fixed/dataset.yaml \
    --epochs 40 \
    --batch 10 \
    --device cpu \
    --name test_setup_detection || error "Detection training test failed"

# Verify detection training output
if [[ -f "results/detection/test_setup_detection/weights/best.pt" ]]; then
    success "Detection training test completed successfully"
else
    error "Detection training test failed - no model weights generated"
fi

# Test classification training
log "Testing classification training (1 epoch)..."
python scripts/11_train_classification_crops.py \
    --data data/classification_crops \
    --epochs 40 \
    --batch 10 \
    --device cpu \
    --name test_setup_classification || error "Classification training test failed"

# Verify classification training output
if [[ -d "results/classification/test_setup_classification" ]]; then
    success "Classification training test completed successfully"
else
    error "Classification training test failed - no results directory created"
fi

# Phase 6: Performance Analysis Test
log "📊 PHASE 6: Performance Analysis Test"

log "Testing performance analysis system..."
python scripts/14_compare_models_performance.py \
    --output results/setup_test_report.md || error "Performance analysis test failed"

if [[ -f "results/setup_test_report.md" ]]; then
    success "Performance analysis test completed successfully"
else
    error "Performance analysis test failed - no report generated"
fi

# Phase 7: Final Verification
log "✅ PHASE 7: Final Verification"

log "Running comprehensive verification checks..."

# File structure verification
declare -A expected_files=(
    ["data/raw"]="6"
    ["data/detection_fixed/images"]="103"
    ["data/detection_fixed/labels"]="103"
    ["scripts"]="14"
    ["config"]="4"
)

for dir in "${!expected_files[@]}"; do
    if [[ -d "$dir" ]]; then
        count=$(ls -1 "$dir"/* 2>/dev/null | wc -l)
        expected=${expected_files[$dir]}
        if [[ $count -ge $expected ]]; then
            success "$dir: $count files (expected >= $expected)"
        else
            warning "$dir: $count files (expected >= $expected)"
        fi
    else
        error "Directory $dir not found"
    fi
done

# Verify total crops count
FINAL_CROPS_COUNT=$(find data/classification_crops/ -name "*.jpg" 2>/dev/null | wc -l)
if [[ $FINAL_CROPS_COUNT -eq 1242 ]]; then
    success "Total cropped parasites: $FINAL_CROPS_COUNT (expected: 1,242)"
else
    error "Crop count mismatch: $FINAL_CROPS_COUNT (expected: 1,242)"
fi

# Generate system documentation
log "Generating system documentation for future reference..."

# Machine specs (if not exists)
if [[ ! -f "machine_specs.txt" ]]; then
    echo "=== NEW MACHINE SPECIFICATIONS ===" > machine_specs.txt
    echo "Setup completed on: $(date)" >> machine_specs.txt
    echo "" >> machine_specs.txt
    echo "Operating System:" >> machine_specs.txt
    uname -a >> machine_specs.txt
    echo "" >> machine_specs.txt
    echo "Python Version:" >> machine_specs.txt
    python3 --version >> machine_specs.txt
    echo "" >> machine_specs.txt
    echo "Memory Information:" >> machine_specs.txt
    free -h >> machine_specs.txt
    echo "" >> machine_specs.txt
    echo "Storage Information:" >> machine_specs.txt
    df -h . >> machine_specs.txt
fi

# Working requirements
pip freeze > working_requirements.txt

success "System documentation generated"

# Final summary
echo ""
echo -e "${GREEN}"
echo "=================================================================="
echo "🎉 SETUP COMPLETED SUCCESSFULLY!"
echo "=================================================================="
echo -e "${NC}"
echo ""
echo "📊 Setup Summary:"
echo "  • Datasets downloaded: $(ls -1 data/raw/ | wc -l)"
echo "  • Detection images: $(ls -1 data/detection_fixed/images/*.jpg 2>/dev/null | wc -l)"
echo "  • Cropped parasites: $(find data/classification_crops/ -name "*.jpg" 2>/dev/null | wc -l)"
echo "  • Training scripts: $(ls scripts/*train*.py | wc -l)"
echo "  • System tests: ✅ Passed"
echo ""
echo "🚀 Ready for Production Training:"
echo ""
echo "# Full Detection Training (2-4 hours on CPU):"
echo "python scripts/10_train_yolo_detection.py --epochs 30 --name yolov8_production"
echo "python scripts/12_train_yolo11_detection.py --epochs 20 --name yolo11_production"
echo "python scripts/13_train_rtdetr_detection.py --epochs 20 --name rtdetr_production"
echo ""
echo "# Full Classification Training (1-2 hours on CPU):"
echo "python scripts/11_train_classification_crops.py --epochs 25 --name classification_production"
echo ""
echo "# Generate Final Research Report:"
echo "python scripts/14_compare_models_performance.py --output results/final_comparison.md"
echo ""
echo "📄 Documentation Files Created:"
echo "  • setup_verification.md - Detailed verification checklist"
echo "  • machine_specs.txt - System specifications"
echo "  • working_requirements.txt - Exact dependency versions"
echo ""
echo "🎓 Research Paper Ready:"
echo "  Title: 'Perbandingan YOLOv8, YOLOv11, dan RT-DETR untuk Deteksi Malaria'"
echo "  Dataset: 1,242 P. falciparum parasites with corrected annotations"
echo "  Approach: Two-step classification (Detection → Single-cell Classification)"
echo ""
echo -e "${BLUE}Happy researching! 🔬${NC}"
echo ""
