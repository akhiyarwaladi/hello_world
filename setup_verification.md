# 🚀 Malaria Detection Pipeline - New Machine Setup Verification

**Target Environment**: Ubuntu/Linux with Python 3.12+
**Estimated Setup Time**: 1-2 hours (depending on internet speed)
**Last Updated**: September 13, 2025

## 🔧 **Phase 1: Environment Setup**

### 1.1 System Requirements Check
- [ ] **Operating System**: Linux Ubuntu 20.04+ (tested on Ubuntu 24.04)
- [ ] **Python Version**: 3.12+ (tested with Python 3.12.3)
- [ ] **Memory**: 8GB+ RAM recommended (tested with 11GB)
- [ ] **Storage**: 50GB+ free space (datasets ~6GB + models ~5GB)
- [ ] **Internet**: Stable connection for dataset downloads
- [ ] **Git**: Installed and configured

### 1.2 Repository Setup
```bash
# Clone repository
git clone https://github.com/akhiyarwaladi/malaria_detection.git
cd malaria_detection

# Verify repository structure
- [ ] Directory `scripts/` exists with 14 Python files
- [ ] Directory `config/` exists with 4 YAML files
- [ ] File `requirements.txt` exists
- [ ] File `README.md` and `CLAUDE.md` exist
```

### 1.3 Virtual Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Verify activation
- [ ] Command prompt shows `(venv)` prefix
- [ ] `which python` points to venv/bin/python
```

### 1.4 Dependencies Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Critical verification tests
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"

- [ ] All import commands succeed without errors
- [ ] PyTorch version >= 2.0.0
- [ ] Ultralytics version >= 8.0.0
- [ ] OpenCV version >= 4.8.0
```

## 📊 **Phase 2: Pipeline Verification**

### 2.1 Data Download Test
```bash
# Test data download (should take 15-30 minutes)
python scripts/01_download_datasets.py

# Verify downloads
ls -la data/raw/
- [ ] Directory `data/raw/nih_cell/` exists with files
- [ ] Directory `data/raw/mp_idb/` exists with subdirectories
- [ ] Directory `data/raw/kaggle_nih/` exists (if Kaggle configured)
- [ ] Directory `data/raw/bbbc041/` exists with files
- [ ] Directory `data/raw/plasmoID/` exists with files
- [ ] Directory `data/raw/iml/` exists with files
- [ ] Total directories: 6+ datasets downloaded
```

### 2.2 Detection Dataset Preparation
```bash
# Generate detection dataset with corrected bounding boxes
python scripts/08_parse_mpid_detection.py --output-path data/detection_fixed

# Verify detection dataset
ls -la data/detection_fixed/
- [ ] Directory `images/` contains exactly 103 .jpg files
- [ ] Directory `labels/` contains exactly 103 .txt files
- [ ] File `dataset.yaml` exists with correct path configuration
- [ ] File `annotations/detection_report.json` shows 1,242 total parasites

# Verify YOLO format
head -3 data/detection_fixed/labels/1305121398-0001-R_S.txt
- [ ] Each line format: "0 center_x center_y width height" (normalized coordinates)
- [ ] All values between 0.0 and 1.0
```

### 2.3 Parasite Cropping Verification
```bash
# Extract individual parasites from detection bounding boxes
python scripts/09_crop_parasites_from_detection.py \
  --detection-path data/detection_fixed \
  --output-path data/classification_crops

# Verify cropped dataset
ls -la data/classification_crops/
- [ ] Directory structure: train/parasite/, val/parasite/, test/parasite/
- [ ] Train images: 869 files (verify: `ls data/classification_crops/train/parasite/ | wc -l`)
- [ ] Val images: 186 files (verify: `ls data/classification_crops/val/parasite/ | wc -l`)
- [ ] Test images: 187 files (verify: `ls data/classification_crops/test/parasite/ | wc -l`)
- [ ] Total images: 1,242 cropped parasites
- [ ] Image dimensions: All 128x128 pixels (verify: `file data/classification_crops/train/parasite/*.jpg | head -3`)
- [ ] Files `dataset_summary.json` and `crop_metadata.json` exist
```

## 🏋️ **Phase 3: Training System Test**

### 3.1 Quick Training Test (Detection)
```bash
# Run 1-epoch test training for YOLOv8 detection
python scripts/10_train_yolo_detection.py \
  --data data/detection_fixed/dataset.yaml \
  --epochs 1 \
  --batch 4 \
  --device cpu \
  --name test_yolov8_detection

# Verify training outputs
ls -la results/detection/test_yolov8_detection/
- [ ] Directory `weights/` contains `best.pt` and `last.pt`
- [ ] File `results.csv` contains training metrics
- [ ] Training completed without CUDA/memory errors
```

### 3.2 Quick Training Test (Classification)
```bash
# Run 1-epoch test training for classification
python scripts/11_train_classification_crops.py \
  --data data/classification_crops \
  --epochs 1 \
  --batch 16 \
  --device cpu \
  --name test_classification

# Verify classification training
ls -la results/classification/test_classification/
- [ ] Directory `weights/` contains model files
- [ ] Training logs show no errors
- [ ] Classification accuracy metrics generated
```

### 3.3 Performance Analysis Test
```bash
# Generate comparison report (will work even with partial training)
python scripts/14_compare_models_performance.py \
  --output results/test_comparison_report.md

# Verify analysis output
ls -la results/
- [ ] File `test_comparison_report.md` generated
- [ ] File `test_comparison_report.json` contains model data
- [ ] Report includes training status information
```

## 🎯 **Phase 4: Full Pipeline Validation**

### 4.1 Expected File Structure After Setup
```
malaria_detection/
├── data/
│   ├── raw/                          # ~6GB datasets
│   ├── detection_fixed/              # 103 images + labels
│   └── classification_crops/         # 1,242 cropped parasites
├── results/
│   ├── detection/                    # Training outputs
│   ├── classification/               # Classification results
│   └── test_comparison_report.md     # Analysis report
├── scripts/                          # 14 pipeline scripts
├── config/                           # 4 configuration files
├── machine_specs.txt                 # This machine specifications
├── working_requirements.txt          # Exact dependency versions
└── README.md                         # Main documentation
```

### 4.2 Verification Commands
```bash
# Count verification
echo "Raw datasets: $(ls data/raw/ | wc -l)"           # Should be 6+
echo "Detection images: $(ls data/detection_fixed/images/ | wc -l)"  # Should be 103
echo "Detection labels: $(ls data/detection_fixed/labels/ | wc -l)"  # Should be 103
echo "Cropped parasites: $(find data/classification_crops/ -name "*.jpg" | wc -l)" # Should be 1,242
echo "Training scripts: $(ls scripts/*train*.py | wc -l)"  # Should be 5+
echo "Config files: $(ls config/*.yaml | wc -l)"       # Should be 4

- [ ] All counts match expected values
- [ ] No error messages in verification commands
```

### 4.3 Memory and Performance Check
```bash
# Check system resources during training
free -h  # Available memory
df -h .  # Available disk space

# Recommended minimum requirements
- [ ] Available RAM: 4GB+ during training
- [ ] Available storage: 20GB+ free space
- [ ] CPU: Multi-core (4+ cores recommended)
```

## ✅ **Phase 5: Ready for Production**

### 5.1 Full Training Commands (After verification passes)
```bash
# Full detection training (2-4 hours on CPU)
python scripts/10_train_yolo_detection.py --epochs 30 --name yolov8_production
python scripts/12_train_yolo11_detection.py --epochs 20 --name yolo11_production
python scripts/13_train_rtdetr_detection.py --epochs 20 --name rtdetr_production

# Full classification training (1-2 hours on CPU)
python scripts/11_train_classification_crops.py --epochs 25 --name classification_production

# Final performance analysis
python scripts/14_compare_models_performance.py --output results/final_comparison.md
```

### 5.2 Success Criteria
- [ ] All training processes complete without errors
- [ ] Model weights (.pt files) generated for all approaches
- [ ] Performance metrics show reasonable accuracy (>80%)
- [ ] Final comparison report generated successfully
- [ ] Research paper data ready for analysis

## 🚨 **Troubleshooting Common Issues**

### Issue 1: Import Errors
```bash
# If import errors occur, reinstall dependencies
pip uninstall -y torch torchvision ultralytics
pip install torch torchvision ultralytics --force-reinstall
```

### Issue 2: Memory Issues During Training
```bash
# Reduce batch size if out of memory
--batch 2    # Instead of default 8
--device cpu # Force CPU if GPU issues
```

### Issue 3: Download Failures
```bash
# Check internet connection and retry
python scripts/01_download_datasets.py --retry 3
```

### Issue 4: Permission Issues
```bash
# Fix file permissions
chmod +x scripts/*.py
chmod -R 755 data/
```

## 📞 **Support Information**

- **Repository**: https://github.com/akhiyarwaladi/malaria_detection
- **Documentation**: README.md (main documentation)
- **Pipeline Reference**: All commands verified on Ubuntu 24.04, Python 3.12.3
- **Hardware Tested**: Intel i7-6700HQ, 11GB RAM, GTX 960M (CPU training mode)

---

**✅ Verification Complete**: If all checkboxes are ticked, your new machine is ready for malaria detection research!