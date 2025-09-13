# 🔒 Critical Files Backup - New Machine Setup

**Purpose**: Essential files that MUST be preserved for complete reproduction
**Last Updated**: September 13, 2025

## 📁 **Files That MUST Be in Repository**

### 1. Configuration Files (CRITICAL)
```
config/
├── dataset_config.yaml    # Dataset URLs and processing parameters
├── class_names.yaml      # Class definitions and colors
├── training.yaml         # Training hyperparameters
└── datasets.yaml         # YOLO dataset configurations
```

### 2. Script Files (CRITICAL)
```
scripts/
├── 01_download_datasets.py           # Data download automation
├── 02_preprocess_data.py             # Image preprocessing (unused in current pipeline)
├── 03_integrate_datasets.py          # Dataset integration (unused in current pipeline)
├── 04_convert_to_yolo.py             # Format conversion (unused in current pipeline)
├── 05_augment_data.py                # Data augmentation (unused in current pipeline)
├── 06_split_dataset.py               # Dataset splitting (unused in current pipeline)
├── 07_train_yolo_quick.py            # Quick training (legacy)
├── 08_parse_mpid_detection.py        # ⭐ MP-IDB detection parsing (CRITICAL)
├── 09_crop_parasites_from_detection.py # ⭐ Parasite cropping (CRITICAL)
├── 10_train_yolo_detection.py        # ⭐ YOLOv8 detection training (CRITICAL)
├── 11_train_classification_crops.py  # ⭐ Classification training (CRITICAL)
├── 12_train_yolo11_detection.py      # ⭐ YOLOv11 detection training (CRITICAL)
├── 13_train_rtdetr_detection.py      # ⭐ RT-DETR detection training (CRITICAL)
├── 14_compare_models_performance.py  # ⭐ Performance analysis (CRITICAL)
└── utils/                            # Utility functions
    ├── __init__.py
    ├── download_utils.py
    ├── image_utils.py
    └── annotation_utils.py
```

### 3. Documentation Files (CRITICAL)
```
├── README.md                    # ⭐ Main documentation (CRITICAL)
├── CLAUDE.md                   # Context for Claude AI
├── requirements.txt            # ⭐ Python dependencies (CRITICAL)
├── setup_verification.md       # ⭐ New machine verification (CRITICAL)
├── quick_setup_new_machine.sh  # ⭐ Automated setup script (CRITICAL)
├── machine_specs.txt           # Current machine specifications
├── working_requirements.txt    # Exact working dependency versions
└── .gitignore                  # Git ignore patterns
```

## 🚫 **Files That Should NOT Be in Repository (Gitignored)**

### 1. Large Data Files
```
data/raw/                    # ~6GB of downloaded datasets
data/detection_fixed/        # 103 images + labels (~500MB)
data/classification_crops/   # 1,242 cropped images (~50MB)
```

### 2. Training Results
```
results/                     # Training outputs, logs, model weights
runs/                        # YOLO training runs
*.pt                         # PyTorch model weights
*.pth                        # PyTorch model files
```

### 3. Environment Files
```
venv/                        # Virtual environment (machine-specific)
__pycache__/                 # Python cache files
*.pyc                        # Compiled Python files
.DS_Store                    # macOS system files
Thumbs.db                    # Windows system files
```

## 📦 **Backup Command for Critical Files**

```bash
# Create backup archive of essential files only
tar -czf malaria_detection_critical_backup.tar.gz \
    --exclude='data/' \
    --exclude='results/' \
    --exclude='runs/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='.git/' \
    .

# List contents of backup
tar -tzf malaria_detection_critical_backup.tar.gz
```

## 🔄 **Restore Process on New Machine**

### 1. Extract Backup
```bash
# Extract critical files
tar -xzf malaria_detection_critical_backup.tar.gz

# Verify extraction
ls -la
```

### 2. Run Automated Setup
```bash
# Make script executable and run
chmod +x quick_setup_new_machine.sh
./quick_setup_new_machine.sh
```

### 3. Manual Verification (if needed)
```bash
# Follow detailed verification checklist
# See: setup_verification.md
```

## 🎯 **Repository Size Optimization**

### Current Repository Size (Critical Files Only)
- **Scripts**: ~500KB (14 Python files)
- **Config**: ~10KB (4 YAML files)
- **Documentation**: ~100KB (Markdown files)
- **Dependencies**: ~5KB (requirements.txt)
- **Total**: ~615KB (without data/results)

### With Data (NOT recommended for repository)
- **Raw Data**: ~6GB (6 datasets)
- **Processed Data**: ~550MB (detection + crops)
- **Results**: ~500MB (model weights + logs)
- **Total with Data**: ~7GB+

## 🔐 **Version Control Best Practices**

### What to Commit
```bash
git add config/
git add scripts/
git add *.md
git add requirements.txt
git add .gitignore
```

### What NOT to Commit
```bash
# These should be in .gitignore
git rm --cached data/
git rm --cached results/
git rm --cached venv/
git rm --cached *.pt
```

## 📋 **Essential Files Checklist for New Machine**

### Before Transfer:
- [ ] All 14 scripts in `scripts/` directory
- [ ] All 4 config files in `config/` directory
- [ ] `requirements.txt` with exact versions
- [ ] `README.md` with updated pipeline documentation
- [ ] `setup_verification.md` for verification process
- [ ] `quick_setup_new_machine.sh` for automated setup
- [ ] `machine_specs.txt` for reference
- [ ] `working_requirements.txt` for exact reproduction

### After Transfer:
- [ ] Repository cloned successfully
- [ ] All critical files present
- [ ] Setup script runs without errors
- [ ] Verification checklist passes
- [ ] Training tests complete successfully

## 🎓 **Research Reproducibility Notes**

### Paper Reference Information
- **Title**: "Perbandingan YOLOv8, YOLOv11, dan RT-DETR untuk Deteksi Malaria"
- **Dataset**: 1,242 P. falciparum parasites from corrected MP-IDB annotations
- **Approach**: Two-step classification (Detection → Single-cell Classification)
- **Models**: YOLOv8, YOLOv11, RT-DETR for detection + YOLOv8-cls for classification

### Key Contributions
1. **Fixed MP-IDB Bounding Boxes**: Corrected coordinate mapping using ground truth masks
2. **Two-Step Pipeline**: Complete automation from detection to classification
3. **Multi-Model Comparison**: Comprehensive analysis of YOLO variants
4. **Reproducible Research**: Full pipeline automation with verification

---

**💡 Tip**: Always test the complete pipeline on a clean machine before submitting research to ensure full reproducibility!