# Malaria Detection Repository Cleanup & Setup Guide

## Current Status Analysis

Training sedang berjalan: Epoch 1/10, Progress ~39%, Loss: 1.053 (turun dari 1.781)

## Repository Cleanup Plan

### Files yang Bisa Dihapus/Reorganisasi

#### 1. **Duplicated/Legacy Files** ❌
```bash
# Files to DELETE
rm -rf malaria_detection/          # Empty/old package directory
rm training_log.txt               # Old training logs
rm download_log.txt               # Old download logs
rm data_audit_backup.json         # Backup file
rm monitor_training.py            # Replaced by integrated training scripts
```

#### 2. **Data Directories to Clean** 🧹
```bash
# Keep only essential data dirs, remove intermediate
rm -rf data/augmented/            # Intermediate augmentation results
rm -rf data/integrated/           # Old integration results
rm -rf data/final/                # Old final results
rm -rf data/processed/            # Old preprocessing results
rm -rf data/cache/                # Cache files

# Keep these:
# data/classification/             # Final training data
# data/final_v2/                   # Latest final data
# data/integrated_v2/              # Latest integration
# data/raw/                        # Source datasets
```

#### 3. **Model/Results Cleanup** 📊
```bash
# Organize results better
mkdir -p results/archive/
mv results/training/old_* results/archive/  # Archive old training results
rm -rf models/rtdetr/             # Remove unused model dirs if no weights
rm -rf models/yolov8/             # Remove unused model dirs if no weights
```

#### 4. **Script Reorganization** 🔧

**Current Structure (Good - Keep):**
```
scripts/
├── 01_download_datasets.py     ✅ Keep
├── 02_preprocess_data.py        ✅ Keep
├── 03_integrate_datasets.py     ✅ Keep
├── 04_convert_to_yolo.py        ✅ Keep
├── 05_augment_data.py           ✅ Keep
├── 06_split_dataset.py          ✅ Keep
└── utils/                       ✅ Keep all
    ├── __init__.py
    ├── download_utils.py
    ├── image_utils.py
    └── annotation_utils.py
```

**Files to Consolidate:**
```bash
# Merge these into main scripts directory
mv quick_train.py scripts/07_train_yolo_quick.py
mv run_pipeline.py scripts/run_full_pipeline.py
mv watch_pipeline.py scripts/watch_pipeline.py
```

### Final Clean Repository Structure

```
malaria_detection/
├── README.md                    # Main documentation
├── requirements.txt             # Dependencies
├── setup.sh                     # Environment setup
├── .gitignore                   # Git ignore rules
├── CLAUDE.md                    # Claude context
├──
├── scripts/                     # All processing scripts
│   ├── 01_download_datasets.py
│   ├── 02_preprocess_data.py
│   ├── 03_integrate_datasets.py
│   ├── 04_convert_to_yolo.py
│   ├── 05_augment_data.py
│   ├── 06_split_dataset.py
│   ├── 07_train_yolo_quick.py  # Quick training (NEW)
│   ├── 08_train_yolo_full.py   # Full training (TO CREATE)
│   ├── 09_train_rtdetr.py      # RT-DETR training
│   ├── run_full_pipeline.py    # Complete pipeline
│   ├── watch_pipeline.py       # Pipeline monitoring
│   └── utils/                  # Utility modules
│       ├── __init__.py
│       ├── download_utils.py
│       ├── image_utils.py
│       └── annotation_utils.py
│
├── config/                      # Configuration files
│   ├── datasets.yaml           # Dataset configurations
│   ├── training.yaml           # Training configurations
│   └── models.yaml             # Model configurations
│
├── data/                        # Data directory (gitignored)
│   ├── raw/                    # Downloaded datasets
│   ├── classification/         # Final classification data
│   └── splits/                 # Train/val/test splits
│
├── results/                     # Results directory (gitignored)
│   ├── classification/         # Classification results
│   ├── detection/              # Detection results
│   ├── weights/                # Trained model weights
│   └── logs/                   # Training logs
│
├── notebooks/                   # Analysis notebooks (if any)
├── tests/                       # Unit tests (if any)
└── venv/                       # Virtual environment (gitignored)
```

---

## Fresh Repository Setup Guide

### 🚀 **Quick Start on New PC (15-30 minutes)**

#### 1. **Clone & Setup Environment**
```bash
# Clone repository
git clone <repository-url>
cd malaria_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 2. **Download Data (30-60 minutes)**
```bash
# Download all datasets (~15-30GB)
python scripts/01_download_datasets.py
```

#### 3. **Quick Classification Training (30 minutes)**
```bash
# Quick preprocessing + training pipeline
python scripts/run_full_pipeline.py --quick

# Or manual steps:
python scripts/02_preprocess_data.py --quick
python scripts/03_integrate_datasets.py --quick
python scripts/04_convert_to_yolo.py --classification
python scripts/07_train_yolo_quick.py
```

#### 4. **Full Training Pipeline (4-8 hours)**
```bash
# Complete preprocessing
python scripts/02_preprocess_data.py

# Integration & conversion
python scripts/03_integrate_datasets.py
python scripts/04_convert_to_yolo.py

# Data augmentation
python scripts/05_augment_data.py

# Training
python scripts/08_train_yolo_full.py  # YOLOv8 classification
python scripts/09_train_rtdetr.py     # RT-DETR detection
```

### 🔧 **Configuration Files**

#### `config/datasets.yaml`
```yaml
datasets:
  nih:
    url: "https://..."
    classes: ["P_falciparum", "P_vivax", "P_malariae", "P_ovale", "Mixed", "Uninfected"]
  mp_idb:
    url: "https://..."
    classes: ["Infected", "Uninfected"]
  # ... other datasets
```

#### `config/training.yaml`
```yaml
quick_training:
  epochs: 10
  batch_size: 32
  img_size: 64
  patience: 3

full_training:
  epochs: 100
  batch_size: 16
  img_size: 224
  patience: 10
```

### 📋 **Dependencies (requirements.txt)**
```txt
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.5.0
Pillow>=8.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.60.0
PyYAML>=6.0
albumentations>=1.3.0
scikit-learn>=1.0.0
```

### 🎯 **Key Commands**

```bash
# Environment
source venv/bin/activate

# Quick test (30 min)
python scripts/07_train_yolo_quick.py

# Check data status
python -c "from pathlib import Path; print('Images:', len(list(Path('data/classification/train').rglob('*.jpg'))))"

# Monitor training
tensorboard --logdir results/classification/

# Clean & restart
python scripts/run_full_pipeline.py --clean --restart
```

### 💡 **Pro Tips**

1. **GPU Usage**: Modify `device='cuda'` in training scripts for GPU acceleration
2. **Memory**: Adjust `batch_size` based on available RAM/VRAM
3. **Quick Preview**: Use `--sample 1000` flags for quick testing with subset
4. **Resume Training**: Models auto-save, can resume from checkpoints
5. **Multiple Experiments**: Use `--name experiment_name` for different runs

### 🔍 **Troubleshooting**

```bash
# Check Python environment
python --version  # Should be 3.8+

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check disk space
df -h

# Clean temporary files
python scripts/utils/clean_temp.py
```

---

## Migration Checklist

- [ ] Run cleanup commands above
- [ ] Test pipeline on subset of data
- [ ] Verify all scripts work with new structure
- [ ] Update CLAUDE.md with new structure
- [ ] Create config files
- [ ] Test on fresh environment
- [ ] Update .gitignore
- [ ] Create proper README.md

Training masih berjalan: ~39% complete, ETA: ~15 menit lagi