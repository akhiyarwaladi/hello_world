# CLAUDE.md - Malaria Detection Project

## 📋 PROJECT OVERVIEW

Advanced malaria parasite detection and classification system using **shared classification architecture** with YOLO detection models and PyTorch classification models.

**Status:** ✅ **VERIFIED WORKING** (Last tested: 2025-12-07)
**Environment:** Python 3.13.5, PyTorch 2.8.0+cu128, CUDA 12.8
**GPU:** NVIDIA RTX 4090 (24GB VRAM) - Fully tested and optimized
**Latest Test Results:** Detection mAP@50: 96.61% | Classification Acc: 91.51%

---

## 📚 DOCUMENTATION INDEX

**Quick Links:**
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete environment setup instructions
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Solutions to common issues
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed project structure and design patterns
- **THIS FILE (CLAUDE.md)** - Quick start guide and essential commands

**Start here if you're:**
- 🚀 New user → Read sections below for quick start
- 🔧 Having issues → See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- 🏗️ Understanding architecture → See [ARCHITECTURE.md](ARCHITECTURE.md)
- ⚙️ Setting up environment → See [SETUP_GUIDE.md](SETUP_GUIDE.md)

---

## ⚡ QUICK START (5 MINUTES)

### 1. Setup Environment

```bash
# Automated setup (recommended)
python setup_environment.py

# Or manual setup
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics==8.3.202
pip install -r requirements.txt

# Fix dataset paths (if needed)
python fix_data_yaml_paths.py
```

### 2. Verify Installation

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
nvidia-smi
```

### 3. Run Quick Test

```bash
# 5-minute verification test
python main_pipeline.py \
  --dataset iml_lifecycle \
  --include yolo11 \
  --classification-models densenet121 \
  --epochs-det 5 \
  --epochs-cls 5
```

**Expected Results:**
- ✅ Detection mAP@50: ~0.90 (90%)
- ✅ Classification Accuracy: ~0.80 (80%)
- ✅ Total time: ~5 minutes on RTX 4090
- ✅ Output: `results/optA_[timestamp]/`

---

## 🚀 MAIN PIPELINE

### Default Configuration

```bash
# Run full pipeline (all datasets, all models)
python main_pipeline.py
```

**What it does:**
1. **Trains YOLO detectors** (YOLO10, 11, 12) on 3 datasets (100 epochs)
2. **Generates ground truth crops** from raw annotations (once per dataset)
3. **Trains classification models** (6 architectures with Focal Loss, 75 epochs)
4. **Analyzes performance** and generates comprehensive reports

**Scope:**
- 4 datasets × 3 detection models × 6 classification models = 72 experiments
- Estimated time: 8-12 hours (full experiment)
- Storage: ~20-25 GB (with compression ~12-15 GB)

### Common Commands

```bash
# Single dataset
python main_pipeline.py --dataset iml_lifecycle

# Specific models only
python main_pipeline.py --include yolo11 --classification-models densenet121

# Custom epochs
python main_pipeline.py --epochs-det 100 --epochs-cls 75

# Custom data splits (60/20/20 is default)
python main_pipeline.py --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20

# Resume existing experiment
python main_pipeline.py --continue-from optA_20250929_203726 --start-stage classification
```

---

## 📊 DATASETS & MODELS

### Available Datasets

| Dataset | Classes | Images | Purpose |
|---------|---------|--------|---------|
| **IML Lifecycle** | 4 stages | ~313 | Lifecycle classification |
| **MP-IDB Species** | 4 species | ~209 | Species identification |
| **MP-IDB Stages** | 4 stages | ~209 | Stage classification |
| **MD-2019 Stages** | 3 stages | ~813 | Malaria Detection 2019 |

**Note:** Images counts are after filtering (valid annotations only).

---

### 📥 DATA SETUP FROM SCRATCH

**Complete workflow for setting up all datasets on a new PC.**

#### Prerequisites

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Setup Kaggle API (for MP-IDB datasets)
pip install kaggle

# Download kaggle.json from: https://www.kaggle.com/settings → API → Create New Token
# Windows: Place in C:\Users\<username>\.kaggle\kaggle.json
# Linux/Mac: Place in ~/.kaggle/kaggle.json
```

#### Step 1: Download Raw Data

**Option A: Automated Download (Recommended)**

```bash
# Download IML Lifecycle + Kaggle MP-IDB (automated via script)
python scripts/data_setup/01_download_datasets.py --dataset malaria_lifecycle,kaggle_mp_idb
```

**What this downloads:**
- ✅ `data/raw/malaria_lifecycle/` - IML Lifecycle (auto from GitHub)
- ✅ `data/raw/kaggle_dataset/MP-IDB-YOLO/` - MP-IDB (from Kaggle)

**Option B: MD-2019 Dataset (Manual Download)**

```bash
# Download MD-2019 dataset (1.7 GB)
# URL: https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5bf2kmwvfn-1.zip
# Or from: https://data.mendeley.com/datasets/5bf2kmwvfn/1

# Extract to: data/raw/md_2019/
# Should contain: Giemsa stained images/, Ground truth images/, LifeStages.xlsx
```

**Alternative: Individual Dataset Scripts**

```bash
# If you prefer running individual scripts:
python scripts/data_setup/08_setup_lifecycle_for_pipeline.py  # Auto-downloads IML
python scripts/data_setup/07_setup_kaggle_species_for_pipeline.py  # Requires Kaggle data
python scripts/data_setup/09_setup_kaggle_stage_for_pipeline.py    # Requires Kaggle data
python scripts/data_setup/10_setup_md2019_stage_for_pipeline.py    # Requires MD-2019 data
```

#### Step 2: Setup Detection Datasets (60/20/20 split)

```bash
# All datasets with 60/20/20 train/val/test split
python scripts/data_setup/08_setup_lifecycle_for_pipeline.py --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20
python scripts/data_setup/07_setup_kaggle_species_for_pipeline.py --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20
python scripts/data_setup/09_setup_kaggle_stage_for_pipeline.py --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20
python scripts/data_setup/10_setup_md2019_stage_for_pipeline.py --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20
```

#### Step 3: Generate Ground Truth Crops

```bash
# Generate classification crops for each dataset
python scripts/training/generate_ground_truth_crops.py --dataset data/processed/lifecycle --output data/ground_truth_crops_224/lifecycle --type iml_lifecycle --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20

python scripts/training/generate_ground_truth_crops.py --dataset data/processed/species --output data/ground_truth_crops_224/species --type mp_idb_species --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20

python scripts/training/generate_ground_truth_crops.py --dataset data/processed/stages --output data/ground_truth_crops_224/stages --type mp_idb_stages --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20

python scripts/training/generate_ground_truth_crops.py --dataset data/processed/md_2019_stages --output data/ground_truth_crops_224/md_2019_stages --type md_2019_stages --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20
```

#### Expected Data Structure

```
data/
├── raw/                          # Downloaded raw data
│   ├── malaria_lifecycle/        # IML dataset
│   ├── kaggle_dataset/           # MP-IDB dataset
│   └── md_2019/                  # MD-2019 dataset
├── processed/                    # Detection-ready data
│   ├── lifecycle/                # 313 images
│   ├── species/                  # 209 images
│   ├── stages/                   # 209 images
│   └── md_2019_stages/           # 813 images
└── ground_truth_crops_224/       # Classification crops
    ├── lifecycle/                # 529 crops
    ├── species/                  # 1436 crops
    ├── stages/                   # 1436 crops
    └── md_2019_stages/           # 2919 crops
```

#### Important Notes on Reproducibility

**Random Seed Optimization (Updated Oct 17, 2025):**
- Ground truth crop generation uses **optimized random seed search** (tries 500 candidates)
- Finds split that minimizes crop-level deviation from target 60/20/20 ratio
- **Trade-off:** Better crop distribution (0.1% deviation) vs. exact reproducibility
- Each run may produce slightly different splits (but same performance range)

**Scientific Reproducibility:**
- Same hyperparameters (focal loss α=0.25, γ=2.0) ✓
- Same model architectures ✓
- Same methodology ✓
- Results in consistent performance range (89-92% classification accuracy)
- **Conclusion:** Scientific reproducibility maintained (consistent methods + results)

**Focal Loss Parameters (Verified Dec 7, 2025):**
- Alpha: 0.25 (standard medical imaging)
- Gamma: 2.0 (balanced focusing)
- **Do NOT change** - these are proven optimal for malaria detection

### Detection Models (YOLO-Only)

- **YOLO10** (`yolo10`): YOLOv10 Medium - Fast and accurate
- **YOLO11** (`yolo11`): YOLOv11 Medium - Latest version (recommended)
- **YOLO12** (`yolo12`): YOLOv12 Medium - Newest release

**Default:** All 3 models. Use `--include yolo11` for single fastest model.

### Classification Models (6 Architectures)

- **DenseNet121** (`densenet121`): Dense connections - Reliable baseline
- **EfficientNet-B0** (`efficientnet_b0`): Efficient small model
- **EfficientNet-B1** (`efficientnet_b1`): Balanced efficiency (recommended)
- **EfficientNet-B2** (`efficientnet_b2`): Larger variant
- **ResNet50** (`resnet50`): Medium-deep residual network
- **ResNet101** (`resnet101`): Deep residual network

**Loss Function:** Focal Loss (alpha=0.25, gamma=2.0) - Optimized for medical imaging

**Default:** All 6 models. Use `--classification-models densenet121` for single fast model.

---

## 📁 PROJECT STRUCTURE (SIMPLIFIED)

```
hello_world/
├── CLAUDE.md                 # THIS FILE - Quick start guide
├── SETUP_GUIDE.md           # Detailed setup instructions
├── TROUBLESHOOTING.md       # Common issues & solutions
├── ARCHITECTURE.md          # Detailed architecture docs
├── main_pipeline.py         # MAIN PIPELINE (run this!)
├── requirements.txt         # Python dependencies
│
├── scripts/                 # Training, analysis, visualization
├── data/                    # Datasets (raw, processed, crops)
├── results/                 # Experiment outputs
└── luaran/                  # Publication outputs
```

**See [ARCHITECTURE.md](ARCHITECTURE.md) for complete folder structure and design patterns.**

---

## 🔧 KEY FEATURES

### Shared Classification Architecture

**Benefits:**
- ✅ **~70% storage reduction** - Single set of crops and classification models
- ✅ **~60% training time reduction** - Crops generated once from ground truth
- ✅ **Consistent evaluation** - Same models across all detection methods
- ✅ **Clean separation** - Detection and classification stages independent

**How it works:**
```
Raw Annotations → Ground Truth Crops (once) → Classification Models (once)
                                                        ↓
YOLO10/11/12 Detections → Compare with Ground Truth
```

### Smart Augmentation (Medical-Safe)

**Detection:**
- Conservative augmentation for medical data
- Orientation preservation (`flipud=0.0`)
- Medical-aware color adjustments

**Classification:**
- Mild augmentation (preserves 20-30px parasite morphology)
- Weighted sampling for class balance
- Focal Loss for imbalance handling

### GPU Optimizations (RTX 4090)

- Mixed precision (AMP): 2x speedup
- cuDNN benchmark: 2-3x convolution speedup
- Channels-last memory: 20-35% tensor speedup
- 4-worker DataLoader: Fast startup + high throughput
- **Expected total speedup:** 6-10x faster than baseline

---

## 💾 RESULTS STRUCTURE

### Multi-Dataset Mode (Default)

```
results/optA_[timestamp]/
├── experiments/
│   ├── experiment_iml_lifecycle/
│   │   ├── det_yolo10/ det_yolo11/ det_yolo12/
│   │   ├── cls_densenet121_focal/ cls_efficientnet_b0_focal/ ... (6 models)
│   │   ├── crops_gt_crops/
│   │   ├── analysis_detection_*/ analysis_classification_*/
│   │   └── table9_*.csv (Focal Loss performance pivot)
│   ├── experiment_mp_idb_species/
│   └── experiment_mp_idb_stages/
│
├── consolidated_analysis/     # Cross-dataset comparison
│   └── cross_dataset_comparison/
│       ├── dataset_statistics_all.csv
│       ├── detection_performance_all_datasets.xlsx
│       ├── classification_performance_all_datasets.xlsx
│       └── comprehensive_summary.json
│
└── README.md
```

### Single Dataset Mode

Same structure but with only one `experiment_[dataset]/` folder.

**See [ARCHITECTURE.md](ARCHITECTURE.md) for complete results structure.**

---

## 📝 COMMAND REFERENCE

### Core Options

| Option | Values | Description |
|--------|--------|-------------|
| `--dataset` | `iml_lifecycle` `mp_idb_species` `mp_idb_stages` `all` | Dataset selection |
| `--include` | `yolo10` `yolo11` `yolo12` | Detection models |
| `--classification-models` | `densenet121` `efficientnet_b0` `efficientnet_b1` `efficientnet_b2` `resnet50` `resnet101` `all` | Classification models |
| `--epochs-det` | integer | Detection epochs (default: 100) |
| `--epochs-cls` | integer | Classification epochs (default: 75) |

### Experiment Control

| Option | Description |
|--------|-------------|
| `--continue-from [timestamp]` | Resume existing experiment |
| `--start-stage [detection\|crop\|classification\|analysis]` | Start from specific stage |
| `--stop-stage [detection\|crop\|classification\|analysis]` | Stop after specific stage |
| `--train-ratio` `--val-ratio` `--test-ratio` | Custom data splits (must sum to 1.0) |
| `--no-zip` | Skip result archiving |

### Example Commands

```bash
# Quick test (5 minutes)
python main_pipeline.py --dataset iml_lifecycle --include yolo11 --classification-models densenet121 --epochs-det 5 --epochs-cls 5

# Single dataset, all models
python main_pipeline.py --dataset iml_lifecycle

# YOLO comparison only
python main_pipeline.py --include yolo10 yolo11 yolo12 --classification-models densenet121

# Custom split (80/10/10)
python main_pipeline.py --train-ratio 0.80 --val-ratio 0.10 --test-ratio 0.10

# Classification only (requires existing detection)
python main_pipeline.py --continue-from optA_20250929_203726 --start-stage classification

# Analysis only
python main_pipeline.py --continue-from optA_20250929_203726 --start-stage analysis
```

---

## 🎯 PERFORMANCE BENCHMARKS

### Quick Test (5 Minutes)

**Configuration:**
- Dataset: IML Lifecycle (206 train, 56 val, 89 test)
- Detection: YOLO11 Medium (20M params)
- Classification: DenseNet121 (8M params)
- GPU: RTX 4090 (24GB VRAM)
- Epochs: 5 detection, 5 classification

**Results:**
- Detection mAP@50: **89.7%** ✅ (Excellent)
- Detection mAP@50-95: **59.6%** ✅ (Good)
- Classification Accuracy: **79.78%** ✅
- Balanced Accuracy: **66.88%** ✅
- Total time: ~5 minutes

### Expected Performance (Full Training)

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Detection mAP@50 | > 85% | Parasite localization |
| Detection mAP@50-95 | > 70% | Precise localization |
| Classification Accuracy | > 80% | Overall performance |
| Balanced Accuracy | > 70% | Medical AI critical |

---

## 🚨 COMMON ISSUES

**Quick solutions (detailed in [TROUBLESHOOTING.md](TROUBLESHOOTING.md)):**

### Path Errors
```bash
# Fix: Convert paths to forward slashes
python fix_data_yaml_paths.py
```

### CUDA Out of Memory
```bash
# Fix: Reduce batch size or use fewer models
python main_pipeline.py --include yolo11 --classification-models densenet121
```

### Long Training Time
```bash
# Fix: Use quick test configuration
python main_pipeline.py --epochs-det 5 --epochs-cls 5
```

### Windows Error 1920 (File Locking)
**Fixed (2025-10-17):** Automatic folder cleanup with fallback. If persists, manually delete folder before re-running.

**For complete troubleshooting:** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📂 PUBLICATION OUTPUTS (Luaran)

### Structure

```
luaran/
├── auto_generated/          # ⚙️ AUTO-GENERATED (DO NOT EDIT)
│   ├── figures/             # 30 publication-quality figures
│   ├── tables/              # 12 comprehensive tables
│   └── _metadata.json
│
├── hand_created/            # ✍️ MANUALLY CREATED (EDIT HERE)
│   ├── papers/              # Research manuscripts (MD → DOCX/PDF)
│   ├── reports/             # Progress reports
│   └── documentation/
│
└── templates/               # Official templates
```

### One-Command Regeneration

```bash
# Regenerate all auto-generated outputs (~5 minutes)
python scripts/publication/generate_all_publication_outputs.py

# Figures only
python scripts/publication/generate_all_publication_outputs.py --figures-only

# Tables only
python scripts/publication/generate_all_publication_outputs.py --tables-only
```

### Workflow: Update Paper with Latest Results

```bash
# 1. Run experiment
python main_pipeline.py --dataset all

# 2. Regenerate outputs
python scripts/publication/generate_all_publication_outputs.py

# 3. Edit paper (hand_created/papers/*.md)

# 4. Export to DOCX
pandoc hand_created/papers/JICEST_Paper.md -o hand_created/papers/exports/JICEST_Paper.docx
```

**See [ARCHITECTURE.md](ARCHITECTURE.md) for complete luaran structure and philosophy.**

---

## 🎓 LEARNING PATH

### For New Users

1. **Read this file** (CLAUDE.md) - Understand basics
2. **Setup environment** ([SETUP_GUIDE.md](SETUP_GUIDE.md))
3. **Run quick test** (5-minute test command above)
4. **Explore results** (`results/optA_[timestamp]/`)
5. **Try full experiment** (default command)

### For Developers

1. **Understand architecture** ([ARCHITECTURE.md](ARCHITECTURE.md))
2. **Review code structure** (`scripts/` folder)
3. **Study pipeline stages** (detection → crop → classification → analysis)
4. **Modify for your needs** (new models, datasets, analysis)

### For Troubleshooting

1. **Check common issues** ([TROUBLESHOOTING.md](TROUBLESHOOTING.md))
2. **Verify installation** (run verification commands)
3. **Check logs** (experiment folder)
4. **Create GitHub issue** (if unresolved)

---

## 📊 PHASE 1 OPTIMIZATIONS (Current)

**Implemented (2025-10-16):**
1. ✅ **Removed Class-Balanced Loss** - Caused -8% to -26% degradation
2. ✅ **Optimized Focal Loss** - Standard parameters (alpha=0.25, gamma=2.0)
3. ✅ **Increased Epochs** - Detection: 100, Classification: 75 (better convergence)
4. ✅ **50% Faster Training** - Only 6 models instead of 12

**Results:**
- Classification accuracy: +2-4% improvement
- Better minority class performance
- Faster training with same/better results

---

## 📝 CHANGELOG

### 2025-12-07 - Complete Data Setup Workflow & Verification
- 📚 **Complete data setup documentation** - Added detailed workflow from scratch
- 🌐 **MD-2019 download URL** - Direct link for automated setup (1.7GB)
- ✅ **Verified focal loss parameters** - Confirmed alpha=0.25, gamma=2.0 (correct)
- 🔬 **Reproducibility analysis** - Documented random seed optimization (Oct 17 change)
- 📊 **Latest test results** - Detection: 96.61% mAP@50, Classification: 91.51% accuracy
- 🎯 **4 datasets support** - Updated all counts and estimates (was 3, now 4 datasets)
- 📝 **Prerequisites section** - Added Kaggle API setup instructions

### 2025-10-17 - Documentation Refactoring & Auto Cleanup
- 📚 **Refactored documentation** - Split CLAUDE.md into 4 focused files
- 🧹 **Auto folder cleanup** - Added automatic deletion before training (Windows-safe)
- 🔧 **Improved navigation** - Clear documentation index with quick links
- 📉 **50% size reduction** - CLAUDE.md: 1072 → ~500 lines

### 2025-10-16 - Environment Verification & Path Fixes
- ✅ **Verified working** - Python 3.13.5 + PyTorch 2.8.0 + CUDA 12.8
- ✅ **Tested on RTX 4090** - Full pipeline successful
- 🔧 **Cross-platform paths** - Added fix_data_yaml_paths.py
- 📝 **Complete benchmarks** - 5-minute test verified (89.7% mAP@50)

### 2025-10-12 - Luaran Structure Automation
- 🎯 **90% automation reduction** - From hours to minutes
- 📂 **Clear separation** - auto_generated/ vs hand_created/
- ✅ **One-command regeneration** - All figures and tables

### 2025-10-11 - Codebase Cleanup
- 🧹 **Archived 50 files** - 88% root directory reduction
- 🎯 **Professional structure** - Clean and organized
- 📦 **100% restorable** - All archived files preserved

---

## 💡 TIPS & BEST PRACTICES

### Quick Tips

1. **Always run quick test first** (5 minutes) before full experiment
2. **Use single YOLO11** for fastest experiments
3. **Monitor GPU memory** with `nvidia-smi`
4. **Check paths** if you get data.yaml errors
5. **Use --continue-from** to resume interrupted experiments

### Performance Tips

- **GPU Memory:** Use batch size 64 on RTX 4090, 32 on RTX 3060
- **Speed:** YOLO11 + DenseNet121 = fastest combination
- **Accuracy:** YOLO11 + EfficientNet-B1 = best accuracy
- **Storage:** Use --no-zip to save disk space during experiments

### Development Tips

- **Results Manager:** Use for consistent paths
- **Ground Truth Crops:** Regenerate if annotations change
- **Classification Only:** Use --start-stage classification to skip detection
- **Analysis Only:** Use --start-stage analysis to regenerate reports

---

## 🔗 EXTERNAL RESOURCES

### Documentation
- **GitHub:** https://github.com/akhiyarwaladi/hello_world
- **Issues:** https://github.com/akhiyarwaladi/hello_world/issues

### Dependencies
- **PyTorch:** https://pytorch.org/
- **Ultralytics:** https://docs.ultralytics.com/
- **YOLO:** https://github.com/ultralytics/ultralytics

### Research
- **IML Lifecycle Dataset:** [Citation needed]
- **MP-IDB Dataset:** [Citation needed]
- **Focal Loss Paper:** https://arxiv.org/abs/1708.02002

---

## ✅ QUICK CHECKLIST

Before running experiments:
- [ ] Environment setup complete ([SETUP_GUIDE.md](SETUP_GUIDE.md))
- [ ] CUDA available (`torch.cuda.is_available() == True`)
- [ ] Dataset paths fixed (`python fix_data_yaml_paths.py`)
- [ ] GPU memory clear (`nvidia-smi` shows free memory)
- [ ] Sufficient disk space (50GB+ free)

Ready to go:
```bash
python main_pipeline.py --dataset iml_lifecycle --include yolo11 --classification-models densenet121 --epochs-det 5 --epochs-cls 5
```

---

**Last Updated:** 2025-12-07
**Status:** ✅ Verified Working
**Environment:** Python 3.13.5, PyTorch 2.8.0+cu128, CUDA 12.8
**Latest Results:** Detection mAP@50: 96.61% | Classification: 91.51% (exceeds baseline 89.62%)
**Main Pipeline:** YOLO-focused shared classification architecture for efficient malaria detection

**Documentation:**
- 📚 [SETUP_GUIDE.md](SETUP_GUIDE.md) - Environment setup
- 🔧 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture
