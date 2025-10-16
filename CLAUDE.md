# CLAUDE.md - Malaria Detection Project

## 📋 PROJECT OVERVIEW

Advanced malaria parasite detection and classification system using **shared classification architecture** with YOLO detection models and PyTorch classification models.

**Status:** ✅ **VERIFIED WORKING** (Last tested: 2025-10-17)
**Environment:** Python 3.13.5, PyTorch 2.8.0+cu128, CUDA 12.8
**GPU:** NVIDIA RTX 4090 (24GB VRAM) - Fully tested and optimized

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
- 3 datasets × 3 detection models × 6 classification models = 54 experiments
- Estimated time: 6-8 hours (full experiment)
- Storage: ~15-18 GB (with compression ~8-12 GB)

### Common Commands

```bash
# Single dataset
python main_pipeline.py --dataset iml_lifecycle

# Specific models only
python main_pipeline.py --include yolo11 --classification-models densenet121

# Custom epochs
python main_pipeline.py --epochs-det 100 --epochs-cls 75

# Custom data splits
python main_pipeline.py --train-ratio 0.66 --val-ratio 0.17 --test-ratio 0.17

# Resume existing experiment
python main_pipeline.py --continue-from optA_20250929_203726 --start-stage classification
```

---

## 📊 DATASETS & MODELS

### Available Datasets

| Dataset | Classes | Images | Purpose |
|---------|---------|--------|---------|
| **IML Lifecycle** | 4 stages | ~350 | Lifecycle classification |
| **MP-IDB Species** | 4 species | ~200 | Species identification |
| **MP-IDB Stages** | 4 stages | ~200 | Stage classification |

**Auto-setup:** Datasets automatically downloaded and prepared on first run.

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

**Last Updated:** 2025-10-17
**Status:** ✅ Verified Working
**Environment:** Python 3.13.5, PyTorch 2.8.0+cu128, CUDA 12.8
**Main Pipeline:** YOLO-focused shared classification architecture for efficient malaria detection

**Documentation:**
- 📚 [SETUP_GUIDE.md](SETUP_GUIDE.md) - Environment setup
- 🔧 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture
