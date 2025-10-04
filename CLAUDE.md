# CLAUDE.md - Malaria Detection Project

## 📋 PROJECT OVERVIEW
Advanced malaria parasite detection and classification system using **Option A: Shared Classification Architecture** with YOLO detection models and PyTorch classification models.

## 🚀 MAIN PIPELINE: OPTION A (YOLO-FOCUSED)

### Option A Pipeline (PRIMARY - Only Pipeline)
```bash
# Default: All datasets, all YOLO models, all classification models
python run_multiple_models_pipeline_OPTION_A.py

# Single dataset with specific models
python run_multiple_models_pipeline_OPTION_A.py --dataset iml_lifecycle --include yolo11 --classification-models densenet121

# Full training with custom epochs
python run_multiple_models_pipeline_OPTION_A.py --epochs-det 100 --epochs-cls 50
```

### **Option A Benefits:**
- ✅ **~70% storage reduction** - Shared classification architecture
- ✅ **~60% training time reduction** - Ground truth crops generated once
- ✅ **Consistent classification** - Same models across all detection methods
- ✅ **Clean organization** - Unified folder structure
- ✅ **YOLO-focused** - Fast and efficient detection models only
- ✅ **Complete analysis** - Comprehensive reporting and visualization

## 📊 AVAILABLE DATASETS

| Dataset | Classes | Purpose | Setup |
|---------|---------|---------|--------|
| **IML Lifecycle** | 4 stages (ring, gametocyte, trophozoite, schizont) | Lifecycle classification | Auto-setup |
| **MP-IDB Species** | 4 species (P_falciparum, P_vivax, P_malariae, P_ovale) | Species classification | Auto-setup |
| **MP-IDB Stages** | 4 stages (ring, schizont, trophozoite, gametocyte) | Stage classification | Auto-setup |

**Default Behavior:** Pipeline runs on **ALL 3 datasets** automatically.

## 🤖 SUPPORTED MODELS

### Detection Models (YOLO-Only for Speed)
- **YOLO10** (`yolo10`): YOLOv10 Medium - Fast and accurate
- **YOLO11** (`yolo11`): YOLOv11 Medium - Latest YOLO version
- **YOLO12** (`yolo12`): YOLOv12 Medium - Newest release

**Default:** All 3 YOLO models (YOLO10, YOLO11, YOLO12)

### Classification Models (6 Architectures × 2 Loss Functions = 12 Models)
- **DenseNet121** (`densenet121`): Dense connections - Focal + Class-Balanced
- **EfficientNet-B1** (`efficientnet_b1`): Efficient architecture - Focal + Class-Balanced
- **EfficientNet-B0** (`efficientnet_b0`): Smaller EfficientNet (5.3M params) - Focal + Class-Balanced
- **ResNet50** (`resnet50`): Medium-deep residual network - Focal + Class-Balanced
- **EfficientNet-B2** (`efficientnet_b2`): Larger EfficientNet - Focal + Class-Balanced
- **ResNet101** (`resnet101`): Deep residual network - Focal + Class-Balanced

**Loss Functions:**
- **Focal Loss**: Handles class imbalance (alpha=0.5, gamma=2.0)
- **Class-Balanced Loss**: Auto-handles extreme imbalance (beta=0.9999)

**Default:** All 12 models (6 architectures × 2 loss functions)

## 📁 PROJECT STRUCTURE

```
hello_world/
├── run_multiple_models_pipeline_OPTION_A.py    # MAIN PIPELINE (Option A)
├── scripts/
│   ├── training/
│   │   ├── generate_ground_truth_crops.py      # Ground truth crop generation
│   │   └── 12_train_pytorch_classification.py  # Classification training
│   ├── data_setup/                            # Dataset preparation scripts
│   └── analysis/                              # Analysis and evaluation tools
├── data/
│   ├── raw/                                   # Raw datasets
│   ├── processed/                             # Processed datasets (YOLO format)
│   └── crops_ground_truth/                    # Ground truth crops (shared)
├── results/                                   # Experiment results
│   ├── optA_[timestamp]/                      # Multi-dataset experiments
│   │   ├── experiments/                       # Individual dataset results
│   │   └── consolidated_analysis/             # Cross-dataset analysis
│   └── exp_optA_[timestamp]_[dataset]/        # Single dataset experiments
└── utils/
    └── results_manager.py                     # Results organization
```

## 🔧 KEY FEATURES

### Option A: Shared Classification Architecture
- **Ground Truth Crops Generated Once**: All detection models use same clean crop data
- **Classification Models Trained Once**: Shared across all detection methods
- **Clean Separation**: Detection and classification stages are independent
- **Efficient Storage**: ~70% reduction vs traditional approach

### Smart Augmentation (Medical-Safe)
**Detection Models:**
- Conservative augmentation for medical data
- Orientation preservation (`flipud=0.0`)
- Medical-aware color adjustments
- Dataset-specific batch sizes and patience

**Classification Models:**
- Enhanced augmentation for minority classes
- Weighted sampling and loss functions
- Focal Loss vs Class-Balanced Loss comparison
- Medical-safe transform strategies

### Comprehensive Analysis Suite

#### Per-Dataset Analysis (Both Single & Multi-Dataset):
- ✅ **Table 9 Classification Pivot**: Focal Loss vs Class-Balanced Loss comparison per dataset
- ✅ **Detection IoU Analysis**: Per-model mAP@0.5, mAP@0.75, mAP@0.5:0.95
- ✅ **Classification Metrics**: Accuracy, balanced accuracy, precision, recall, F1-score per class
- ✅ **Model Performance Reports**: Individual analysis for each trained model

#### Multi-Dataset Consolidated Analysis (ONLY with `--dataset all`):
**Automatically Generated (9 files):**
1. ✅ `dataset_statistics_all.csv` - Augmentation effects across all datasets
2. ✅ `detection_performance_all_datasets.csv/xlsx` - YOLO comparison (CSV + Excel)
3. ✅ `classification_focal_loss_all_datasets.csv` - Focal Loss results across datasets
4. ✅ `classification_class_balanced_all_datasets.csv` - Class-Balanced results across datasets
5. ✅ `classification_performance_all_datasets.xlsx` - Combined Excel (2 sheets: Focal + CB)
6. ✅ `comprehensive_summary.json` - Complete data in JSON format (34 KB)
7. ✅ `README.md` - Overview with detailed tables

**Dataset Statistics Example:**
```
| Dataset | Original Train | Detection Aug | Classification Aug | Det Multiplier | Cls Multiplier |
|---------|----------------|---------------|-------------------|----------------|----------------|
| iml_lifecycle | 218 | 956 | 765 | 4.4x | 3.5x |
| mp_idb_species | 146 | 640 | 512 | 4.4x | 3.5x |
| mp_idb_stages | 146 | 640 | 512 | 4.4x | 3.5x |
```

**Detection Performance Example:**
```
| Dataset | Model | mAP@50 | mAP@50-95 | Precision | Recall |
|---------|-------|--------|-----------|-----------|--------|
| iml_lifecycle | YOLO11 | 0.9457 | 0.8006 | 0.9160 | 0.9510 |
| mp_idb_species | YOLO11 | 0.9288 | 0.5575 | 0.8868 | 0.8957 |
```

**Classification Performance Example (Table 9 Summary):**
```
### IML_LIFECYCLE
Focal Loss:
- densenet121: 0.6629
- efficientnet_b0: 0.7191
- efficientnet_b1: 0.8090
- efficientnet_b2: 0.6854

Class-Balanced Loss:
- densenet121: 0.3820
- efficientnet_b0: 0.6517
- efficientnet_b1: 0.8202
- efficientnet_b2: 0.7191
```

## 🚀 PERFORMANCE OPTIMIZATIONS

### YOLO-Focused Detection
- Standard learning rate (0.0005) for all YOLO models
- GPU-optimized batch sizes per dataset
- Conservative augmentation for medical data
- Early stopping to prevent overfitting

### Classification Training
- Stratified sampling for class balance
- Weighted loss functions
- Mixed precision training (RTX 3060 optimized)
- Focal Loss vs Class-Balanced Loss comparison

### Data Quality
- Ground truth crops eliminate detection noise
- Stratified train/val/test splits (customizable, default: 66%/17%/17%)
- Class balance maintained across splits
- Medical-specific augmentation strategies

## 📈 EXAMPLE WORKFLOWS

### Default Full Experiment
```bash
# Run everything: 3 datasets × 3 detection × 12 classification = 108 experiments
python run_multiple_models_pipeline_OPTION_A.py
```

### Quick Test Run
```bash
# Single dataset, single models
python run_multiple_models_pipeline_OPTION_A.py \
  --dataset iml_lifecycle \
  --include yolo11 \
  --classification-models densenet121 \
  --epochs-det 5 \
  --epochs-cls 5 \
  --no-zip
```

### Specific Experiments
```bash
# YOLO comparison on single dataset
python run_multiple_models_pipeline_OPTION_A.py \
  --dataset mp_idb_species \
  --include yolo10 yolo11 yolo12 \
  --classification-models densenet121 efficientnet_b1

# Full training with more epochs
python run_multiple_models_pipeline_OPTION_A.py \
  --epochs-det 100 \
  --epochs-cls 50
```

### Continue Existing Experiment
```bash
# Resume from specific stage
python run_multiple_models_pipeline_OPTION_A.py \
  --continue-from optA_20250929_203726 \
  --start-stage classification
```

## 🔄 PIPELINE STAGES

1. **Detection Training**: Train YOLO models (10, 11, 12) on parasite detection
2. **Ground Truth Crops**: Generate crops from raw annotations (not detection results)
3. **Classification Training**: Train PyTorch models on clean crop data (12 models)
4. **Analysis**: Comprehensive performance evaluation and visualization

## 💾 RESULTS STRUCTURE

### 🔹 Multi-Dataset Mode (Default: `--dataset all`)
**Folder Pattern**: `optA_[timestamp]/`
**Manager**: `ParentStructureManager` (nested with `experiments/`)

```
results/optA_20251001_183508/                   ← Parent folder
├── experiments/                                 ← Container for all datasets
│   ├── experiment_iml_lifecycle/               ← Dataset folder
│   │   ├── det_yolo10/                         ← Detection models
│   │   ├── det_yolo11/
│   │   ├── det_yolo12/
│   │   ├── cls_densen_ce_classification/       ← Classification models (12 total)
│   │   ├── cls_densen_focal_classification/
│   │   ├── cls_efficientnet_b1_ce_classification/
│   │   ├── cls_efficientnet_b1_focal_classification/
│   │   ├── ... (8 more classification models)
│   │   ├── crops_gt_crops/                     ← Ground truth crops (shared)
│   │   ├── analysis_detection_yolo10/          ← Analysis folders
│   │   ├── analysis_classification_*/
│   │   ├── table9_focal_loss.csv               ← Table 9 pivots
│   │   ├── table9_class_balanced.csv
│   │   └── table9_classification_pivot.xlsx
│   ├── experiment_mp_idb_species/              ← Same structure
│   └── experiment_mp_idb_stages/               ← Same structure
├── consolidated_analysis/                      ← **Cross-dataset comparison (ONLY in multi-dataset)**
│   └── cross_dataset_comparison/
│       ├── dataset_statistics_all.csv          ← Augmentation effects
│       ├── detection_performance_all_datasets.csv/xlsx  ← YOLO comparison
│       ├── classification_focal_loss_all_datasets.csv
│       ├── classification_class_balanced_all_datasets.csv
│       ├── classification_performance_all_datasets.xlsx  ← 2 sheets (Focal + CB)
│       ├── comprehensive_summary.json          ← Complete data (34 KB)
│       └── README.md                           ← Overview with tables
├── README.md
└── optA_20251001_183508.zip                    ← Auto-generated archive
```

**Why Nested Structure?**
- Organized comparison across multiple datasets
- Consolidated analysis for cross-dataset insights
- Clean separation of dataset-specific results

---

### 🔸 Single Dataset Mode (`--dataset iml_lifecycle`)
**Folder Pattern**: `optA_[timestamp]/`
**Manager**: `ParentStructureManager` (unified structure, same as multi-dataset)

```
results/optA_20251004_114731/                    ← Parent folder
├── experiments/                                 ← Container for experiments
│   └── experiment_iml_lifecycle/               ← Dataset folder
│       ├── det_yolo11/                         ← Detection models
│       ├── cls_densenet121_focal/              ← Classification models (12 total)
│       ├── cls_densenet121_cb/
│       ├── cls_efficientnet_b0_focal/
│       ├── cls_efficientnet_b0_cb/
│       ├── cls_efficientnet_b1_focal/
│       ├── cls_efficientnet_b1_cb/
│       ├── ... (6 more classification models)
│       ├── crops_gt_crops/                     ← Ground truth crops
│       ├── analysis_detection_yolo11/          ← Analysis folders
│       ├── analysis_classification_*/
│       ├── analysis_dataset_statistics/
│       ├── analysis_option_a_summary/
│       ├── table9_focal_loss.csv               ← Table 9 pivots
│       ├── table9_class_balanced.csv
│       └── table9_classification_pivot.xlsx
├── consolidated_analysis/                      ← Empty (for consistency)
├── master_summary.json                         ← Experiment summary
├── master_summary.xlsx                         ← Excel summary
├── README.md                                   ← Overview
└── optA_20251004_114731.zip                    ← Auto-generated archive
```

**Why Unified Structure?**
- ✅ Consistent organization (same as multi-dataset)
- ✅ Ready for future cross-dataset comparison
- ✅ Cleaner navigation with parent/experiments hierarchy
- ✅ Includes consolidated_analysis/ folder (empty but prepared)
- ✅ Master summary with accurate component counts

---

### 📋 Structure Comparison Summary

| Aspect | Multi-Dataset | Single Dataset |
|--------|--------------|----------------|
| **Pattern** | `optA_[timestamp]/` | `optA_[timestamp]/` |
| **Structure** | Nested (`experiments/`) | Nested (`experiments/`) ✅ UNIFIED |
| **Manager** | `ParentStructureManager` | `ParentStructureManager` |
| **Consolidated Analysis** | ✅ Yes (cross-dataset) | ⚪ Empty (ready for future) |
| **Use Case** | Compare performance across datasets | Focused single dataset study |
| **Command** | `--dataset all` (default) | `--dataset iml_lifecycle` |
| **Master Summary** | ✅ Accurate counts | ✅ Accurate counts |

## 📝 COMMAND REFERENCE

### Core Options
- `--dataset`: Dataset selection (`iml_lifecycle`, `mp_idb_species`, `mp_idb_stages`, `all`)
- `--include`: Detection models (`yolo10`, `yolo11`, `yolo12`)
- `--classification-models`: Classification models (`densenet121`, `efficientnet_b1`, etc., `all`)
- `--epochs-det`: Detection training epochs (default: 50)
- `--epochs-cls`: Classification training epochs (default: 30)

### Data Split Options (NEW!)
- `--train-ratio`: Training set ratio (default: 0.66 = 66%)
- `--val-ratio`: Validation set ratio (default: 0.17 = 17%)
- `--test-ratio`: Test set ratio (default: 0.17 = 17%)
  - **Note**: Ratios must sum to 1.0

### Experiment Control
- `--experiment-name`: Custom experiment name (default: `optA`)
- `--continue-from`: Resume existing experiment
- `--start-stage`: Start from specific stage (`detection`, `crop`, `classification`, `analysis`)
- `--stop-stage`: Stop after specific stage
- `--no-zip`: Skip result archiving

### Example Commands

#### Basic Usage
```bash
# Default full experiment
python run_multiple_models_pipeline_OPTION_A.py

# Single dataset
python run_multiple_models_pipeline_OPTION_A.py --dataset iml_lifecycle

# Specific YOLO models
python run_multiple_models_pipeline_OPTION_A.py --include yolo11 yolo12

# Specific classification models
python run_multiple_models_pipeline_OPTION_A.py --classification-models densenet121 efficientnet_b1
```

#### Custom Data Splits
```bash
# Custom 66/17/17 split (as requested)
python run_multiple_models_pipeline_OPTION_A.py \
  --train-ratio 0.66 \
  --val-ratio 0.17 \
  --test-ratio 0.17

# Custom 80/10/10 split
python run_multiple_models_pipeline_OPTION_A.py \
  --dataset iml_lifecycle \
  --train-ratio 0.80 \
  --val-ratio 0.10 \
  --test-ratio 0.10
```

#### Stage Control
```bash
# Detection only
python run_multiple_models_pipeline_OPTION_A.py --stop-stage detection

# Classification only (requires existing detection)
python run_multiple_models_pipeline_OPTION_A.py \
  --continue-from optA_20250929_203726 \
  --start-stage classification \
  --stop-stage classification

# Analysis only
python run_multiple_models_pipeline_OPTION_A.py \
  --continue-from optA_20250929_203726 \
  --start-stage analysis
```

#### Manual Analysis
```bash
# Dataset statistics (standalone)
python scripts/analysis/dataset_statistics_analyzer.py --output analysis_results

# Model performance comparison
python scripts/analysis/compare_models_performance.py \
  --iou-from-results \
  --results-csv path/to/detection/results.csv \
  --output comparison_results
```

## 🎯 PERFORMANCE METRICS

### Default Experiment Scope
- **Datasets**: 3 (IML Lifecycle, MP-IDB Species, MP-IDB Stages)
- **Detection Models**: 3 (YOLO10, YOLO11, YOLO12)
- **Classification Models**: 12 (6 architectures × 2 loss functions)
- **Total Experiments**: 108 (3 × 3 × 12)
- **Estimated Time**: 6-8 hours (full experiment)
- **Storage**: ~15-18 GB (with compression ~8-12 GB)

### Efficiency Gains vs Traditional Approach
- **Storage Reduction**: ~70% (shared classification architecture)
- **Training Time Reduction**: ~60% (ground truth crops generated once)
- **Analysis Enhancement**: Comprehensive automated reporting

## 🚨 IMPORTANT NOTES

- **YOLO-Only Pipeline**: RT-DETR removed for faster execution
- **Option A is Primary**: Only actively maintained pipeline
- **GPU Optimized**: RTX 3060 tested and optimized
- **Medical-Safe Augmentation**: Preserves diagnostic features
- **Automatic Setup**: Datasets auto-download and setup

## 🔧 TROUBLESHOOTING

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use fewer models
2. **Long training time**: Use `--include yolo11` for fastest single model
3. **Storage space**: Use `--no-zip` and clean results folder regularly
4. **Missing datasets**: Pipeline auto-downloads, ensure internet connection

### Performance Tips
1. **Quick test**: `--dataset iml_lifecycle --include yolo11 --classification-models densenet121`
2. **YOLO comparison**: `--include yolo10 yolo11 yolo12 --classification-models densenet121`
3. **Full analysis**: Default command (no parameters)

---
*Last Updated: 2025-09-30*
*Option A Pipeline: YOLO-focused shared classification architecture for efficient malaria detection*