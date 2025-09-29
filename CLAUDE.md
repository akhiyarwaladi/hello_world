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
- **DenseNet121** (`densenet121`): Dense connections - CE + Focal Loss
- **EfficientNet-B1** (`efficientnet_b1`): Efficient architecture - CE + Focal Loss
- **ConvNeXt-Tiny** (`convnext_tiny`): Modern CNN - CE + Focal Loss
- **MobileNet-V3-Large** (`mobilenet_v3_large`): Mobile-optimized - CE + Focal Loss
- **EfficientNet-B2** (`efficientnet_b2`): Larger EfficientNet - CE + Focal Loss
- **ResNet101** (`resnet101`): Deep residual network - CE + Focal Loss

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
- Cross-entropy vs Focal Loss comparison
- Medical-safe transform strategies

### Comprehensive Analysis Suite
**Pipeline Automatically Generates:**
- **Table 9 Classification Pivot**: Cross-Entropy vs Focal Loss comparison
- **Dataset Statistics**: Before/after augmentation effects (~4.4x detection, ~3.5x classification)
- **Detection Models Comparison**: Performance across all YOLO models
- **Individual Model Analysis**: Per-model IoU and classification metrics
- **Multi-Dataset Analysis**: Cross-dataset insights and optimal model recommendations

**Example Dataset Statistics Output:**
```
       Dataset  Original_Train  Original_Val  Original_Test  Augmented_Train  Augmented_Total  Multiplier
 iml_lifecycle             218            62             33              956             1051        4.4x
mp_idb_species             146            42             21              640              703        4.4x
 mp_idb_stages             146            42             21              640              703        4.4x
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
- Focal Loss vs Cross-Entropy comparison

### Data Quality
- Ground truth crops eliminate detection noise
- Stratified train/val/test splits (70%/20%/10%)
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

### Multi-Dataset Experiments (Default)
```
results/optA_20250929_203726/
├── experiments/
│   ├── experiment_iml_lifecycle/       # Dataset-specific results
│   ├── experiment_mp_idb_species/
│   └── experiment_mp_idb_stages/
├── consolidated_analysis/              # Cross-dataset comparison
│   └── cross_dataset_comparison/
├── README.md
└── optA_20250929_203726.zip           # Auto-generated archive
```

### Single Dataset Experiments
```
results/exp_optA_20250929_203726_iml_lifecycle/
├── det_yolo10/                        # Detection model results
├── det_yolo11/
├── det_yolo12/
├── cls_denset_ce/                     # Classification model results
├── cls_denset_focal/
├── cls_effnet_ce/
├── ... (12 classification models total)
├── crops_gt_crops/                    # Shared ground truth crops
├── analysis_*/                       # Individual analysis results
├── table9_classification_pivot.xlsx   # Cross-Entropy vs Focal comparison
└── exp_optA_20250929_203726_iml_lifecycle.zip
```

## 📝 COMMAND REFERENCE

### Core Options
- `--dataset`: Dataset selection (`iml_lifecycle`, `mp_idb_species`, `mp_idb_stages`, `all`)
- `--include`: Detection models (`yolo10`, `yolo11`, `yolo12`)
- `--classification-models`: Classification models (`densenet121`, `efficientnet_b1`, etc., `all`)
- `--epochs-det`: Detection training epochs (default: 50)
- `--epochs-cls`: Classification training epochs (default: 30)

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