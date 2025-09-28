# CLAUDE.md - Malaria Detection Project

## 📋 PROJECT OVERVIEW
Advanced malaria parasite detection and classification system using multi-model pipeline with YOLO detection and PyTorch classification.

## 🚀 MAIN PIPELINES

### 1. Standard Pipeline (Detection-based Crops)
```bash
python run_multiple_models_pipeline.py --dataset iml_lifecycle --include yolo11 --epochs-det 50 --epochs-cls 30
```

### 2. Ground Truth Pipeline (Recommended)
```bash
python run_multiple_models_pipeline_ground_truth_version.py --dataset iml_lifecycle --include yolo11 --epochs-det 50 --epochs-cls 30
```

**Ground Truth Pipeline Benefits:**
- ✅ Uses raw annotations for cleaner classification training
- ✅ Eliminates detection noise from classification data
- ✅ Stratified train/val/test splits (70%/20%/10%)
- ✅ Better classification accuracy

## 📊 AVAILABLE DATASETS

| Dataset | Classes | Purpose | Setup Command |
|---------|---------|---------|---------------|
| **IML Lifecycle** | 4 stages (ring, gametocyte, trophozoite, schizont) | Lifecycle classification | Auto-setup in pipeline |
| **MP-IDB Species** | 4 species (P_falciparum, P_vivax, P_malariae, P_ovale) | Species classification | Auto-setup in pipeline |
| **MP-IDB Stages** | 4 stages (ring, schizont, trophozoite, gametocyte) | Stage classification | Auto-setup in pipeline |

## 🤖 SUPPORTED MODELS

### Detection Models
- **YOLO10** (`yolo10`): Fast and accurate
- **YOLO11** (`yolo11`): Latest YOLO version
- **YOLO12** (`yolo12`): Newest release
- **RT-DETR** (`rtdetr`): Transformer-based detector

### Classification Models
- **DenseNet121** (`densenet121`): Dense connections
- **EfficientNet-B1** (`efficientnet_b1`): Efficient architecture
- **ConvNeXt-Tiny** (`convnext_tiny`): Modern CNN
- **MobileNet-V3-Large** (`mobilenet_v3_large`): Mobile-optimized
- **EfficientNet-B2** (`efficientnet_b2`): Larger EfficientNet
- **ResNet101** (`resnet101`): Deep residual network

## 📁 PROJECT STRUCTURE

```
hello_world/
├── run_multiple_models_pipeline.py              # Standard pipeline
├── run_multiple_models_pipeline_ground_truth_version.py  # Enhanced pipeline
├── scripts/
│   ├── training/
│   │   ├── generate_ground_truth_crops.py       # Ground truth crop generation
│   │   ├── 12_train_pytorch_classification.py   # Classification training
│   │   └── 11_crop_detections.py               # Detection-based crops
│   ├── data_setup/                             # Dataset preparation scripts
│   └── analysis/                               # Analysis and evaluation
├── data/
│   ├── raw/                                    # Raw datasets
│   ├── processed/                              # Processed datasets
│   └── crops_ground_truth/                     # Ground truth crops
└── results/                                    # Experiment results
```

## 🔧 KEY FEATURES

### Ground Truth Crop Generation
- **Fixed Duplicate Bug**: No longer generates 2x crops
- **Auto-Cleanup**: Removes old folders before generation
- **Stratified Splits**: Balanced train/val/test splits
- **Class Balance**: All classes represented in each split

### Smart Augmentation
**Detection Models:**
- Conservative augmentation for medical data
- Orientation preservation (`flipud=0.0`)
- Medical-aware color adjustments
- Dataset-specific batch sizes and patience

**Classification Models:**
- Base augmentation for majority classes
- Enhanced augmentation for minority classes
- Weighted sampling and loss functions
- Class-specific transform strategies

### Dataset Statistics Analysis
**Comprehensive Dataset Analysis**: Pipeline automatically generates:
- **Train/Val/Test Split Analysis**: Count and percentage for all datasets
- **Augmentation Effects**: Estimated data variety increase (~4.4x detection, ~3.5x classification)
- **Before/After Comparison**: Original vs effective training data volume
- **Medical-Safe Parameters**: Documentation of conservative augmentation settings
- **Fair Evaluation Guarantee**: Val/test never augmented for unbiased metrics

**Example Output** (2 Separate Tables):

**Detection Model Dataset Statistics:**
```
       Dataset  Original_Train  Original_Val  Original_Test  Augmented_Train  Augmented_Total  Multiplier
 iml_lifecycle             218            62             33              956             1051        4.4x
mp_idb_species             146            42             21              640              703        4.4x
 mp_idb_stages             146            42             21              640              703        4.4x
```

**Classification Model Dataset Statistics:**
```
       Dataset  Original_Train  Original_Val  Original_Test  Augmented_Train  Augmented_Total  Multiplier
 iml_lifecycle             218            62             33              765              860        3.5x
mp_idb_species             146            42             21              512              575        3.5x
 mp_idb_stages             146            42             21              512              575        3.5x
```

### Model-Specific Optimizations
- **RT-DETR**: Lower learning rate (0.0001) for transformer architecture
- **YOLO**: Standard learning rate (0.0005) for CNN architecture
- **Mixed Precision**: RTX 3060 optimized training
- **Early Stopping**: Prevents overfitting

## 📈 EXAMPLE WORKFLOWS

### Quick Test Run
```bash
python run_multiple_models_pipeline_ground_truth_version.py \
  --dataset iml_lifecycle \
  --include yolo11 \
  --classification-models densenet121 \
  --epochs-det 5 \
  --epochs-cls 5 \
  --no-zip
```

### Full Training
```bash
python run_multiple_models_pipeline_ground_truth_version.py \
  --dataset iml_lifecycle \
  --include yolo11 rtdetr \
  --classification-models all \
  --epochs-det 100 \
  --epochs-cls 50
```

### Continue Existing Experiment
```bash
python run_multiple_models_pipeline_ground_truth_version.py \
  --continue-from exp_multi_pipeline_20250927_194449_iml_lifecycle \
  --start-stage classification
```

## 🎯 PERFORMANCE OPTIMIZATIONS

### Detection Training
- GPU-optimized batch sizes per dataset
- Adaptive patience based on complexity
- Conservative augmentation for small datasets
- Model-specific learning rate tuning

### Classification Training
- Stratified sampling for class balance
- Weighted loss functions
- Enhanced augmentation for minority classes
- Mixed precision training for speed

### Data Quality
- Ground truth crops eliminate detection noise
- Proper train/val/test splits
- Class balance maintained across splits
- Medical-specific augmentation strategies

## 🔄 PIPELINE STAGES

1. **Detection Training**: Train YOLO/RT-DETR models on parasite detection
2. **Ground Truth Crops**: Generate crops from raw annotations (not detection results)
3. **Classification Training**: Train PyTorch models on clean crop data
4. **Analysis**: Performance evaluation and visualization

## 💾 RESULTS MANAGEMENT

Results are automatically organized in centralized structure:
```
results/exp_[name]_[timestamp]_[dataset]/
├── detection/          # Detection model weights and logs
├── crop_data/         # Generated crop datasets
├── models/            # Classification model weights
└── analysis/          # Performance analysis
```

## 🚨 IMPORTANT NOTES

- Use **Ground Truth Pipeline** for better results
- **Original Pipeline** preserved for compatibility
- All models support **GPU acceleration** (RTX 3060 optimized)
- **Stratified splits** ensure class balance
- **Medical-aware augmentation** preserves diagnostic features

## 🔧 TROUBLESHOOTING

### Common Issues
1. **CUDA out of memory**: Reduce batch size in pipeline
2. **Class imbalance**: Use ground truth pipeline with weighted sampling
3. **Low accuracy**: Increase epochs and use enhanced augmentation
4. **Missing dependencies**: Check environment setup

### Dataset Issues
1. **Raw data not found**: Check data/raw/ directory structure
2. **Processed data missing**: Pipeline will auto-setup datasets
3. **Empty splits**: Use stratified splitting in ground truth pipeline

## 📝 COMMAND REFERENCE

### Pipeline Options
- `--dataset`: Choose dataset (iml_lifecycle, mp_idb_species, mp_idb_stages)
- `--include`: Select detection models
- `--classification-models`: Select classification models
- `--epochs-det/--epochs-cls`: Training epochs
- `--experiment-name`: Custom experiment name
- `--continue-from`: Resume existing experiment
- `--start-stage`: Start from specific stage (detection, crop, classification, analysis)
- `--stop-stage`: Stop after completing specific stage (detection, crop, classification, analysis)
- `--no-zip`: Skip result archiving

### Example Commands

#### Basic Usage
```bash
# List available experiments
python run_multiple_models_pipeline_ground_truth_version.py --list-experiments

# Full multi-model training
python run_multiple_models_pipeline_ground_truth_version.py \
  --dataset iml_lifecycle \
  --include yolo11 yolo12 rtdetr \
  --classification-models densenet121 efficientnet_b1 convnext_tiny \
  --epochs-det 100 \
  --epochs-cls 50

# Quick validation run
python run_multiple_models_pipeline_ground_truth_version.py \
  --dataset mp_idb_species \
  --include yolo11 \
  --classification-models densenet121 \
  --epochs-det 10 \
  --epochs-cls 10 \
  --no-zip
```

#### Stage Control Commands
```bash
# Detection training only
python run_multiple_models_pipeline_ground_truth_version.py \
  --dataset iml_lifecycle \
  --include yolo11 \
  --epochs-det 50 \
  --stop-stage detection

# Crop generation only (requires existing detection)
python run_multiple_models_pipeline_ground_truth_version.py \
  --continue-from exp_multi_pipeline_20250928_130011_iml_lifecycle \
  --start-stage detection \
  --stop-stage crop

# Classification training only
python run_multiple_models_pipeline_ground_truth_version.py \
  --continue-from exp_multi_pipeline_20250928_130011_iml_lifecycle \
  --start-stage classification \
  --stop-stage classification \
  --epochs-cls 30

# Analysis only (requires existing models)
python run_multiple_models_pipeline_ground_truth_version.py \
  --continue-from exp_multi_pipeline_20250928_130011_iml_lifecycle \
  --start-stage analysis
```

#### Manual Analysis Commands
```bash
# Dataset statistics analysis (standalone)
python scripts/analysis/dataset_statistics_analyzer.py \
  --output dataset_analysis_results

# Fast IoU analysis from training results (no re-testing)
python scripts/analysis/compare_models_performance.py \
  --iou-from-results \
  --results-csv path/to/detection/results.csv \
  --output iou_analysis_results \
  --experiment-name experiment_name

# Deprecated: IoU analysis with re-testing (slow)
python scripts/analysis/compare_models_performance.py \
  --iou-analysis \
  --model path/to/detection/best.pt \
  --output iou_analysis_results
```

---
*Last Updated: 2025-09-27*
*Ground Truth Pipeline: Enhanced version with stratified splits and cleaner classification training*