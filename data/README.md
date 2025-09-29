# Data Directory Structure

## Overview
Clean separation between raw source data and processed pipeline-ready data for **Option A: Shared Classification Architecture**.

## Structure

```
data/
├── raw/                           # Raw source datasets (original downloads)
│   ├── kaggle_dataset/            # Original Kaggle MP-IDB YOLO dataset
│   │   └── MP-IDB-YOLO/          # 209 images, 16 classes (4 species × 4 stages)
│   ├── malaria_lifecycle/         # IML Lifecycle dataset (GitHub)
│   │   ├── IML_Malaria/          # 345 images with 38,000 tagged cells
│   │   └── annotations.json      # JSON annotations for lifecycle stages
│   ├── mp_idb/                   # MP-IDB original dataset (GitHub)
│   │   └── MP-IDB/               # Original MP-IDB image database
│   └── README.md files           # Dataset documentation
│
├── processed/                     # Pipeline-ready processed data (YOLO format)
│   ├── species/                  # Species classification (4 classes)
│   │   ├── train/, val/, test/   # Stratified splits
│   │   └── data.yaml            # YOLO dataset configuration
│   ├── stages/                   # Stage classification (4 stages)
│   │   ├── train/, val/, test/   # Stratified splits
│   │   └── data.yaml            # YOLO dataset configuration
│   ├── lifecycle/                # Lifecycle detection (parasite stages)
│   │   ├── train/, val/, test/   # Stratified splits
│   │   └── data.yaml            # YOLO dataset configuration
│   └── crops_ground_truth/       # Shared ground truth crops (Option A)
│       ├── train/, val/, test/   # Classification-ready crop images
│       └── metadata.csv         # Crop generation metadata
│
└── README.md                     # This file
```

## Option A Pipeline Data Flow

### Stage 1: Automatic Dataset Setup
```bash
# Run Option A pipeline - auto-downloads and processes datasets
python run_multiple_models_pipeline_OPTION_A.py

# Pipeline automatically:
# 1. Downloads raw datasets if missing
# 2. Converts to YOLO format in data/processed/[dataset]/
# 3. Creates stratified train/val/test splits (70%/20%/10%)
```

### Stage 2: Detection Training (YOLO Models)
```bash
# Pipeline trains YOLO models on processed datasets
# Output: Detection models in results/*/det_yolo*/
#
# Models trained:
# - YOLO10 Medium (yolov10m.pt)
# - YOLO11 Medium (yolo11m.pt)
# - YOLO12 Medium (yolo12m.pt)
```

### Stage 3: Ground Truth Crop Generation (Shared)
```bash
# Pipeline generates crops from RAW ANNOTATIONS (not detection results)
# Output: data/processed/crops_ground_truth/[dataset]/
#
# Benefits:
# - Clean crops without detection noise
# - Generated ONCE and shared across all detection models
# - ~70% storage reduction vs traditional approach
```

### Stage 4: Classification Training (Shared Models)
```bash
# Pipeline trains classification models on shared ground truth crops
# Output: Classification models in results/*/cls_*/
#
# Models trained (6 architectures × 2 loss functions = 12):
# - DenseNet121 (Cross-Entropy + Focal Loss)
# - EfficientNet-B1 (Cross-Entropy + Focal Loss)
# - ConvNeXt-Tiny (Cross-Entropy + Focal Loss)
# - MobileNet-V3-Large (Cross-Entropy + Focal Loss)
# - EfficientNet-B2 (Cross-Entropy + Focal Loss)
# - ResNet101 (Cross-Entropy + Focal Loss)
```

### Stage 5: Comprehensive Analysis
```bash
# Pipeline automatically generates analysis reports
# Output: results/*/analysis_*/
#
# Analysis includes:
# - Table 9 Classification Pivot (Cross-Entropy vs Focal Loss)
# - Dataset Statistics (before/after augmentation)
# - Detection Models Comparison
# - Individual model performance metrics
# - Multi-dataset consolidated analysis
```

## Dataset Types

### IML Lifecycle (Lifecycle Detection)
- **Classes**: 4 lifecycle stages (ring, gametocyte, trophozoite, schizont)
- **Source**: IML Malaria dataset from GitHub
- **Images**: 345 microscopic images with 38,000 tagged cells
- **Purpose**: Parasite lifecycle stage detection and classification
- **Format**: YOLO detection format with JSON annotations

### MP-IDB Species (Species Classification)
- **Classes**: 4 species (P_falciparum, P_vivax, P_malariae, P_ovale)
- **Source**: Kaggle MP-IDB YOLO dataset
- **Images**: 209 images with expert pathologist annotations
- **Purpose**: Malaria species identification
- **Format**: YOLO classification format

### MP-IDB Stages (Stage Classification)
- **Classes**: 4 stages (ring, schizont, trophozoite, gametocyte)
- **Source**: Kaggle MP-IDB dataset (stage-focused extraction)
- **Images**: 209 images with stage-specific annotations
- **Purpose**: Parasite development stage classification
- **Format**: YOLO classification format

## Data Management Guidelines

### Raw Data (`data/raw/`)
- **Never modify** files in raw directories
- Original source datasets preserved for reproducibility
- Auto-downloaded by pipeline if missing
- Backed up and version controlled

### Processed Data (`data/processed/`)
- **Auto-generated** by pipeline from raw data
- Can be regenerated if corrupted or lost
- YOLO-compatible formats with data.yaml configurations
- Stratified splits ensure balanced train/val/test distributions

### Ground Truth Crops (Option A Shared)
- **Generated once** from raw annotations (not detection results)
- **Shared across all detection models** for consistency
- Clean, noise-free crop images for classification training
- Significant storage savings vs traditional approach

### Results Data (`results/`)
- **Experiment outputs** organized by timestamp and dataset
- Detection model weights, logs, and performance metrics
- Classification model weights and training results
- Comprehensive analysis reports and visualizations

## Pipeline Integration

### Automatic Workflow
The Option A pipeline handles all data management automatically:

1. **Dataset Detection**: Checks for raw data, downloads if missing
2. **Format Conversion**: Converts to YOLO format with proper splits
3. **Quality Validation**: Ensures data integrity and class balance
4. **Crop Generation**: Creates clean ground truth crops once
5. **Model Training**: Trains all models on consistent data
6. **Results Organization**: Structures outputs for easy analysis

### Manual Data Operations
```bash
# Standalone dataset setup (if needed)
python scripts/data_setup/setup_iml_lifecycle_for_pipeline.py
python scripts/data_setup/setup_mp_idb_species_for_pipeline.py
python scripts/data_setup/setup_mp_idb_stages_for_pipeline.py

# Ground truth crop generation (standalone)
python scripts/training/generate_ground_truth_crops.py \
  --dataset data/raw/malaria_lifecycle \
  --output data/processed/crops_ground_truth/iml_lifecycle \
  --type iml_lifecycle \
  --crop_size 224

# Dataset statistics analysis
python scripts/analysis/dataset_statistics_analyzer.py \
  --output analysis_results
```

## Storage Optimization

### Option A Efficiency Gains
- **Ground Truth Crops**: Generated once vs 3× (per detection model)
- **Shared Classification**: Single model set vs 3× (per detection model)
- **Clean Architecture**: Eliminates redundant intermediate files
- **Compressed Archives**: Auto-generated ZIP files for experiments

### Storage Requirements
- **Raw Datasets**: ~500MB (auto-downloaded)
- **Processed Data**: ~1GB (YOLO format + crops)
- **Single Experiment**: ~3-5GB (before compression)
- **Full Multi-Dataset**: ~15-18GB (before compression)
- **Compressed Archives**: ~50% reduction via ZIP

---
*Updated: 2025-09-30*
*Optimized for Option A: Shared Classification Architecture Pipeline*