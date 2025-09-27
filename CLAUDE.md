# Malaria Detection Project - Context for Claude

## Project Overview
Comprehensive malaria detection system using YOLOv10, YOLOv11, YOLOv12, and RT-DETR models for microscopy image analysis.

**Current Status: PRODUCTION READY - Complete Pipeline with Multiple Datasets**

**Latest Update**: September 27, 2025 - 3-Dataset Architecture with Unicode Fixes

**Current Status (Sep 27, 2025)**:
- **3 Dataset Support**: Species, Stages, and Lifecycle detection
- **Unicode Issues Fixed**: All emoticons removed, clean pipeline execution
- **Class Imbalance Identified**: IML lifecycle dataset analysis completed
- **Production Ready**: mp_idb_stages recommended for balanced training

## Available Datasets

### 1. **mp_idb_species** (4 Species Classification - STABLE)
- **Classes**: P_falciparum, P_vivax, P_malariae, P_ovale
- **Size**: 209 images, 1,436 objects
- **Usage**: `--use-kaggle-dataset`
- **Status**: Balanced, production ready
- **Path**: `kaggle_pipeline_ready/`

### 2. **mp_idb_stages** (4 Stage Classification - RECOMMENDED)
- **Classes**: ring, schizont, trophozoite, gametocyte
- **Size**: 342 images
- **Usage**: `--dataset mp_idb_stages`
- **Status**: Best balanced dataset
- **Path**: `kaggle_stage_pipeline_ready/`

### 3. **iml_lifecycle** (4 Parasite Stages - FOCUSED)
- **Classes**: ring, gametocyte, trophozoite, schizont (no red blood cells)
- **Size**: 313 images, 529 parasite objects
- **Usage**: `--dataset iml_lifecycle`
- **Status**: Focused on parasite stages only (balanced)
- **Path**: `lifecycle_pipeline_ready/`
- **Note**: Red blood cells removed for focused parasite classification

## Quick Start Commands

### 1. **mp_idb_species** (4 Species - STABLE)
```bash
# Production Pipeline
python run_multiple_models_pipeline.py --use-kaggle-dataset --exclude-detection rtdetr --epochs-det 40 --epochs-cls 30

# Quick Test
python run_multiple_models_pipeline.py --use-kaggle-dataset --include yolo11 --test-mode --epochs-det 5 --epochs-cls 5
```

### 2. **mp_idb_stages** (4 Stages - RECOMMENDED)
```bash
# Best Balanced Pipeline
python run_multiple_models_pipeline.py --dataset mp_idb_stages --exclude-detection rtdetr --epochs-det 40 --epochs-cls 30

# Quick Test
python run_multiple_models_pipeline.py --dataset mp_idb_stages --include yolo11 --test-mode --epochs-det 5 --epochs-cls 5
```

### 3. **iml_lifecycle** (4 Parasite Stages - FOCUSED)
```bash
# Focused Parasite Training
python run_multiple_models_pipeline.py --dataset iml_lifecycle --include yolo11 --epochs-det 20 --epochs-cls 15

# Quick Test
python run_multiple_models_pipeline.py --dataset iml_lifecycle --include yolo11 --test-mode --epochs-det 5 --epochs-cls 5
```

### Status Monitoring
```bash
python scripts/monitoring/training_status.py        # Check training progress
python pipeline_manager.py list                     # List experiments
python pipeline_manager.py status EXPERIMENT_NAME   # Detailed status
```

## 3-Stage Workflow (CORE PIPELINE)

### STAGE 1: Train Detection Model
```bash
python pipeline.py train yolo11_detection --name auto_yolo11_det --epochs 40
```
**Output**: `results/current_experiments/training/detection/[model_type]/[experiment_name]/weights/best.pt`

### STAGE 2: Generate Crops from Detection
```bash
python scripts/training/11_crop_detections.py --model yolo11 --experiment auto_yolo11_det
```
**Auto-finds model**: Automatically locates detection model from Stage 1
**Output**: `data/crops_from_[detection_type]_[experiment]/`

### STAGE 3: Train Classification Model
```bash
python pipeline.py train yolo11_classification --name auto_yolo11_cls --data data/crops_from_yolo11_auto_yolo11_det/
```
**Uses crop data**: Automatically uses crops generated from Stage 2

## Multiple Models Pipeline

### Parameter Reference
```bash
# Dataset Selection
--use-kaggle-dataset          # mp_idb_species (4 species)
--dataset mp_idb_stages       # Stage classification (4 stages)
--dataset iml_lifecycle       # Lifecycle detection (5 classes)

# Model Selection
--include yolo10 yolo11       # Specific models only
--exclude-detection rtdetr    # Skip slow RT-DETR
--test-mode                   # Reduced epochs for testing

# Epoch Configuration
--epochs-det 40               # Detection training epochs
--epochs-cls 30               # Classification training epochs
```

### Recommended Starting Point
```bash
# Best balanced dataset for initial testing
python run_multiple_models_pipeline.py --dataset mp_idb_stages --include yolo11 --epochs-det 15 --epochs-cls 10
```

## Available Models

### Detection Models (Stage 1)
- **YOLOv10** (`yolo10`) - Balanced speed/accuracy
- **YOLOv11** (`yolo11`) - Current best performer
- **YOLOv12** (`yolo12`) - Experimental newest version
- **RT-DETR** (`rtdetr`) - Transformer-based (slower, exclude recommended)

### Classification Models (Stage 3)
- **PyTorch Models**: DenseNet121, EfficientNet-B1, ConvNeXt-Tiny, MobileNetV3-Large, EfficientNet-B2, ResNet101
- **YOLO Classification**: Built-in YOLO classifiers

## 🔗 Automatic Data Flow

### Stage Connection Logic
```python
# Stage 1 → Stage 2: Auto-discovery
--model yolo11 --experiment my_det → finds: yolov11_detection/my_det/weights/best.pt

# Stage 2 → Stage 3: Auto-path generation
Stage 2 output: data/crops_from_yolo11_my_det/
Stage 3 input: --data automatically set to crop path
```

### Key Automation Features
- **Structured Paths**: All outputs follow consistent naming patterns
- **Pattern Matching**: Scripts auto-discover previous stage outputs
- **Parameter Passing**: Data paths automatically passed between stages
- **No Manual Copy-Paste**: Everything connected through folder structure

## 🔧 Key Scripts & Files

### Core Pipeline
```
run_multiple_models_pipeline.py    # Multi-model automation with analysis
run_complete_pipeline.py           # Single model full automation
pipeline.py                        # Main CLI interface
pipeline_manager.py                # Experiment management
```

### Stage Scripts
```
scripts/training/11_crop_detections.py           # Stage 2: Generate crops
scripts/monitoring/training_status.py           # Training progress monitoring
scripts/analysis/unified_journal_analysis.py    # Publication-ready analysis
```

### Configuration
```
config/models.yaml                 # Model definitions & parameters
config/datasets.yaml               # Dataset sources & classes
utils/results_manager.py           # Auto folder organization
```

## 📁 Output Organization

### Training Outputs
```
results/current_experiments/
├── training/              # Normal mode (full epochs)
│   ├── detection/         # Detection models
│   └── classification/    # Classification models
└── validation/            # Test mode (--test-mode)
```

### Analysis & Reports
```
results/
├── exp_multi_pipeline_*/  # Complete experiment analyses
│   ├── analysis/          # Confusion matrix, metrics, reports
│   ├── experiment_summary.json
│   └── experiment_summary.md
└── archive/               # Historical experiments
```

### Data Storage
```
data/
├── lifecycle_pipeline_ready/        # iml_lifecycle (5 classes, imbalanced)
├── kaggle_pipeline_ready/           # mp_idb_species (4 species, stable)
├── kaggle_stage_pipeline_ready/     # mp_idb_stages (4 stages, recommended)
└── crops_from_*/                    # Generated crops from detection
```

## Current Status: 3-DATASET PRODUCTION READY

**3 Dataset Architecture**:
- **mp_idb_species**: 4 species classification (stable)
- **mp_idb_stages**: 4 stage classification (recommended)
- **iml_lifecycle**: 5 class detection (experimental, imbalanced)

**Key Features**:
- 3-Stage workflow with automatic data flow
- Multiple model support (YOLOv10/11/12, RT-DETR)
- 6 PyTorch classification models
- Unicode issues resolved
- Class imbalance analysis completed

**SEPTEMBER 27, 2025 STATUS**:
- **Production Ready**: mp_idb_stages recommended for balanced training
- **Issue Resolution**: Unicode encoding errors fixed
- **Dataset Analysis**: Class imbalance documented (IML lifecycle 98.6% red blood cells)
- **Script Fixes**: Crop detection updated for lifecycle dataset support