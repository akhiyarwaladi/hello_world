# Malaria Detection Project - Context for Claude

## 🎯 Project Overview
Comprehensive malaria detection system using YOLOv10, YOLOv11, YOLOv12, and RT-DETR models for microscopy image analysis.

**Current Status: ✅ PRODUCTION READY - Complete Pipeline with Multiple Datasets**

**Latest Update**: September 26, 2025 - Lifecycle dataset integration + 5-class malaria stage detection

**Latest Updates (Sep 26, 2025)**:
- ✅ **New Lifecycle Dataset**: 5-class detection (red_blood_cell, ring, gametocyte, trophozoite, schizont) - 345 labels
- ✅ **Kaggle Stage Dataset**: 4-class parasite stages (ring, schizont, trophozoite, gametocyte) - 342 images
- ✅ **Active Training**: Lifecycle dataset pipeline running in background (YOLOv10/11/12)
- ✅ **Multi-Dataset Support**: --dataset flag for different dataset selection
- ✅ Production-ready pipeline with full automation

**Live Experiments**:
- 🔄 **Running**: Lifecycle dataset pipeline (ID: 9a96d2) - 5-class detection training
- ✅ **Completed**: `exp_multi_pipeline_20250924_233409` - Complete analysis available
- ✅ **Ready Datasets**: kaggle_pipeline_ready (209 images), lifecycle_pipeline_ready (345 labels)

## 📊 Available Datasets

### Primary Datasets (Ready for Training)
1. **`lifecycle_pipeline_ready`** (NEWEST) - Complete malaria lifecycle detection
   - **Classes**: 5 classes (red_blood_cell, ring, gametocyte, trophozoite, schizont)
   - **Size**: 345 label files
   - **Usage**: `--dataset lifecycle`

2. **`kaggle_pipeline_ready`** (STABLE) - Parasite stage classification
   - **Classes**: 4 classes (ring, schizont, trophozoite, gametocyte)
   - **Size**: 209 images, 1436 objects
   - **Usage**: `--use-kaggle-dataset`

3. **`kaggle_stage_pipeline_ready`** - Stage-specific detection
   - **Classes**: 4 classes (ring, schizont, trophozoite, gametocyte)
   - **Size**: 342 images
   - **Usage**: `--dataset stage`

## 🚀 Quick Start Commands

### Complete Pipeline Automation (RECOMMENDED)
```bash
# NEWEST: Lifecycle dataset (5 classes) - exclude slow RT-DETR
python run_multiple_models_pipeline.py --dataset lifecycle --exclude-detection rtdetr --epochs-det 40 --epochs-cls 30

# STABLE: Kaggle dataset (4 classes) - most tested
python run_multiple_models_pipeline.py --use-kaggle-dataset --exclude-detection rtdetr --epochs-det 40 --epochs-cls 30

# QUICK TEST: Single model validation
python run_multiple_models_pipeline.py --include yolo11 --test-mode --use-kaggle-dataset
```

### Status Monitoring
```bash
python scripts/monitoring/training_status.py        # Check training progress
python pipeline_manager.py list                     # List experiments
python pipeline_manager.py status EXPERIMENT_NAME   # Detailed status
```

## ⚡ 3-Stage Workflow (CORE PIPELINE)

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

## 🔄 Multiple Models Pipeline

### Exclusion/Inclusion Support
```bash
# RECOMMENDED: Exclude slow models
python run_multiple_models_pipeline.py --exclude-detection rtdetr --epochs-det 40 --epochs-cls 30

# Include specific models only
python run_multiple_models_pipeline.py --include yolo10 yolo11 --epochs-det 40 --epochs-cls 30

# Test mode (reduced epochs)
python run_multiple_models_pipeline.py --include yolo11 --test-mode --epochs-det 5 --epochs-cls 5
```

### Continue/Resume Functionality
```bash
# List experiments that can be continued
python pipeline_manager.py list

# Continue from specific stage (SAFE - no overwrite)
python run_multiple_models_pipeline.py --continue-from EXPERIMENT_NAME --start-stage analysis --include yolo11

# Continue from earlier stage (OVERWRITES)
python run_multiple_models_pipeline.py --continue-from EXPERIMENT_NAME --start-stage crop --include yolo11 --epochs-cls 30
```

## 🎯 Available Models

### Detection Models (Stage 1)
- **YOLOv10** (`yolo10`) - Latest YOLO, balanced speed/accuracy
- **YOLOv11** (`yolo11`) - Current best performer
- **YOLOv12** (`yolo12`) - Experimental newest version
- **RT-DETR** (`rtdetr`) - Transformer-based (slower, high accuracy)

### Classification Models (Stage 3)
- **YOLO Classification** (`yolo10`, `yolo11`) - Fast YOLO-based classifiers
- **PyTorch Models** - ResNet18, EfficientNet-B0, DenseNet121, MobileNetV2

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
├── lifecycle_pipeline_ready/     # 5-class lifecycle dataset
├── kaggle_pipeline_ready/        # 4-class kaggle dataset
├── kaggle_stage_pipeline_ready/  # 4-class stage dataset
└── crops_from_*/                 # Generated crops from detection
```

## 🎉 Current Status: ADVANCED PIPELINE - PRODUCTION READY

✅ **Multiple dataset support** (lifecycle, kaggle, stage datasets)
✅ **5-class malaria lifecycle detection** (newest capability)
✅ **3-Stage workflow with automatic data flow**
✅ **Multiple model support with exclusion/inclusion**
✅ **Timestamp-based experiment naming**
✅ **Advanced monitoring and status tracking**
✅ **Error resilience and continuation**
✅ **Full automation available (one command)**
✅ **Auto folder organization & model discovery**

**CURRENT ACTIVE WORK**: Lifecycle dataset pipeline running in background (5-class detection: red_blood_cell + 4 parasite stages)

**SEPTEMBER 26, 2025 STATUS**:
- ✅ **Multi-dataset architecture**: Support for 3 different dataset configurations
- ✅ **Enhanced classification**: From 4-class to 5-class detection capability
- ✅ **Production stability**: All pipelines tested and operational
- ✅ **Background processing**: Long-running pipelines with monitoring support
- ✅ **Documentation streamlined**: Removed outdated sections, focused on current capabilities