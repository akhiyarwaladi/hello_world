# Malaria Detection Project - Context for Claude

## 🎯 Project Overview
Comprehensive malaria detection system using YOLOv8, YOLOv11, YOLOv12, and RT-DETR models for microscopy image analysis.

**Current Status: ✅ ADVANCED PIPELINE - Multiple Models with Exclusion Support**

**Latest Update**: September 20, 2025 - Added advanced pipeline management with model exclusion/inclusion support

## 🚀 Simplified 3-Stage Workflow

### STAGE 1: Train Detection Model
```bash
# Use venv for all commands
source venv/bin/activate

python pipeline.py train yolov8_detection --name auto_yolov8_det --epochs 40
python pipeline.py train yolov11_detection --name auto_yolov11_det --epochs 40
python pipeline.py train yolo12_detection --name auto_yolo12_det --epochs 40
python pipeline.py train rtdetr_detection --name auto_rtdetr_det --epochs 40
```
**Output**: `results/current_experiments/training/detection/[model_type]/[experiment_name]/weights/best.pt`

### STAGE 2: Generate Crops from Detection
```bash
python scripts/10_crop_detections.py --model yolo8 --experiment auto_yolov8_det
python scripts/10_crop_detections.py --model yolo11 --experiment auto_yolov11_det
python scripts/10_crop_detections.py --model rtdetr --experiment auto_rtdetr_det
```
**Auto-finds model**: Automatically locates detection model from Stage 1
**Output**: `data/crops_from_[detection_type]_[experiment]/`

### STAGE 3: Train Classification Model
```bash
python pipeline.py train yolov8_classification --name auto_yolov8_cls --data data/crops_from_yolo8_auto_yolov8_det/
python pipeline.py train yolov11_classification --name auto_yolov11_cls --data data/crops_from_yolo11_auto_yolov11_det/
```
**Uses crop data**: Automatically uses crops generated from Stage 2

## ⚡ COMPLETE AUTOMATION - ADVANCED PIPELINE COMMANDS

### 🚀 NEW: Multiple Models with Exclusion Support
```bash
# ALL MODELS except specific ones (RECOMMENDED for large runs)
source venv/bin/activate
python run_multiple_models_pipeline.py --exclude rtdetr --epochs-det 40 --epochs-cls 30

# INCLUDE only specific models
python run_multiple_models_pipeline.py --include yolo8 yolo11 --epochs-det 40 --epochs-cls 30

# ALL MODELS (no exclusions)
python run_multiple_models_pipeline.py --epochs-det 40 --epochs-cls 30
```

### 🎯 Single Model Pipeline
```bash
# SINGLE MODEL: Train one detection model with classification
source venv/bin/activate
python run_complete_pipeline.py --detection yolo8 --epochs-det 40 --epochs-cls 30
```

### 📊 Monitoring & Status Tools
```bash
# Check training status clearly
python scripts/check_training_status.py

# Experiment management
python scripts/experiment_manager.py
```

**NEW Features**:
- **Model Exclusion**: Skip specific models (e.g., exclude RT-DETR)
- **Model Inclusion**: Run only specific models
- **Timestamp Naming**: Auto-generated unique experiment names
- **Clear Monitoring**: Unambiguous training status tracking
- **Error Handling**: Continue with other models if one fails

## 🔗 AUTOMATIC DATA FLOW BETWEEN STAGES

### How Stage 2 Finds Detection Model from Stage 1
```python
# Function: find_detection_model() in scripts/13_full_detection_classification_pipeline.py
# Pattern: results/current_experiments/training/detection/[model_type]/[experiment]/weights/best.pt

--model yolo8 → searches in yolov8_detection/ folder
--experiment auto_yolov8_det → searches for auto_yolov8_det/ subfolder
→ Auto-finds: results/current_experiments/training/detection/yolov8_detection/auto_yolov8_det/weights/best.pt
```

### How Stage 3 Finds Crop Data from Stage 2
```python
# Crop generation creates structured output
Stage 2 output: data/crops_from_[detection_type]_[experiment]/
Stage 3 input: --data parameter automatically passed to classification training

# Example:
Stage 1: train yolov8_detection --name my_detector
Stage 2: generates crops → data/crops_from_yolo8_my_detector/
Stage 3: --data data/crops_from_yolo8_my_detector/ (auto-passed)
```

### Key Automation Features
- **Structured Paths**: All outputs follow consistent naming patterns
- **Pattern Matching**: Scripts auto-discover previous stage outputs
- **Parameter Passing**: Data paths automatically passed between stages
- **No Manual Copy-Paste**: Everything connected through folder structure

## 📁 Organized Results Structure (AUTO-GENERATED)

```
results/
├── current_experiments/    # Active training/validation
│   ├── validation/        # Quick tests & validation
│   ├── training/          # Full training experiments
│   └── comparison/        # Model comparisons
├── completed_models/      # Production-ready models
├── publications/         # Publication-ready exports
├── archive/              # Historical experiments
└── experiment_logs/      # All experiment logs
```

## 🎯 Available Models & Workflows

### Detection Models (Stage 1)
- `yolov8_detection` - Fast and accurate
- `yolov11_detection` - Latest YOLO version
- `yolo12_detection` - Newest YOLO version (experimental)
- `rtdetr_detection` - Transformer-based (slower but high accuracy)

### Classification Models (Stage 3)
- `yolov8_classification` - Species classification
- `yolov11_classification` - Latest classifier
- `pytorch_classification` - Various CNN architectures (ResNet, DenseNet, etc.)

## ⚡ Quick Commands

### Check Status
```bash
python pipeline.py list                      # List available models
python pipeline.py status                    # Check system status
```

### Complete Workflow Options

#### Option 1: TRUE Full Automation (RECOMMENDED)
```bash
python run_complete_pipeline.py --detection yolo8 --epochs-det 50 --epochs-cls 30
```
**What it does sequentially**:
1. **Trains detection model** → saves to structured path
2. **Auto-finds detection model** → generates crops to structured path
3. **Auto-passes crop data** → trains classification model

#### Option 2: Bulk Processing (if detection models exist)
```bash
python scripts/13_full_detection_classification_pipeline.py --detection yolo8
```
**What it does** (assumes detection models already trained):
1. **Finds existing detection model** → generates crops
2. **Trains multiple classification models** on crops

#### Option 3: Manual 3-Stage Process (FULL CONTROL)
```bash
# Stage 1: Train detection (saves to structured path)
python pipeline.py train yolov8_detection --name my_detector --epochs 50
# → results/current_experiments/training/detection/yolov8_detection/my_detector/weights/best.pt

# Stage 2: Generate crops (auto-finds model from stage 1)
python scripts/10_crop_detections.py --model yolo8 --experiment my_detector
# → data/crops_from_yolo8_my_detector/

# Stage 3: Train classification (use crops from stage 2)
python pipeline.py train yolov8_classification --name my_classifier --data data/crops_from_yolo8_my_detector/
# → results/current_experiments/training/classification/yolov8_classification/my_classifier/
```

#### Option 3: Data Setup (if needed)
```bash
python scripts/01_download_datasets.py       # Download all datasets
python scripts/02_preprocess_data.py         # Preprocess images
python scripts/03_integrate_datasets.py      # Integrate datasets
```

## 🔧 Key Files & Scripts

### Main Scripts (CORE WORKFLOW)
```
run_multiple_models_pipeline.py          # NEW: Multiple models with exclusion support
run_complete_pipeline.py                 # Single model full automation (all 3 stages)
pipeline.py                               # Main interface for training
scripts/10_crop_detections.py            # Stage 2: Generate crops from detection
scripts/check_training_status.py         # NEW: Clear training status monitoring
scripts/experiment_manager.py            # NEW: Experiment organization & tracking
```

### Data Setup Scripts (if needed)
```
scripts/01_download_datasets.py          # Download datasets
scripts/02_preprocess_data.py            # Preprocess images
scripts/03_integrate_datasets.py         # Integrate datasets
```

### Utilities
```
utils/results_manager.py                 # Auto folder organization
config/models.yaml                       # Model configurations
```

## 🎯 Current Workflow Status
- **3-Stage Pipeline**: Detection → Crop → Classification
- **Multiple Model Support**: Train all models with exclusions
- **Timestamp Naming**: Clear experiment identification
- **Full Automation**: Single command runs everything
- **Organized Results**: Auto-organized folder structure
- **Advanced Monitoring**: Clear status tracking tools
- **Error Resilience**: Continue on failures

## 🚨 Important Notes for Claude

### 🆕 FRESH MACHINE SETUP (FROM ZERO TO RESULTS)
**Complete setup from fresh clone to results:**

```bash
# 1. Clone repository & setup environment
git clone [repository-url]
cd hello_world
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Setup data pipeline (4 steps) - CORRECTED PATH
python scripts/data_setup/01_download_datasets.py --dataset mp_idb  # Download MP-IDB (~500MB)
python scripts/data_setup/02_preprocess_data.py                     # Preprocess images
python scripts/data_setup/03_integrate_datasets.py                  # Integrate datasets
python scripts/data_setup/04_convert_to_yolo.py                     # Convert to YOLO format

# 3. Run complete 3-stage workflow (RECOMMENDED)
python run_multiple_models_pipeline.py --exclude rtdetr --epochs-det 40 --epochs-cls 30

# Alternative: Single model pipeline
python run_complete_pipeline.py --detection yolo8 --epochs-det 40 --epochs-cls 30
```

**NOTE**: The old `quick_setup_new_machine.sh` references missing scripts:
- `scripts/08_parse_mpid_detection.py` ❌ → Use data_setup pipeline above ✅
- `scripts/09_crop_parasites_from_detection.py` ❌ → Use `scripts/training/10_crop_detections.py` ✅

### CURRENT WORKFLOW - 3 STAGES WITH AUTOMATIC DATA FLOW
1. **Train Detection Model** → `pipeline.py train [detection_model]` → saves to structured path
2. **Generate Crops** → `scripts/10_crop_detections.py` → auto-finds detection model → generates crops
3. **Train Classification** → `pipeline.py train [classification_model]` → uses crop data from stage 2

### AUTOMATIC DATA CONNECTIONS
- **Stage 1→2**: Detection model auto-discovered via `find_detection_model()` function
- **Stage 2→3**: Crop data path auto-passed to classification training
- **No manual path copying**: Everything connected through consistent folder structure

### RECOMMENDED COMMANDS (UPDATED)
```bash
# BEST: Multiple models with exclusion (current running)
source venv/bin/activate
python run_multiple_models_pipeline.py --exclude rtdetr --epochs-det 40 --epochs-cls 30

# SINGLE MODEL pipeline
python run_complete_pipeline.py --detection yolo8 --epochs-det 40 --epochs-cls 30

# STATUS MONITORING
python scripts/check_training_status.py
```

### Error Handling & Troubleshooting
- **IMPORTANT**: Always use `source venv/bin/activate` before running any command
- **Data issues**: Run `python scripts/01_download_datasets.py` first
- **Memory issues**: Use `--batch 4` for CPU training
- **Model not found**: Script auto-searches in structured paths, check experiment names
- **Crop generation fails**: Check if detection model exists at expected path
- **Training status**: Use `python scripts/check_training_status.py` for clear status
- **Experiment tracking**: Use `python scripts/experiment_manager.py` for organization

### Data Flow Paths (AUTOMATIC)
```
Stage 1 Output: results/current_experiments/training/detection/[model_type]/[experiment_name]/weights/best.pt
Stage 2 Input:  Auto-found via pattern matching
Stage 2 Output: data/crops_from_[detection_type]_[experiment]/
Stage 3 Input:  Auto-passed crop data path
Stage 3 Output: results/current_experiments/training/classification/[model_type]/[experiment_name]/
```

### Pattern Matching Logic
```python
# Stage 2 finds Stage 1 model:
--model yolo8 → searches yolov8_detection/ folder
--experiment my_exp → searches my_exp/ subfolder
→ finds: yolov8_detection/my_exp/weights/best.pt

# Stage 3 gets Stage 2 data:
crops generated → data/crops_from_yolo8_my_exp/
classification training → --data automatically set to crop path
```

---

## 🎉 Status: ADVANCED PIPELINE - PRODUCTION READY

✅ **3-Stage workflow with automatic data flow**
✅ **Multiple model support with exclusion/inclusion**
✅ **Timestamp-based experiment naming (no ambiguity)**
✅ **Advanced monitoring and status tracking**
✅ **Error resilience and continuation**
✅ **Full automation available (one command)**
✅ **Auto folder organization**
✅ **Auto model/data discovery**

**CURRENT STATUS**: YOLOv8 training in progress (epoch 18/40), YOLOv11 and YOLOv12 queued, RT-DETR excluded.

**LATEST FEATURES**:
- Model exclusion: `--exclude rtdetr`
- Model inclusion: `--include yolo8 yolo11`
- Clear status monitoring
- Experiment management tools
- Virtual environment integration
