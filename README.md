# 🦠 Malaria Detection Pipeline
**Two-Stage Deep Learning Approach: Detection → Classification**

## 🎯 What This Does
Automatically detects malaria parasites in blood smear images and classifies them into 4 species:
- P. falciparum
- P. vivax
- P. ovale
- P. malariae

## 🚀 Quick Start (3 Steps)

### 1. Setup & Download Data
```bash
# Setup environment
source venv/bin/activate

# RECOMMENDED: Download Kaggle YOLO dataset (ready-to-use)
python scripts/data_setup/01_download_datasets.py --dataset kaggle_mp_idb
python scripts/data_setup/setup_kaggle_dataset.py

# Alternative: Original MP-IDB (requires processing)
# python scripts/data_setup/01_download_datasets.py --dataset mp_idb
```

### 2. Run Complete Pipeline
```bash
# Using Kaggle dataset (RECOMMENDED) - Multiple models automation
python run_multiple_models_pipeline.py --use-kaggle-dataset --exclude rtdetr --epochs-det 40 --epochs-cls 30

# Alternative: Single model only
python run_multiple_models_pipeline.py --use-kaggle-dataset --include yolo8 --epochs-det 40 --epochs-cls 30
```

### 3. Check Results
Results saved in `results/current_experiments/` and `experiments/`

## 📁 Project Structure (Organized & Clean)

```
📦 malaria-detection/
├── 🔥 run_multiple_models_pipeline.py     # Main pipeline interface (RECOMMENDED)
├── 📋 pipeline_manager.py                 # Pipeline management & continue functionality
├── 🛠️ setup_kaggle_for_pipeline.py       # Kaggle dataset setup helper
├── 📂 scripts/
│   ├── 📁 data_setup/                     # Step 1: Data preparation
│   │   ├── 01_download_datasets.py        # Download datasets (includes Kaggle)
│   │   ├── setup_kaggle_dataset.py        # Setup Kaggle YOLO dataset
│   │   ├── 02_preprocess_data.py          # Clean and process images
│   │   ├── 03_integrate_datasets.py       # Combine multiple datasets
│   │   ├── 04_convert_to_yolo.py          # Convert to YOLO format
│   │   ├── 05_augment_data.py             # Data augmentation
│   │   └── 06_split_dataset.py            # Train/val/test split
│   └── 📁 training/                       # Step 2-3: Model training
│       ├── 10_crop_detections.py          # Generate crops from detection
│       └── 11b_train_pytorch_classification.py  # PyTorch classifiers
│   └── 📁 analysis/                       # Performance analysis
│       └── 14_compare_models_performance.py
├── 📂 config/                             # All configurations
│   ├── models.yaml                        # Model settings
│   ├── datasets.yaml                      # Data configurations
│   └── dataset_config.yaml                # Download settings
├── 📂 data/                               # All datasets and results
│   ├── raw/mp_idb/                        # Downloaded MP-IDB dataset
│   └── kaggle_dataset/                    # Kaggle YOLO dataset
└── 📂 results/                            # Training results and analysis
```

## 🔄 Three Ways to Use

### Option 1: 🔥 Multiple Models Pipeline (RECOMMENDED)
```bash
# Train all models (exclude slow RT-DETR) with Kaggle dataset
python run_multiple_models_pipeline.py --use-kaggle-dataset --exclude rtdetr --epochs-det 40 --epochs-cls 30

# Include specific models only
python run_multiple_models_pipeline.py --use-kaggle-dataset --include yolo8 yolo11 --epochs-det 40 --epochs-cls 30
```

### Option 2: 🎯 Single Model Only
```bash
# Train only specific model: detection training → crop generation → classification training
python run_multiple_models_pipeline.py --use-kaggle-dataset --include yolo8 --epochs-det 40 --epochs-cls 30
```

### Option 3: ⚙️ Manual 3-Stage Control
```bash
# Stage 1: Use single model pipeline with specific settings
python run_multiple_models_pipeline.py --include yolo8 --epochs-det 50 --epochs-cls 30

# Alternative: Use individual training scripts
# Stage 2: Generate crops (if needed separately)
python scripts/training/10_crop_detections.py --model yolo8 --input data/kaggle_detection_ready/test/images --output data/manual_crops

# Stage 3: Train classification on custom crops
python scripts/training/11b_train_pytorch_classification.py --data data/manual_crops
```

### Option 4: 🔧 Custom Training
```bash
# View pipeline management and continue options
python pipeline_manager.py list

# Custom training with specific parameters
python run_multiple_models_pipeline.py --include rtdetr --epochs-det 100 --epochs-cls 50 --test-mode
```

## 📊 Available Models

**Detection Models (Stage 1):**
- `yolov8_detection` - Fast and accurate (RECOMMENDED)
- `yolov11_detection` - Latest YOLO version
- `yolo12_detection` - Newest YOLO variant
- `rtdetr_detection` - Transformer-based (slower)

**Classification Models (Stage 3):**
- `yolov8_classification` - YOLO classifier (RECOMMENDED)
- `yolov11_classification` - Latest YOLO classifier
- `pytorch_classification` - Multiple CNN architectures:
  - ResNet18, DenseNet121, EfficientNet-B0, MobileNetV2

## 📈 Workflow Explained

### Stage 1: Detection
- **Input**: Full blood smear images
- **Output**: Bounding boxes around infected cells
- **Models**: YOLOv8, YOLOv11, RT-DETR

### Stage 2: Crop Generation
- **Input**: Detection results + original images
- **Output**: Individual cell crops (128x128 pixels)
- **Automatic**: Finds detection model → generates crops

### Stage 3: Classification
- **Input**: Cell crops from Stage 2
- **Output**: Species classification (4 classes)
- **Models**: YOLO, ResNet, DenseNet, etc.

## 🗂️ Results Organization

```
results/current_experiments/
├── training/
│   ├── detection/[model_type]/[experiment_name]/
│   └── classification/[model_type]/[experiment_name]/
└── validation/
```

## 🛠️ Scripts Organization

### Data Setup Scripts (`scripts/data_setup/`)
- **01_download_datasets.py** - Download MP-IDB and other datasets
- **02_preprocess_data.py** - Clean and standardize images
- **03_integrate_datasets.py** - Combine multiple datasets
- **04_convert_to_yolo.py** - Convert annotations to YOLO format
- **05_augment_data.py** - Apply data augmentation
- **06_split_dataset.py** - Create train/val/test splits

### Training Scripts (`scripts/`)
- **training/10_crop_detections.py** - Generate crops from detection results
- **training/11b_train_pytorch_classification.py** - Train PyTorch classifiers
- **Main Pipeline Interface:**
  - `run_multiple_models_pipeline.py` - Primary automation interface
  - `pipeline_manager.py` - Experiment management & continue functionality
  - `setup_kaggle_for_pipeline.py` - Dataset setup helper

### Analysis Scripts (`scripts/analysis/`)
- **14_compare_models_performance.py** - Generate performance comparison

## 💡 Tips

**For First Time Use:**
1. Setup Kaggle API credentials first (required for dataset download)
2. Start with `kaggle_mp_idb` dataset (ready-to-use, RECOMMENDED)
3. Use `run_multiple_models_pipeline.py` for comprehensive results
4. Exclude RT-DETR initially (`--exclude rtdetr`) as it's slower

**For Research:**
- Use multiple models pipeline to compare all approaches
- Experiment with epoch numbers (detection: 40-100, classification: 30-50)
- Check `results/` and `experiments/` folders for comprehensive analysis
- Use `--use-kaggle-dataset` flag for optimized dataset

**Troubleshooting:**
- If Kaggle download fails: Setup API credentials in `~/.kaggle/kaggle.json`
- If models not found: Check experiment names match between stages
- If out of memory: Reduce batch size with `--batch 4`
- If data missing: Re-run download script with correct dataset name

## 🎓 Based on Research
Implementation follows the two-stage approach from:
*"Automated Identification of Malaria-Infected Cells and Classification of Human Malaria Parasites Using a Two-Stage Deep Learning Technique"*

---
**Ready to detect malaria with AI! 🔬🤖**