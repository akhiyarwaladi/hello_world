# 🦠 Malaria Detection Pipeline
**Two-Stage Deep Learning Approach: Detection → Classification**

## 🎯 What This Does
Automatically detects malaria parasites in blood smear images and classifies them into 4 species:
- P. falciparum
- P. vivax
- P. ovale
- P. malariae

## 🚀 Quick Start (3 Steps)

### 1. Download Data
```bash
source venv/bin/activate
python scripts/data_setup/01_download_datasets.py --dataset mp_idb
```

### 2. Run Complete Pipeline
```bash
# Train detection → generate crops → train classification (all automatic)
python run_complete_pipeline.py --detection yolo8 --epochs-det 50 --epochs-cls 30
```

### 3. Check Results
Results saved in `results/current_experiments/`

## 📁 Project Structure (Organized & Clean)

```
📦 malaria-detection/
├── 🎮 pipeline.py                    # Main interface for training models
├── 🚀 run_complete_pipeline.py       # Full automation (recommended)
├── 📂 scripts/
│   ├── 📁 data_setup/                # Step 1: Data preparation
│   │   ├── 01_download_datasets.py   # Download MP-IDB dataset
│   │   ├── 02_preprocess_data.py     # Clean and process images
│   │   ├── 03_integrate_datasets.py  # Combine multiple datasets
│   │   ├── 04_convert_to_yolo.py     # Convert to YOLO format
│   │   ├── 05_augment_data.py        # Data augmentation
│   │   └── 06_split_dataset.py       # Train/val/test split
│   ├── 📁 training/                  # Step 2-3: Model training
│   │   ├── 07_train_yolo_detection.py    # YOLOv8 detection
│   │   ├── 08_train_yolo11_detection.py  # YOLOv11 detection
│   │   ├── 09_train_rtdetr_detection.py  # RT-DETR detection
│   │   ├── 10_crop_detections.py         # Generate crops from detection
│   │   ├── 11_train_classification_crops.py     # Train YOLO classification
│   │   ├── 11b_train_pytorch_classification.py  # PyTorch classifiers
│   │   └── 13_full_detection_classification_pipeline.py  # Bulk automation
│   └── 📁 analysis/                  # Performance analysis
│       └── 14_compare_models_performance.py
├── 📂 config/                        # All configurations
│   ├── models.yaml                   # Model settings
│   ├── datasets.yaml                 # Data configurations
│   ├── dataset_config.yaml           # Download settings
│   ├── class_names.yaml              # Class definitions
│   └── results_structure.yaml        # Results organization
├── 📂 data/                          # All datasets and results
│   └── raw/mp_idb/                   # Downloaded MP-IDB dataset
└── 📂 archive_unused/                # Complex analysis files (archived)
```

## 🔄 Three Ways to Use

### Option 1: 🎯 Full Automation (EASIEST)
```bash
# Everything automatic: detection training → crop generation → classification training
python run_complete_pipeline.py --detection yolo8 --epochs-det 50 --epochs-cls 30
```

### Option 2: ⚙️ Manual 3-Stage Control
```bash
# Stage 1: Train detection model
python pipeline.py train yolov8_detection --name my_detector --epochs 50

# Stage 2: Generate crops (auto-finds detection model)
python scripts/training/10_crop_detections.py --model yolo8 --experiment my_detector

# Stage 3: Train classification (auto-uses crops)
python pipeline.py train yolov8_classification --name my_classifier --epochs 30
```

### Option 3: 🔧 Custom Training
```bash
# List all available models
python pipeline.py list

# Train specific models with custom parameters
python pipeline.py train rtdetr_detection --name rtdetr_test --epochs 100 --batch 8
```

## 📊 Available Models

**Detection Models (Stage 1):**
- `yolov8_detection` - Fast and accurate
- `yolov11_detection` - Latest YOLO version
- `rtdetr_detection` - Transformer-based

**Classification Models (Stage 3):**
- `yolov8_classification` - YOLO classifier
- `yolov11_classification` - Latest YOLO classifier
- `pytorch_resnet18_classification` - ResNet18
- `pytorch_densenet121_classification` - DenseNet121
- `pytorch_efficientnet_b0_classification` - EfficientNet-B0
- `pytorch_mobilenet_v2_classification` - MobileNetV2

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

### Training Scripts (`scripts/training/`)
- **07_train_yolo_detection.py** - Train YOLOv8 detection
- **08_train_yolo11_detection.py** - Train YOLOv11 detection
- **09_train_rtdetr_detection.py** - Train RT-DETR detection
- **10_crop_detections.py** - Generate crops from detection results
- **11_train_classification_crops.py** - Train YOLO classification
- **11b_train_pytorch_classification.py** - Train PyTorch classifiers
- **13_full_detection_classification_pipeline.py** - Bulk automation

### Analysis Scripts (`scripts/analysis/`)
- **14_compare_models_performance.py** - Generate performance comparison

## 💡 Tips

**For First Time Use:**
1. Start with `mp_idb` dataset (most important)
2. Use `run_complete_pipeline.py` for simplicity
3. Try `yolo8` models first (good balance of speed/accuracy)

**For Research:**
- Use different detection models and compare results
- Experiment with epoch numbers (detection: 50-100, classification: 30-50)
- Check `results/` folder for training logs and model weights

**Troubleshooting:**
- If models not found: Check experiment names match between stages
- If out of memory: Reduce batch size with `--batch 4`
- If data missing: Re-run download script

## 🎓 Based on Research
Implementation follows the two-stage approach from:
*"Automated Identification of Malaria-Infected Cells and Classification of Human Malaria Parasites Using a Two-Stage Deep Learning Technique"*

---
**Ready to detect malaria with AI! 🔬🤖**