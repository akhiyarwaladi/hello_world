# Malaria Detection Project - Context for Claude

## Project Overview
This is a comprehensive malaria detection system using YOLOv8 and RT-DETR models for microscopy image analysis. The project implements a **TWO-STAGE PIPELINE**:

### Stage 1: Detection (YOLO Object Detection)
- **Purpose**: Locate malaria parasites in blood smear images
- **Input**: Full microscopy images (1024x1024 typical)
- **Output**: Bounding boxes around detected parasites
- **Classes**: Binary detection (parasite vs background)
- **Dataset**: `data/detection/` - images with YOLO format annotations

### Stage 2: Classification (6-Class Species Classification)
- **Purpose**: Classify detected parasites into specific species
- **Input**: Cropped parasite regions from Stage 1 detection
- **Output**: Species classification with confidence scores
- **Classes**: 6-class system:
  1. **P_falciparum** - Most dangerous, causes severe malaria
  2. **P_vivax** - Common, causes recurring malaria
  3. **P_malariae** - Rare, chronic infections
  4. **P_ovale** - Rare, similar to vivax
  5. **Mixed_infection** - Multiple species present
  6. **Uninfected** - Normal red blood cells
- **Dataset**: `data/classification_crops/` - cropped images organized by species folders

### Two-Stage Architecture Benefits
- **Higher Accuracy**: Specialized models for each task
- **Efficient Processing**: Focus classification on detected regions only
- **Clinical Relevance**: Matches diagnostic workflow (find → identify)
- **Scalable**: Can handle full slide images efficiently

## Project Status (Updated: December 12, 2024)
- ✅ **All major codebase issues FIXED**
- ✅ **Data download completed** - 6 datasets successfully downloaded
- ✅ **Initial preprocessing COMPLETED** - 56,754 images processed  
- ✅ **Integration COMPLETED** - Unified dataset created
- ❌ **Species mapping issue FIXED** - Added proper P_falciparum, P_vivax processing
- 🔄 **Re-preprocessing running** (~15% complete with corrected species mapping)
- 🔄 **YOLOv8 training active** - Multiple training processes running
- 🤖 **Pipeline watcher active** - Monitoring all processes

## Key Scripts Fixed & Implemented
- `scripts/02_preprocess_data.py` - **UPDATED** - Added NIH thick smear species-specific processing
- `scripts/03_integrate_datasets.py` - **UPDATED** - Fixed species mapping for 6-class system  
- `scripts/04_convert_to_yolo.py` - Fixed from detector to MalariaYOLOConverter
- `scripts/05_augment_data.py` - Created comprehensive MalariaDataAugmenter
- `scripts/06_split_dataset.py` - Created MalariaDatasetSplitter
- `scripts/utils/` - Complete utils package with download, image, annotation helpers
- `scripts/07_train_yolo.py` - **NEW** - YOLOv8 training script
- `scripts/08_train_rtdetr.py` - **NEW** - RT-DETR training script

## Pipeline Architecture
1. **Data Download** (`01_download_datasets.py`) - Downloads NIH, MP-IDB, BBBC041, PlasmoID, IML, Uganda datasets
2. **Preprocessing** (`02_preprocess_data.py`) - Quality assessment, resizing, normalization with CLAHE
3. **Integration** (`03_integrate_datasets.py`) - Maps species to unified 6-class system
4. **YOLO Conversion** (`04_convert_to_yolo.py`) - Converts to YOLO training format
5. **Data Augmentation** (`05_augment_data.py`) - Uses Albumentations for minority class balancing  
6. **Dataset Splitting** (`06_split_dataset.py`) - Creates stratified train/val/test splits

## Automation Features
- **Pipeline Watcher** (`watch_pipeline.py`) - Monitors preprocessing completion and auto-runs remaining pipeline
- **Pipeline Runner** (`run_pipeline.py`) - Runs complete pipeline with error handling
- **Background Processing** - Multiple stages running in parallel

## Data Structure
```
data/
├── raw/                    # Downloaded datasets (gitignored)
├── processed/             # Preprocessed images & CSV metadata  
├── integrated/            # Unified dataset format
├── augmented/             # Augmented training data
└── splits/                # Train/validation/test splits
```

## Virtual Environment
- **Location**: `venv/` directory
- **Activation**: `source venv/bin/activate`  
- **Dependencies**: All required libraries installed (PyTorch, OpenCV, Albumentations, etc.)

## Background Processes Running
1. **Re-preprocessing** - Processing with corrected species mapping (~15% complete)
2. **YOLOv8 Training** - Multiple training processes active on CPU
3. **Integration** - Re-running with updated preprocessing data
4. **Pipeline Watcher** - Monitoring all processes automatically

## Usage Commands
```bash
# Activate environment
source venv/bin/activate

# Check pipeline status
python -c "from pathlib import Path; print('Processed images:', len(list(Path('data/processed/images').glob('*.jpg'))))"

# Manual pipeline execution
python watch_pipeline.py

# Individual script execution
python scripts/02_preprocess_data.py
```

## Current Training Status
1. **YOLOv8 Classification** - Active training on CPU (multiple processes)
2. **Species Distribution** - Fixed to include all 6 classes properly
3. **Performance Monitoring** - Real-time training progress tracking

## Next Steps
1. Complete corrected preprocessing with full species data
2. Evaluate training results and model performance
3. RT-DETR detection model training
4. Model optimization and deployment preparation

## Important Notes
- Large datasets are gitignored for repo size management
- All scripts have comprehensive error handling and logging
- Pipeline supports both automatic and manual execution modes
- Progress tracking available through todo lists and status messages