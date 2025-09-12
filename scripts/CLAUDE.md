# Scripts Directory - Context for Claude

## Purpose
Contains all Python scripts for the malaria detection pipeline.

## Pipeline Scripts (Fixed & Working)
1. `01_download_datasets.py` - **MalariaDatasetDownloader** - Downloads from multiple sources (NIH, MP-IDB, etc.)
2. `02_preprocess_data.py` - **MalariaDataPreprocessor** - **UPDATED** - Added NIH thick smear species processing
3. `03_integrate_datasets.py` - **MalariaDatasetIntegrator** - **UPDATED** - Fixed species mapping for 6 classes  
4. `04_convert_to_yolo.py` - **MalariaYOLOConverter** - Converts to YOLO training format
5. `05_augment_data.py` - **MalariaDataAugmenter** - Albumentations for minority class balancing
6. `06_split_dataset.py` - **MalariaDatasetSplitter** - Stratified train/val/test splits
7. `07_train_yolo.py` - **YOLOTrainer** - **NEW** - YOLOv8 classification training
8. `08_train_rtdetr.py` - **RTDETRTrainer** - **NEW** - RT-DETR detection training

## Utility Scripts
- `run_pipeline.py` - Manual pipeline runner with error handling
- `watch_pipeline.py` - **ACTIVE** - Auto-monitors and continues pipeline execution

## Major Fixes Applied
- Scripts 02, 03, 04 had WRONG content (trainer/detector code instead of intended functionality)
- Scripts 05, 06 were essentially empty (1 line each)
- **ALL FIXED** with proper classes and comprehensive functionality
- Added complete error handling, logging, progress bars

## Current Status (Updated: December 12, 2024)
- **Preprocessing** (02) **UPDATED & RE-RUNNING** - Fixed species mapping, ~15% complete
- **Integration** (03) **COMPLETED & RE-RUNNING** - Processing with corrected data
- **Training** (07,08) **ACTIVE** - YOLOv8 training processes running on CPU
- **Pipeline Watcher** active - monitoring all background processes

## Usage
```bash
source venv/bin/activate
python scripts/02_preprocess_data.py    # Currently running in background
python watch_pipeline.py                # Currently monitoring automatically
```