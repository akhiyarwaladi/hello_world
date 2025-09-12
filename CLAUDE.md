# Malaria Detection Project - Context for Claude

## Project Overview
This is a comprehensive malaria detection system using YOLOv8 and RT-DETR models for microscopy image analysis. The project processes multiple datasets for malaria parasite detection and classification into 6 classes: P_falciparum, P_vivax, P_malariae, P_ovale, Mixed_infection, and Uninfected.

## Project Status (Last Updated)
- ✅ **All major codebase issues FIXED**
- ✅ **Data download completed** 
- 🔄 **Preprocessing running** (13% complete, ~27k images processed)
- 🤖 **Automated pipeline active** - will auto-run remaining steps
- 📁 **Ready for**: Integration → YOLO Conversion → Augmentation → Dataset Splitting

## Key Scripts Fixed & Implemented
- `scripts/02_preprocess_data.py` - Fixed from trainer to proper MalariaDataPreprocessor
- `scripts/03_integrate_datasets.py` - Fixed from trainer to MalariaDatasetIntegrator  
- `scripts/04_convert_to_yolo.py` - Fixed from detector to MalariaYOLOConverter
- `scripts/05_augment_data.py` - Created comprehensive MalariaDataAugmenter
- `scripts/06_split_dataset.py` - Created MalariaDatasetSplitter
- `scripts/utils/` - Complete utils package with download, image, annotation helpers

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
1. **Preprocessing** - Processing NIH dataset images with progress bars
2. **Integration** - Waiting for CSV file from preprocessing
3. **Pipeline Watcher** - Monitoring for auto-continuation

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

## Next Steps After Pipeline Completion
1. Model training with YOLOv8/RT-DETR
2. Performance evaluation  
3. Model optimization and deployment

## Important Notes
- Large datasets are gitignored for repo size management
- All scripts have comprehensive error handling and logging
- Pipeline supports both automatic and manual execution modes
- Progress tracking available through todo lists and status messages