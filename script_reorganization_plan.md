# Script Reorganization Plan

## Current Scripts vs New Names

### Data Processing
- `01_download_datasets.py` → `download_datasets.py` ✓
- `02_preprocess_data.py` → `preprocess_images.py`
- `03_integrate_datasets.py` → `integrate_datasets.py` ✓
- `04_convert_to_yolo.py` → `convert_to_yolo.py` ✓
- `05_augment_data.py` → `augment_data.py` ✓
- `06_split_dataset.py` → `split_dataset.py` ✓

### Detection Training
- `09_train_detection.py` → `train_detection_legacy.py` (remove?)
- `10_train_yolo_detection.py` → `train_yolo_detection.py`
- `12_train_yolo11_detection.py` → `train_yolo11_detection.py`
- `13_train_rtdetr_detection.py` → `train_rtdetr_detection.py`

### Classification Training
- `07_train_yolo_quick.py` → `train_classification_quick.py`
- `11_train_classification_crops.py` → `train_classification_crops.py`

### Data Preparation
- `08_parse_mpid_detection.py` → `parse_mpid_annotations.py`
- `09_crop_parasites_from_detection.py` → `crop_parasites_from_detection.py`
- `10_crop_detections.py` → `crop_detections.py`

### Analysis & Comparison
- `14_compare_models_performance.py` → `compare_model_performance.py`

### Pipeline Management
- `run_full_pipeline.py` → `run_pipeline.py` ✓
- `watch_pipeline.py` → `watch_pipeline.py` ✓

## Reorganization Strategy

1. First, update the enhanced pipeline to use new script names
2. Rename scripts one by one, testing each rename
3. Update any references in documentation
4. Remove duplicate/obsolete scripts

## Priority Order
1. Fix pipeline script references
2. Rename core training scripts (most important for user)
3. Rename data processing scripts
4. Clean up duplicates