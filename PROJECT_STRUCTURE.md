# Malaria Detection Project - Clean Structure

## 📁 Current Active Files

### Core Training Scripts
- `train_multispecies.py` - **MAIN TRAINING SCRIPT** (clean, no NNPACK warnings)
- `create_multispecies_dataset.py` - Dataset creation from MP-IDB

### Pipeline & Scripts
- `pipeline_enhanced.py` - Active pipeline (descriptive script names)
- `scripts/train_yolo_detection.py` - YOLO detection training
- `scripts/train_classification_crops.py` - Classification training
- `scripts/crop_detections.py` - Create classification crops

### Current Datasets
- `data/detection_multispecies/` - **NEW** Multi-species YOLO detection dataset
- `data/classification_multispecies/` - **NEW** 4-class classification dataset
  - falciparum: 1,210 crops
  - vivax: 59 crops
  - malariae: 43 crops
  - ovale: 33 crops

### Configuration
- `.bashrc_ml` - ML environment settings (NNPACK_DISABLE=1)
- `CLAUDE.md` - Project documentation

## 🗂️ Archived Files

### archive/debug/
- `analyze_classification_problem.py` - Used to find 100% accuracy issue
- `debug_crop_species.py` - Species mapping debugging
- `test_validation.py` - Validation testing

### archive/pipelines/
- `malaria_pipeline.py` - Old pipeline version
- `pipeline.py` - Original pipeline

### archive/old_scripts/
- `download_complete_mp_idb.py` - MP-IDB download script
- `convert_masks_to_bbox.py` - Mask to bbox conversion

## ✅ Problem Resolution Status

1. **100% Classification Accuracy** - FIXED ✅
   - Root cause: Single-class dataset ("parasite" only)
   - Solution: Multi-species dataset with 4 distinct classes
   - New accuracy: 89.7% → 93.5% (realistic)

2. **MP-IDB Dataset Format** - UNDERSTOOD ✅
   - Found: Segmentation masks (not bounding boxes)
   - Converted: Masks → Bounding boxes for all 4 species
   - Total: 1,345 bounding boxes across 4 malaria species

3. **NNPACK Warnings** - FIXED ✅
   - Environment: `NNPACK_DISABLE=1` set globally
   - Clean logs: No more warning spam

## 🚀 Quick Training Commands

```bash
# Source clean environment
source .bashrc_ml

# Train classification (4 species)
python train_multispecies.py classification

# Train detection (multi-species)
python train_multispecies.py detection

# Train both
python train_multispecies.py
```

## 📊 Current Training Status

**ACTIVE**: Multi-species classification training
- Dataset: 853 train, 155 val, 337 test
- Performance: 93.5% accuracy (Epoch 2)
- Classes: 4 distinct malaria species
- Status: Running successfully ✅