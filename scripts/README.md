# Scripts Directory

## Overview

This directory contains all Python scripts organized by function. Total: **50 scripts** across 7 categories.

## Directory Structure

```
scripts/
├── analysis/          # Performance analysis and statistics (6 scripts)
├── data_setup/        # Dataset preparation pipeline (12 scripts)
├── monitoring/        # Experiment tracking (2 scripts)
├── pipeline/          # Visualization orchestration (3 scripts)
├── publication/       # Paper output generation (8 scripts)
├── training/          # Model training (3 scripts)
├── visualization/     # Figure generation (16 scripts)
└── utils/             # Shared utilities (empty - use root utils/)
```

## Entry Points

### Primary Entry Point
| Script | Purpose | When to Use |
|--------|---------|-------------|
| `../main_pipeline.py` | Full training pipeline | Training YOLO + Classification models |

### Secondary Entry Points
| Script | Purpose | When to Use |
|--------|---------|-------------|
| `publication/generate_all_publication_outputs.py` | Generate all paper outputs | After experiments complete |
| `pipeline/generate_all_visualizations.py` | Batch visualization | Generate all figures at once |
| `data_setup/01_download_datasets.py` | Start data pipeline | Setting up new environment |

---

## Scripts by Category

### 1. Data Setup (`data_setup/`) - 12 scripts

Sequential pipeline for dataset preparation:

```
01_download_datasets.py     → Download raw data
02_preprocess_data.py       → Basic preprocessing
03_integrate_datasets.py    → Merge datasets
04_convert_to_yolo.py       → Convert to YOLO format
05_augment_data.py          → Apply augmentations
06_split_dataset.py         → Train/val/test split
```

**Dataset-specific setup (run ONE per dataset):**
```
07_setup_kaggle_species_for_pipeline.py  → MP-IDB Species
08_setup_lifecycle_for_pipeline.py       → IML Lifecycle
09_setup_kaggle_stage_for_pipeline.py    → MP-IDB Stages
10_setup_md2019_stage_for_pipeline.py    → MD-2019 Stages
```

**Utilities:**
```
generate_high_quality_crops.py  → High-res crop generation
upscale_existing_crops.py       → Upscale existing crops
```

**Typical Usage:**
```bash
# New PC: Setup lifecycle dataset with 60/20/20 split
python scripts/data_setup/08_setup_lifecycle_for_pipeline.py \
    --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20
```

---

### 2. Training (`training/`) - 3 scripts

| Script | Purpose |
|--------|---------|
| `12_train_pytorch_classification.py` | Main classification training (DenseNet, EfficientNet, ResNet) |
| `advanced_losses.py` | Loss functions (Focal Loss, Class-Balanced) |
| `generate_ground_truth_crops.py` | Generate classification crops from annotations |

**Typical Usage:**
```bash
# Generate crops for a dataset
python scripts/training/generate_ground_truth_crops.py \
    --dataset data/processed/lifecycle \
    --output data/ground_truth_crops_224/lifecycle \
    --type iml_lifecycle \
    --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20
```

---

### 3. Analysis (`analysis/`) - 6 scripts

| Script | Purpose |
|--------|---------|
| `analyze_md2019_dataset.py` | MD-2019 dataset analysis |
| `compare_models_performance.py` | Cross-model comparison |
| `dataset_statistics_analyzer.py` | General dataset statistics |
| `generate_comprehensive_consolidated_analysis.py` | Multi-dataset analysis |
| `generate_table2_from_experiment.py` | Generate Table 2 for paper |
| `verify_md2019_bbox_source.py` | Validate MD-2019 annotations |

---

### 4. Visualization (`visualization/`) - 16 scripts

**Detection Visualizations:**
| Script | Purpose |
|--------|---------|
| `generate_detection_only.py` | Predicted detections (green boxes) |
| `generate_detection_only_with_metadata.py` | Detections + metadata |
| `generate_gt_detection.py` | Ground truth detections |

**Classification Visualizations:**
| Script | Purpose |
|--------|---------|
| `generate_classification_only.py` | Predicted classifications |
| `generate_classification_only_with_metadata.py` | Classifications + metadata |
| `generate_gt_classification.py` | Ground truth classifications |

**Combined Visualizations:**
| Script | Purpose |
|--------|---------|
| `generate_detection_classification_figures.py` | Detection + Classification combined |
| `generate_all_detection_classification_figures.py` | Batch all visualizations |

**Specialized:**
| Script | Purpose |
|--------|---------|
| `generate_improved_gradcam.py` | Grad-CAM attention maps |
| `generate_pipeline_architecture_diagram.py` | Pipeline diagram |
| `generate_professional_training_curves_final.py` | Training curves |
| `generate_combined_4datasets_augmentation.py` | Augmentation comparison |
| `generate_compact_augmentation_figures.py` | Compact augmentation figures |

**Wrappers:**
| Script | Purpose |
|--------|---------|
| `run_detection_classification_on_experiment.py` | Run on experiment folder |
| `run_improved_gradcam_on_experiments.py` | Run Grad-CAM on experiments |

---

### 5. Publication (`publication/`) - 8 scripts

| Script | Purpose |
|--------|---------|
| `generate_all_publication_outputs.py` | **MAIN** - Generate all outputs |
| `export_tables_to_luaran.py` | Export tables to luaran/ |
| `copy_images_with_errors.py` | Copy error analysis images |
| `copy_selected_images_3x2_layout.py` | 3x2 grid layout |
| `copy_selected_images_to_paper_folder.py` | Copy to paper folder |
| `enhance_pipeline_figure.py` | Enhance pipeline diagram |
| `generate_gt_vs_pred_comparison.py` | GT vs Prediction comparison |
| `verify_publication_data.py` | Validate publication data |

**Typical Usage:**
```bash
# Generate all publication outputs
python scripts/publication/generate_all_publication_outputs.py
```

---

### 6. Pipeline (`pipeline/`) - 3 scripts

| Script | Purpose |
|--------|---------|
| `generate_all_visualizations.py` | Orchestrate all visualizations |
| `generate_separate_model_visualizations.py` | Per-model visualizations |
| `generate_visualizations_with_metadata.py` | Visualizations with metadata |

---

### 7. Monitoring (`monitoring/`) - 2 scripts

| Script | Purpose |
|--------|---------|
| `experiment_manager.py` | Manage experiment lifecycle |
| `training_status.py` | Monitor training progress |

---

## Dependency Graph

```
main_pipeline.py
    ├── scripts/training/12_train_pytorch_classification.py
    ├── scripts/training/generate_ground_truth_crops.py
    ├── scripts/analysis/generate_comprehensive_consolidated_analysis.py
    └── scripts/visualization/generate_all_detection_classification_figures.py

publication/generate_all_publication_outputs.py
    ├── scripts/publication/export_tables_to_luaran.py
    ├── scripts/visualization/generate_professional_training_curves_final.py
    └── scripts/visualization/generate_improved_gradcam.py
```

---

## Shared Utilities

Common functions are in `../utils/`:

| Module | Functions |
|--------|-----------|
| `annotation_utils.py` | load_json_annotations, load_yolo_annotations, convert_annotations_format |
| `image_utils.py` | draw_boxes, yolo_to_absolute, resize_image, create_image_grid |
| `results_manager.py` | ResultsManager class for experiment paths |
| `download_utils.py` | Dataset download functions |

**Usage:**
```python
from utils.image_utils import draw_boxes, yolo_to_absolute
from utils.annotation_utils import load_yolo_annotations
```

---

## Quick Reference

**Setup new dataset:**
```bash
python scripts/data_setup/08_setup_lifecycle_for_pipeline.py --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20
```

**Generate crops:**
```bash
python scripts/training/generate_ground_truth_crops.py --dataset data/processed/lifecycle --output data/ground_truth_crops_224/lifecycle --type iml_lifecycle
```

**Run full pipeline:**
```bash
python main_pipeline.py --dataset iml_lifecycle --include yolo11 --classification-models densenet121
```

**Generate publication outputs:**
```bash
python scripts/publication/generate_all_publication_outputs.py
```

---

*Last Updated: 2025-12-07*
