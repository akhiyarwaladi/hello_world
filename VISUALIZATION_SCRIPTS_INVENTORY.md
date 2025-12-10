# 📊 VISUALIZATION SCRIPTS INVENTORY
**Complete list of ALL visualization scripts in the project**
**Date**: 2025-12-10
**Status**: Verified & Categorized

---

## ✅ **ACTIVE SCRIPTS** (Currently Used)

### **Main Entry Point**
- `scripts/visualization/generate_all_centralized.py` ⭐ **MAIN SCRIPT**
  - ONE centralized generator for ALL visualizations
  - Features:
    - ✅ Publication-quality confusion matrices (NO rotation, dataset colors)
    - ✅ Complete training curves (accuracy + loss)
    - ✅ Selected error cases
    - ✅ Metadata reports
  - Output: `visualization_outputs/` (ONE folder for everything)

### **Core Generators** (Called by main script)
- `scripts/visualization/regenerate_publication_quality_confusion_matrices.py` ✅ **NEW**
  - **NO 45-degree rotation** (rotation=0)
  - **Dataset-specific colormaps**: Blues, Greens, Purples, Oranges
  - **Dual annotations**: count + percentage
  - **High DPI**: 400

- `scripts/visualization/generate_training_curves.py` ✅ **FIXED**
  - Accuracy + Loss curves
  - **NO bbox_inches='tight'** (consistent sizing)
  - Color-blind friendly
  - 400 DPI publication quality

- `scripts/visualization/generate_consolidated_confusion_matrices.py`
  - Generates 2x2 grid (best models)
  - Loads existing individual CMs and combines them

### **Test Image Visualizations**
- `scripts/visualization/generate_detection_only_with_metadata.py`
  - GT detection boxes + Predicted detection boxes
  - Generates metadata CSVs (TP/FP/FN counts, confidence scores)

- `scripts/visualization/generate_classification_only_with_metadata.py`
  - GT classification + Predicted classification
  - Generates metadata CSVs (accuracy, confidence, per-class metrics)

- `scripts/visualization/generate_gt_detection.py`
  - Ground truth detection boxes only (for paper)

- `scripts/visualization/generate_gt_classification.py`
  - Ground truth classification labels only (for paper)

### **Special Features**
- `scripts/visualization/generate_improved_gradcam.py`
  - GradCAM visualization for model interpretability
  - Separate feature (not part of main pipeline)

- `scripts/visualization/generate_combined_4datasets_augmentation.py`
  - Data augmentation comparison figure (4 datasets)

- `scripts/visualization/generate_compact_augmentation_figures.py`
  - Compact augmentation visualization

- `scripts/visualization/generate_pipeline_architecture_diagram.py`
  - Architecture diagram for paper (Figure 2)

### **Support Modules**
- `scripts/visualization/selectors/` - Error case selection logic
  - `base_selector.py` - Abstract base class
  - `detection_error_selector.py` - FP, FN, Mixed, Perfect cases
  - `classification_error_selector.py` - All Wrong, High Conf Error, Mixed, etc.

- `scripts/visualization/reporters/` - Output formatters
  - `base_reporter.py` - Abstract base
  - `csv_reporter.py` - CSV export
  - `markdown_reporter.py` - Human-readable reports
  - `json_reporter.py` - Machine-readable metadata

---

## 🗄️ **ARCHIVED SCRIPTS** (Obsolete/Superseded)

### **Old Confusion Matrix Generators** (Superseded)
- `scripts/visualization/regenerate_beautiful_confusion_matrices.py` ❌ **OBSOLETE**
  - Old version WITH 45-degree rotation
  - Replaced by: `regenerate_publication_quality_confusion_matrices.py`

- `scripts/visualization/beautify_existing_confusion_matrices.py` ❌ **OBSOLETE**
  - Failed approach (tried to extract CM from images)
  - Doesn't work for imbalanced test splits

### **Archived on 2025-12-10**
Location: `archive/scripts_visualization_old_20251210/`

- `consolidate_all_outputs.py` ❌
  - Old consolidation approach (manual copying)
  - Replaced by: `generate_all_centralized.py`

- `generate_all_test_visualizations.py` ❌
  - Old orchestrator
  - Replaced by: `generate_all_centralized.py`

- `generate_all_detection_classification_figures.py` ❌
  - Old combined generator
  - Replaced by: separate `generate_*_with_metadata.py`

- `generate_detection_only.py` ❌
  - No metadata version
  - Replaced by: `generate_detection_only_with_metadata.py`

- `generate_classification_only.py` ❌
  - No metadata version
  - Replaced by: `generate_classification_only_with_metadata.py`

- `generate_detection_classification_figures.py` ❌
  - Old version without metadata
  - Replaced by: modular approach

- `generate_professional_training_curves_final.py` ❌
  - Old training curves (hardcoded paths)
  - Replaced by: `generate_training_curves.py` (modular)

- `run_detection_classification_on_experiment.py` ❌
  - One-off runner script
  - Not needed (use main pipeline)

- `run_improved_gradcam_on_experiments.py` ❌
  - One-off runner
  - Use: `generate_improved_gradcam.py` directly

### **Other Archived Visualization Scripts**
Location: `archive/scripts/visualization/`

- `generate_augmentation_no_title.py`
- `generate_gradcam.py` (old version)
- `generate_high_quality_augmentation_figure.py`
- `visualize_augmentation.py`
- `augmentation_generators_old/` (entire folder)

---

## 📁 **PUBLICATION SCRIPTS** (For Paper/Laporan)
Location: `scripts/publication/`

- `generate_all_publication_outputs.py` ⭐
  - Generates ALL publication tables and figures
  - Outputs to: `luaran/auto_generated/`

- `copy_images_with_errors.py`
  - Copies selected error images to paper folder

- `copy_selected_images_3x2_layout.py`
  - Creates figure layouts for paper

- `copy_selected_images_to_paper_folder.py`
  - Organizes images for publication

- `enhance_pipeline_figure.py`
  - Post-processes pipeline architecture diagram

- `export_tables_to_luaran.py`
  - Exports Excel tables to luaran folder

- `generate_gt_vs_pred_comparison.py`
  - Side-by-side GT vs Prediction comparison

- `verify_publication_data.py`
  - Validates publication data integrity

---

## 🔄 **PIPELINE INTEGRATION SCRIPTS**
Location: `scripts/pipeline/`

- `generate_all_visualizations.py`
  - Part of main training pipeline
  - Called after training completes

- `generate_separate_model_visualizations.py`
  - Generates visualizations for each model separately

- `generate_visualizations_with_metadata.py`
  - Metadata-aware visualization generation

---

## 📝 **TEMPLATE & UTILITY SCRIPTS**
Location: `archive/scripts/temp_utilities/`

- `select_qualitative_images.py`
  - Selects best images for qualitative analysis
  - Generated: `luaran/templates/figures/qualitative_classification/`

- `select_classification_visualizations.py`
  - Selects classification visualization examples

- `create_detailed_metadata.py`
  - Creates detailed metadata for images

- `create_image_metadata.py`
  - Image-level metadata generation

---

## 🎯 **CURRENT WORKFLOW**

### **1. Main Visualization Generation**
```bash
# ONE command for EVERYTHING
python scripts/visualization/generate_all_centralized.py
```

**Generates:**
- ✅ 24 publication-quality confusion matrices (NO rotation, dataset colors)
- ✅ 8 training curves (4 accuracy + 4 loss, consistent sizing)
- ✅ Top 20 detection + 20 classification error cases
- ✅ Metadata CSVs + Reports
- ✅ README guide

**Output location:** `visualization_outputs/` (ONE centralized folder)

### **2. Publication Outputs**
```bash
# Generate all publication tables and figures
python scripts/publication/generate_all_publication_outputs.py
```

**Output location:** `luaran/auto_generated/`

---

## ✅ **KEY FEATURES IMPLEMENTED**

### **Confusion Matrices**
- ✅ **NO 45-degree rotation** (rotation=0 for all axes)
- ✅ **Dataset-specific colormaps**:
  - IML Lifecycle: **Blues**
  - MP-IDB Species: **Greens**
  - MP-IDB Stages: **Purples**
  - MD-2019 Stages: **Oranges**
- ✅ **Dual annotations** (count + percentage)
- ✅ **High resolution** (400 DPI)
- ✅ **Large figure size** (12x10 inches for readability)

### **Training Curves**
- ✅ **Complete set** (accuracy + loss for all 4 datasets)
- ✅ **Consistent sizing** (NO bbox_inches='tight')
- ✅ **Color-blind friendly** palette
- ✅ **Professional styling** (400 DPI, journal-ready)

### **Test Visualizations**
- ✅ **Metadata-rich** (TP/FP/FN counts, confidence scores)
- ✅ **Error case selection** (paper score ranking)
- ✅ **Modular selectors** (easy to add new criteria)
- ✅ **Multiple output formats** (CSV, Markdown, JSON)

---

## 🗑️ **SCRIPTS TO DELETE** (If Confirmed Obsolete)

After verification, these can be deleted:
1. `scripts/visualization/regenerate_beautiful_confusion_matrices.py` (has 45° rotation)
2. `scripts/visualization/beautify_existing_confusion_matrices.py` (doesn't work)

Already archived (safe to keep in archive):
- `archive/scripts_visualization_old_20251210/*.py` (9 files)
- `archive/scripts/visualization/*.py` (old versions)

---

## 📊 **STATISTICS**

**Active scripts:** 14 core + 6 support modules = **20 total**
**Archived scripts:** 9 recent + ~10 old = **~19 total**
**Publication scripts:** 8
**Pipeline scripts:** 3
**Template/Utility scripts:** 4

**Total Python visualization scripts in project:** ~54

---

## 🎯 **RECOMMENDATIONS**

1. ✅ **Keep using:** `generate_all_centralized.py` as main entry point
2. ✅ **Keep:** `regenerate_publication_quality_confusion_matrices.py` (NO rotation)
3. ❌ **Delete:** `regenerate_beautiful_confusion_matrices.py` (old, has rotation)
4. ❌ **Delete:** `beautify_existing_confusion_matrices.py` (doesn't work)
5. ✅ **Keep archived:** All scripts in `archive/` for reference

---

**Last Updated:** 2025-12-10 01:35 WIB
**Status:** ✅ Complete inventory, all scripts categorized
**Main Script:** `scripts/visualization/generate_all_centralized.py`
**Confusion Matrix Generator:** `scripts/visualization/regenerate_publication_quality_confusion_matrices.py`
