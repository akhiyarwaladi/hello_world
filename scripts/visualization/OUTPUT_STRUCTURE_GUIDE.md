# 📁 **COMPLETE OUTPUT STRUCTURE GUIDE**

**Panduan lengkap lokasi semua output visualisasi - TERPUSAT & MUDAH**

---

## 🎯 **3 LOKASI UTAMA OUTPUT**

### **1. Per-Model Output (Hasil Training)**
📂 `results/optA_[timestamp]/experiments/experiment_[dataset]/`

```
experiment_iml_lifecycle/
├── cls_efficientnet_b1_focal/
│   ├── confusion_matrix.png          ← Confusion matrix model ini
│   ├── training_curves.png            ← Loss/accuracy curves
│   └── results.csv                    ← Numeric results
│
├── det_yolo11/
│   ├── confusion_matrix.png          ← Detection confusion matrix
│   └── confusion_matrix_normalized.png
│
└── visualizations/                     ← TEST IMAGE VISUALIZATIONS
    ├── pred_detection_yolo11/
    │   ├── PA171697.png              ← Image dengan bounding boxes
    │   ├── PA171698.png
    │   └── detection_metadata.csv    ← Metadata (TP, FP, FN, confidence)
    │
    └── pred_classification_efficientnet_b1_focal/
        ├── PA171697.png              ← Classification predictions
        └── classification_metadata_images.csv
```

**📌 ISI:** Individual model results, per-image visualizations, metadata

---

### **2. Consolidated Analysis (Selected Cases)**
📂 `results/optA_[timestamp]/visualization_summary/`

```
visualization_summary/
├── selected_detection_errors.csv      ← Best detection error cases (FP, FN, Mixed)
├── selected_classification_errors.csv ← Best classification error cases
├── selected_error_images.csv          ← Combined (format lama)
├── visualization_report.md            ← Human-readable summary
└── visualization_metadata.json        ← Machine-readable metadata
```

**📌 ISI:** Error case selection, best images untuk paper, analysis summary

---

### **3. Publication Figures (Journal-Ready)**
📂 `luaran/auto_generated/figures/`

```
luaran/auto_generated/figures/
├── performance/
│   └── confusion_matrices.png        ← 2x2 GRID CONSOLIDATED ⭐
│
└── training_curves/
    ├── accuracy_iml_lifecycle.png
    ├── accuracy_md_2019_stages.png
    ├── accuracy_mp_idb_species.png
    └── accuracy_mp_idb_stages.png
```

**📌 ISI:** Publication-ready figures (400 DPI, journal styling)

---

## 🚀 **ONE-COMMAND GENERATION**

### **Generate SEMUA Visualizations:**

```bash
# Full pipeline - generates everything
python scripts/visualization/generate_all_test_visualizations.py \
  --experiment-dir results/optA_20251207_233941

# Output:
# ✅ Selected error cases → visualization_summary/
# ✅ Training curves → luaran/auto_generated/figures/training_curves/
# ✅ Consolidated confusion matrices → luaran/auto_generated/figures/performance/
```

### **Generate Confusion Matrices Only:**

```bash
python scripts/visualization/generate_consolidated_confusion_matrices.py \
  --experiment-dir results/optA_20251207_233941/experiments

# Output:
# ✅ confusion_matrices.png (2x2 grid) → luaran/auto_generated/figures/performance/
```

### **Generate Training Curves Only:**

```bash
python scripts/visualization/generate_training_curves.py \
  --experiment-dir results/optA_20251207_233941/experiments \
  --output-dir luaran/auto_generated/figures/training_curves

# Output:
# ✅ 4 accuracy curves → training_curves/
```

---

## 📊 **IMAGE OUTPUT TYPES**

### **1. Test Image Visualizations (Bounding Boxes)**
- **Location:** `experiments/experiment_*/visualizations/pred_*/`
- **Count:** ~100-400 images per model per dataset
- **Format:** PNG with drawn bounding boxes
- **Colors:**
  - 🟢 Green = True Positive / Correct
  - 🔴 Red = False Positive / Wrong
  - 🟡 Yellow = False Negative / Missed

### **2. Confusion Matrices**

#### **Individual (Per-Model):**
- **Location:** `experiments/experiment_*/cls_*_focal/confusion_matrix.png`
- **Count:** 1 per classification model (~24 files total)
- **Format:** Heatmap dengan labels

#### **Consolidated (2x2 Grid):**
- **Location:** `luaran/auto_generated/figures/performance/confusion_matrices.png`
- **Count:** 1 file (4 best models dalam 1 figure)
- **Format:** Publication-quality 2x2 grid
- **Content:**
  - Top-left: Species best model
  - Top-right: Species second best
  - Bottom-left: Lifecycle best model
  - Bottom-right: Lifecycle second best

### **3. Training Curves**
- **Location:** `luaran/auto_generated/figures/training_curves/`
- **Count:** 4 files (1 per dataset)
- **Format:** Accuracy curves (best vs worst model comparison)
- **Style:** Journal-quality (400 DPI, color-blind friendly)

---

## 🎯 **QUICK ACCESS PATHS**

### **Untuk Paper/Laporan:**

```bash
# Confusion matrices (publication)
luaran/auto_generated/figures/performance/confusion_matrices.png

# Training curves (publication)
luaran/auto_generated/figures/training_curves/*.png

# Selected error cases (CSV for filtering)
results/optA_20251207_233941/visualization_summary/selected_*.csv

# Human-readable summary
results/optA_20251207_233941/visualization_summary/visualization_report.md
```

### **Untuk Analysis Mendalam:**

```bash
# Per-model confusion matrices
results/optA_*/experiments/experiment_*/cls_*_focal/confusion_matrix.png

# All test visualizations dengan bounding boxes
results/optA_*/experiments/experiment_*/visualizations/pred_*/

# Metadata lengkap (TP/FP/FN/confidence)
results/optA_*/experiments/experiment_*/visualizations/pred_*/detection_metadata.csv
results/optA_*/experiments/experiment_*/visualizations/pred_*/classification_metadata_images.csv
```

---

## ⚡ **SIMPLIFIED WORKFLOW**

### **Scenario 1: Generate untuk Paper**

```bash
# 1. Generate training curves
python scripts/visualization/generate_training_curves.py

# 2. Generate consolidated confusion matrices
python scripts/visualization/generate_consolidated_confusion_matrices.py

# 3. Generate selected error cases
python scripts/visualization/generate_all_test_visualizations.py
```

**Output:** All publication figures ready in `luaran/auto_generated/figures/`

### **Scenario 2: Analyze Specific Model**

```bash
# Go to model folder
cd results/optA_20251207_233941/experiments/experiment_iml_lifecycle

# Check confusion matrix
open cls_efficientnet_b1_focal/confusion_matrix.png

# Check test visualizations
open visualizations/pred_classification_efficientnet_b1_focal/PA171697.png

# Check metadata
open visualizations/pred_classification_efficientnet_b1_focal/classification_metadata_images.csv
```

### **Scenario 3: Select Images for Publication**

```bash
# 1. Generate error selection
python scripts/visualization/generate_all_test_visualizations.py

# 2. Open CSV in Excel
open results/optA_20251207_233941/visualization_summary/selected_detection_errors.csv
open results/optA_20251207_233941/visualization_summary/selected_classification_errors.csv

# 3. Sort by:
#    - paper_score (descending) - highest first
#    - error_category - group by type
#    - n_false_positives / n_false_negatives - by severity

# 4. Copy image paths from 'image_file' column
```

---

## 📋 **OUTPUT CHECKLIST**

### **After Full Pipeline Run:**

✅ **Per-Model Outputs** (in `experiments/`)
- [ ] Confusion matrices for all classification models (~24 files)
- [ ] Training curves for all models (~24 files)
- [ ] Test visualizations for all models (~5,000+ images)
- [ ] Metadata CSVs for all visualizations (~60 files)

✅ **Consolidated Analysis** (in `visualization_summary/`)
- [ ] selected_detection_errors.csv (~200-300 cases)
- [ ] selected_classification_errors.csv (~500-600 cases)
- [ ] visualization_report.md (human-readable)
- [ ] visualization_metadata.json (machine-readable)

✅ **Publication Figures** (in `luaran/auto_generated/figures/`)
- [ ] confusion_matrices.png (2x2 grid)
- [ ] training_curves/*.png (4 files)

---

## 🗂️ **FOLDER STRUCTURE SUMMARY**

```
📁 PROJECT ROOT
│
├── 📁 results/optA_[timestamp]/
│   └── 📁 experiments/                    ← PER-MODEL TRAINING OUTPUTS
│       ├── experiment_iml_lifecycle/
│       ├── experiment_md_2019_stages/
│       ├── experiment_mp_idb_species/
│       └── experiment_mp_idb_stages/
│
│   └── 📁 visualization_summary/          ← ERROR CASE ANALYSIS
│       ├── selected_*.csv
│       ├── visualization_report.md
│       └── visualization_metadata.json
│
└── 📁 luaran/auto_generated/figures/      ← PUBLICATION-READY FIGURES
    ├── performance/
    │   └── confusion_matrices.png        ⭐ 2x2 GRID
    └── training_curves/
        └── *.png                          ⭐ 4 CURVES
```

---

## 💡 **KEY POINTS**

1. **3 tiers output:**
   - Tier 1: Individual model results (detailed)
   - Tier 2: Selected error cases (curated)
   - Tier 3: Publication figures (polished)

2. **Confusion matrices:**
   - Individual: In each `cls_*_focal/` folder
   - Consolidated: In `luaran/auto_generated/figures/performance/`

3. **Test visualizations:**
   - All in `experiments/experiment_*/visualizations/`
   - Organized by model (`pred_detection_*`, `pred_classification_*`)
   - With metadata CSVs for filtering

4. **One-command generation:**
   - `generate_all_test_visualizations.py` - Complete pipeline
   - `generate_consolidated_confusion_matrices.py` - Confusion matrices only
   - `generate_training_curves.py` - Training curves only

5. **Images only (no LaTeX/PDF/web dashboard):**
   - All outputs are PNG images
   - Plus CSV/Markdown/JSON for metadata
   - No complex dependencies

---

**Last Updated:** 2025-12-10
**Status:** Complete & Tested ✅
