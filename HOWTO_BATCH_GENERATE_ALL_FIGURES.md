# 🚀 Batch Generate All Detection + Classification Figures

**Script:** `scripts/visualization/generate_all_detection_classification_figures.py`

## 🎯 Purpose

Automatically generate detection + classification figures for **ALL datasets** and **ALL model combinations** in an Option A experiment folder.

This script is perfect when you want to generate figures for:
- Multiple datasets (mp_idb_species, mp_idb_stages, iml_lifecycle)
- Multiple detection models (YOLO10, YOLO11, YOLO12)
- Multiple classification models (all 12 variants)

## 📋 Prerequisites

You need an existing Option A experiment folder with trained models:
```
results/optA_YYYYMMDD_HHMMSS/
├── experiments/
│   ├── experiment_mp_idb_species/
│   │   ├── det_yolo10/
│   │   ├── det_yolo11/
│   │   ├── det_yolo12/
│   │   ├── cls_densen_ce/
│   │   ├── cls_densen_focal/
│   │   ├── ... (10 more classification models)
│   │   └── crops_gt_crops/
│   ├── experiment_mp_idb_stages/
│   └── experiment_iml_lifecycle/
└── consolidated_analysis/
```

---

## 🚀 Usage

### **1. Generate ALL Figures (All Datasets, All Models)**

```bash
conda activate malaria

# Process ALL test images with ALL model combinations
python scripts/visualization/generate_all_detection_classification_figures.py \
  --parent-folder results/optA_20251001_183508
```

**Result:**
- Processes: 2 datasets × 3 detection models × 12 classification models = 72 combinations
- Output structure:
```
paper_figures_all/
├── mp_idb_species/
│   ├── yolo10_densen_ce/
│   │   ├── gt_detection/
│   │   ├── pred_detection/
│   │   ├── gt_classification/
│   │   └── pred_classification/
│   ├── yolo10_densen_focal/
│   ├── ... (36 combinations)
└── mp_idb_stages/
    ├── yolo10_densen_ce/
    └── ... (36 combinations)
```

---

### **2. Generate for Specific Detection Model**

```bash
# Only YOLO11
python scripts/visualization/generate_all_detection_classification_figures.py \
  --parent-folder results/optA_20251001_183508 \
  --detection-models yolo11
```

**Result:** 2 datasets × 1 detection × 12 classification = 24 combinations

---

### **3. Generate for Specific Classification Models**

```bash
# Only DenseNet with both loss functions
python scripts/visualization/generate_all_detection_classification_figures.py \
  --parent-folder results/optA_20251001_183508 \
  --classification-models densen_ce densen_focal
```

**Result:** 2 datasets × 3 detection × 2 classification = 12 combinations

---

### **4. Generate for Single Dataset**

```bash
# Only mp_idb_species
python scripts/visualization/generate_all_detection_classification_figures.py \
  --parent-folder results/optA_20251001_183508 \
  --datasets mp_idb_species
```

---

### **5. Quick Test (Limited Images)**

```bash
# Process only first 5 images per combination
python scripts/visualization/generate_all_detection_classification_figures.py \
  --parent-folder results/optA_20251001_183508 \
  --max-images 5 \
  --detection-models yolo11 \
  --classification-models densen_ce
```

**Perfect for:** Testing before running full generation

---

### **6. Specific Combination**

```bash
# YOLO11 + DenseNet CE only
python scripts/visualization/generate_all_detection_classification_figures.py \
  --parent-folder results/optA_20251001_183508 \
  --detection-models yolo11 \
  --classification-models densen_ce \
  --output-base paper_figures_best_model
```

**Result:** 2 datasets × 1 detection × 1 classification = 2 combinations (fastest!)

---

## 📋 Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--parent-folder` | ✅ Yes | - | Option A experiment folder path |
| `--output-base` | No | `paper_figures_all` | Base output directory |
| `--detection-models` | No | `['all']` | Detection models: `yolo10`, `yolo11`, `yolo12`, or `all` |
| `--classification-models` | No | `['all']` | Classification models: specific names or `all` |
| `--datasets` | No | `['all']` | Datasets to process: dataset names or `all` |
| `--max-images` | No | `None` | Max images per combination (None = all) |
| `--det-conf-threshold` | No | `0.25` | Detection confidence threshold |

---

## 🎯 Classification Model Names

Use these names for `--classification-models`:

| Model Name | Description |
|------------|-------------|
| `densen_ce` | DenseNet121 + Cross-Entropy |
| `densen_focal` | DenseNet121 + Focal Loss |
| `efficientnet_b1_ce` | EfficientNet-B1 + Cross-Entropy |
| `efficientnet_b1_focal` | EfficientNet-B1 + Focal Loss |
| `efficientnet_b2_ce` | EfficientNet-B2 + Cross-Entropy |
| `efficientnet_b2_focal` | EfficientNet-B2 + Focal Loss |
| `convne_ce` | ConvNeXt-Tiny + Cross-Entropy |
| `convne_focal` | ConvNeXt-Tiny + Focal Loss |
| `mobile_ce` | MobileNet-V3-Large + Cross-Entropy |
| `mobile_focal` | MobileNet-V3-Large + Focal Loss |
| `resnet_ce` | ResNet101 + Cross-Entropy |
| `resnet_focal` | ResNet101 + Focal Loss |

---

## 💡 Recommended Workflows

### **For Paper - Best Model Comparison**

```bash
# Step 1: Generate for best performing models only
python scripts/visualization/generate_all_detection_classification_figures.py \
  --parent-folder results/optA_20251001_183508 \
  --detection-models yolo11 \
  --classification-models densen_ce efficientnet_b1_focal \
  --output-base paper_figures_best_models

# Step 2: Review outputs
ls paper_figures_best_models/*/yolo11_*/pred_classification/

# Step 3: Select 2-3 best representative images per dataset
```

---

### **For Supplementary - Full Model Comparison**

```bash
# Generate ALL combinations for comprehensive analysis
python scripts/visualization/generate_all_detection_classification_figures.py \
  --parent-folder results/optA_20251001_183508 \
  --output-base paper_figures_supplementary
```

**Note:** This will generate ~72 combinations. Takes ~2-4 hours for full test sets.

---

### **Quick Sanity Check**

```bash
# Test with 3 images before full run
python scripts/visualization/generate_all_detection_classification_figures.py \
  --parent-folder results/optA_20251001_183508 \
  --max-images 3 \
  --detection-models yolo11 \
  --classification-models densen_ce \
  --output-base test_figures
```

---

## 📊 Output Structure

```
<output-base>/
├── mp_idb_species/                          # Dataset 1
│   ├── yolo10_densen_ce/                    # Detection + Classification combo
│   │   ├── gt_detection/                    # 42 images (blue boxes + "parasite")
│   │   ├── pred_detection/                  # 42 images (green boxes + confidence)
│   │   ├── gt_classification/               # 42 images (blue boxes + class labels)
│   │   └── pred_classification/             # 42 images (green/red + predictions)
│   ├── yolo10_densen_focal/
│   ├── yolo10_efficientnet_b1_ce/
│   └── ... (more combinations)
├── mp_idb_stages/                           # Dataset 2
│   ├── yolo10_densen_ce/
│   └── ... (same structure)
└── iml_lifecycle/                           # Dataset 3 (if available)
    └── ...
```

**Each combination generates 4 × N images** (where N = number of test images)

---

## 📈 Estimated Time & Storage

| Scope | Combinations | Images Generated | Time (Est.) | Storage (Est.) |
|-------|-------------|------------------|-------------|----------------|
| **Quick test** (3 imgs) | 2 | 24 | ~2 min | ~50 MB |
| **Best models** (full) | 4 | ~672 | ~30 min | ~2 GB |
| **ALL models** (full) | 72 | ~12,096 | ~4 hours | ~30 GB |

**Assumptions:**
- 2 datasets × ~42 test images each
- Using all 3 YOLO models and 12 classification models

---

## 🔧 Troubleshooting

### **Issue:** No experiments found

**Solution:**
```bash
# Check parent folder structure
ls results/optA_20251001_183508/experiments/

# Should show: experiment_mp_idb_species/, experiment_mp_idb_stages/, etc.
```

---

### **Issue:** No models selected after filtering

**Cause:** Model name filter too strict

**Solution:**
```bash
# Use partial names (script uses "contains" matching)
--classification-models densen      # Matches both densen_ce and densen_focal
--classification-models efficientnet  # Matches all efficientnet variants
```

---

### **Issue:** Out of disk space

**Solution:**
```bash
# Process in batches by dataset
python scripts/visualization/generate_all_detection_classification_figures.py \
  --parent-folder results/optA_20251001_183508 \
  --datasets mp_idb_species  # Process one dataset at a time

# Clean up after reviewing
rm -rf paper_figures_all/mp_idb_species/yolo10_*  # Keep only best models
```

---

### **Issue:** Process takes too long

**Solution:**
```bash
# Reduce scope
--max-images 10                  # Limit images
--detection-models yolo11         # Use best detection model only
--classification-models densen_ce  # Use best classification model only
```

---

## ✅ Quality Checklist

Before full generation:
- [ ] Confirmed parent folder path is correct
- [ ] Tested with `--max-images 3` first
- [ ] Checked available disk space (30+ GB for full run)
- [ ] Decided which model combinations to generate
- [ ] Set appropriate `--output-base` folder

After generation:
- [ ] Verified all 4 output types generated
- [ ] Checked labels are visible and correct
- [ ] Selected best 2-3 images per dataset
- [ ] Documented which model combinations were used

---

## 🎯 Summary

**Use this script when you need:**
- ✅ Figures for multiple datasets
- ✅ Comparison across multiple models
- ✅ Batch processing to save time
- ✅ Organized output by dataset and model

**Don't use this script if:**
- ❌ You only need 1 specific combination → Use `generate_detection_classification_figures.py` directly
- ❌ You want custom processing logic → Write custom script

---

## 📝 Example Commands

```bash
# 1. Best for paper (quick, focused)
python scripts/visualization/generate_all_detection_classification_figures.py \
  --parent-folder results/optA_20251001_183508 \
  --detection-models yolo11 \
  --classification-models densen_ce \
  --output-base paper_figures_main

# 2. Model comparison (moderate)
python scripts/visualization/generate_all_detection_classification_figures.py \
  --parent-folder results/optA_20251001_183508 \
  --detection-models yolo11 \
  --classification-models densen_ce densen_focal efficientnet_b1_ce \
  --output-base paper_figures_comparison

# 3. Full analysis (comprehensive, slow)
python scripts/visualization/generate_all_detection_classification_figures.py \
  --parent-folder results/optA_20251001_183508 \
  --output-base paper_figures_all_models
```

---

**Created:** 2025-10-02
**Script:** `scripts/visualization/generate_all_detection_classification_figures.py`
**Related:** `HOWTO_GENERATE_FIGURES_DETECTION_CLASSIFICATION.md` (single combination usage)
