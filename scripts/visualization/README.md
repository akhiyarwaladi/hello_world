# Visualization Scripts for Malaria Detection Pipeline

This folder contains visualization scripts for generating publication-ready figures from the malaria detection and classification pipeline.

## 📁 Available Scripts

### 1. **Detection + Classification Visualization** ⭐ MAIN SCRIPT
**File:** `generate_detection_classification_figures.py`

**Purpose:** Generate 4 separate visualizations per test image showing detection and classification results.

**Key Features:**
- ✅ **4 output types per image:**
  - `gt_detection/` - Ground truth boxes + "parasite" labels (blue)
  - `pred_detection/` - Predicted detection boxes + confidence scores (green)
  - `gt_classification/` - Ground truth boxes + class labels (blue)
  - `pred_classification/` - GT boxes + predicted class labels (color-coded: green=correct, red=wrong)

- ✅ **Critical design:** `pred_classification` uses GT boxes (not detection predictions) to evaluate pure classification performance
- ✅ **Supports all 6 model architectures:** DenseNet121, EfficientNet-B1/B2, ConvNeXt-Tiny, MobileNet-V3-Large, ResNet101
- ✅ **Large, readable labels:** font_scale=1.5, thickness=4
- ✅ **Publication-ready:** 300 dpi output
- ✅ **Default: Process ALL test images**

**Usage:**
```bash
python scripts/visualization/generate_detection_classification_figures.py \
  --detection-model results/optA_XXX/experiments/experiment_YYY/det_yolo11/weights/best.pt \
  --classification-model results/optA_XXX/experiments/experiment_YYY/cls_densen_ce/best.pt \
  --test-images data/processed/dataset/test/images \
  --test-labels data/processed/dataset/test/labels \
  --gt-crops results/optA_XXX/experiments/experiment_YYY/crops_gt_crops \
  --output paper_figures
```

**Documentation:** See `../../HOWTO_GENERATE_FIGURES_DETECTION_CLASSIFICATION.md`

---

### 2. **Augmentation Visualization**
**File:** `visualize_augmentation.py`

**Purpose:** Generate visualizations of data augmentation techniques for paper figures.

**Key Features:**
- ✅ Shows 6 augmentation types in compact comparison format
- ✅ **Edge replication** for rotation (no black borders)
- ✅ Medical-safe augmentation examples
- ✅ Publication-ready output (300 dpi)
- ✅ Comparison mode for paper + full examples for supplementary

**Usage:**
```bash
# Compact comparison for paper
python scripts/visualization/visualize_augmentation.py \
  --image data/ground_truth_crops_224/lifecycle/crops/test/gametocyte/PA171697_crop_000.jpg \
  --output paper_figures \
  --comparison-only

# Full examples
python scripts/visualization/visualize_augmentation.py \
  --image data/ground_truth_crops_224/lifecycle/crops/test/gametocyte/PA171697_crop_000.jpg \
  --output augmentation_examples
```

**Documentation:**
- `README_AUGMENTATION_VISUALIZATION.md` (detailed script docs)
- `../../HOWTO_GENERATE_AUGMENTATION_FIGURES.md` (usage guide)

---

## 🎯 Quick Start for Paper Figures

### Generate Detection + Classification Figures (RECOMMENDED)

```bash
# 1. Activate environment
conda activate malaria

# 2. Generate figures for one dataset (example: species)
python scripts/visualization/generate_detection_classification_figures.py \
  --detection-model results/optA_20251001_183508/experiments/experiment_mp_idb_species/det_yolo11/weights/best.pt \
  --classification-model results/optA_20251001_183508/experiments/experiment_mp_idb_species/cls_densen_ce/best.pt \
  --test-images data/processed/species/test/images \
  --test-labels data/processed/species/test/labels \
  --gt-crops results/optA_20251001_183508/experiments/experiment_mp_idb_species/crops_gt_crops \
  --output paper_figures/species

# 3. Review outputs and select best 2-3 representative images
ls paper_figures/species/pred_classification/

# 4. For augmentation figure (optional)
python scripts/visualization/visualize_augmentation.py \
  --image data/ground_truth_crops_224/lifecycle/crops/test/gametocyte/PA171697_crop_000.jpg \
  --output paper_figures \
  --comparison-only
```

---

## 📊 Understanding Output Colors

### Detection + Classification Script:
- **Blue boxes:** Ground truth annotations
- **Green boxes:** Correct predictions / Model detections
- **Red boxes:** Incorrect predictions
- **Labels format:**
  - Detection: `parasite 0.95` (confidence)
  - Classification: `class_name conf ✓` (correct) or `pred_class conf (GT:true_class)` (wrong)

---

## 🔧 Requirements

```bash
# Core dependencies
pip install torch torchvision ultralytics opencv-python pillow numpy matplotlib pandas

# Or use project environment
conda activate malaria  # All dependencies already installed
```

---

## 📝 File Structure

```
scripts/visualization/
├── README.md                                    # This file
├── generate_detection_classification_figures.py # Main visualization script ⭐
├── visualize_augmentation.py                    # Augmentation visualization
└── README_AUGMENTATION_VISUALIZATION.md         # Augmentation script docs
```

---

## 💡 Tips for Paper Figures

1. **Main Figure:** Use `gt_classification/` (top) + `pred_classification/` (bottom) side by side
2. **Supplementary:** Use `gt_detection/` + `pred_detection/` to show detection performance
3. **Select 2-3 images:** Best case (all green), average case (mixed), challenging case (errors)
4. **Caption tips:**
   - Explain color coding (blue=GT, green=correct, red=wrong)
   - Include performance metrics (mAP, accuracy)
   - Note that pred_classification uses GT boxes to isolate classification performance

---

## 🎯 Design Rationale

### Why pred_classification uses GT boxes?

Classification models in this pipeline are trained on crops from **ground truth annotations**, not on crops from detection predictions. To evaluate **pure classification performance** without mixing in detection errors:

- ✅ Use GT boxes for classification evaluation
- ✅ Color-code correctness (green/red)
- ✅ Show predicted vs actual labels

This design ensures:
1. **Fair evaluation:** Classification is evaluated on same data distribution it was trained on
2. **Clear separation:** Detection errors don't contaminate classification metrics
3. **Interpretability:** Easy to see which species/stages are confused

---

## ✅ Quality Checklist

Before submitting paper:
- [ ] Generated all 4 visualization types
- [ ] Labels are large and readable
- [ ] Colors are correct (blue=GT, green=correct, red=wrong)
- [ ] Selected 2-3 representative images
- [ ] Figure captions explain methodology
- [ ] Augmentation figure included (optional)
- [ ] Resolution is 300 dpi or higher

---

**Last Updated:** 2025-10-02
**Maintainer:** Malaria Detection Pipeline Team
