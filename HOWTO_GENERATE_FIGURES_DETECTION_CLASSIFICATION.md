# 📊 Detection + Classification Visualization for Paper Figures

**Script:** `scripts/visualization/generate_detection_classification_figures.py`

## 🎯 Overview

Generates **4 separate visualizations** for each test image to clearly show detection and classification results for publication.

### **Critical Design Decision:**
**`pred_classification` uses GROUND TRUTH boxes**, not predicted detection boxes. This evaluates **pure classification performance** without mixing in detection errors, matching the training methodology where classification models were trained on GT crops.

---

## 📁 Output Structure

```
paper_figures/
├── gt_detection/          # Ground truth detection annotations
│   └── image001.png       # Blue boxes + "parasite" label
├── pred_detection/        # Predicted detection results
│   └── image001.png       # Green boxes + confidence scores
├── gt_classification/     # Ground truth classification
│   └── image001.png       # Blue boxes + class labels (ring, schizont, etc.)
└── pred_classification/   # Predicted classification on GT boxes
    └── image001.png       # Green (correct) / Red (wrong) + predicted labels
```

---

## 🎨 Visualization Details

### 1. **GT Detection** (`gt_detection/`)
- **Purpose:** Show ground truth parasite locations
- **Boxes:** Ground truth annotations
- **Color:** Blue
- **Labels:** "parasite" (generic detection label)
- **Use in paper:** Supplementary - show detection ground truth

### 2. **Predicted Detection** (`pred_detection/`)
- **Purpose:** Show YOLO detection results
- **Boxes:** Model predictions
- **Color:** Green
- **Labels:** "parasite" + confidence score (e.g., "parasite 0.95")
- **Use in paper:** Supplementary - show detection performance

### 3. **GT Classification** (`gt_classification/`)
- **Purpose:** Show ground truth parasite species/stages
- **Boxes:** Ground truth annotations
- **Color:** Blue
- **Labels:** Class names (e.g., "P_falciparum", "ring", "schizont")
- **Use in paper:** Main figure - show what model should predict

### 4. **Predicted Classification** (`pred_classification/`)
- **Purpose:** Evaluate pure classification performance
- **Boxes:** Ground truth annotations (NOT detection predictions!)
- **Color:**
  - 🟢 Green = Correct prediction
  - 🔴 Red = Incorrect prediction
- **Labels:**
  - Correct: "predicted_class confidence ✓"
  - Wrong: "predicted_class confidence (GT:true_class)"
- **Use in paper:** Main figure - show classification results

**Why GT boxes?** Classification models were trained on crops from GT boxes. Using predicted detection boxes would mix detection errors with classification errors, making it impossible to isolate classification performance.

---

## 🚀 Usage

### **Process ALL Test Images (Default)**

```bash
conda activate malaria

python scripts/visualization/generate_detection_classification_figures.py \
  --detection-model results/optA_20251001_183508/experiments/experiment_mp_idb_species/det_yolo11/weights/best.pt \
  --classification-model results/optA_20251001_183508/experiments/experiment_mp_idb_species/cls_densen_ce/best.pt \
  --test-images data/processed/species/test/images \
  --test-labels data/processed/species/test/labels \
  --gt-crops results/optA_20251001_183508/experiments/experiment_mp_idb_species/crops_gt_crops \
  --output paper_figures
```

**Result:** Processes ALL test images (e.g., 42 images × 4 outputs = 168 files)

---

### **Process Limited Images**

```bash
# Process first 5 images only
python scripts/visualization/generate_detection_classification_figures.py \
  --detection-model path/to/det_model.pt \
  --classification-model path/to/cls_model.pt \
  --test-images data/processed/species/test/images \
  --test-labels data/processed/species/test/labels \
  --gt-crops path/to/crops_gt_crops \
  --output paper_figures \
  --max-images 5
```

---

### **Adjust Detection Confidence**

```bash
# Lower confidence threshold (for models with fewer training epochs)
python scripts/visualization/generate_detection_classification_figures.py \
  --detection-model path/to/det_model.pt \
  --classification-model path/to/cls_model.pt \
  --test-images data/processed/species/test/images \
  --test-labels data/processed/species/test/labels \
  --gt-crops path/to/crops_gt_crops \
  --output paper_figures \
  --det-conf-threshold 0.1
```

---

## 📋 Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--detection-model` | ✅ Yes | - | Path to YOLO detection model (`best.pt`) |
| `--classification-model` | ✅ Yes | - | Path to classification model (`best.pt`) |
| `--test-images` | ✅ Yes | - | Directory containing test images |
| `--test-labels` | ✅ Yes | - | Directory containing YOLO format labels |
| `--gt-crops` | ✅ Yes | - | Path to ground truth crops directory |
| `--output` | No | `paper_figures` | Output base directory |
| `--det-conf-threshold` | No | `0.25` | Detection confidence threshold |
| `--max-images` | No | `None` | Maximum images to process (None = all) |

---

## 🤖 Supported Classification Models

✅ **All 6 architectures supported:**
- DenseNet121
- EfficientNet-B1
- EfficientNet-B2
- ConvNeXt-Tiny
- MobileNet-V3-Large
- ResNet101

✅ **Both loss functions supported:**
- Cross-Entropy
- Focal Loss

**Total:** 12 model variants automatically detected and loaded.

---

## 💡 Workflow for Paper Figures

### **Step 1: Generate All Figures**
```bash
# Process all test images
python scripts/visualization/generate_detection_classification_figures.py \
  --detection-model path/to/best_detection_model.pt \
  --classification-model path/to/best_classification_model.pt \
  --test-images data/processed/dataset/test/images \
  --test-labels data/processed/dataset/test/labels \
  --gt-crops path/to/crops_gt_crops \
  --output paper_figures
```

### **Step 2: Review and Select**
Browse generated images and select **2-3 representative examples:**
- ✅ **Best case:** All predictions correct (green boxes)
- ⚠️ **Average case:** Mix of correct and incorrect
- ❌ **Challenging case:** Complex/crowded images with errors

### **Step 3: Create Paper Figures**

#### **Main Figure: Classification Results**
- Use `gt_classification/` (top row) to show ground truth
- Use `pred_classification/` (bottom row) to show predictions
- Caption should explain color coding (green=correct, red=wrong)

#### **Supplementary Figure: Detection Results**
- Use `gt_detection/` (left) to show ground truth
- Use `pred_detection/` (right) to show detection results
- Caption should note confidence scores on predictions

---

## 📝 Example Figure Captions

### **Main Figure Caption:**
```
Figure X: Malaria parasite classification performance on test images.
Top row: Ground truth annotations with species labels (blue boxes).
Bottom row: Model predictions using [model_name] (green = correct prediction,
red = incorrect prediction). Predicted labels show class name, confidence
score, and ground truth (for errors). The model achieves 88.3% classification
accuracy on test set.
```

### **Supplementary Figure Caption:**
```
Figure SX: Malaria parasite detection performance using YOLO11.
Left: Ground truth detection annotations (blue boxes).
Right: Model predictions with confidence scores (green boxes).
The model achieves 92.9% mAP@50 on the test set.
```

---

## 🔧 Troubleshooting

### **Issue:** No pred_classification labels showing

**Cause:** GT class mapping not found

**Solution:** Check that `ground_truth_crop_metadata.csv` exists in crops directory:
```bash
ls path/to/crops_gt_crops/ground_truth_crop_metadata.csv
```

---

### **Issue:** No predicted boxes in pred_detection

**Cause:** Detection confidence threshold too high or model undertrained

**Solution:** Lower confidence threshold:
```bash
--det-conf-threshold 0.1
```

---

### **Issue:** Classification model fails to load

**Cause:** Model architecture not recognized

**Solution:** Ensure model checkpoint has `model_name` key. Supported:
- `densenet121`, `efficientnet_b1`, `efficientnet_b2`
- `convnext_tiny`, `mobilenet_v3_large`, `resnet101`

---

### **Issue:** Out of memory

**Solution:** Process in batches:
```bash
--max-images 20  # Process 20 images at a time
```

---

## ✅ Quality Checklist

- [ ] Generated all 4 visualization types
- [ ] Labels are large and readable (font_scale=1.5)
- [ ] Colors are correct (blue=GT, green=correct pred, red=wrong pred)
- [ ] pred_classification shows GT boxes (not detection boxes)
- [ ] Confidence scores visible on predictions
- [ ] Selected 2-3 representative images for paper
- [ ] Figure captions explain color coding and methodology

---

## 🎯 Key Advantages

✅ **Separates detection from classification** - Clear evaluation of each stage

✅ **4 distinct visualizations** - Easy comparison and interpretation

✅ **Color-coded correctness** - Instantly see which predictions are wrong

✅ **GT boxes for classification** - Isolates classification performance from detection errors

✅ **Confidence scores** - Shows model certainty for each prediction

✅ **Publication-ready** - High-quality images with clear labels

✅ **Processes all images** - No manual selection needed

---

**Created:** 2025-10-02
**Last Updated:** 2025-10-02
**Script:** `scripts/visualization/generate_detection_classification_figures.py`
