# 📊 How to Generate Detection + Classification Figures for Paper

## ✅ Script Baru (Improved)

Script: `scripts/visualization/generate_detection_classification_figures.py`

## 🎯 Output Structure

**Untuk SETIAP test image, generates 4 separate figures:**

```
paper_figures/
├── gt_detection/          ← Ground Truth detection boxes (blue, no labels)
│   ├── PA171697.png
│   ├── PA171699.png
│   └── ...
├── pred_detection/        ← Predicted detection boxes (green, no classification)
│   ├── PA171697.png
│   ├── PA171699.png
│   └── ...
├── gt_classification/     ← GT boxes + GT class labels (blue + text)
│   ├── PA171697.png
│   ├── PA171699.png
│   └── ...
└── pred_classification/   ← Predicted boxes + classification (green + text)
    ├── PA171697.png
    ├── PA171699.png
    └── ...
```

### **4 Types of Visualizations:**

1. **GT Detection** (`gt_detection/`)
   - Ground truth bounding boxes ONLY
   - Blue boxes, no class labels
   - Shows detection annotations

2. **Predicted Detection** (`pred_detection/`)
   - Model detection results ONLY (YOLO)
   - Green boxes, no classification
   - Shows detection performance

3. **GT Classification** (`gt_classification/`)
   - Ground truth boxes WITH class labels
   - Blue boxes + class names
   - Shows ground truth parasites with their types

4. **Predicted Classification** (`pred_classification/`)
   - Model detection + classification results
   - Green boxes + predicted class + confidence
   - Complete end-to-end pipeline results

---

## 🚀 Quick Start

### **Default: Process ALL Test Images**

```bash
# Activate environment
conda activate malaria

# Generate ALL figures (default behavior)
python scripts/visualization/generate_detection_classification_figures.py \
  --detection-model results/optA_20251001_210647/experiments/experiment_iml_lifecycle/det_yolo12/weights/best.pt \
  --classification-model results/optA_20251001_210647/experiments/experiment_iml_lifecycle/cls_efficientnet_b1_focal/best.pt \
  --test-images data/processed/lifecycle/test/images \
  --test-labels data/processed/lifecycle/test/labels \
  --gt-crops results/optA_20251001_210647/experiments/experiment_iml_lifecycle/crops_gt_crops \
  --output paper_figures
```

**Result:**
- Processes ALL 66 test images
- Generates 66 × 4 = 264 total images
- Organized in 4 folders

---

### **Limit Number of Images**

```bash
# Generate for first 5 images only
python scripts/visualization/generate_detection_classification_figures.py \
  --detection-model results/optA_20251001_210647/experiments/experiment_iml_lifecycle/det_yolo12/weights/best.pt \
  --classification-model results/optA_20251001_210647/experiments/experiment_iml_lifecycle/cls_efficientnet_b1_focal/best.pt \
  --test-images data/processed/lifecycle/test/images \
  --test-labels data/processed/lifecycle/test/labels \
  --gt-crops results/optA_20251001_210647/experiments/experiment_iml_lifecycle/crops_gt_crops \
  --output paper_figures \
  --max-images 5
```

**Result:**
- Processes first 5 test images
- Generates 5 × 4 = 20 images

---

### **Lower Detection Confidence Threshold**

If model trained with few epochs, use lower confidence:

```bash
python scripts/visualization/generate_detection_classification_figures.py \
  --detection-model results/optA_20251001_210647/experiments/experiment_iml_lifecycle/det_yolo12/weights/best.pt \
  --classification-model results/optA_20251001_210647/experiments/experiment_iml_lifecycle/cls_efficientnet_b1_focal/best.pt \
  --test-images data/processed/lifecycle/test/images \
  --test-labels data/processed/lifecycle/test/labels \
  --gt-crops results/optA_20251001_210647/experiments/experiment_iml_lifecycle/crops_gt_crops \
  --output paper_figures \
  --det-conf-threshold 0.1 \
  --max-images 10
```

---

## 📊 Example Output

For test image `PA171697.jpg`, generates:

### 1. GT Detection (`gt_detection/PA171697.png`)
```
┌────────────────────────────┐
│  [Original Image]          │
│                            │
│  📦 Blue box               │
│  📦 Blue box               │
│  📦 Blue box               │
│                            │
│  No class labels           │
└────────────────────────────┘
```

### 2. Predicted Detection (`pred_detection/PA171697.png`)
```
┌────────────────────────────┐
│  [Original Image]          │
│                            │
│  📦 Green box              │
│  📦 Green box              │
│  📦 Green box              │
│                            │
│  No classification yet     │
└────────────────────────────┘
```

### 3. GT Classification (`gt_classification/PA171697.png`)
```
┌────────────────────────────┐
│  [Original Image]          │
│                            │
│  📦 Gametocyte (blue)      │
│  📦 Ring (blue)            │
│  📦 Trophozoite (blue)     │
│                            │
│  Shows GT class labels     │
└────────────────────────────┘
```

### 4. Predicted Classification (`pred_classification/PA171697.png`)
```
┌────────────────────────────┐
│  [Original Image]          │
│                            │
│  📦 Gametocyte 0.95 (green)│
│  📦 Ring 0.87 (green)      │
│  📦 Trophozoite 0.72 (green)│
│                            │
│  Complete pipeline output  │
└────────────────────────────┘
```

---

## 📝 Command Line Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--detection-model` | YOLO detection model path | ✅ Yes | - |
| `--classification-model` | Classification model path | ✅ Yes | - |
| `--test-images` | Test images directory | ✅ Yes | - |
| `--test-labels` | Test labels directory (YOLO format) | ✅ Yes | - |
| `--gt-crops` | Ground truth crops directory | ✅ Yes | - |
| `--output` | Output base directory | No | `paper_figures` |
| `--det-conf-threshold` | Detection confidence threshold | No | `0.25` |
| `--max-images` | Maximum images to process | No | `None` (all) |

---

## 💡 Usage Tips

### **For Paper - Recommended Workflow:**

1. **Generate ALL figures first** (to see all results)
   ```bash
   python scripts/visualization/generate_detection_classification_figures.py \
     --detection-model path/to/det_model.pt \
     --classification-model path/to/cls_model.pt \
     --test-images path/to/images \
     --test-labels path/to/labels \
     --gt-crops path/to/crops \
     --output paper_figures
   ```

2. **Review outputs** - Pick best representative images

3. **For paper, use:**
   - **Figure 1**: `gt_classification/` - Show what ground truth looks like
   - **Figure 2**: `pred_classification/` - Show complete pipeline results
   - **Supplementary**: `gt_detection/` and `pred_detection/` - Show detection performance

4. **Select 2-3 representative images:**
   - Best case (all correct)
   - Average case (some errors)
   - Challenging case (complex/crowded)

---

## 📋 Paper Figure Examples

### **Main Figure Caption:**

```
Figure X: End-to-end malaria detection and classification pipeline.
(Top) Ground truth annotations showing parasite locations and species labels
(gametocyte, ring, trophozoite, schizont). (Bottom) Model predictions using
YOLO12 for detection and EfficientNet-B1 (Focal Loss) for classification.
Green boxes with labels show predicted parasite species and confidence scores.
The pipeline achieves 94% detection mAP@50 and 88.9% classification accuracy
on the test set.
```

### **Supplementary Figure Caption:**

```
Figure SX: Detection performance comparison.
(Left) Ground truth detection annotations showing parasite locations without
species labels. (Right) YOLO12 detection results showing predicted bounding
boxes. The model achieves high detection accuracy with IoU > 0.5 for most
parasites.
```

---

## 🔧 Troubleshooting

### No detections / Empty pred_detection images

**Cause**: Detection confidence threshold too high or model needs more training

**Solution**:
```bash
# Lower confidence threshold
--det-conf-threshold 0.1
```

### Classification labels not showing

**Cause**: GT crops directory structure incorrect

**Solution**:
```bash
# Check structure
ls results/.../crops_gt_crops/crops/test/
# Should show: gametocyte/ ring/ schizont/ trophozoite/
```

### Out of memory

**Solution**:
```bash
# Process in batches
--max-images 20  # Process 20 at a time
```

---

## 📦 Requirements

```bash
pip install torch torchvision ultralytics opencv-python pillow numpy
```

---

## ✅ Checklist for Paper

- [ ] Generate all 4 types of figures for test set
- [ ] Review outputs and select 2-3 representative images
- [ ] Use `gt_classification/` to show ground truth
- [ ] Use `pred_classification/` to show pipeline results
- [ ] Optional: Use `gt_detection/` and `pred_detection/` in supplementary
- [ ] Write clear figure captions explaining color coding
- [ ] Include performance metrics in caption
- [ ] Ensure 300 dpi resolution (default in script)

---

## 🎯 Summary

**Key Advantages of This Script:**

✅ **Separates concerns**: Detection vs Classification
✅ **4 clear visualizations**: Easy to compare and understand
✅ **Process all images**: No manual selection needed
✅ **Organized output**: Clean folder structure
✅ **Publication-ready**: High resolution (300 dpi via cv2.imwrite)
✅ **Default behavior**: Process ALL test images automatically

**Perfect for paper figures!** 🎉

---

**Created:** 2025-10-02
**Script:** `scripts/visualization/generate_detection_classification_figures.py`
**Purpose:** Generate separate detection and classification visualizations for publication
