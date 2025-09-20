# YOLO12 Complete Pipeline Results (Organized)

**Original Experiment**: multi_pipeline_20250920_131500
**Organized**: 2025-09-20 14:38:51

## 📊 Performance Summary

### Detection Performance
- **mAP50**: 0.995
- **mAP50-95**: 0.995
- **Precision**: 0.977
- **Recall**: 0.986
- **Training Time**: 20.7 minutes
- **Epochs**: 39

### IoU Variation Analysis (NMS Thresholds)

| IoU Threshold | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---------------|---------|--------------|-----------|--------|
| 0.3 | 0.965 | 0.965 | 0.958 | 0.950 |
| 0.5 | 0.964 | 0.964 | 0.958 | 0.950 |
| 0.7 | 0.964 | 0.964 | 0.958 | 0.950 |

**Best Performance**: mAP@0.5=0.965 at IoU=0.3

### Total Training Time: 20.7 minutes

## 📁 Organized Structure
```
yolo12_complete/
├── 01_detection_results/     # Detection training results (symlinked)
├── 02_crop_data/            # Generated crop dataset (symlinked)
├── 03_classification_results/ # Classification results (symlinked)
├── 04_analysis/             # Confusion matrix & deep analysis
└── 05_summary/              # This summary
```

## 🔗 Original Results Locations
- **Detection**: results/current_experiments/training/detection/yolo12_detection/multi_pipeline_20250920_131500_yolo12_det

---
*Organized by Existing Results Organizer*
