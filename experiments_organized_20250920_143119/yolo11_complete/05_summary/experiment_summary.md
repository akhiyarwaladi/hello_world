# YOLO11 Complete Pipeline Results (Organized)

**Original Experiment**: multi_pipeline_20250920_131500
**Organized**: 2025-09-20 14:31:56

## 📊 Performance Summary

### Detection Performance
- **mAP50**: 0.985
- **mAP50-95**: 0.978
- **Precision**: 0.910
- **Recall**: 0.934
- **Training Time**: 13.4 minutes
- **Epochs**: 33

### IoU Variation Analysis (NMS Thresholds)

| IoU Threshold | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---------------|---------|--------------|-----------|--------|
| 0.3 | 0.995 | 0.995 | 0.938 | 0.975 |
| 0.5 | 0.995 | 0.995 | 0.938 | 0.975 |
| 0.7 | 0.995 | 0.995 | 0.938 | 0.975 |

**Best Performance**: mAP@0.5=0.995 at IoU=0.3

### Total Training Time: 13.4 minutes

## 📁 Organized Structure
```
yolo11_complete/
├── 01_detection_results/     # Detection training results (symlinked)
├── 02_crop_data/            # Generated crop dataset (symlinked)
├── 03_classification_results/ # Classification results (symlinked)
├── 04_analysis/             # Confusion matrix & deep analysis
└── 05_summary/              # This summary
```

## 🔗 Original Results Locations
- **Detection**: results/current_experiments/training/detection/yolo11_detection/multi_pipeline_20250920_131500_yolo11_det

---
*Organized by Existing Results Organizer*
