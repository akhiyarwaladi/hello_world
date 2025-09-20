# YOLO8 Complete Pipeline Results (Organized)

**Original Experiment**: multi_pipeline_20250920_131500
**Organized**: 2025-09-20 14:30:03

## 📊 Performance Summary

### Detection Performance
- **mAP50**: 0.995
- **mAP50-95**: 0.972
- **Precision**: 0.948
- **Recall**: 0.953
- **Training Time**: 11.9 minutes
- **Epochs**: 29

### Classification Performance
- **Top-1 Accuracy**: 0.857
- **Top-5 Accuracy**: 1.000 (Expected 1.0 for 4-class)
- **Training Time**: 0.4 minutes
- **Epochs**: 16

### Total Training Time: 12.3 minutes

## 📁 Organized Structure
```
yolo8_complete/
├── 01_detection_results/     # Detection training results (symlinked)
├── 02_crop_data/            # Generated crop dataset (symlinked)
├── 03_classification_results/ # Classification results (symlinked)
├── 04_analysis/             # Confusion matrix & deep analysis
└── 05_summary/              # This summary
```

## 🔗 Original Results Locations
- **Detection**: results/current_experiments/training/detection/yolov8_detection/multi_pipeline_20250920_131500_yolo8_det
- **Classification**: results/current_experiments/training/classification/yolov8_classification/multi_pipeline_20250920_131500_yolo8_cls
- **Crops**: data/crops_from_yolo8_multi_pipeline_20250920_131500_yolo8_det

---
*Organized by Existing Results Organizer*
