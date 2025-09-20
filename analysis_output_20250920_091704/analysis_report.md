# Malaria Detection Pipeline Analysis Report

**Generated**: 2025-09-20 09:17:06

## Executive Summary

This report analyzes the performance of the complete malaria detection pipeline including detection models, crop generation, and classification training.

## Detection Results Summary

**Models Analyzed**: 10

### Detection Performance
- **Best mAP50**: yolov8 (0.852)
- **Average mAP50**: 0.333
- **Training Time Range**: 41.3s - 753.9s

### Model Comparison
| Model | mAP50 | mAP50-95 | Precision | Recall | Time(s) |
|-------|-------|----------|-----------|---------|---------|
| yolov8 | 0.760 | 0.371 | 0.70236 | 0.727 | 753.9 |
| yolov8 | 0.024 | 0.009 | 0.02515 | 0.626 | 87.4 |
| yolov8 | 0.019 | 0.007 | 0.01258 | 0.560 | 173.7 |
| yolov8 | 0.852 | 0.410 | 0.82893 | 0.835 | 653.7 |
| yolov8 | 0.320 | 0.173 | 0.00358 | 1.000 | 41.3 |
| yolov8 | 0.374 | 0.180 | 0.52723 | 0.337 | 67.6 |
| yolov8 | 0.320 | 0.173 | 0.00358 | 1.000 | 41.7 |
| yolov8 | 0.374 | 0.180 | 0.52723 | 0.337 | 53.8 |
| yolo11 | 0.272 | 0.197 | 0.00347 | 1.000 | 42.0 |
| rtdetr | 0.013 | 0.010 | 0.00363 | 0.667 | 377.4 |


## Crop Generation Results

**Total Crops Generated**: 382
**Models with Crops**: 2

### Crop Distribution
- **yolo8_auto_pipeline_yolo8_det_split**: 0 crops
- **yolo8_auto_pipeline_yolo8_det**: 382 crops
  - val: 27 crops
  - test: 33 crops


## Classification Results Summary

**Classification Experiments**: 18
**Total Epochs**: 129

### Classification Performance
- **auto_yolov8_cls**: 0.935 accuracy (4 epochs)
- **combo_yolov8det_yolov8m_cls**: 0.897 accuracy (3 epochs)
- **yolo11_multispecies_classification**: 0.729 accuracy (3 epochs)
- **ground_truth_to_yolo8s_cls**: 1.000 accuracy (8 epochs)
- **auto_pipeline_yolo8_cls**: 1.000 accuracy (5 epochs)
- **combo_yolov11det_yolov8s_cls**: 0.903 accuracy (3 epochs)
- **species_aware_to_yolo11_cls**: 1.000 accuracy (8 epochs)
- **auto_yolov11_cls**: 0.961 accuracy (5 epochs)
- **yolo8_det_to_yolov8n_cls.pt_cls**: 0.000 accuracy (10 epochs)
- **ground_truth_to_yolo8_cls**: 1.000 accuracy (10 epochs)
- **species_aware_yolo11_cls**: 1.000 accuracy (10 epochs)
- **ground_truth_to_yolo11_cls_v2**: 1.000 accuracy (8 epochs)
- **ground_truth_to_yolo8m_cls**: 1.000 accuracy (8 epochs)
- **species_aware_to_yolo8m_cls**: 1.000 accuracy (8 epochs)
- **species_aware_yolo8_cls**: 1.000 accuracy (10 epochs)
- **species_aware_to_yolo8n_cls**: 1.000 accuracy (8 epochs)
- **ground_truth_to_yolo11_cls**: 1.000 accuracy (10 epochs)
- **species_aware_to_yolo8s_cls**: 1.000 accuracy (8 epochs)


## Recommendations

### Detection Models
1. **YOLOv8**: Best overall performance for small datasets
2. **YOLOv11**: Good mAP50-95, suitable for precision tasks
3. **RT-DETR**: Needs optimization for small datasets

### Next Steps
1. **Increase Training Epochs**: Run full training with 50+ epochs
2. **Optimize RT-DETR**: Use smaller model variant and better hyperparameters
3. **Classification Training**: Complete full training on generated crops
4. **Cross-Validation**: Implement k-fold validation for robust evaluation

## Technical Details

### Dataset Information
- **Training Images**: 140 microscopy images
- **Validation Images**: 28 images
- **Classes**: 4 malaria species (P. falciparum, P. malariae, P. ovale, P. vivax)
- **Crop Size**: 128x128 pixels

### Training Configuration
- **Device**: CPU training
- **Framework**: Ultralytics YOLO
- **Data Augmentation**: Standard YOLO augmentations

---
*Report generated automatically from experimental results*
