# Malaria Detection Models Comparison Report

Generated on: 2025-09-13 21:58:59

## Executive Summary

This report compares the performance of YOLOv8, YOLOv11, YOLOv12, and RT-DETR models for malaria parasite detection and classification.

### Dataset Summary
- **Detection Dataset**: 103 microscopy images, 1,242 parasites (P. falciparum)
- **Classification Dataset**: 1,242 cropped parasite images (128x128px)
- **Source**: MP-IDB dataset with corrected bounding box annotations

## Model Performance Comparison

### Detection Models
\n| Model | mAP50 | mAP50-95 | Precision | Recall | Epochs | Batch Size |\n|-------|-------|----------|-----------|---------|---------|------------|\n| YOLOv11 | nan | nan | nan | nan | 20 | 8 |\n| RT-DETR | nan | nan | nan | nan | 20 | 6 |\n| YOLOv8 | 0.000 | 0.000 | 0.000 | 0.006 | 30 | 8 |\n| YOLOv11 | N/A | N/A | N/A | N/A | 1 | 8 |\n\n### Classification Models\n\n| Model | Top-1 Accuracy | Epochs | Batch Size |\n|-------|----------------|---------|------------|\n| unknown | N/A | 10 | 16 |\n| unknown | N/A | 10 | 32 |\n| unknown | N/A | 3 | 32 |\n| unknown | N/A | 5 | 16 |\n| unknown | N/A | 8 | 16 |\n| unknown | N/A | 25 | 32 |\n

## Key Findings

### Detection Performance
- **Best mAP50**: 0.000
- **Best Precision**: 0.000
- **Best Recall**: 0.006

### Classification Performance
- **Best Accuracy**: N/A

## Training Status
- **Note**: Training was interrupted due to system crash
- **Completed Epochs**: Most models completed 1-3 epochs only
- **Recommendation**: Resume training for full evaluation

## Conclusions

### Model Comparison for Malaria Detection Research

1. **YOLOv8**: Baseline performance, well-established architecture
2. **YOLOv11**: Newer version with potential improvements
3. **RT-DETR**: Transformer-based approach, different detection paradigm

### Recommendations

Based on the preliminary results:
- **Resume Training**: Complete full epoch training for fair comparison
- For deployment: Consider model size vs accuracy trade-off
- For research: Compare inference speed and computational requirements
- For clinical use: Prioritize precision to minimize false positives

## Technical Details

### Experimental Setup
- **Framework**: Ultralytics YOLO
- **Hardware**: CPU training
- **Dataset Split**: 70% train, 15% val, 15% test
- **Image Preprocessing**: CLAHE enhancement, normalization

### Data Quality Improvements
- ✅ **Bounding Box Correction**: Fixed coordinate mapping from MP-IDB CSV annotations
- ✅ **Ground Truth Validation**: Used binary masks for accurate parasite localization
- ✅ **Proper Cropping**: Individual parasite cells for classification
- ✅ **Dataset Ready**: 1,242 cropped parasites, 103 detection images

---

*This report was generated automatically from training experiment results.*
