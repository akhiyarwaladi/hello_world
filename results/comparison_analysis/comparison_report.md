# Detection vs Ground Truth Classification Comparison Report

Generated on: 2025-09-15 14:43:55

## Executive Summary

- **Total Experiments Analyzed**: 2
- **Crop Sources**: yolo11_detection
- **Model Types**: YOLO
- **Best Performance**: 100.00% (yolo11_det_to_efficientnet)

### Performance by Crop Source:
- **yolo11_detection**: 100.00% ± 0.00% (2.0 experiments)

## Crop Generation Analysis

### Yolo8 Detection
- **Total Crops**: 1,806
- **Source Images**: 207
- **Avg Crops per Image**: 8.72
- **Avg Detection Confidence**: 0.620
- **Confidence Range**: 0.250 - 0.990
- **Split Distribution**:
  - train: 1,153 crops
  - test: 355 crops
  - val: 298 crops

### Yolo11 Detection
- **Total Crops**: 407
- **Source Images**: 168
- **Avg Crops per Image**: 2.42
- **Avg Detection Confidence**: 0.414
- **Confidence Range**: 0.251 - 0.944
- **Split Distribution**:
  - train: 284 crops
  - val: 62 crops
  - test: 61 crops

### Ground Truth
- **Total Crops**: 1,345
- **Source Images**: 208
- **Avg Crops per Image**: 6.47
- **Avg Detection Confidence**: 1.000
- **Confidence Range**: 1.000 - 1.000
- **Split Distribution**:
  - train: 854 crops
  - test: 282 crops
  - val: 209 crops

## Detailed Experiment Results

| Experiment | Source | Model | Accuracy | Loss | Training Time |
|------------|--------|-------|----------|------|---------------|
| yolo11_det_to_efficientnet | yolo11_detection | efficientnet_b0 | 100.00% | N/A | 47.8 min |
| yolo11_det_to_resnet18 | yolo11_detection | resnet18 | 100.00% | N/A | 45.8 min |

## Key Findings

1. **Best Crop Source**: Yolo11 Detection achieved highest average accuracy (100.00%)
2. **Best Model Type**: YOLO achieved highest average accuracy (100.00%)
3. **Crop Generation**: Generated 3,558 total crops across all sources
4. **Detection vs Ground Truth**: YOLOv8 generated 1,806 crops vs 1,345 from ground truth

## Recommendations

1. Use the best-performing crop source and model combination for production
2. Consider the trade-off between crop quantity (detection) and quality (ground truth)
3. Investigate failure cases in lower-performing combinations
4. Consider ensemble methods combining multiple approaches
