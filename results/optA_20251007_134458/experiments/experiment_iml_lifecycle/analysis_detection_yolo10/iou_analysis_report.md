# IoU Variation Analysis - NO RE-TESTING

**Experiment**: detection_yolo10
**Source**: Pre-computed training validation results
**Advantage**: No model loading or prediction required

## Performance at Different IoU Thresholds (TRAINING VALIDATION)

| IoU Threshold | mAP | mAP@0.5:0.95 | Precision | Recall | Source |
|---------------|-----|--------------|-----------|--------|--------|
| 0.5 | 0.967 | 0.777 | 0.914 | 0.937 | training_validation |
| 0.75 | 0.932 | 0.777 | 0.914 | 0.937 | estimated_from_training |
| 0.5:0.95 | 0.777 | 0.777 | 0.914 | 0.937 | training_validation |

## Analysis Results - TRAINING VALIDATION BASED

**PRE-COMPUTED METRICS** (from best training epoch 19.0):
- **mAP@0.5**: 0.966810 (training validation)
- **mAP@0.75**: 0.932136 (estimated from mAP50-95)
- **mAP@0.5:0.95**: 0.776780 (training validation)

**Performance Pattern**: Uses validation metrics from training
**Behavior**: No re-testing required, instant analysis

## Summary
- **Performance Range**: mAP@0.5=0.967, mAP@0.75=0.932, mAP@0.5:0.95=0.777
- **Best Epoch**: 19.0 out of 100 total epochs
- **Source**: Training validation results (no additional testing)

## Advantages of Pre-computed Analysis
- [OK] **No Model Loading**: Skips expensive model initialization
- [OK] **No Re-prediction**: Uses existing validation results
- [OK] **Instant Results**: Analysis completes in seconds
- [OK] **Consistent Data**: Same validation set used during training

## Files Generated
- `iou_variation_results.json`: Raw metrics data
- `iou_comparison_table.csv`: Comparison table
- `iou_analysis_report.md`: This report
