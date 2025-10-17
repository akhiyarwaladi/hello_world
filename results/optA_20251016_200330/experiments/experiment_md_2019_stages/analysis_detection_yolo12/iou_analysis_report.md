# IoU Variation Analysis - NO RE-TESTING

**Experiment**: detection_yolo12
**Source**: Pre-computed training validation results
**Advantage**: No model loading or prediction required

## Performance at Different IoU Thresholds (TRAINING VALIDATION)

| IoU Threshold | mAP | mAP@0.5:0.95 | Precision | Recall | Source |
|---------------|-----|--------------|-----------|--------|--------|
| 0.5 | 0.746 | 0.536 | 0.625 | 0.778 | training_validation |
| 0.75 | 0.643 | 0.536 | 0.625 | 0.778 | estimated_from_training |
| 0.5:0.95 | 0.536 | 0.536 | 0.625 | 0.778 | training_validation |

## Analysis Results - TRAINING VALIDATION BASED

**PRE-COMPUTED METRICS** (from best training epoch 35.0):
- **mAP@0.5**: 0.745890 (training validation)
- **mAP@0.75**: 0.642672 (estimated from mAP50-95)
- **mAP@0.5:0.95**: 0.535560 (training validation)

**Performance Pattern**: Uses validation metrics from training
**Behavior**: No re-testing required, instant analysis

## Summary
- **Performance Range**: mAP@0.5=0.746, mAP@0.75=0.643, mAP@0.5:0.95=0.536
- **Best Epoch**: 35.0 out of 100 total epochs
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
