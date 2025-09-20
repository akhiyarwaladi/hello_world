# IoU Variation Analysis

## Performance at Different IoU Thresholds (TEST SET)

| IoU Threshold | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---------------|---------|--------------|-----------|--------|
| 0.3 | 0.292 | 0.180 | 0.009 | 0.902 |
| 0.5 | 0.265 | 0.177 | 0.004 | 1.000 |
| 0.7 | 0.240 | 0.178 | 0.004 | 1.000 |

## Summary
- **Best Performance**: mAP@0.5=0.292 at IoU=0.3
- **Model**: best.pt
- **Evaluation**: TEST SET (independent)

## Files Generated
- `iou_variation_results.json`: Raw metrics data
- `iou_comparison_table.csv`: Comparison table
- `iou_analysis_report.md`: This report
