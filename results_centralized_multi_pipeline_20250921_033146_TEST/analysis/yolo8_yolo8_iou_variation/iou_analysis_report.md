# IoU Variation Analysis

## Performance at Different IoU Thresholds (TEST SET)

| IoU Threshold | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---------------|---------|--------------|-----------|--------|
| 0.3 | 0.151 | 0.076 | 0.009 | 0.887 |
| 0.5 | 0.178 | 0.103 | 0.005 | 1.000 |
| 0.7 | 0.153 | 0.088 | 0.004 | 1.000 |

## Summary
- **Best Performance**: mAP@0.5=0.178 at IoU=0.5
- **Model**: best.pt
- **Evaluation**: TEST SET (independent)

## Files Generated
- `iou_variation_results.json`: Raw metrics data
- `iou_comparison_table.csv`: Comparison table
- `iou_analysis_report.md`: This report
