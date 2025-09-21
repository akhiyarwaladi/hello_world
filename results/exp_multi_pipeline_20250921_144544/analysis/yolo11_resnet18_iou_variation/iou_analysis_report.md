# IoU Variation Analysis

## Performance at Different IoU Thresholds (TEST SET)

| IoU Threshold | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---------------|---------|--------------|-----------|--------|
| 0.3 | 0.019 | 0.006 | 0.002 | 0.648 |
| 0.5 | 0.018 | 0.006 | 0.002 | 0.714 |
| 0.7 | 0.016 | 0.006 | 0.002 | 0.746 |

## Summary
- **Best Performance**: mAP@0.5=0.019 at IoU=0.3
- **Model**: best.pt
- **Evaluation**: TEST SET (independent)

## Files Generated
- `iou_variation_results.json`: Raw metrics data
- `iou_comparison_table.csv`: Comparison table
- `iou_analysis_report.md`: This report
