# IoU Variation Analysis

## Performance at Different IoU Thresholds (TEST SET)

| IoU Threshold | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---------------|---------|--------------|-----------|--------|
| 0.3 | 0.864 | 0.583 | 0.944 | 0.783 |
| 0.5 | 0.865 | 0.579 | 0.946 | 0.773 |
| 0.7 | 0.855 | 0.569 | 0.941 | 0.778 |

## Summary
- **Best Performance**: mAP@0.5=0.865 at IoU=0.5
- **Model**: best.pt
- **Evaluation**: TEST SET (independent)

## Files Generated
- `iou_variation_results.json`: Raw metrics data
- `iou_comparison_table.csv`: Comparison table
- `iou_analysis_report.md`: This report
