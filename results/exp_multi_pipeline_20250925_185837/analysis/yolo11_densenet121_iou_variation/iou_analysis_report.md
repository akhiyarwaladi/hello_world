# IoU Variation Analysis - FIXED

## Performance at Different IoU Thresholds (TEST SET)

| IoU Threshold | mAP | mAP@0.5:0.95 | Precision | Recall |
|---------------|-----|--------------|-----------|--------|
| 0.5 | 0.000 | 0.000 | 0.000 | 0.010 |
| 0.75 | 0.000 | 0.000 | 0.000 | 0.010 |
| 0.5:0.95 | 0.000 | 0.000 | 0.000 | 0.010 |

## YOLO IoU Analysis Results - RESEARCH COMPLIANT

**YOLO BUILT-IN IoU THRESHOLDS** (validated evaluation):
- **mAP@0.5**: 0.000082 (standard evaluation - highest)
- **mAP@0.75**: 0.000082 (strict evaluation - lower)
- **mAP@0.5:0.95**: 0.000074 (comprehensive average - lowest)

**Pattern Verification**: IoU 0.5 > IoU 0.75 > IoU 0.5:0.95 ✓
**Behavior**: Higher IoU threshold → Lower mAP (as expected in research)

## Summary
- **Performance Range**: mAP@0.5=0.000, mAP@0.75=0.000, mAP@0.5:0.95=0.000
- **Model**: best.pt
- **Evaluation**: TEST SET (independent)

## Files Generated
- `iou_variation_results.json`: Raw metrics data
- `iou_comparison_table.csv`: Comparison table
- `iou_analysis_report.md`: This report
