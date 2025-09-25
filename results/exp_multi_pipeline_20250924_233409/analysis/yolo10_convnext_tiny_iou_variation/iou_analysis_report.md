# IoU Variation Analysis - FIXED

## Performance at Different IoU Thresholds (TEST SET)

| IoU Threshold | mAP | mAP@0.5:0.95 | Precision | Recall |
|---------------|-----|--------------|-----------|--------|
| 0.5 | 0.932 | 0.673 | 0.883 | 0.876 |
| 0.75 | 0.865 | 0.673 | 0.883 | 0.876 |
| 0.5:0.95 | 0.673 | 0.673 | 0.883 | 0.876 |

## YOLO IoU Analysis Results - RESEARCH COMPLIANT

**YOLO BUILT-IN IoU THRESHOLDS** (validated evaluation):
- **mAP@0.5**: 0.932292 (standard evaluation - highest)
- **mAP@0.75**: 0.864536 (strict evaluation - lower)
- **mAP@0.5:0.95**: 0.673448 (comprehensive average - lowest)

**Pattern Verification**: IoU 0.5 > IoU 0.75 > IoU 0.5:0.95 ✓
**Behavior**: Higher IoU threshold → Lower mAP (as expected in research)

## Summary
- **Performance Range**: mAP@0.5=0.932, mAP@0.75=0.865, mAP@0.5:0.95=0.673
- **Model**: best.pt
- **Evaluation**: TEST SET (independent)

## Files Generated
- `iou_variation_results.json`: Raw metrics data
- `iou_comparison_table.csv`: Comparison table
- `iou_analysis_report.md`: This report
