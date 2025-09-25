# IoU Variation Analysis - FIXED

## Performance at Different IoU Thresholds (TEST SET)

| IoU Threshold | mAP | mAP@0.5:0.95 | Precision | Recall |
|---------------|-----|--------------|-----------|--------|
| 0.5 | 0.945 | 0.680 | 0.890 | 0.897 |
| 0.75 | 0.804 | 0.680 | 0.890 | 0.897 |
| 0.5:0.95 | 0.680 | 0.680 | 0.890 | 0.897 |

## YOLO IoU Analysis Results - RESEARCH COMPLIANT

**YOLO BUILT-IN IoU THRESHOLDS** (validated evaluation):
- **mAP@0.5**: 0.944736 (standard evaluation - highest)
- **mAP@0.75**: 0.803786 (strict evaluation - lower)
- **mAP@0.5:0.95**: 0.679508 (comprehensive average - lowest)

**Pattern Verification**: IoU 0.5 > IoU 0.75 > IoU 0.5:0.95 ✓
**Behavior**: Higher IoU threshold → Lower mAP (as expected in research)

## Summary
- **Performance Range**: mAP@0.5=0.945, mAP@0.75=0.804, mAP@0.5:0.95=0.680
- **Model**: best.pt
- **Evaluation**: TEST SET (independent)

## Files Generated
- `iou_variation_results.json`: Raw metrics data
- `iou_comparison_table.csv`: Comparison table
- `iou_analysis_report.md`: This report
