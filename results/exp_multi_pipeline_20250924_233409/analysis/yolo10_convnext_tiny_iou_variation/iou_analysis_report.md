# IoU Variation Analysis - FIXED

## Performance at Different IoU Thresholds (TEST SET)

| IoU Threshold | mAP | mAP@0.5:0.95 | Precision | Recall |
|---------------|-----|--------------|-----------|--------|
| 0.5 | 0.932 | 0.673 | 0.883 | 0.876 |
| 0.75 | 0.865 | 0.673 | 0.883 | 0.876 |
| 0.5:0.95 | 0.673 | 0.673 | 0.883 | 0.876 |

## CORRECTED IoU Analysis Results

**YOLO EVALUATION IoU THRESHOLDS** (only 3 available):
- **mAP@0.5**: 0.932292 (standard evaluation - should be highest)
- **mAP@0.75**: 0.864536 (strict evaluation - should be lower)
- **mAP@0.5:0.95**: 0.673448 (comprehensive average - most comprehensive)

**Pattern Verification**: IoU 0.5 > IoU 0.75 > IoU 0.5:0.95 ✓
**Behavior**: Higher IoU threshold → Lower mAP (as expected in research)

## Summary
- **Standard Performance**: mAP@IoU0.5=0.932
- **Model**: best.pt
- **Evaluation**: TEST SET (independent)

## Files Generated
- `iou_variation_results.json`: Raw metrics data
- `iou_comparison_table.csv`: Comparison table
- `iou_analysis_report.md`: This report
