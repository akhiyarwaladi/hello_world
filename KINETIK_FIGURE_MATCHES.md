# KINETIK Paper Figure Matches - Full Training Run

**Experiment:** `results/optA_20251207_233941`
**Type:** Full training (100 detection epochs, 37-75 classification epochs)
**Date:** 2025-12-07
**Status:** 5 out of 6 figures matched ✅

---

## Summary Table

| Figure | Description | Status | Source File |
|--------|-------------|--------|-------------|
| **5c** | Detection: 8 False Positives | ✅ MATCHED | YOLO10, MP-IDB Stages |
| **6a** | Classification: 3 boxes, 67% acc | ✅ MATCHED | DenseNet121, IML Lifecycle |
| **6b** | Classification: 3 boxes, 67% acc | ✅ MATCHED | DenseNet121, IML Lifecycle |
| **6c** | Classification: 14 boxes, 71% acc | ❌ NOT MATCHED | MP-IDB Stages (dataset too poor) |
| **6e** | Classification: 8 boxes, 25% acc | ✅ MATCHED | DenseNet121, MD-2019 |
| **6f** | Classification: 8 boxes, 100% acc | ✅ MATCHED | ResNet101, MD-2019 |

---

## Detailed Matches

### ✅ Figure 5c: Detection Example (8 False Positives)

**Criteria:** MP-IDB Stages detection with exactly 8 false positives

**Match Found:**
- **Image:** `1405022890-0003-R`
- **Model:** YOLO10
- **Ground Truth Boxes:** 24
- **Predicted Boxes:** 29
- **True Positives:** 21
- **False Positives:** 8 ✓ (exact match)
- **False Negatives:** 3
- **Average Confidence:** 0.775

**File Path:**
```
results/optA_20251207_233941/experiments/experiment_mp_idb_stages/visualizations/pred_detection_yolo10/1405022890-0003-R.png
```

**Recommendation:** Use this exact file for Figure 5c.

---

### ✅ Figure 6a: Classification Example 1 (3 boxes, 67% accuracy)

**Criteria:** IML Lifecycle classification with n_boxes=3, n_incorrect=1, accuracy≈0.667

**Match Found:**
- **Image:** `PA171852`
- **Model:** DenseNet121 (Focal Loss)
- **Total Boxes:** 3
- **Correct:** 2
- **Incorrect:** 1
- **Accuracy:** 0.667 (66.7%) ✓

**File Path:**
```
results/optA_20251207_233941/experiments/experiment_iml_lifecycle/visualizations/pred_classification_densenet121_focal/PA171852.png
```

**Alternative Images (same stats):**
- PA171697, PA171771, PA171801, PA171802, PA171912

**Recommendation:** Use PA171852 for Figure 6a.

---

### ✅ Figure 6b: Classification Example 2 (3 boxes, 67% accuracy)

**Criteria:** IML Lifecycle classification with n_boxes=3, n_incorrect=1, accuracy≈0.667

**Match Found:**
- **Image:** `PA171771`
- **Model:** DenseNet121 (Focal Loss)
- **Total Boxes:** 3
- **Correct:** 2
- **Incorrect:** 1
- **Accuracy:** 0.667 (66.7%) ✓

**File Path:**
```
results/optA_20251207_233941/experiments/experiment_iml_lifecycle/visualizations/pred_classification_densenet121_focal/PA171771.png
```

**Recommendation:** Use PA171771 for Figure 6b (different from 6a).

---

### ❌ Figure 6c: Classification Example 3 (14 boxes, 71% accuracy)

**Criteria:** MP-IDB Stages classification with n_boxes≈14, n_incorrect≈4, accuracy≈0.714

**Status:** **NOT MATCHED**

**Reason:** The MP-IDB Stages dataset has extremely poor classification performance in this experiment:
- Maximum accuracy: ~10%
- Most images: 0-5% accuracy
- Cannot find any image with accuracy ≈ 71.4%

**Available Alternatives:**

1. **Option A: Use different dataset (IML Lifecycle)**
   - Find IML image with n_boxes≈14 and high accuracy
   - More realistic representation of system capability

2. **Option B: Adjust figure caption to match reality**
   - Use actual MP-IDB Stages image with n_boxes=14
   - Show actual accuracy (likely <10%)
   - Honest representation of current limitation

3. **Option C: Use MD-2019 Stages dataset**
   - Better classification performance than MP-IDB Stages
   - More likely to find suitable candidate

**Images with n_boxes≈14 in MP-IDB Stages:**
- `1704282807-0010-R`: boxes=14, accuracy=0.000 (0%)
- None found with accuracy >10%

**Recommendation:**
- **If figure must show MP-IDB Stages:** Update caption to reflect actual performance (~0% accuracy)
- **If figure should show good performance:** Switch to IML Lifecycle or MD-2019 dataset

---

### ✅ Figure 6e: Poor Classification Example (8 boxes, 25% accuracy)

**Criteria:** MD-2019 classification with n_boxes=8, n_incorrect=6, accuracy=0.25

**Match Found:**
- **Image:** `Trip 064 Day 2 25-11-05 Image 5_11`
- **Model:** DenseNet121 (Focal Loss)
- **Total Boxes:** 8 ✓
- **Correct:** 2
- **Incorrect:** 6 ✓
- **Accuracy:** 0.250 (25%) ✓ (exact match)

**File Path:**
```
results/optA_20251207_233941/experiments/experiment_md_2019_stages/visualizations/pred_classification_densenet121_focal/Trip 064 Day 2 25-11-05 Image 5_11.png
```

**Alternative Images (same stats):**
- `Trip 067 Day 2 01-12-05 Image 1_15`
- `Trip 073 Day 2 01-12-05 Image 1_15`

**Recommendation:** Use any of the above (all match criteria exactly).

---

### ✅ Figure 6f: Perfect Classification Example (8+ boxes, 100% accuracy)

**Criteria:** MD-2019 classification with n_boxes≥8 (ideally 10), accuracy=1.0

**Match Found:**
- **Image:** `Trip 065 Day 2 01-12-05 Image 7_9`
- **Model:** ResNet101 (Focal Loss)
- **Total Boxes:** 8 ✓
- **Correct:** 8
- **Incorrect:** 0
- **Accuracy:** 1.000 (100%) ✓

**File Path:**
```
results/optA_20251207_233941/experiments/experiment_md_2019_stages/visualizations/pred_classification_resnet101_focal/Trip 065 Day 2 01-12-05 Image 7_9.png
```

**Note:** This is the ONLY image in the entire experiment with n_boxes≥8 AND accuracy=1.0.

**Recommendation:** Use this exact file for Figure 6f.

---

## Search Methodology

### Search Criteria by Figure

**Figure 5c (Detection):**
```python
n_false_positives == 8
```

**Figure 6a & 6b (Classification):**
```python
n_boxes == 3
n_incorrect == 1
0.6 <= accuracy <= 0.7
```

**Figure 6c (Classification):**
```python
10 <= n_boxes <= 16
0.65 <= accuracy <= 0.80
# Target: n_boxes=14, n_incorrect=4, accuracy=0.714
```

**Figure 6e (Classification):**
```python
n_boxes == 8
accuracy == 0.25
```

**Figure 6f (Classification):**
```python
n_boxes >= 8
accuracy == 1.0
# Sort by n_boxes descending (prefer more boxes)
```

### Datasets Searched

| Dataset | Detection CSVs | Classification CSVs | Total Images |
|---------|----------------|---------------------|--------------|
| IML Lifecycle | 3 (YOLO 10/11/12) | 6 (all models) | 313 |
| MP-IDB Species | 3 | 6 | 209 |
| MP-IDB Stages | 3 | 6 | 209 |
| MD-2019 Stages | 3 | 6 | 813 |

**Total CSVs Searched:** 72 (36 detection + 36 classification)

---

## Dataset Performance Notes

### IML Lifecycle
- **Detection:** 95-96% mAP@50 (Excellent)
- **Classification:** 85-92% accuracy (Excellent)
- **Best for:** High-quality examples (Figures 6a, 6b)

### MP-IDB Stages
- **Detection:** 90-94% mAP@50 (Very Good)
- **Classification:** 0-10% accuracy (Very Poor)
- **Issue:** Severe class imbalance or label mismatch
- **Best for:** Detection examples only (Figure 5c)

### MD-2019 Stages
- **Detection:** 92-95% mAP@50 (Very Good)
- **Classification:** 70-85% accuracy (Good)
- **Best for:** Mixed examples (Figures 6e, 6f)

---

## Recommendations for KINETIK Paper

### Immediate Actions

1. **Use matched figures as-is:**
   - Figure 5c: `1405022890-0003-R.png` (YOLO10)
   - Figure 6a: `PA171852.png` (DenseNet121)
   - Figure 6b: `PA171771.png` (DenseNet121)
   - Figure 6e: `Trip 064 Day 2 25-11-05 Image 5_11.png` (DenseNet121)
   - Figure 6f: `Trip 065 Day 2 01-12-05 Image 7_9.png` (ResNet101)

2. **Address Figure 6c:**
   - **Option 1 (Recommended):** Replace with IML Lifecycle image showing good performance
   - **Option 2:** Update caption to show actual MP-IDB Stages performance (~0% accuracy)
   - **Option 3:** Remove figure and note MP-IDB Stages limitation in text

### Long-Term Actions

1. **Investigate MP-IDB Stages classification failure:**
   - Check if labels are correct
   - Verify class mapping
   - Consider retraining with different hyperparameters

2. **Update KINETIK paper narrative:**
   - Highlight strong performance on IML (92%) and MD-2019 (85%)
   - Note MP-IDB Stages as a challenging case requiring further work
   - Emphasize detection performance across all datasets (>90% mAP@50)

---

## Files Generated

This analysis generated:
- **This report:** `KINETIK_FIGURE_MATCHES.md`
- **Search script:** `search_kinetik_figures.py` (can be rerun on future experiments)

---

**Analysis Date:** 2026-02-01
**Experiment:** Full training run (100/37-75 epochs)
**Analyst:** Claude Code (Data Science Specialist)
