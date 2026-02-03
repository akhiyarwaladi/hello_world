# Figure Cases Identification Summary
## Experiment: optA_20251207_233941

This document identifies specific images from the experiment metadata that match the error case criteria for **Figure 5** (Detection Errors) and **Figure 6** (Classification Errors) in the journal paper.

---

## FIGURE 5: Detection Error Cases (YOLO11)

### (a) IML Lifecycle - False Positive Only Case
**Image:** `PA171785`
**File Path:** `results\optA_20251207_233941\experiments\experiment_iml_lifecycle\visualizations\pred_detection_yolo11\PA171785.png`

**Metrics:**
- Ground Truth Boxes: **1**
- Predicted Boxes: **4**
- False Positives: **3** (overdetection)
- False Negatives: **0**
- Average Confidence: **0.867**

**Error Type:** Pure overdetection case - model detected 3 additional false parasites with high confidence.

---

### (b) IML Lifecycle - False Negative Only Case
**Image:** `PA171699`
**File Path:** `results\optA_20251207_233941\experiments\experiment_iml_lifecycle\visualizations\pred_detection_yolo11\PA171699.png`

**Metrics:**
- Ground Truth Boxes: **2**
- Predicted Boxes: **0**
- False Positives: **0**
- False Negatives: **2** (complete miss)
- Average Confidence: **0.000**

**Error Type:** Complete detection failure - model failed to detect any of the 2 parasites present.

---

### (c) MP-IDB Stages - Overdetection Case (High FP)
**Image:** `1307210661-0007-R`
**File Path:** `results\optA_20251207_233941\experiments\experiment_mp_idb_stages\visualizations\pred_detection_yolo11\1307210661-0007-R.png`

**Metrics:**
- Ground Truth Boxes: **31**
- Predicted Boxes: **36**
- False Positives: **6** (overdetection in crowded scene)
- False Negatives: **1**
- Average Confidence: **0.702**

**Error Type:** Overdetection in densely populated image - model added 6 false detections while missing only 1 true parasite.

---

### (d) MP-IDB Species - Mixed Error Case
**Image:** `1701151546-0015-R_T`
**File Path:** `results\optA_20251207_233941\experiments\experiment_mp_idb_species\visualizations\pred_detection_yolo11\1701151546-0015-R_T.png`

**Metrics:**
- Ground Truth Boxes: **37**
- Predicted Boxes: **39**
- False Positives: **11**
- False Negatives: **9**
- Average Confidence: **0.688**

**Error Type:** Heavy mixed errors - both overdetection (11 FP) and missed detections (9 FN) in a very crowded scene with 37 parasites.

---

### (e) MD-2019 Stages - Crowded Mixed Error Case
**Image:** `Trip 073 Day 2 01-12-05 Image 1_8`
**File Path:** `results\optA_20251207_233941\experiments\experiment_md_2019_stages\visualizations\pred_detection_yolo11\Trip 073 Day 2 01-12-05 Image 1_8.png`

**Metrics:**
- Ground Truth Boxes: **18**
- Predicted Boxes: **17**
- False Positives: **3**
- False Negatives: **4**
- Average Confidence: **0.735**

**Error Type:** Mixed errors in crowded scene - model made both types of errors (total 7 errors) despite reasonable confidence.

---

### (f) MD-2019 Stages - False Negative Case (Atypical Morphology)
**Image:** `Trip 073 Day 2 01-12-05 Image 1_10`
**File Path:** `results\optA_20251207_233941\experiments\experiment_md_2019_stages\visualizations\pred_detection_yolo11\Trip 073 Day 2 01-12-05 Image 1_10.png`

**Metrics:**
- Ground Truth Boxes: **14**
- Predicted Boxes: **10**
- False Positives: **0**
- False Negatives: **4** (atypical morphology)
- Average Confidence: **0.702**

**Error Type:** Pure underdetection - model missed 4 parasites (likely atypical morphology) while making zero false alarms.

---

## FIGURE 6: Classification Error Cases (EfficientNet-B0 Focal)

### (a) IML Lifecycle - Single Error Case
**Image:** `PA171852`
**File Path:** `results\optA_20251207_233941\experiments\experiment_iml_lifecycle\visualizations\pred_classification_efficientnet_b0_focal\PA171852.png`

**Metrics:**
- Total Boxes: **3**
- Correct Classifications: **2**
- Incorrect Classifications: **1**
- Accuracy: **66.7%**
- Average Confidence: **0.643**

**Error Type:** Minimal error - 1 out of 3 parasites misclassified (moderate confidence).

---

### (b) IML Lifecycle - Moderate Error Case
**Image:** `PA171802`
**File Path:** `results\optA_20251207_233941\experiments\experiment_iml_lifecycle\visualizations\pred_classification_efficientnet_b0_focal\PA171802.png`

**Metrics:**
- Total Boxes: **3**
- Correct Classifications: **2**
- Incorrect Classifications: **1**
- Accuracy: **66.7%**
- Average Confidence: **0.581**

**Error Type:** Moderate error with lower confidence - 1 misclassification among 3 parasites (similar to 6a but different image).

---

### (c) MP-IDB Stages - Heavy Error Case (High Box Count)
**Image:** `1704282807-0019-R_G`
**File Path:** `results\optA_20251207_233941\experiments\experiment_mp_idb_stages\visualizations\pred_classification_efficientnet_b0_focal\1704282807-0019-R_G.png`

**Metrics:**
- Total Boxes: **41**
- Correct Classifications: **1**
- Incorrect Classifications: **40**
- Accuracy: **2.4%**
- Average Confidence: **0.503**

**Error Type:** Catastrophic failure - almost complete misclassification (40/41 wrong) in extremely crowded scene with moderate confidence.

---

### (d) MP-IDB Species - Complete Failure (Species Confusion)
**Image:** `1305121398-0003-R`
**File Path:** `results\optA_20251207_233941\experiments\experiment_mp_idb_species\visualizations\pred_classification_efficientnet_b0_focal\1305121398-0003-R.png`

**Metrics:**
- Total Boxes: **1**
- Correct Classifications: **0**
- Incorrect Classifications: **1**
- Accuracy: **0.0%**
- Average Confidence: **0.811**

**Error Type:** Complete failure with high confidence - model confidently misidentified the species (100% wrong).

---

### (e) MD-2019 Stages - Heavy Error Case
**Image:** `Trip 073 Day 2 01-12-05 Image 1_8`
**File Path:** `results\optA_20251207_233941\experiments\experiment_md_2019_stages\visualizations\pred_classification_efficientnet_b0_focal\Trip 073 Day 2 01-12-05 Image 1_8.png`

**Metrics:**
- Total Boxes: **18**
- Correct Classifications: **3**
- Incorrect Classifications: **15**
- Accuracy: **16.7%**
- Average Confidence: **0.639**

**Error Type:** Heavy error - only 3 out of 18 parasites correctly classified (83% error rate) despite moderate confidence.

---

### (f) MD-2019 Stages - Perfect Case (High Complexity)
**Image:** `Trip 065 Day 2 01-12-05 Image 7_9`
**File Path:** `results\optA_20251207_233941\experiments\experiment_md_2019_stages\visualizations\pred_classification_resnet101_focal\Trip 065 Day 2 01-12-05 Image 7_9.png`
**Model:** ResNet101 Focal Loss

**Metrics:**
- Total Boxes: **8**
- Correct Classifications: **8**
- Incorrect Classifications: **0**
- Accuracy: **100.0%**
- Average Confidence: **0.913**

**Error Type:** Perfect classification - all 8 parasites correctly classified in a moderately crowded scene with high confidence.

**Note:** This case was found using ResNet101 model (not EfficientNet-B0). EfficientNet-B0/B1/B2 and DenseNet121 achieved perfect accuracy only on images with ≤6 boxes in the test set.

---

## Summary Statistics

### Detection Errors (Figure 5)
| Panel | Dataset | Error Type | GT Boxes | Pred Boxes | FP | FN | Total Errors |
|-------|---------|------------|----------|------------|----|----|--------------|
| (a) | IML Lifecycle | FP Only | 1 | 4 | 3 | 0 | 3 |
| (b) | IML Lifecycle | FN Only | 2 | 0 | 0 | 2 | 2 |
| (c) | MP-IDB Stages | Overdetection | 31 | 36 | 6 | 1 | 7 |
| (d) | MP-IDB Species | Mixed | 37 | 39 | 11 | 9 | 20 |
| (e) | MD-2019 Stages | Crowded Mixed | 18 | 17 | 3 | 4 | 7 |
| (f) | MD-2019 Stages | FN (Atypical) | 14 | 10 | 0 | 4 | 4 |

### Classification Errors (Figure 6)
| Panel | Dataset | Error Type | Boxes | Correct | Incorrect | Accuracy | Confidence | Model |
|-------|---------|------------|-------|---------|-----------|----------|------------|-------|
| (a) | IML Lifecycle | Single Error | 3 | 2 | 1 | 66.7% | 0.643 | EfficientNet-B0 |
| (b) | IML Lifecycle | Moderate | 3 | 2 | 1 | 66.7% | 0.581 | EfficientNet-B0 |
| (c) | MP-IDB Stages | Catastrophic | 41 | 1 | 40 | 2.4% | 0.503 | EfficientNet-B0 |
| (d) | MP-IDB Species | Complete Fail | 1 | 0 | 1 | 0.0% | 0.811 | EfficientNet-B0 |
| (e) | MD-2019 Stages | Heavy Error | 18 | 3 | 15 | 16.7% | 0.639 | EfficientNet-B0 |
| (f) | MD-2019 Stages | Perfect | 8 | 8 | 0 | 100.0% | 0.913 | ResNet101 |

---

## How to Use This Document

1. **For Figure 5:** Use the PNG files from `pred_detection_yolo11` folders for each identified image.
2. **For Figure 6:** Use the PNG files from the respective classification model folders:
   - Panels (a)-(e): `pred_classification_efficientnet_b0_focal`
   - Panel (f): `pred_classification_resnet101_focal`
3. **Image Files:** All PNG files contain bounding boxes and labels overlaid on the original microscopy images.
4. **All cases found:** ✓ Complete set of 12 images identified (6 detection + 6 classification)

---

**Generated:** 2026-02-01
**Experiment:** optA_20251207_233941
**Detection Model:** YOLO11 Medium
**Classification Model:** EfficientNet-B0 Focal Loss
**Datasets:** IML Lifecycle, MP-IDB Species, MP-IDB Stages, MD-2019 Stages
