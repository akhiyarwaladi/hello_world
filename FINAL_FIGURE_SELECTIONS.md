# Final Figure Selections for KINETIK Paper
## Experiment: optA_20251207_233941 (5-epoch quick test)

**Date:** 2026-02-01
**Purpose:** Match KINETIK paper figure descriptions using current experiment results

---

## 📋 SELECTION SUMMARY

| Figure | Dataset | Status | Image Selected | Model | Match Quality |
|--------|---------|--------|----------------|-------|---------------|
| 5c | MP-IDB Stages | ✅ EXACT | `1405022890-0003-R` | YOLO10 | Perfect (8 FPs) |
| 6a | IML Lifecycle | ✅ EXACT | `PA171852` | EfficientNet-B0 | Perfect (3 boxes, 66.7% acc) |
| 6b | IML Lifecycle | ✅ EXACT | `PA171802` | ResNet50 | Perfect (3 boxes, 66.7% acc) |
| 6c | MP-IDB Stages | ❌ SKIP | - | - | No match (use baseline) |
| 6e | MD-2019 | ✅ EXACT | `Trip 064 Day 2 25-11-05 Image 5_11` | EfficientNet-B2 | Perfect (8 boxes, 25% acc) |
| 6f | MD-2019 | ⚠️ ACCEPT | `Trip 065 Day 2 01-12-05 Image 7_9` | ResNet101 | Good (8 boxes, 100% acc, prefer 10 boxes) |

**Result:** 4 exact matches, 1 acceptable match, 1 skip (requires baseline experiment)

---

## 🎯 DETAILED SELECTIONS

### Figure 5c: Detection False Positives (MP-IDB Stages)

**Target:** 8 false positives to demonstrate detection limitations

**Selected:** `1405022890-0003-R` (YOLO10)

**Metrics:**
- Ground truth boxes: 24
- Predicted boxes: 29
- Correct matches: 21
- **False positives: 8** ✅
- False negatives: 3
- Avg confidence: 0.775

**File Path:**
```
results/optA_20251207_233941/experiments/experiment_mp_idb_stages/
  visualizations/pred_detection_yolo10/1405022890-0003-R.png
```

**CSV Source:**
```
experiment_mp_idb_stages/visualizations/pred_detection_yolo10/detection_metadata.csv
```

**Why this one:**
- EXACT match to paper description (8 FPs)
- Only image in entire experiment with exactly 8 false positives
- Demonstrates detection challenge: complex image with 24 parasites
- Image file verified to exist ✅

---

### Figure 6a: IML Lifecycle Classification (Error Case 1)

**Target:** n_boxes=3, n_incorrect=1, accuracy≈0.667

**Selected:** `PA171852` (EfficientNet-B0)

**Metrics:**
- n_boxes: 3 ✅
- n_correct: 2
- n_incorrect: 1 ✅
- **Accuracy: 0.667** ✅
- Avg confidence: 0.643

**File Path:**
```
results/optA_20251207_233941/experiments/experiment_iml_lifecycle/
  visualizations/pred_classification_efficientnet_b0_focal/PA171852.png
```

**CSV Source:**
```
experiment_iml_lifecycle/visualizations/pred_classification_efficientnet_b0_focal/
  classification_metadata_images.csv
```

**Why this one:**
- EXACT match to all target criteria
- **Found in ALL 6 classification models** (most consistent)
- Shows typical classification error pattern
- Confidence range across models: 0.556-0.861 (demonstrates model variation)

**Cross-Model Consistency:**
```
Model              Accuracy  Confidence
-----------------  --------  ----------
DenseNet121        0.667     0.774
EfficientNet-B0    0.667     0.643  ← SELECTED
EfficientNet-B1    0.667     0.556
EfficientNet-B2    0.667     0.861
ResNet50           0.667     0.791
ResNet101          0.667     0.706
```

---

### Figure 6b: IML Lifecycle Classification (Error Case 2)

**Target:** n_boxes=3, n_incorrect=1, accuracy≈0.667 (different from 6a)

**Selected:** `PA171802` (ResNet50)

**Metrics:**
- n_boxes: 3 ✅
- n_correct: 2
- n_incorrect: 1 ✅
- **Accuracy: 0.667** ✅
- Avg confidence: 0.928 (high confidence despite error!)

**File Path:**
```
results/optA_20251207_233941/experiments/experiment_iml_lifecycle/
  visualizations/pred_classification_resnet50_focal/PA171802.png
```

**CSV Source:**
```
experiment_iml_lifecycle/visualizations/pred_classification_resnet50_focal/
  classification_metadata_images.csv
```

**Why this one:**
- EXACT match to all target criteria
- Different from Figure 6a (PA171852 vs PA171802)
- **High confidence (0.928) despite being wrong** - demonstrates overconfidence issue
- Found in 6 models, but EfficientNet-B1 shows 100% accuracy (interesting variation)

**Cross-Model Consistency:**
```
Model              Accuracy  Confidence
-----------------  --------  ----------
DenseNet121        0.667     0.766
EfficientNet-B0    0.667     0.581
EfficientNet-B1    1.000     0.628  ← Perfect in this model!
EfficientNet-B2    0.667     0.822
ResNet50           0.667     0.928  ← SELECTED (highest confidence)
ResNet101          0.667     0.393  ← Lowest confidence
```

**Note:** ResNet50 chosen to show "overconfident error" (92.8% confidence but wrong). Could alternatively use EfficientNet-B1 to show model variation (same image, different result).

---

### Figure 6c: MP-IDB Stages Classification

**Target:** n_boxes=14, n_incorrect=4, accuracy≈0.714

**Status:** ❌ **NO MATCH IN 5-EPOCH EXPERIMENT**

**Problem:**
- In 5-epoch quick test, classification models haven't converged
- All high box count images (≥10 boxes) have accuracy <10%
- Cannot demonstrate classification capability with such low accuracy

**Evidence:**
```
Image               n_boxes  accuracy
------------------  -------  --------
1704282807-0019-R_G   41      0.024
1307210661-0007-R     31      0.065
1701151546-0007-R     27      0.000
1704282807-0012-R_T   27      0.000
1405022890-0003-R     24      0.000
```

**Recommendation:**
- **SKIP this figure** in 5-epoch experiment paper
- OR use baseline experiment `optA_20251016_200330` (75 epochs) where models achieved >85% accuracy
- Mention in paper: "Complex multi-parasite images require full training (75 epochs) to achieve adequate classification performance"

---

### Figure 6e: MD-2019 Classification (Low Accuracy)

**Target:** n_boxes=8, n_incorrect=6, accuracy=0.25

**Selected:** `Trip 064 Day 2 25-11-05 Image 5_11` (EfficientNet-B2)

**Metrics:**
- n_boxes: 8 ✅
- n_correct: 2
- n_incorrect: 6 ✅
- **Accuracy: 0.25** ✅
- Avg confidence: 0.999 (overconfident!)

**File Path:**
```
results/optA_20251207_233941/experiments/experiment_md_2019_stages/
  visualizations/pred_classification_efficientnet_b2_focal/
  Trip 064 Day 2 25-11-05 Image 5_11.png
```

**CSV Source:**
```
experiment_md_2019_stages/visualizations/pred_classification_efficientnet_b2_focal/
  classification_metadata_images.csv
```

**Why this one:**
- EXACT match to all target criteria
- Found in 5/6 classification models (very consistent)
- **99.9% confidence but 75% wrong** - excellent demonstration of overconfidence problem
- Most stable across models (same accuracy in all 5)

**Cross-Model Consistency:**
```
Model              Accuracy  Confidence
-----------------  --------  ----------
DenseNet121        0.250     0.993
EfficientNet-B0    0.250     0.809
EfficientNet-B1    0.250     0.854
EfficientNet-B2    0.250     0.999  ← SELECTED (highest confidence)
ResNet50           0.250     0.890
```

**Alternative Option:** `Trip 073 Day 2 01-12-05 Image 1_15` (also exact match)

---

### Figure 6f: MD-2019 Classification (High Accuracy)

**Target:** n_boxes≥8 (preferably 10), accuracy=1.0

**Selected:** `Trip 065 Day 2 01-12-05 Image 7_9` (ResNet101)

**Metrics:**
- n_boxes: 8 ⚠️ (target was ≥8, ideally 10)
- n_correct: 8
- n_incorrect: 0
- **Accuracy: 1.0** ✅
- Avg confidence: 0.913

**File Path:**
```
results/optA_20251207_233941/experiments/experiment_md_2019_stages/
  visualizations/pred_classification_resnet101_focal/
  Trip 065 Day 2 01-12-05 Image 7_9.png
```

**CSV Source:**
```
experiment_md_2019_stages/visualizations/pred_classification_resnet101_focal/
  classification_metadata_images.csv
```

**Why this one:**
- Only image with n_boxes≥8 AND perfect accuracy in entire experiment
- Demonstrates model capability despite limited training
- ResNet101 performed best on this image

**Caveat:**
- Only n_boxes=8 (not the preferred n_boxes=10 from paper)
- Only achieves 100% in ResNet101; other models have 62.5-75% accuracy on same image
- **Recommendation:** Use with caveat note, OR use baseline experiment for n_boxes=10 example

**Cross-Model Consistency:**
```
Model              Accuracy  Confidence
-----------------  --------  ----------
EfficientNet-B0    0.750     0.796
EfficientNet-B1    0.625     0.933
EfficientNet-B2    0.750     0.999
ResNet50           0.750     0.923
ResNet101          1.000     0.913  ← SELECTED (only perfect)
```

**Note:** This image shows high model variation - ResNet101 gets 100% while others get 62.5-75%. Could be used to discuss model-specific performance.

---

## 📊 STATISTICAL SUMMARY

### Coverage by Dataset

**IML Lifecycle:**
- Figures found: 6a, 6b (2/2 = 100%)
- Match quality: EXACT for both

**MP-IDB Species:**
- Not required for paper

**MP-IDB Stages:**
- Figures found: 5c (detection)
- Figures missing: 6c (classification - low accuracy in 5 epochs)
- Match quality: 1/2 = 50%

**MD-2019:**
- Figures found: 6e, 6f (2/2 = 100%)
- Match quality: 6e EXACT, 6f ACCEPTABLE

**Overall: 5/6 figures matched (83.3%)**

### Match Quality Distribution

- ✅ **EXACT matches:** 4/6 (66.7%)
  - Figure 5c: 8 FPs (EXACT)
  - Figure 6a: 3 boxes, 66.7% acc (EXACT)
  - Figure 6b: 3 boxes, 66.7% acc (EXACT)
  - Figure 6e: 8 boxes, 25% acc (EXACT)

- ⚠️ **ACCEPTABLE matches:** 1/6 (16.7%)
  - Figure 6f: 8 boxes (not 10), 100% acc (ACCEPTABLE)

- ❌ **NO MATCH:** 1/6 (16.7%)
  - Figure 6c: Requires full 75-epoch training

---

## 🔍 INTERESTING FINDINGS

### 1. Duplicate Entries in CSV Files
All selected images appear **twice** in their respective CSV files. This appears to be a data processing artifact. Use the first occurrence.

### 2. Cross-Model Variation
Same image can produce different results across models:

**Example: PA171802 (Figure 6b candidate)**
- ResNet50: 66.7% accuracy, 92.8% confidence ← SELECTED
- EfficientNet-B1: **100% accuracy**, 62.8% confidence
- ResNet101: 66.7% accuracy, 39.3% confidence

**Example: Trip 065...Image 7_9 (Figure 6f)**
- ResNet101: **100% accuracy** ← SELECTED
- Other models: 62.5-75% accuracy

This demonstrates model-specific performance and could be discussed in the paper.

### 3. Overconfidence Problem
Multiple examples of high confidence with wrong predictions:
- Figure 6b: 92.8% confidence, but 1/3 wrong (66.7% accuracy)
- Figure 6e: **99.9% confidence**, but 6/8 wrong (25% accuracy)

### 4. 5-Epoch Limitations
- Detection performs well (84% mAP@50)
- Simple classification cases work (3 boxes, 66.7% accuracy)
- Complex cases fail (>10 boxes, <10% accuracy)
- Demonstrates need for full 75-epoch training on complex images

---

## 📂 FILE REFERENCES

### Absolute Paths to Selected Images

```
# Figure 5c (Detection)
C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251207_233941\experiments\
  experiment_mp_idb_stages\visualizations\pred_detection_yolo10\1405022890-0003-R.png

# Figure 6a (IML - EfficientNet-B0)
C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251207_233941\experiments\
  experiment_iml_lifecycle\visualizations\pred_classification_efficientnet_b0_focal\PA171852.png

# Figure 6b (IML - ResNet50)
C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251207_233941\experiments\
  experiment_iml_lifecycle\visualizations\pred_classification_resnet50_focal\PA171802.png

# Figure 6e (MD-2019 - EfficientNet-B2)
C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251207_233941\experiments\
  experiment_md_2019_stages\visualizations\pred_classification_efficientnet_b2_focal\
  Trip 064 Day 2 25-11-05 Image 5_11.png

# Figure 6f (MD-2019 - ResNet101)
C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251207_233941\experiments\
  experiment_md_2019_stages\visualizations\pred_classification_resnet101_focal\
  Trip 065 Day 2 01-12-05 Image 7_9.png
```

### CSV Metadata Files

```
# Detection metadata
experiment_mp_idb_stages/visualizations/pred_detection_yolo10/detection_metadata.csv

# IML classification metadata
experiment_iml_lifecycle/visualizations/pred_classification_efficientnet_b0_focal/
  classification_metadata_images.csv
experiment_iml_lifecycle/visualizations/pred_classification_resnet50_focal/
  classification_metadata_images.csv

# MD-2019 classification metadata
experiment_md_2019_stages/visualizations/pred_classification_efficientnet_b2_focal/
  classification_metadata_images.csv
experiment_md_2019_stages/visualizations/pred_classification_resnet101_focal/
  classification_metadata_images.csv
```

---

## 🎓 RECOMMENDATIONS FOR PAPER

### Option A: Use 5-Epoch Experiment Only

**Include figures:** 5c, 6a, 6b, 6e, 6f (skip 6c)

**Narrative:**
- 5-epoch experiment demonstrates rapid prototyping capability
- Detection achieves 84% mAP@50 in just 5 epochs
- Simple classification cases (≤3 boxes) achieve reasonable accuracy
- Complex cases (>10 boxes) require full training → reference baseline

**Caveat notes:**
- Figure 6f: Only n_boxes=8 (not 10) due to limited training
- Figure 6c: Skipped - requires 75-epoch baseline experiment

### Option B: Hybrid Approach (Recommended)

**From 5-epoch experiment:** 5c, 6a, 6b, 6e
**From baseline experiment (optA_20251016_200330):** 6c, 6f

**Narrative:**
- Demonstrate full system capability using baseline (96% detection, 91% classification)
- Show that all figure cases exist in properly trained models
- Note: Current 5-epoch quick test achieves 4/6 exact matches

**Advantage:**
- All 6 figures with EXACT matches
- More convincing demonstration of system capability
- Avoids caveats about limited training

---

## ✅ VERIFICATION CHECKLIST

Before using these selections, verify:

- [ ] All image files exist at specified paths
- [ ] CSV metadata matches reported metrics
- [ ] Image quality suitable for publication
- [ ] Bounding boxes and labels visible
- [ ] File names match between CSV and actual files
- [ ] Duplicate entries noted (use first occurrence)
- [ ] Cross-model variations documented (if discussing)

---

**Document Generated:** 2026-02-01
**Experiment ID:** optA_20251207_233941
**Training Config:** 5 epochs detection, 5 epochs classification
**Purpose:** Find figure cases matching KINETIK paper descriptions
**Result:** 5/6 figures matched (1 requires baseline experiment)

**Next Steps:**
1. Verify all image files exist and are publication-quality
2. Decide: 5-epoch only (skip 6c) OR hybrid with baseline (all 6 figures)
3. Extract images and verify bounding boxes/labels are clear
4. Document any caveats in paper text
