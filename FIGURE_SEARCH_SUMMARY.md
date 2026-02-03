# Figure Case Search Summary
## Experiment: optA_20251207_233941

---

## ✅ EXACT MATCHES FOUND

### Figure 5c: Detection - MP-IDB Stages (8 False Positives)

**🎯 EXACT MATCH - YOLO10**
- **Image:** `1405022890-0003-R`
- **File:** `experiment_mp_idb_stages/visualizations/pred_detection_yolo10/detection_metadata.csv`
- **Stats:**
  - Ground truth boxes: 24
  - Predicted boxes: 29
  - Correct matches: 21
  - False positives: **8** ✅
  - False negatives: 3
  - Avg confidence: 0.775

**Recommendation:** Use this exact image - perfect match for paper description.

---

### Figure 6a & 6b: Classification - IML Lifecycle (n_boxes=3, accuracy=0.667)

**Multiple exact matches found across ALL classification models!**

**Top Candidates:**

#### Option 1: PA171852 (Most consistent across models)
- n_boxes: 3
- n_correct: 2
- n_incorrect: 1
- accuracy: 0.667 ✅
- Found in: DenseNet121, EfficientNet-B0, B1, B2, ResNet101, ResNet50
- Avg confidence range: 0.556-0.861

#### Option 2: PA171802
- n_boxes: 3
- n_correct: 2
- n_incorrect: 1
- accuracy: 0.667 ✅
- Found in: DenseNet121, EfficientNet-B0, B2, ResNet101, ResNet50
- Avg confidence range: 0.393-0.928

#### Option 3: PA171771
- n_boxes: 3
- n_correct: 2
- n_incorrect: 1
- accuracy: 0.667 ✅
- Found in: DenseNet121, EfficientNet-B0, B1, B2, ResNet101, ResNet50
- Avg confidence range: 0.467-0.851

**Recommendation:**
- **6a:** Use PA171852 (most stable, found in all models)
- **6b:** Use PA171802 (different from 6a, wide confidence range shows model variation)

---

### Figure 6e: Classification - MD-2019 (n_boxes=8, n_incorrect=6, accuracy=0.25)

**🎯 EXACT MATCHES FOUND - Multiple options!**

#### Primary Option: Trip 064 Day 2 25-11-05 Image 5_11
- n_boxes: 8 ✅
- n_correct: 2
- n_incorrect: 6 ✅
- accuracy: 0.25 ✅
- Found in: DenseNet121, EfficientNet-B0, B1, B2, ResNet50
- Avg confidence range: 0.809-0.999

#### Alternative Option: Trip 073 Day 2 01-12-05 Image 1_15
- n_boxes: 8 ✅
- n_correct: 2
- n_incorrect: 6 ✅
- accuracy: 0.25 ✅
- Found in: EfficientNet-B1, B2, ResNet50
- Avg confidence range: 0.716-0.989

**Recommendation:** Use "Trip 064 Day 2 25-11-05 Image 5_11" - found in most models.

---

## ⚠️ NO EXACT MATCHES - ALTERNATIVES NEEDED

### Figure 6c: Classification - MP-IDB Stages (n_boxes=14, n_incorrect=4, accuracy≈0.714)

**Problem:** In this experiment, high box count images (≥10 boxes) have VERY LOW accuracy (0-6.5%).

**Why:** The 5-epoch quick test didn't give classification models enough training time to handle complex multi-parasite images. These images require ~75 epochs to achieve the 71% accuracy described in the paper.

**Evidence from distribution analysis:**
- Images with n_boxes≥10: ALL have accuracy <10%
- Highest box count (n_boxes=41): accuracy = 2.4%
- Second highest (n_boxes=31): accuracy = 6.5%

**Options:**

#### Option A: Use different experiment (RECOMMENDED)
Use images from `optA_20251016_200330` (100/75 epoch baseline) where classification models were fully trained. That experiment should have cases matching the paper's description.

#### Option B: Accept closest match from current experiment
**Closest available:**
- Image: `1307210661-0007-R`
- n_boxes: 31 (higher than target 14)
- n_correct: 2
- n_incorrect: 29 (higher than target 4)
- accuracy: 0.065 (much lower than target 0.714)
- **NOT RECOMMENDED** - accuracy too low to demonstrate classification capability

**Recommendation:** Skip this figure from the current 5-epoch experiment, or use baseline experiment results.

---

### Figure 6f: Classification - MD-2019 (n_boxes≥8, accuracy=1.0)

**Problem:** No images with n_boxes≥9 and perfect accuracy in this 5-epoch experiment.

**Best available:**

#### Only Option: Trip 065 Day 2 01-12-05 Image 7_9
- n_boxes: 8 (target was ≥8, preferably 10) ⚠️
- n_correct: 8
- n_incorrect: 0
- accuracy: 1.0 ✅
- Found in: ResNet101 only
- Avg confidence: 0.913

**Why only n_boxes=8:** With 5 training epochs, the models haven't learned to confidently predict on larger cell counts.

**Options:**

#### Option A: Accept n_boxes=8 (ACCEPTABLE)
- Still meets minimum requirement (n_boxes≥8)
- Perfect accuracy demonstrates capability
- Caveat: Lower box count than paper's original example

#### Option B: Use baseline experiment
Use `optA_20251016_200330` which should have n_boxes=10+ examples with perfect accuracy.

**Recommendation:** Accept n_boxes=8 for 5-epoch experiment with caveat in paper, OR use baseline experiment for n_boxes=10.

---

## 📊 SUMMARY TABLE

| Figure | Target | Status | Match Quality | Recommendation |
|--------|--------|--------|---------------|----------------|
| 5c (Detection) | 8 FPs | ✅ EXACT | Perfect | Use YOLO10 `1405022890-0003-R` |
| 6a (IML) | n_boxes=3, acc=0.667 | ✅ EXACT | Perfect | Use PA171852 |
| 6b (IML) | n_boxes=3, acc=0.667 | ✅ EXACT | Perfect | Use PA171802 |
| 6c (MP-IDB Stages) | n_boxes=14, acc=0.714 | ❌ NO MATCH | Poor | Use baseline experiment |
| 6e (MD-2019) | n_boxes=8, acc=0.25 | ✅ EXACT | Perfect | Use Trip 064...Image 5_11 |
| 6f (MD-2019) | n_boxes≥8, acc=1.0 | ⚠️ PARTIAL | Acceptable | Use Trip 065...Image 7_9 (n=8) |

**Overall:** 4/6 exact matches, 1 acceptable partial match, 1 requires baseline experiment.

---

## 🎯 FINAL RECOMMENDATIONS

### For 5-Epoch Experiment Paper (optA_20251207_233941)

**Use these figures:**
1. ✅ Figure 5c: YOLO10 detection with 8 FPs
2. ✅ Figure 6a: PA171852 (IML, 3 boxes, 66.7% acc)
3. ✅ Figure 6b: PA171802 (IML, 3 boxes, 66.7% acc)
4. ✅ Figure 6e: Trip 064 Day 2 25-11-05 Image 5_11 (MD-2019, 8 boxes, 25% acc)
5. ⚠️ Figure 6f: Trip 065 Day 2 01-12-05 Image 7_9 (MD-2019, 8 boxes, 100% acc) - with caveat about n_boxes=8 instead of 10

**Skip or modify:**
- ❌ Figure 6c: Classification accuracy too low in 5-epoch test to demonstrate capability

**Caveat for paper:** Note that Figure 6c and optimal Figure 6f require full training (75 epochs) and should reference baseline experiment results instead.

### For Full Paper (Using Baseline optA_20251016_200330)

Search the baseline experiment for:
- Figure 6c: Should find n_boxes=14, accuracy≈71% cases
- Figure 6f: Should find n_boxes=10+ with perfect accuracy

---

## 📂 FILES FOR VERIFICATION

To verify any case, check these CSV files:

**Detection:**
```
results/optA_20251207_233941/experiments/experiment_[dataset]/visualizations/
  pred_detection_[model]/detection_metadata.csv
```

**Classification:**
```
results/optA_20251207_233941/experiments/experiment_[dataset]/visualizations/
  pred_classification_[model]_focal/classification_metadata_images.csv
```

**Baseline (for 6c and 6f):**
```
results/optA_20251016_200330/experiments/...
```

---

**Generated:** 2026-02-01
**Experiment:** optA_20251207_233941 (5 epochs detection, 5 epochs classification)
**Purpose:** Find cases matching KINETIK paper figure descriptions
