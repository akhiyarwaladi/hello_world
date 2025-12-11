# Selected Cases - Final Selection
**Based on**: FIGURE_METADATA_MAPPING.md criteria
**Date**: 2025-12-11
**Experiment**: optA_20251207_233941
**Detection Model**: YOLO11 (best)
**Classification Model**: EfficientNet-B1 Focal Loss (best)

---

## DETECTION CASES (6 types)

### [1] SIMPLE FP - Few false positives among correct detections
**Image**: `Trip 065 Day 2 01-12-05 Image 7_2`
**Dataset**: MD-2019 Stages
**Metadata**:
- Ground Truth: 7 boxes
- Correct: 7 TP
- False Positives: 1 FP
- False Negatives: 0 FN
- Confidence: 0.881 (High)
- **Interpretation**: Model correctly detected all 7 parasites but added 1 false positive (debris/artifact mistaken as parasite)

**Source**: `results/optA_20251207_233941/experiments/experiment_md_2019_stages/visualizations/pred_detection_yolo11/Trip 065 Day 2 01-12-05 Image 7_2.png`

---

### [2] PURE FN - Only missing parasites (no false positives)
**Image**: `Trip 067 Day 2 01-12-05 Image 1_3`
**Dataset**: MD-2019 Stages
**Metadata**:
- Ground Truth: 10 boxes
- Correct: 6 TP
- False Positives: 0 FP
- False Negatives: 4 FN (40% missed)
- **Interpretation**: Model missed 4 out of 10 parasites without adding any false positives. Likely atypical morphology or low contrast parasites.

**Source**: `results/optA_20251207_233941/experiments/experiment_md_2019_stages/visualizations/pred_detection_yolo11/Trip 067 Day 2 01-12-05 Image 1_3.png`

---

### [3] HEAVY FP - Many false positives (overdetection)
**Image**: `1701151546-0015-R_T`
**Dataset**: MP-IDB Species
**Metadata**:
- Ground Truth: 37 boxes
- Correct: 28 TP (75.7%)
- False Positives: 11 FP
- False Negatives: 9 FN
- **Interpretation**: Model detected 39 total predictions (28 correct + 11 FP). Heavy overdetection with 11 artifacts/debris mistaken as parasites. This is the worst FP case in the dataset.

**Source**: `results/optA_20251207_233941/experiments/experiment_mp_idb_species/visualizations/pred_detection_yolo11/1701151546-0015-R_T.png`

---

### [4] MIXED FP+FN - Both error types simultaneously
**Image**: `1701151546-0015-R_T`
**Dataset**: MP-IDB Species
**Metadata**:
- Ground Truth: 37 boxes
- Correct: 28 TP
- False Positives: 11 FP
- False Negatives: 9 FN
- **Interpretation**: Same image as Heavy FP - demonstrates both overdetection (11 FP) AND missing detection (9 FN) simultaneously. Model struggles with crowded field and morphology variation.

**Source**: `results/optA_20251207_233941/experiments/experiment_mp_idb_species/visualizations/pred_detection_yolo11/1701151546-0015-R_T.png`

---

### [5] CROWDED FIELD - Dense parasitemia (10+ GT boxes) with few errors
**Image**: `1704282807-0012-R_T`
**Dataset**: MP-IDB Stages
**Metadata**:
- Ground Truth: 27 boxes (HIGH density)
- Correct: 27 TP (100%)
- False Positives: 1 FP (minimal)
- False Negatives: 0 FN
- **Interpretation**: Excellent performance on highly crowded field. Model correctly detected all 27 parasites with only 1 minor false positive. Shows model robustness in high-density scenarios.

**Source**: `results/optA_20251207_233941/experiments/experiment_mp_idb_stages/visualizations/pred_detection_yolo11/1704282807-0012-R_T.png`

---

### [6] ATYPICAL FN - Missing parasite with atypical morphology
**Image**: `Trip 073 Day 2 01-12-05 Image 1_10`
**Dataset**: MD-2019 Stages
**Metadata**:
- Ground Truth: 14 boxes
- Correct: 10 TP (71.4%)
- False Positives: 0 FP
- False Negatives: 4 FN (28.6% missed)
- **Interpretation**: Model missed 4 parasites (no false positives). MD-2019 dataset contains multi-patient samples with morphological variation, explaining higher FN rate. Parasites may have atypical appearance or be in transition stages.

**Source**: `results/optA_20251207_233941/experiments/experiment_md_2019_stages/visualizations/pred_detection_yolo11/Trip 073 Day 2 01-12-05 Image 1_10.png`

---

## CLASSIFICATION CASES (6 types)

### [1] SINGLE ERROR - 66.7% accuracy (1 wrong out of 3)
**Image**: `Trip 804 Day 1 02-12-05 Image 3_11`
**Dataset**: MD-2019 Stages
**Metadata**:
- Total boxes: 3
- Correct: 2 (66.7%)
- Incorrect: 1 (33.3%)
- Confidence: 0.993 (Very High)
- **Interpretation**: Model correctly classified 2 out of 3 parasites with very high confidence. The single error shows that even with high confidence, stage confusion can occur on morphologically ambiguous parasites.

**Source**: `results/optA_20251207_233941/experiments/experiment_md_2019_stages/visualizations/pred_classification_efficientnet_b1_focal/Trip 804 Day 1 02-12-05 Image 3_11.png`

---

### [2] MODERATE ERROR - 50% accuracy (moderate error rate)
**Image**: `1409171742-0009-R`
**Dataset**: MP-IDB Species
**Metadata**:
- Total boxes: 6
- Correct: 3 (50%)
- Incorrect: 3 (50%)
- **Interpretation**: Half of the parasites misclassified. Shows species confusion (P. falciparum vs P. vivax/ovale) on moderate-sized image. Model struggles with species-level differentiation when morphological features overlap.

**Source**: `results/optA_20251207_233941/experiments/experiment_mp_idb_species/visualizations/pred_classification_efficientnet_b1_focal/1409171742-0009-R.png`

---

### [3] STAGE TRANSITION - Trophozoite→Ring confusion
**Image**: `1704282807-0019-R_G`
**Dataset**: MP-IDB Stages
**Metadata**:
- Total boxes: 41 (LARGE)
- Correct: 4 (9.8%)
- Incorrect: 37 (90.2%)
- Accuracy: 0.098 (Very Low)
- **Interpretation**: **SEVERE classification failure** on MP-IDB Stages. Model only correctly classified 4 out of 41 parasites (90% error rate). This demonstrates that while detection works well (YOLO11 detected 41 boxes correctly), stage classification is extremely challenging. Likely widespread Ring↔Trophozoite↔Schizont confusion due to morphological similarity during transitions.

**Source**: `results/optA_20251207_233941/experiments/experiment_mp_idb_stages/visualizations/pred_classification_efficientnet_b1_focal/1704282807-0019-R_G.png`

---

### [4] SPECIES CONFUSION - P. vivax/ovale mix (0% accuracy)
**Image**: `1701151546-0015-R_T`
**Dataset**: MP-IDB Species
**Metadata**:
- Total boxes: 37
- Correct: 0 (0%)
- Incorrect: 37 (100%)
- Accuracy: 0.000 (ALL WRONG)
- **Interpretation**: **COMPLETE classification failure**. All 37 parasites misclassified. This is the worst species classification case. Model likely confused all P. vivax as P. ovale (or vice versa) due to similar morphology. Shows critical limitation in species differentiation.

**Source**: `results/optA_20251207_233941/experiments/experiment_mp_idb_species/visualizations/pred_classification_efficientnet_b1_focal/1701151546-0015-R_T.png`

---

### [5] HEAVY ERROR - 0% accuracy on large image
**Image**: `1701151546-0015-R_T`
**Dataset**: MP-IDB Species
**Metadata**:
- Total boxes: 37
- Correct: 0 (0%)
- Incorrect: 37 (100%)
- **Interpretation**: Same as Species Confusion case above. Demonstrates complete systematic error - model consistently misclassified ALL parasites in the same direction, suggesting learned feature misalignment rather than random errors.

**Source**: `results/optA_20251207_233941/experiments/experiment_mp_idb_species/visualizations/pred_classification_efficientnet_b1_focal/1701151546-0015-R_T.png`

---

### [6] PERFECT CROWDED - 100% accuracy on 10+ parasites
**Image**: `1704282807-0020-R_T_S`
**Dataset**: MP-IDB Species
**Metadata**:
- Total boxes: 14 (Crowded)
- Correct: 14 (100%)
- Incorrect: 0
- Accuracy: 1.000 (PERFECT)
- **Interpretation**: **Excellent performance** - model correctly classified all 14 parasites in a crowded field. Shows model capability when parasites have clear morphological features and good image quality. This is the best classification case demonstrating model potential.

**Source**: `results/optA_20251207_233941/experiments/experiment_md_2019_stages/visualizations/pred_classification_efficientnet_b1_focal/1704282807-0020-R_T_S.png`

---

## NOTES

### Dataset-Specific Observations:
1. **MP-IDB Stages**: Classification accuracy extremely low (3-10%) - model struggles with stage differentiation
2. **MP-IDB Species**: Mixed performance (0-100%) - species confusion common
3. **MD-2019 Stages**: Better classification than MP-IDB Stages despite multi-patient variation
4. **IML Lifecycle**: Most balanced performance across all metrics

### Key Findings:
- **Detection (YOLO11)**: Generally robust with 310/672 perfect detections (46%)
- **Classification (EfficientNet-B1)**: Highly variable - struggles with:
  - Stage transitions (Ring↔Trophozoite↔Schizont)
  - Species differentiation (P. vivax/ovale confusion)
  - Crowded fields with morphological variation
- **Success factors**: Clear morphology, good image quality, well-separated parasites
- **Failure factors**: Atypical morphology, transition stages, species overlap, poor contrast

### Image Duplication:
Note that some images appear in multiple categories:
- `1701151546-0015-R_T`: Heavy FP (det), Mixed FP+FN (det), Species Confusion (cls), Heavy Error (cls)
- This is intentional - the same image can exemplify multiple error types

---

**Generated**: 2025-12-11
**Purpose**: Final case selection for publication figures based on comprehensive metadata analysis
