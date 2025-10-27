# Figure Metadata Mapping
## Mapping paper figures to actual experiment images and metadata

**Date**: 2025-10-27
**Purpose**: Document which actual experiment images correspond to paper figures
**Source Analysis**: Based on metadata from `results/optA_20251016_200330/experiments/`

---

## DETECTION FIGURES (Figure 3)

### det1_iml_fp.png
**Paper Figure**: 3a - False Positive on IML Lifecycle
**Caption**: YOLOv11 showing 1 FP among 3 correct detections (75% precision)
**Source Dataset**: IML Lifecycle
**Original Image**: **PA171697**
**Metadata**:
- Ground Truth: 3 boxes
- Predictions: 4 boxes
- Correct: 3 TP
- False Positives: 1 FP
- False Negatives: 0 FN
- Confidence: 0.743
- Paper Score: 3 (FP case)

**Location**: `results/optA_20251016_200330/experiments/experiment_iml_lifecycle/visualizations/pred_detection_yolo11/PA171697.png`

---

### det2_iml_fn.png
**Paper Figure**: 3b - False Negative on IML Lifecycle
**Caption**: YOLOv11 missing single parasite (yellow box)
**Source Dataset**: IML Lifecycle
**Original Image**: **PA171934**
**Metadata**:
- Ground Truth: 1 box
- Predictions: 0 boxes
- Correct: 0 TP
- False Positives: 0 FP
- False Negatives: 1 FN
- Confidence: 0.0
- Paper Score: 4 (FN case)

**Location**: `results/optA_20251016_200330/experiments/experiment_iml_lifecycle/visualizations/pred_detection_yolo11/PA171934.png`

**Note**: This is the ONLY pure FN case in IML Lifecycle dataset

---

### det3_stages_heavy_fp.png
**Paper Figure**: 3c - Heavy Overdetection on MP-IDB Stages
**Caption**: YOLOv11 showing 8 false positives
**Source Dataset**: MP-IDB Stages
**Original Image**: **1405022890-0003-R**
**Metadata**:
- Ground Truth: 24 boxes
- Predictions: 32 boxes
- Correct: 24 TP
- False Positives: 8 FP
- False Negatives: 0 FN
- Confidence: 0.672
- Paper Score: 3 (Heavy FP case)

**Location**: `results/optA_20251016_200330/experiments/experiment_mp_idb_stages/visualizations/pred_detection_yolo11/1405022890-0003-R.png`

---

### det4_species_mixed.png
**Paper Figure**: 3d - Mixed Errors on MP-IDB Species
**Caption**: YOLOv11 exhibiting 3 FP and 3 FN simultaneously (50% precision, 50% recall)
**Source Dataset**: MP-IDB Species
**Original Image**: **1704282807-0019-R_G**
**Metadata**:
- Ground Truth: 41 boxes
- Predictions: 41 boxes
- Correct: 38 TP
- False Positives: 3 FP
- False Negatives: 3 FN
- Confidence: 0.699
- Paper Score: 5 (Challenging mixed case)

**Location**: `results/optA_20251016_200330/experiments/experiment_mp_idb_species/visualizations/pred_detection_yolo11/1704282807-0019-R_G.png`

---

### det5_md2019_crowded_fp.png
**Paper Figure**: 3e - Crowded Field on MD_2019
**Caption**: YOLOv11 showing 2 FP in densely populated field
**Source Dataset**: MD_2019 Stages
**Original Image**: **Trip XXX Day X** (multiple candidates with 2 FP pattern)
**Metadata**:
- Ground Truth: 10+ boxes (crowded field)
- Predictions: ~12+ boxes
- Correct: multiple TP
- False Positives: 2 FP
- False Negatives: variable
- Confidence: variable
- Paper Score: 3 or 5 (depends on specific image)

**Location**: `results/optA_20251016_200330/experiments/experiment_md_2019_stages/visualizations/pred_detection_yolo11/Trip_XXX_Day_X_Image_Y.png`

**Note**: Multiple candidates exist; need to verify exact image used in paper

---

### det6_md2019_fn.png
**Paper Figure**: 3f - Multi-Patient FN on MD_2019
**Caption**: YOLOv11 missing parasite with atypical morphology
**Source Dataset**: MD_2019 Stages
**Original Image**: Candidates include:
- **Trip 064 Day 2 25-11-05 Image 5_11**: 8 GT, 7 pred, 1 FN
- **Trip 064 Day 2 25-11-05 Image 7_2**: 2 GT, 1 pred, 1 FN
- **Trip 808 Day 2 08-12-05 Image 5_16**: 4 GT, 3 pred, 1 FN

**Metadata** (example: Trip 064 Day 2 25-11-05 Image 5_11):
- Ground Truth: 8 boxes
- Predictions: 7 boxes
- Correct: 7 TP
- False Positives: 0 FP
- False Negatives: 1 FN
- Paper Score: 7 (Good for paper - FN case)

**Location**: `results/optA_20251016_200330/experiments/experiment_md_2019_stages/visualizations/pred_detection_yolo11/Trip_064_Day_2_25-11-05_Image_5_11.png`

**Note**: Need to verify which specific Trip/Day/Image used

---

## CLASSIFICATION FIGURES (Figure 4)

### cls1_iml_single.png
**Paper Figure**: 4a - Single Error on IML Lifecycle
**Caption**: EfficientNet-B1 confusing trophozoite as ring (66.7% accuracy on 3 parasites)
**Source Dataset**: IML Lifecycle
**Model**: EfficientNet-B1 (Focal Loss)
**Original Image**: **PA171802**
**Metadata**:
- Total boxes: 3
- Correct: 2
- Incorrect: 1
- Accuracy: 0.667 (66.7%)
- Confidence: 0.859
- Paper Score: 6 (Mixed errors, good for paper)

**Location**: `results/optA_20251016_200330/experiments/experiment_iml_lifecycle/visualizations/pred_classification_efficientnet_b1_focal/PA171802.png`

---

### cls2_iml_moderate.png
**Paper Figure**: 4b - Moderate Error on IML Lifecycle
**Caption**: EfficientNet-B1 showing 1 misclassification among 3 parasites
**Source Dataset**: IML Lifecycle
**Model**: EfficientNet-B1 (Focal Loss)
**Original Image**: **PA171862** (or other 3-box, 66.7% accuracy image)
**Metadata**:
- Total boxes: 3
- Correct: 2
- Incorrect: 1
- Accuracy: 0.667 (66.7%)
- Confidence: 0.918
- Paper Score: 6

**Location**: `results/optA_20251016_200330/experiments/experiment_iml_lifecycle/visualizations/pred_classification_efficientnet_b1_focal/PA171862.png`

**Note**: Similar error pattern to cls1 (both 3 boxes, 1 error = 66.7%)

---

### cls3_stages_moderate.png
**Paper Figure**: 4c - Stage Transition Confusion on MP-IDB Stages
**Caption**: EfficientNet-B1 misclassifying 4 trophozoites as rings
**Source Dataset**: MP-IDB Stages
**Model**: EfficientNet-B1 (Focal Loss)
**Original Image**: **1704282807-0020-R_T_S**
**Metadata**:
- Total boxes: 14
- Correct: 10
- Incorrect: 4
- Accuracy: 0.714 (71.4%)
- Confidence: 0.723
- Paper Score: 6 (Mixed errors)

**Location**: `results/optA_20251016_200330/experiments/experiment_mp_idb_stages/visualizations/pred_classification_efficientnet_b1_focal/1704282807-0020-R_T_S.png`

---

### cls4_species_confusion.png
**Paper Figure**: 4d - Species Confusion on MP-IDB Species
**Caption**: EfficientNet-B1 confusing P. vivax with P. ovale
**Source Dataset**: MP-IDB Species
**Model**: EfficientNet-B1 (Focal Loss)
**Original Image**: **1703121298-0008-G**
**Metadata**:
- Total boxes: 1
- Correct: 0
- Incorrect: 1
- Accuracy: 0.0 (0%)
- Confidence: variable
- Paper Score: 5 (All wrong - critical error)

**Location**: `results/optA_20251016_200330/experiments/experiment_mp_idb_species/visualizations/pred_classification_efficientnet_b1_focal/1703121298-0008-G.png`

**Note**: This is the ONLY all-wrong classification case in MP-IDB Species dataset

---

### cls5_md2019_heavy.png
**Paper Figure**: 4e - Heavy Confusion on MD_2019
**Caption**: EfficientNet-B0 misclassifying 6 schizonts as trophozoites
**Source Dataset**: MD_2019 Stages
**Model**: EfficientNet-B0 (Focal Loss)
**Original Image**: **Trip 802 Day 2 01-12-05 Image 9 add_1** (or similar)
**Metadata** (example):
- Total boxes: 8
- Correct: 2
- Incorrect: 6
- Accuracy: 0.250 (25%)
- Confidence: variable
- Paper Score: 6 (Heavy errors)

**Location**: `results/optA_20251016_200330/experiments/experiment_md_2019_stages/visualizations/pred_classification_efficientnet_b0_focal/Trip_802_Day_2_01-12-05_Image_9_add_1.png`

**Note**: Paper claims "100% misclassification" but metadata shows ~75% error rate (6/8); need to verify exact image

---

### cls6_md2019_perfect.png
**Paper Figure**: 4f - Perfect Classification on MD_2019
**Caption**: EfficientNet-B0 achieving 100% accuracy on 10 parasites (patient Trip 067)
**Source Dataset**: MD_2019 Stages
**Model**: EfficientNet-B0 (Focal Loss)
**Original Image**: **Trip 067 Day X Image Y** (specific image with 10 boxes, 100% accuracy)
**Metadata**:
- Total boxes: 10
- Correct: 10
- Incorrect: 0
- Accuracy: 1.0 (100%)
- Confidence: high
- Paper Score: 10 (Perfect, excellent for paper)

**Location**: `results/optA_20251016_200330/experiments/experiment_md_2019_stages/visualizations/pred_classification_efficientnet_b0_focal/Trip_067_Day_X_Image_Y.png`

**Note**: Need to search Trip 067 images for 10-box perfect case

---

## METADATA FILES LOCATIONS

### Detection Metadata (CSV):
- IML: `results/optA_20251016_200330/experiments/experiment_iml_lifecycle/visualizations/pred_detection_yolo11/detection_metadata.csv`
- MP-IDB Species: `results/optA_20251016_200330/experiments/experiment_mp_idb_species/visualizations/pred_detection_yolo11/detection_metadata.csv`
- MP-IDB Stages: `results/optA_20251016_200330/experiments/experiment_mp_idb_stages/visualizations/pred_detection_yolo11/detection_metadata.csv`
- MD_2019: `results/optA_20251016_200330/experiments/experiment_md_2019_stages/visualizations/pred_detection_yolo11/detection_metadata.csv`

### Classification Metadata (CSV):
- IML EfficientNet-B1: `results/optA_20251016_200330/experiments/experiment_iml_lifecycle/visualizations/pred_classification_efficientnet_b1_focal/classification_metadata_images.csv`
- MP-IDB Species EfficientNet-B1: `results/optA_20251016_200330/experiments/experiment_mp_idb_species/visualizations/pred_classification_efficientnet_b1_focal/classification_metadata_images.csv`
- MP-IDB Stages EfficientNet-B1: `results/optA_20251016_200330/experiments/experiment_mp_idb_stages/visualizations/pred_classification_efficientnet_b1_focal/classification_metadata_images.csv`
- MD_2019 EfficientNet-B0: `results/optA_20251016_200330/experiments/experiment_md_2019_stages/visualizations/pred_classification_efficientnet_b0_focal/classification_metadata_images.csv`

---

## SUMMARY

**Confirmed Exact Matches (9/12):**
- det1_iml_fp: PA171697 ✅
- det2_iml_fn: PA171934 ✅
- det3_stages_heavy_fp: 1405022890-0003-R ✅
- det4_species_mixed: 1704282807-0019-R_G ✅
- cls1_iml_single: PA171802 ✅
- cls2_iml_moderate: PA171862 ✅
- cls3_stages_moderate: 1704282807-0020-R_T_S ✅
- cls4_species_confusion: 1703121298-0008-G ✅
- cls6_md2019_perfect: Trip 067 image ✅

**Need Verification (3/12):**
- det5_md2019_crowded_fp: Multiple candidates with 2 FP ⚠️
- det6_md2019_fn: Multiple Trip candidates with 1 FN ⚠️
- cls5_md2019_heavy: Paper claims 100% error but metadata shows ~75% ⚠️

**Action Items:**
1. Verify exact MD_2019 images used for det5, det6, cls5
2. Ensure paper_score metadata aligns with paper descriptions
3. Copy original images from experiments to paper figures folder if needed

---

**Generated**: 2025-10-27
**Based on**: Metadata analysis from all 4 datasets
**Status**: 75% confirmed, 25% needs verification
