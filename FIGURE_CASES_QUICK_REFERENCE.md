# Figure Cases Quick Reference
## Experiment: optA_20251207_233941

Quick lookup table for all 12 figure cases identified from the experiment metadata.

---

## FIGURE 5: Detection Error Cases (YOLO11)

| Panel | Image Name | Dataset | GT | Pred | FP | FN | Error Type |
|-------|------------|---------|----|----|----|----|------------|
| **(a)** | `PA171785` | IML Lifecycle | 1 | 4 | 3 | 0 | FP only (overdetection) |
| **(b)** | `PA171699` | IML Lifecycle | 2 | 0 | 0 | 2 | FN only (complete miss) |
| **(c)** | `1307210661-0007-R` | MP-IDB Stages | 31 | 36 | 6 | 1 | Overdetection (crowded) |
| **(d)** | `1701151546-0015-R_T` | MP-IDB Species | 37 | 39 | 11 | 9 | Mixed errors (heavy) |
| **(e)** | `Trip 073 Day 2 01-12-05 Image 1_8` | MD-2019 Stages | 18 | 17 | 3 | 4 | Crowded mixed |
| **(f)** | `Trip 073 Day 2 01-12-05 Image 1_10` | MD-2019 Stages | 14 | 10 | 0 | 4 | FN (atypical) |

**File Path Pattern:**
```
results\optA_20251207_233941\experiments\experiment_[dataset]\visualizations\pred_detection_yolo11\[image_name].png
```

**Datasets:**
- IML Lifecycle → `experiment_iml_lifecycle`
- MP-IDB Species → `experiment_mp_idb_species`
- MP-IDB Stages → `experiment_mp_idb_stages`
- MD-2019 Stages → `experiment_md_2019_stages`

---

## FIGURE 6: Classification Error Cases

| Panel | Image Name | Dataset | Boxes | Correct | Incorrect | Acc | Model | Error Type |
|-------|------------|---------|-------|---------|-----------|-----|-------|------------|
| **(a)** | `PA171852` | IML Lifecycle | 3 | 2 | 1 | 66.7% | EfficientNet-B0 | Single error |
| **(b)** | `PA171802` | IML Lifecycle | 3 | 2 | 1 | 66.7% | EfficientNet-B0 | Moderate error |
| **(c)** | `1704282807-0019-R_G` | MP-IDB Stages | 41 | 1 | 40 | 2.4% | EfficientNet-B0 | Catastrophic |
| **(d)** | `1305121398-0003-R` | MP-IDB Species | 1 | 0 | 1 | 0.0% | EfficientNet-B0 | Complete fail |
| **(e)** | `Trip 073 Day 2 01-12-05 Image 1_8` | MD-2019 Stages | 18 | 3 | 15 | 16.7% | EfficientNet-B0 | Heavy error |
| **(f)** | `Trip 065 Day 2 01-12-05 Image 7_9` | MD-2019 Stages | 8 | 8 | 0 | 100.0% | ResNet101 | Perfect |

**File Path Patterns:**

*For panels (a)-(e) using EfficientNet-B0:*
```
results\optA_20251207_233941\experiments\experiment_[dataset]\visualizations\pred_classification_efficientnet_b0_focal\[image_name].png
```

*For panel (f) using ResNet101:*
```
results\optA_20251207_233941\experiments\experiment_md_2019_stages\visualizations\pred_classification_resnet101_focal\Trip 065 Day 2 01-12-05 Image 7_9.png
```

---

## Copy-Paste File Paths

### Figure 5 (Detection)

```
# (a) IML FP only
results\optA_20251207_233941\experiments\experiment_iml_lifecycle\visualizations\pred_detection_yolo11\PA171785.png

# (b) IML FN only
results\optA_20251207_233941\experiments\experiment_iml_lifecycle\visualizations\pred_detection_yolo11\PA171699.png

# (c) MP-IDB Stages Overdetection
results\optA_20251207_233941\experiments\experiment_mp_idb_stages\visualizations\pred_detection_yolo11\1307210661-0007-R.png

# (d) MP-IDB Species Mixed
results\optA_20251207_233941\experiments\experiment_mp_idb_species\visualizations\pred_detection_yolo11\1701151546-0015-R_T.png

# (e) MD-2019 Crowded Mixed
results\optA_20251207_233941\experiments\experiment_md_2019_stages\visualizations\pred_detection_yolo11\Trip 073 Day 2 01-12-05 Image 1_8.png

# (f) MD-2019 FN Atypical
results\optA_20251207_233941\experiments\experiment_md_2019_stages\visualizations\pred_detection_yolo11\Trip 073 Day 2 01-12-05 Image 1_10.png
```

### Figure 6 (Classification)

```
# (a) IML Single Error
results\optA_20251207_233941\experiments\experiment_iml_lifecycle\visualizations\pred_classification_efficientnet_b0_focal\PA171852.png

# (b) IML Moderate Error
results\optA_20251207_233941\experiments\experiment_iml_lifecycle\visualizations\pred_classification_efficientnet_b0_focal\PA171802.png

# (c) MP-IDB Stages Catastrophic
results\optA_20251207_233941\experiments\experiment_mp_idb_stages\visualizations\pred_classification_efficientnet_b0_focal\1704282807-0019-R_G.png

# (d) MP-IDB Species Complete Fail
results\optA_20251207_233941\experiments\experiment_mp_idb_species\visualizations\pred_classification_efficientnet_b0_focal\1305121398-0003-R.png

# (e) MD-2019 Heavy Error
results\optA_20251207_233941\experiments\experiment_md_2019_stages\visualizations\pred_classification_efficientnet_b0_focal\Trip 073 Day 2 01-12-05 Image 1_8.png

# (f) MD-2019 Perfect
results\optA_20251207_233941\experiments\experiment_md_2019_stages\visualizations\pred_classification_resnet101_focal\Trip 065 Day 2 01-12-05 Image 7_9.png
```

---

## Notes

1. **All files verified:** ✓ All 12 PNG files exist and contain annotated visualizations
2. **Detection model:** YOLO11 Medium (consistent across all Figure 5 panels)
3. **Classification models:**
   - EfficientNet-B0 Focal Loss (panels a-e in Figure 6)
   - ResNet101 Focal Loss (panel f in Figure 6 - only model achieving 100% on 8+ boxes)
4. **Image quality:** All PNGs are high-resolution with bounding boxes and class labels
5. **Metadata source:** CSVs in `visualizations/*/detection_metadata.csv` and `classification_metadata_images.csv`

---

## Statistical Highlights

**Figure 5 Error Patterns:**
- Pure overdetection (FP only): 1 case (PA171785, 3 FPs)
- Pure underdetection (FN only): 2 cases (PA171699, Trip 073...1_10)
- Mixed errors: 3 cases (ranging from 7-20 total errors in crowded scenes)
- Most challenging: 1701151546-0015-R_T (37 GT boxes, 20 total errors)

**Figure 6 Error Patterns:**
- Perfect classification: 1 case (8 boxes, 100% accuracy, ResNet101)
- Near-perfect (66.7%): 2 cases (single error in 3-box images)
- Heavy errors (2-17%): 3 cases (catastrophic failures in crowded/ambiguous scenes)
- Interesting case: 1305121398-0003-R (100% wrong but 81.1% confidence - species confusion)

---

**Generated:** 2026-02-01
**Experiment:** optA_20251207_233941
**Total Cases:** 12 (6 detection + 6 classification)
**All Files Verified:** ✓ Yes
