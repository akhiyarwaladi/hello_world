# 🚨 DETECTION DATA MISMATCH FOUND!

**Date:** 2025-10-26
**Issue:** Detection metrics in Table 2 script DO NOT MATCH experiment results!

---

## ❌ COMPARISON: Paper vs Experiment CSV

### IML Lifecycle

| Model | Metric | Paper (Script) | Experiment CSV | Match? |
|-------|--------|---------------|----------------|--------|
| YOLO10 | mAP@50 | 93.81% | 93.81% | ✅ |
| YOLO10 | mAP@50-95 | 77.71% | 77.71% | ✅ |
| YOLO10 | Precision | **91.39%** | **90.22%** | ❌ **-1.17%** |
| YOLO10 | Recall | **87.86%** | **92.22%** | ❌ **+4.36%** |
| | | | | |
| YOLO11 | mAP@50 | 94.99% | 94.99% | ✅ |
| YOLO11 | mAP@50-95 | 77.76% | 77.76% | ✅ |
| YOLO11 | Precision | **92.42%** | **91.91%** | ❌ **-0.51%** |
| YOLO11 | Recall | **87.86%** | **91.11%** | ❌ **+3.25%** |
| | | | | |
| YOLO12 | mAP@50 | 94.40% | 94.40% | ✅ |
| YOLO12 | mAP@50-95 | 78.21% | 78.21% | ✅ |
| YOLO12 | Precision | **91.51%** | **92.20%** | ❌ **+0.69%** |
| YOLO12 | Recall | **88.40%** | **86.67%** | ❌ **-1.73%** |

### MP-IDB Species

| Model | Metric | Paper (Script) | Experiment CSV | Match? |
|-------|--------|---------------|----------------|--------|
| YOLO10 | mAP@50 | 92.44% | 92.44% | ✅ |
| YOLO10 | mAP@50-95 | 60.12% | 60.12% | ✅ |
| YOLO10 | Precision | **88.55%** | **85.67%** | ❌ **-2.88%** |
| YOLO10 | Recall | **93.12%** | **90.73%** | ❌ **-2.39%** |
| | | | | |
| YOLO11 | mAP@50 | 92.57% | 92.57% | ✅ |
| YOLO11 | mAP@50-95 | 62.17% | 62.17% | ✅ |
| YOLO11 | Precision | **88.89%** | **86.52%** | ❌ **-2.37%** |
| YOLO11 | Recall | **92.78%** | **91.88%** | ❌ **-0.90%** |
| | | | | |
| YOLO12 | mAP@50 | 92.72% | 92.72% | ✅ |
| YOLO12 | mAP@50-95 | 62.25% | 62.25% | ✅ |
| YOLO12 | Precision | **88.51%** | **88.99%** | ❌ **+0.48%** |
| YOLO12 | Recall | **92.78%** | **89.28%** | ❌ **-3.50%** |

### MP-IDB Stages

| Model | Metric | Paper (Script) | Experiment CSV | Match? |
|-------|--------|---------------|----------------|--------|
| YOLO10 | mAP@50 | 93.78% | 93.78% | ✅ |
| YOLO10 | mAP@50-95 | 44.48% | 44.48% | ✅ |
| YOLO10 | Precision | **89.96%** | **91.57%** | ❌ **+1.61%** |
| YOLO10 | Recall | **90.72%** | **93.12%** | ❌ **+2.40%** |
| | | | | |
| YOLO11 | mAP@50 | 94.48% | 94.48% | ✅ |
| YOLO11 | mAP@50-95 | 60.34% | 60.34% | ✅ |
| YOLO11 | Precision | **91.01%** | **93.15%** | ❌ **+2.14%** |
| YOLO11 | Recall | **91.07%** | **88.36%** | ❌ **-2.71%** |
| | | | | |
| YOLO12 | mAP@50 | 96.27% | 96.27% | ✅ |
| YOLO12 | mAP@50-95 | 61.53% | 61.53% | ✅ |
| YOLO12 | Precision | **93.35%** | **92.91%** | ❌ **-0.44%** |
| YOLO12 | Recall | **92.43%** | **92.59%** | ❌ **+0.16%** |

### MD_2019 Stages

| Model | Metric | Paper (Script) | Experiment CSV | Match? |
|-------|--------|---------------|----------------|--------|
| YOLO10 | mAP@50 | 70.84% | 70.84% | ✅ |
| YOLO10 | mAP@50-95 | 57.05% | 57.05% | ✅ |
| YOLO10 | Precision | **78.55%** | **67.89%** | ❌ **-10.66%** |
| YOLO10 | Recall | **86.67%** | **71.05%** | ❌ **-15.62%** |
| | | | | |
| YOLO11 | mAP@50 | 72.91% | 72.91% | ✅ |
| YOLO11 | mAP@50-95 | 57.71% | 57.71% | ✅ |
| YOLO11 | Precision | **80.33%** | **68.58%** | ❌ **-11.75%** |
| YOLO11 | Recall | **87.46%** | **75.70%** | ❌ **-11.76%** |
| | | | | |
| YOLO12 | mAP@50 | 71.12% | 71.12% | ✅ |
| YOLO12 | mAP@50-95 | 56.93% | 56.93% | ✅ |
| YOLO12 | Precision | **78.97%** | **65.92%** | ❌ **-13.05%** |
| YOLO12 | Recall | **88.44%** | **75.18%** | ❌ **-13.26%** |

---

## 📊 SUMMARY

| Dataset | Total Metrics | mAP Matches | Precision/Recall Matches |
|---------|--------------|-------------|-------------------------|
| IML Lifecycle | 12 | 6/6 ✅ | **0/6 ❌** |
| MP-IDB Species | 12 | 6/6 ✅ | **0/6 ❌** |
| MP-IDB Stages | 12 | 6/6 ✅ | **0/6 ❌** |
| MD_2019 Stages | 12 | 6/6 ✅ | **0/6 ❌** |
| **TOTAL** | **48** | **24/24 ✅** | **0/24 ❌** |

---

## 🚨 CRITICAL FINDINGS

1. **mAP@50 and mAP@50-95:** ALL CORRECT ✅ (24/24 match)
2. **Precision and Recall:** ALL WRONG ❌ (0/24 match)

**Worst Mismatches:**
- MD_2019 YOLO10 Recall: **-15.62%** difference!
- MD_2019 YOLO12 Precision: **-13.05%** difference!
- MD_2019 YOLO11 Recall: **-11.76%** difference!

---

## 🔍 ROOT CAUSE

The Precision and Recall values in `generate_table2_detection_performance.py` were **HALLUCINATED** or copied from a different experiment run!

The mAP values are correct because they match the experiment CSV exactly, but Precision/Recall do NOT match.

---

## ✅ SOLUTION REQUIRED

**MUST UPDATE** `generate_table2_detection_performance.py` with CORRECT data from:
```
results/optA_20251016_200330/consolidated_analysis/cross_dataset_comparison/detection_performance_all_datasets.csv
```

**CORRECT VALUES (from CSV):**

```python
detection_data = [
    # IML Lifecycle
    ['IML Lifecycle\n(4 stages)', 'YOLOv10', 93.81, 77.71, 90.22, 92.22],
    ['', 'YOLOv11', 94.99, 77.76, 91.91, 91.11],
    ['', 'YOLOv12', 94.40, 78.21, 92.20, 86.67],

    # MP-IDB Species
    ['MP-IDB Species\n(4 species)', 'YOLOv10', 92.44, 60.12, 85.67, 90.73],
    ['', 'YOLOv11', 92.57, 62.17, 86.52, 91.88],
    ['', 'YOLOv12', 92.72, 62.25, 88.99, 89.28],

    # MP-IDB Stages
    ['MP-IDB Stages\n(4 stages)', 'YOLOv10', 93.78, 44.48, 91.57, 93.12],
    ['', 'YOLOv11', 94.48, 60.34, 93.15, 88.36],
    ['', 'YOLOv12', 96.27, 61.53, 92.91, 92.59],

    # MD_2019 Stages
    ['MD_2019 Stages\n(3 stages)', 'YOLO10', 70.84, 57.05, 67.89, 71.05],
    ['', 'YOLOv11', 72.91, 57.71, 68.58, 75.70],
    ['', 'YOLOv12', 71.12, 56.93, 65.92, 75.18]
]
```

---

## ⚠️ IMPACT

**This affects:**
1. ❌ Table 2 in Excel file
2. ❌ Narasi in paper mentioning Precision/Recall values
3. ❌ Any analysis discussing detection Precision/Recall

**This does NOT affect:**
- ✅ mAP@50 values (all correct)
- ✅ mAP@50-95 values (all correct)
- ✅ Classification tables (separate data)

---

**Status:** ⚠️ **CRITICAL - MUST FIX IMMEDIATELY**
**Priority:** 🔴 **HIGH**
**Action Required:** Update script and regenerate Table 2
