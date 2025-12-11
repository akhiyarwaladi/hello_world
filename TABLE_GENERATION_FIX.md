# Table Generation Fix - December 11, 2025

## Issue Identified

**User Report:** "tabelnya kenapa jadi format yang lama lagi ya :((("

### Problems Found:

1. **Wrong Format**: Tables were using old format (overall metrics) instead of KINETIK journal format (per-class metrics with 2-level headers)
2. **Wrong Data Source**: Tables script was hardcoded to use old experiment `optA_20251016_200330` instead of current experiment `optA_20251207_233941`
3. **Narrative Mismatch**: Report paragraphs describing table data would be incorrect due to wrong data source

## Root Cause

**Wrong Script Usage:**
- OLD (incorrect): `scripts/reporting/generate_professional_excel_tables_v2.py`
  - Uses overall metrics (mAP@50, Precision, Recall, F1)
  - Hardcoded to old experiment data
  - Wrong format for classification tables

**Correct Approach:**
- NEW: Dynamic scripts that read from specified experiment folder
- KINETIK format: Per-class metrics (Precision, F1 for each class)
- 2-level headers with class names and sample counts

## Solution Implemented

### New Scripts Created:

#### 1. `scripts/reporting/generate_classification_tables_from_experiment.py`
**Purpose:** Generate Tables 3-6 (Classification Performance) with KINETIK format

**Features:**
- Reads from specified experiment folder (default: optA_20251207_233941)
- Dynamically extracts data from `table9_focal_loss.csv` files
- Extracts actual training times from `results.txt` files
- Generates 2-level headers: Class names (row 1) + Metric names (row 2)
- Per-class metrics: Precision and F1 for each class
- Includes sample counts: `Class Name (n=X)`

**Usage:**
```bash
python scripts/reporting/generate_classification_tables_from_experiment.py --experiment optA_20251207_233941
```

**Outputs:**
- `Table3_iml_lifecycle.xlsx` - IML Lifecycle Classification
- `Table4_mp_idb_species.xlsx` - MP-IDB Species Classification
- `Table5_mp_idb_stages.xlsx` - MP-IDB Stages Classification
- `Table6_md_2019_stages.xlsx` - MD-2019 Stages Classification

#### 2. `scripts/reporting/generate_tables12_detection.py`
**Purpose:** Generate Tables 1-2 (Dataset Statistics & Detection Performance)

**Features:**
- Reads from consolidated analysis CSV files
- Generates professional Excel tables with styling
- Uses correct experiment folder

**Usage:**
```bash
python scripts/reporting/generate_tables12_detection.py --experiment optA_20251207_233941
```

**Outputs:**
- `Table1_Dataset_Statistics.xlsx` - Dataset Statistics
- `Table2_Detection_Performance.xlsx` - Detection Performance (all 4 datasets)

## KINETIK Format Specification

### Classification Tables (Tables 3-6)

**Structure:**
```
| Model          | Parameters (M) | Training Time (min) | Accuracy | Balanced Acc | [Class 1 (n=X)]      | [Class 2 (n=Y)]      | ...
|                |                |                     |          |              | Precision | F1       | Precision | F1       | ...
|----------------|----------------|---------------------|----------|--------------|-----------|----------|-----------|----------|
| DENSENET121    | 8.0            | 4.1                 | 0.9340   | 0.9379       | 0.9787    | 0.9583   | 0.9429    | 0.9565   | ...
| EFFICIENTNET-B0| 5.3            | 2.4                 | 0.8962   | 0.8830       | 0.9592    | 0.9592   | 0.9412    | 0.9412   | ...
```

**Key Features:**
- 2-level headers (merged cells for class names)
- Class names with sample counts: `gametocyte (n=49)`
- Per-class metrics: Precision and F1 (NOT overall F1)
- Decimal format: 0.0000 for metrics
- Training times from actual experiment results

## Verification

### Data Source Verification:
```bash
# Report references:
grep "Sumber Data Eksperimen" luaran/laporan_akhir/Laporan_Akhir_Penelitian.md
# Output: **Sumber Data Eksperimen**: optA_20251207_233941

# Tables now use:
# - Source: results/optA_20251207_233941/
# - Classification: table9_focal_loss.csv (per-class metrics)
# - Training times: results.txt (actual times)
```

### All 6 Tables Regenerated:
- ✅ Table1_Dataset_Statistics.xlsx (22:37)
- ✅ Table2_Detection_Performance.xlsx (22:37)
- ✅ Table3_iml_lifecycle.xlsx (22:36)
- ✅ Table4_mp_idb_species.xlsx (22:36)
- ✅ Table5_mp_idb_stages.xlsx (22:36)
- ✅ Table6_md_2019_stages.xlsx (22:36)

## Next Steps Required

### CRITICAL: Update Report Narratives

The user correctly identified: **"berarti narasi paragraf kita sekarang salah semua juga ya ultrathink"**

All narrative paragraphs in `luaran/laporan_akhir/Laporan_Akhir_Penelitian.md` that describe table data must be updated to match the new values from experiment `optA_20251207_233941`.

**Sections to Update:**
1. Dataset statistics descriptions (Table 1)
2. Detection performance descriptions (Table 2)
3. Classification performance descriptions (Tables 3-6)
4. All accuracy, precision, F1 values mentioned in text
5. Training time comparisons
6. Best model identifications

**Approach:**
1. Read new table data
2. Compare with current narrative text
3. Update all mismatched values
4. Verify consistency between tables and narrative

## Benefits of New Approach

1. **✅ Dynamic Data Loading**: No more hardcoded experiment paths
2. **✅ Correct Format**: KINETIK journal format with per-class metrics
3. **✅ Actual Training Times**: Read from experiment results, not estimated
4. **✅ Reusable**: Can regenerate tables for any experiment folder
5. **✅ Maintainable**: Clear separation of concerns (detection vs classification)

## Commands for Future Use

```bash
# Regenerate all tables from latest experiment
python scripts/reporting/generate_tables12_detection.py --experiment optA_20251207_233941
python scripts/reporting/generate_classification_tables_from_experiment.py --experiment optA_20251207_233941

# Or from a different experiment
python scripts/reporting/generate_tables12_detection.py --experiment optA_YYYYMMDD_HHMMSS
python scripts/reporting/generate_classification_tables_from_experiment.py --experiment optA_YYYYMMDD_HHMMSS
```

## Files Modified/Created

**New Scripts:**
- `scripts/reporting/generate_classification_tables_from_experiment.py`
- `scripts/reporting/generate_tables12_detection.py`

**Tables Regenerated:**
- `luaran/laporan_akhir/tables/Table1_Dataset_Statistics.xlsx`
- `luaran/laporan_akhir/tables/Table2_Detection_Performance.xlsx`
- `luaran/laporan_akhir/tables/Table3_iml_lifecycle.xlsx`
- `luaran/laporan_akhir/tables/Table4_mp_idb_species.xlsx`
- `luaran/laporan_akhir/tables/Table5_mp_idb_stages.xlsx`
- `luaran/laporan_akhir/tables/Table6_md_2019_stages.xlsx`

**Documentation:**
- `TABLE_GENERATION_FIX.md` (this file)

---

**Date:** December 11, 2025 22:37 WIB
**Status:** ✅ Tables Fixed - Correct Format & Correct Data
**Next:** Update narrative paragraphs in report
