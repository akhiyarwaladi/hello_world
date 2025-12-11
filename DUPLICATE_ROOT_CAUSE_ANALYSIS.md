# Root Cause Analysis: Table1 Duplicate Rows

**Date:** 2025-12-11
**Issue:** Table1_Dataset_Statistics.xlsx has 16 rows instead of 4 (4× duplication)

---

## 🔍 ROOT CAUSE IDENTIFIED

### The Bug Chain

1. **main_pipeline.py:1779-1783** - Analyzer called WITHOUT dataset parameter:
   ```python
   dataset_stats_cmd = [
       "python", "scripts/analysis/dataset_statistics_analyzer.py",
       "--output", str(dataset_stats_path)
   ]
   # ❌ MISSING: --dataset and --dataset-path arguments!
   ```

2. **dataset_statistics_analyzer.py:18-36** - Defaults to analyzing ALL datasets:
   ```python
   def __init__(self, specific_dataset=None, specific_dataset_path=None):
       # Default: analyze all known datasets
       self.datasets = {
           "iml_lifecycle": "data/processed/lifecycle",
           "md_2019_stages": "data/processed/md_2019_stages",
           "mp_idb_species": "data/processed/species",
           "mp_idb_stages": "data/processed/stages"
       }

       # Single dataset mode (ONLY if parameters provided)
       if specific_dataset and specific_dataset_path:
           self.datasets = {specific_dataset: specific_dataset_path}
   ```

3. **Result:** Each experiment folder gets a CSV with ALL 4 datasets:
   ```csv
   Dataset,Original_Train,Original_Val,Original_Test,...
   iml_lifecycle,372,128,126,...
   md_2019_stages,936,312,378,...
   mp_idb_species,250,84,84,...
   mp_idb_stages,250,84,84,...
   ```

   - `experiment_iml_lifecycle/analysis_dataset_statistics/dataset_statistics_summary.csv` → 4 rows
   - `experiment_mp_idb_species/analysis_dataset_statistics/dataset_statistics_summary.csv` → 4 rows
   - `experiment_mp_idb_stages/analysis_dataset_statistics/dataset_statistics_summary.csv` → 4 rows
   - `experiment_md_2019_stages/analysis_dataset_statistics/dataset_statistics_summary.csv` → 4 rows

4. **generate_comprehensive_consolidated_analysis.py:103-105** - Accumulates duplicates:
   ```python
   for folder in dataset_folders:  # 4 folders
       stats = load_dataset_statistics(folder)  # Returns 4 rows each time
       if stats:
           all_dataset_stats.extend(stats)  # 4 folders × 4 rows = 16 rows!
   ```

5. **Final result:** `dataset_statistics_all.csv` has 16 rows (4× duplication)

---

## ✅ THE FIX

**File:** `main_pipeline.py` line 1779-1783

**BEFORE (BROKEN):**
```python
dataset_stats_cmd = [
    "python", "scripts/analysis/dataset_statistics_analyzer.py",
    "--output", str(dataset_stats_path)
]
```

**AFTER (FIXED):**
```python
# Dataset mapping
dataset_path_map = {
    "iml_lifecycle": "data/processed/lifecycle",
    "mp_idb_species": "data/processed/species",
    "mp_idb_stages": "data/processed/stages",
    "md_2019_stages": "data/processed/md_2019_stages"
}

dataset_stats_cmd = [
    "python", "scripts/analysis/dataset_statistics_analyzer.py",
    "--output", str(dataset_stats_path),
    "--dataset", args.dataset,
    "--dataset-path", dataset_path_map.get(args.dataset, "")
]
```

---

## 📊 Expected Result After Fix

Each experiment folder will have a CSV with **ONLY its own dataset** (1 row):

- `experiment_iml_lifecycle/.../dataset_statistics_summary.csv` → 1 row (iml_lifecycle only)
- `experiment_mp_idb_species/.../dataset_statistics_summary.csv` → 1 row (mp_idb_species only)
- `experiment_mp_idb_stages/.../dataset_statistics_summary.csv` → 1 row (mp_idb_stages only)
- `experiment_md_2019_stages/.../dataset_statistics_summary.csv` → 1 row (md_2019_stages only)

**Consolidated result:** `dataset_statistics_all.csv` → 4 rows (NO duplicates)

---

## 🎯 Why Drop_Duplicates Was Wrong

User's concern was valid:
> "kenapa drop duplicates nya manual ya, berarti sumber masalahnya kenapa duplicate belum tau"

**Problems with drop_duplicates() approach:**
1. ❌ **Symptom treatment** - Masks the real bug
2. ❌ **Inefficient** - Generates 16 rows, then drops 12
3. ❌ **Fragile** - If stats differ slightly, drop_duplicates() might keep wrong row
4. ❌ **Misleading** - Future developers won't know why duplicates exist

**Proper fix:**
1. ✅ **Root cause fix** - Prevent duplicates from being generated
2. ✅ **Efficient** - Generate only 4 rows total
3. ✅ **Correct** - Each experiment analyzes only its own dataset
4. ✅ **Clear** - Code intent is obvious

---

## 🔧 Implementation Status

- [x] Root cause identified
- [x] Fix designed
- [x] Fix applied to main_pipeline.py (lines 1780-1794)
- [x] Removed drop_duplicates() workaround from generate_professional_excel_tables_v2.py (line 236-237)
- [ ] Test with regeneration (requires re-running pipeline analysis stage)
- [ ] Verify no duplicates

**Changes Made:**
1. `main_pipeline.py:1780-1794` - Added `--dataset` and `--dataset-path` parameters to analyzer call
2. `generate_professional_excel_tables_v2.py:236-237` - Removed drop_duplicates() and added comment explaining root cause fix

---

**Generated:** 2025-12-11
**Analysis by:** Claude Sonnet 4.5
