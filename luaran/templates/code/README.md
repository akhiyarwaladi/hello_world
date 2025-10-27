# 📊 Table Generation Scripts

This folder contains **FINAL and ACTIVE** scripts to generate all 7 tables used in the KINETIK journal paper.

---

## 📁 Files in This Folder

| Script | Output | Description |
|--------|--------|-------------|
| `generate_table1_dataset_augmentation.py` | `Table1_Dataset_Augmentation.xlsx` | Dataset statistics with train/val/test split and augmentation impact |
| `generate_table2_detection_performance.py` | `Table2_Detection_Performance.xlsx` | YOLO detection performance (v10/v11/v12) across 4 datasets |
| `generate_tables3456_classification.py` | `Table3-6_Classification.xlsx` | Classification performance on 4 datasets (IML, MP-IDB Species, MP-IDB Stages, MD_2019) |
| `generate_table7_sota_comparison.py` | `Table7_Comparison_SOTA.xlsx` | Comparison with 5 state-of-the-art papers (2022-2024) |
| `README.md` | - | This file (documentation) |

---

## 🚀 How to Use

### Prerequisites
```bash
pip install openpyxl pandas
```

### Generate Individual Tables

**Table 1: Dataset Augmentation**
```bash
cd luaran/templates/code
python generate_table1_dataset_augmentation.py
```
→ Output: `../tables/Table1_Dataset_Augmentation.xlsx`

**Table 2: Detection Performance**
```bash
cd luaran/templates/code
python generate_table2_detection_performance.py
```
→ Output: `../tables/Table2_Detection_Performance.xlsx`

**Tables 3-6: Classification Performance**
```bash
cd luaran/templates/code
python generate_tables3456_classification.py
```
→ Output:
- `../tables/Table3_IML_Classification.xlsx`
- `../tables/Table4_Species_Classification.xlsx`
- `../tables/Table5_Stages_Classification.xlsx`
- `../tables/Table6_MD2019_Classification.xlsx`

**Table 7: SOTA Comparison**
```bash
cd luaran/templates/code
python generate_table7_sota_comparison.py
```
→ Output: `../tables/Table7_Comparison_SOTA.xlsx`

### Generate All Tables at Once
```bash
cd luaran/templates/code
python generate_table1_dataset_augmentation.py
python generate_table2_detection_performance.py
python generate_tables3456_classification.py
python generate_table7_sota_comparison.py
```

---

## 📊 Table Descriptions

### Table 1: Dataset Statistics and Augmentation
- **Purpose:** Show original 60/20/20 split and augmentation impact
- **Structure:** Multi-level headers with 13 columns
- **Key Info:**
  - Detection training: 4.4× augmentation on train only
  - Classification training: 3.5× augmentation on train only
  - Val/Test: NEVER augmented

### Table 2: YOLO Detection Performance
- **Purpose:** Compare YOLOv10/v11/v12 detection across 4 datasets
- **Metrics:** mAP@50, mAP@50-95, Precision, Recall
- **Key Results:**
  - IML Lifecycle: 94.99% mAP@50 (YOLO11)
  - MP-IDB Stages: 96.27% mAP@50 (YOLO12)
  - MD_2019: 72.91% mAP@50 (YOLO11)

### Tables 3-6: Classification Performance
- **Purpose:** Compare 6 CNN architectures (DenseNet121, EfficientNet-B0/B1/B2, ResNet50/101)
- **Datasets:** IML Lifecycle, MP-IDB Species, MP-IDB Stages, MD_2019
- **Metrics:** Accuracy, Balanced Accuracy, Per-class Precision & F1
- **Key Results:**
  - IML: 91.51% (EfficientNet-B1)
  - MP-IDB Species: 98.28% (EfficientNet-B1)
  - MP-IDB Stages: 96.13% (ResNet50)
  - MD_2019: 86.45% (EfficientNet-B0)

### Table 7: SOTA Comparison
- **Purpose:** Compare with 5 recent papers (2022-2024) using SAME datasets
- **Papers:** Arshad 2022, Loddo 2022, Zedda 2023, Zedda 2022, Sukumarran 2024
- **Focus:** Fair apples-to-apples comparison (IML Lifecycle and MP-IDB only)

---

## 📝 Data Sources

All scripts use data from:
```
results/optA_20251016_200330/
├── experiments/
│   ├── experiment_iml_lifecycle/
│   ├── experiment_mp_idb_species/
│   ├── experiment_mp_idb_stages/
│   └── experiment_md_2019_stages/
└── consolidated_analysis/
    └── cross_dataset_comparison/
```

---

## ⚙️ Script Modifications

### To Update Data

**Table 1 (Dataset Augmentation):**
- Edit lines 88-107 in `generate_table1_dataset_augmentation.py`
- Update `data` array with new train/val/test numbers
- Verify against `dataset_statistics_detailed.csv`

**Table 2 (Detection):**
- Edit lines 22-44 in `generate_table2_detection_performance.py`
- Update `detection_data` array with new metrics
- Extract from experiment `detection_results.json` files

**Tables 3-6 (Classification):**
- Edit lines 145-303 in `generate_tables3456_classification.py`
- Update each dataset's `models_data` dictionary
- Extract from experiment `classification_results.json` files

**Table 7 (SOTA):**
- Edit lines 34-93 in `generate_table7_sota_comparison.py`
- Update `data` array with new comparison papers
- Add/remove rows as needed

---

## 🎨 Styling

All tables use consistent styling:
- **Header:** Dark blue (#366092), white bold text
- **Subheader:** Light blue (#DCE6F1), bold text
- **Data:** Black text, centered alignment
- **Borders:** Thin borders on all cells
- **Font:** Size 10-11, Arial/Calibri

---

## ✅ Verification

After generating tables, verify:
1. ✅ File name matches paper reference (e.g., `Table2_Detection_Performance.xlsx`)
2. ✅ Data matches experiment results
3. ✅ Caption in paper describes table accurately
4. ✅ Narasi (text before/after table) consistent with data
5. ✅ All numbers formatted correctly (2 decimal places for percentages)

---

## 📚 Related Files

**In Paper:**
- `luaran/templates/KINETIK_PAPER_DRAFT_UPDATED_2025.md`
  - Table references: Lines 87, 119, 128, 133, 138, 143, 222
  - Captions and narasi for all 7 tables

**Verification:**
- `luaran/templates/tables/TABLE_VERIFICATION_COMPLETE.md`
  - Complete verification of all tables, captions, and narasi

---

## 🔧 Troubleshooting

**Problem:** ModuleNotFoundError: No module named 'openpyxl'
```bash
pip install openpyxl
```

**Problem:** FileNotFoundError: [Errno 2] No such file or directory
```bash
# Make sure to run from luaran/templates/code/ folder
cd luaran/templates/code
python generate_tableX_....py
```

**Problem:** Output path '../tables/TableX.xlsx' doesn't work
```bash
# Check that tables folder exists
mkdir -p ../tables
```

---

## 📅 Last Updated

**Date:** 2025-10-26
**Status:** ✅ All scripts verified and working
**Paper Version:** KINETIK_PAPER_DRAFT_UPDATED_2025.md

---

## 🎯 Quick Generate All

To regenerate ALL 7 tables quickly:
```bash
cd luaran/templates/code
for script in generate_*.py; do python "$script"; done
```

Or on Windows:
```cmd
cd luaran\templates\code
for %f in (generate_*.py) do python %f
```

---

**Note:** These scripts generate the FINAL tables used in the journal paper. Any modifications should be documented in the paper's narasi sections to maintain consistency.
