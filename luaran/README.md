# 📂 LUARAN - RESEARCH OUTPUTS

**Project**: Malaria Detection & Classification using YOLO + CNN
**Date**: October 2025
**Experiment**: optA_20251007_134458
**Status**: ✅ Complete

---

## 📄 DOKUMEN UTAMA (2 files)

### 1. **Laporan Kemajuan BISMA**
- **File**: `Laporan_Kemajuan_FINAL_WITH_TABLES.md` (87 KB)
- **Isi**: Progress report lengkap (Sections C-H), 9 tables, 24 references
- **Status**: ✅ Ready for submission

### 2. **JICEST Paper**
- **File**: `JICEST_Paper_FINAL_WITH_TABLES.md` (35 KB)
- **Isi**: Full paper IMRaD structure, bilingual abstracts, 4 tables, 10 figures
- **Status**: ✅ Ready for submission

---

## 📖 PANDUAN (1 file)

### 3. **Guide Lengkap**
- **File**: `GUIDE_ULTRATHINK.md` (17 KB)
- **Isi**: Comprehensive usage guide untuk semua materials
- **Topics**: Markdown editing, table/figure insertion, DOCX generation, troubleshooting

---

## 📊 DATA TABLES (CSV format)

**Folder**: `tables/`

4 CSV files dengan experimental data:
1. `Table1_Detection_Performance_UPDATED.csv` - 9 YOLO models
2. `Table2_Classification_Performance_UPDATED.csv` - 18 CNN models
3. `Table3_Dataset_Statistics_UPDATED.csv` - 3 datasets
4. `Table4_Minority_Class_Performance_UPDATED.csv` - 12 minority classes

**Source**: Extracted from experiment optA_20251007_134458 (including table9 data)

---

## 🖼️ FIGURES (PNG 300 DPI)

**Folder**: `figures/`

### Main Figures (10 files):
- `figure1-10.png` - Sample images, confusion matrices, detection examples, architecture diagram, PR curves, model comparison, training curves, Grad-CAM

### Supplementary Figures:
- `supplementary/` - 15 files (S1-S15)

---

## 🚀 QUICK START

### **Edit Markdown:**
```bash
code luaran/Laporan_Kemajuan_FINAL_WITH_TABLES.md
code luaran/JICEST_Paper_FINAL_WITH_TABLES.md
```

### **Generate DOCX (Optional):**
```bash
python generate_docx_from_markdown.py
```

### **Insert Tables/Figures:**
See `GUIDE_ULTRATHINK.md` for detailed instructions

---

## 📈 KEY METRICS

- **Datasets**: 731 images (3 datasets: IML Lifecycle 313, MP-IDB Species/Stages 209 each)
- **Detection**: 95.71% mAP@50 (YOLOv12 on IML Lifecycle)
- **Classification**: 98.8% accuracy (EfficientNet-B1 on MP-IDB Species)
- **Efficiency**: 70% storage reduction, 60% training time reduction
- **Inference**: <25ms/image (40+ FPS on RTX 3060)

---

## ✅ FILE SUMMARY

```
luaran/
├── Laporan_Kemajuan_FINAL_WITH_TABLES.md  (87 KB) ← LAPORAN
├── JICEST_Paper_FINAL_WITH_TABLES.md      (35 KB) ← PAPER
├── GUIDE_ULTRATHINK.md                    (17 KB) ← PANDUAN
├── README.md                               (this)  ← OVERVIEW
├── tables/                                 (4 CSV files)
└── figures/                                (10 main + 15 supp)
```

**Total**: 4 markdown files (clean & organized!)

---

*Last updated: 2025-10-08*
*Status: ✅ Complete - Ready for submission*
