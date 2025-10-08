# 📁 Luaran Penelitian - Malaria Detection

## 📄 Dokumen Utama

### 1. **JICEST_Paper.docx** (48 KB)
Paper penelitian lengkap untuk submission jurnal SINTA 3 (JICEST/JISEBI)

**Isi:**
- Bilingual abstracts (English + Indonesian)
- Complete IMRaD structure
- 24 verified references (2016-2025)
- Ready for SINTA 3 journal submission

### 2. **Laporan_Kemajuan_Malaria_Detection.docx** (47 KB)
Laporan kemajuan penelitian untuk BISMA

**Isi:**
- Section C-H lengkap dengan Lampiran
- Timeline 12 bulan (Section E)
- 24 referensi terverifikasi
- Ready for BISMA submission

### 3. **IMPROVEMENTS_SUMMARY.md** (8 KB)
Dokumentasi lengkap semua enhancement yang diterapkan

---

## 📊 Folder Pendukung

### 📁 **figures/**
25 visualisasi publication-quality (300 DPI)

**Main figures (10):**
1. Detection Performance Comparison
2. Classification Accuracy Heatmap
3. Species F1 Comparison
4. Stages F1 Comparison
5. Class Imbalance Distribution
6. Model Efficiency Analysis
7. Precision-Recall Tradeoff
8. Confusion Matrices
9. Training Curves
10. Pipeline Architecture

**Supplementary figures (15):**
- S1: Data Augmentation Examples
- S2-S3: Confusion Matrices
- S4: Training Curves
- S5-S6: Detection Ground Truth Bounding Boxes
- S7: Detection PR Curve
- S8-S9: Detection Prediction Bounding Boxes
- S10: Detection Training Results
- S11: Grad-CAM Species Composite
- S12: Grad-CAM Stages Composite
- S13: Grad-CAM Explanation
- S14-S15: Training Curves (alternatives)

### 📁 **tables/**
6 tabel statistik terstruktur (CSV format)

1. Table1_Detection_Performance.csv
2. Table2_Species_Classification.csv
3. Table3_Stages_Classification.csv
4. Table4_Species_F1_Scores.csv
5. Table5_Stages_F1_Scores.csv
6. Table6_Dataset_Statistics.csv

---

## ✅ Status

**JICEST Paper**: ✅ Ready for submission
**Laporan Kemajuan**: ✅ Ready for BISMA
**Visualizations**: ✅ 25/25 complete
**Tables**: ✅ 6/6 complete

---

## 🎯 Hasil Penelitian

### Dataset
- MP-IDB Species & Stages (209 images each)
- Split: 146/42/21 (train/val/test)
- Augmentation: 4.4× detection, 3.5× classification

### Performance
- **Detection**: 93.09% mAP@50 (YOLOv11)
- **Species Classification**: 98.8% accuracy (DenseNet121, EfficientNet-B1)
- **Stages Classification**: 94.31% accuracy (EfficientNet-B0)
- **Inference Time**: <100ms per image (RTX 3060)

### Efficiency
- Storage reduction: 70% (45GB → 14GB)
- Training time reduction: 60% (450h → 180h)
- Shared classification architecture

---

*Last updated: 2025-10-08*
*All documents ready for submission*
