# PANDUAN LENGKAP ULTRATHINK - MALARIA DETECTION RESEARCH

**Date**: 2025-10-08
**Status**: ✅ Complete
**Experiment Source**: optA_20251007_134458

---

## 📚 DAFTAR ISI

1. [Dokumen Utama](#dokumen-utama)
2. [Struktur Folder](#struktur-folder)
3. [CSV Tables](#csv-tables)
4. [Figures/Gambar](#figuresgambar)
5. [Cara Menggunakan Markdown](#cara-menggunakan-markdown)
6. [Cara Insert Tables](#cara-insert-tables)
7. [Cara Insert Figures](#cara-insert-figures)
8. [Generate DOCX](#generate-docx-optional)
9. [Key Metrics Summary](#key-metrics-summary)
10. [Troubleshooting](#troubleshooting)

---

## 📄 DOKUMEN UTAMA

### **File yang Digunakan:**

**1. Laporan Kemajuan BISMA**
- **File**: `Laporan_Kemajuan_FINAL_WITH_TABLES.md` (87 KB)
- **Isi**: Sections C-H lengkap, 9 tabel, 24 references
- **Status**: ✅ Complete, siap digunakan

**2. JICEST Paper**
- **File**: `JICEST_Paper_FINAL_WITH_TABLES.md` (35 KB)
- **Isi**: IMRaD structure, bilingual abstracts, 4 tables, 10 figures
- **Status**: ✅ Complete, siap digunakan

**3. Panduan Ini**
- **File**: `GUIDE_ULTRATHINK.md` (file ini)
- **Isi**: Panduan lengkap penggunaan semua materials

**4. Overview Folder**
- **File**: `README.md` (2.6 KB)
- **Isi**: Quick overview folder luaran

---

## 📁 STRUKTUR FOLDER

```
luaran/
│
├── Laporan_Kemajuan_FINAL_WITH_TABLES.md    ← LAPORAN KEMAJUAN (87 KB)
├── JICEST_Paper_FINAL_WITH_TABLES.md        ← JICEST PAPER (35 KB)
├── GUIDE_ULTRATHINK.md                      ← PANDUAN INI
├── README.md                                ← Overview singkat
│
├── tables/                                  ← CSV TABLES
│   ├── Table1_Detection_Performance_UPDATED.csv
│   ├── Table2_Classification_Performance_UPDATED.csv
│   ├── Table3_Dataset_Statistics_UPDATED.csv
│   └── Table4_Minority_Class_Performance_UPDATED.csv
│
└── figures/                                 ← GAMBAR 300 DPI
    ├── figure1_sample_images.png
    ├── figure2_confusion_matrix_species.png
    ├── figure3_confusion_matrix_stages.png
    ├── figure4_detection_examples_iml.png
    ├── figure5_detection_examples_mpidb.png
    ├── figure6_architecture_diagram.png
    ├── figure7_pr_curves_iml.png
    ├── figure8_model_comparison_bar_chart.png
    ├── figure9_training_curves_classification.png
    ├── figure10_gradcam_visualizations.png
    └── supplementary/                       ← 15 supplementary figures
```

---

## 📊 CSV TABLES

**Lokasi**: `luaran/tables/`

### Table 1: Detection Performance
- **File**: `Table1_Detection_Performance_UPDATED.csv`
- **Isi**: 9 YOLO models (YOLOv10/11/12 × 3 datasets)
- **Columns**: Dataset, Model, Epochs, mAP@50, mAP@50-95, Precision, Recall, Training_Time_Hours
- **Best**: YOLOv12 95.71% mAP@50 (IML Lifecycle)

### Table 2: Classification Performance
- **File**: `Table2_Classification_Performance_UPDATED.csv`
- **Isi**: 18 CNN models (6 architectures × 3 datasets)
- **Columns**: Dataset, Model, Loss, Epochs, Accuracy, Balanced_Accuracy, Training_Time_Hours
- **Best**: EfficientNet-B1 98.8% accuracy (MP-IDB Species)

### Table 3: Dataset Statistics
- **File**: `Table3_Dataset_Statistics_UPDATED.csv`
- **Isi**: 3 datasets dengan augmentation details
- **Columns**: Dataset, Total_Images, Train, Val, Test, Classes, Detection_Aug_Train, Classification_Aug_Train, Det_Multiplier, Cls_Multiplier
- **Total**: 731 images (313 + 209 + 209)

### Table 4: Minority Class Performance
- **File**: `Table4_Minority_Class_Performance_UPDATED.csv`
- **Isi**: 12 minority classes (<20 samples)
- **Columns**: Dataset, Class, Support, Best_Model, Precision, Recall, F1_Score, Challenge_Level
- **Key**: Kelas dengan <10 samples F1=51-77%

---

## 🖼️ FIGURES/GAMBAR

**Lokasi**: `luaran/figures/`

### Main Figures (10 files):

1. **figure1_sample_images.png** - Sample microscopy images dari 3 datasets
2. **figure2_confusion_matrix_species.png** - Confusion matrix MP-IDB Species
3. **figure3_confusion_matrix_stages.png** - Confusion matrix MP-IDB Stages
4. **figure4_detection_examples_iml.png** - YOLO detection pada IML Lifecycle
5. **figure5_detection_examples_mpidb.png** - YOLO detection pada MP-IDB
6. **figure6_architecture_diagram.png** - Option A architecture flowchart
7. **figure7_pr_curves_iml.png** - Precision-Recall curves IML Lifecycle
8. **figure8_model_comparison_bar_chart.png** - Bar chart 6 CNN models
9. **figure9_training_curves_classification.png** - Training curves best models
10. **figure10_gradcam_visualizations.png** - Grad-CAM attention maps

### Supplementary Figures (15 files):
**Lokasi**: `luaran/figures/supplementary/`
- S1-S3: Additional confusion matrices
- S4-S7: Detection metrics (IoU, mAP curves)
- S8-S10: Per-class classification metrics
- S11-S13: Grad-CAM for all 6 CNN models
- S14-S15: Augmentation examples

**Format**: Semua PNG 300 DPI (publication quality)

---

## 📝 CARA MENGGUNAKAN MARKDOWN

### **Opsi 1: Edit Langsung di Text Editor**

```bash
# VSCode (recommended)
code luaran/Laporan_Kemajuan_FINAL_WITH_TABLES.md

# Atau text editor lain
nano luaran/JICEST_Paper_FINAL_WITH_TABLES.md
```

**Preview di VSCode**: Ctrl+Shift+V

### **Opsi 2: View di GitHub**
- Push ke GitHub (sudah done ✅)
- Buka repository di browser
- Navigate ke `luaran/` folder
- GitHub akan render markdown otomatis

### **Opsi 3: Convert ke PDF (tanpa Word)**
```bash
# Install pandoc terlebih dahulu
pandoc luaran/Laporan_Kemajuan_FINAL_WITH_TABLES.md -o Laporan.pdf

# Dengan styling
pandoc luaran/JICEST_Paper_FINAL_WITH_TABLES.md -o JICEST.pdf --pdf-engine=xelatex
```

---

## 📊 CARA INSERT TABLES

### **Method 1: Copy CSV → Paste ke Word**

**Step-by-step:**

1. **Buka CSV di Excel/LibreOffice**
   ```bash
   excel luaran/tables/Table1_Detection_Performance_UPDATED.csv
   ```

2. **Select All → Copy**
   - Ctrl+A (select all)
   - Ctrl+C (copy)

3. **Paste ke Word Document**
   - Di Word, posisikan cursor di lokasi table
   - Ctrl+V (paste)
   - Pilih "Keep Source Formatting"

4. **Format Table di Word**
   ```
   - Table Tools → Design → "Grid Table 4 - Accent 1"
   - Layout → AutoFit → AutoFit to Contents
   - Bold header row (first row)
   - Center-align kolom numeric
   ```

### **Method 2: Import CSV di Word (Automatic)**

**Step-by-step:**

1. **Di Word: Insert → Table → Convert Text to Table**
2. **Browse ke CSV file** (pilih Table1_Detection_Performance_UPDATED.csv)
3. **Set delimiter**: Comma (,)
4. **Click OK** → Table akan ter-generate otomatis
5. **Apply formatting** (table style, alignment)

### **Method 3: Markdown Table (Edit di Markdown)**

Bila ingin edit table di markdown dulu:
```bash
# Convert CSV to Markdown table
# Online: https://www.convertcsv.com/csv-to-markdown.htm
# Or pandoc:
pandoc luaran/tables/Table1.csv -o table1.md
```

Copy hasil markdown table, paste di .md file, edit sesuai kebutuhan.

---

## 🖼️ CARA INSERT FIGURES

### **Inserting in Word/LibreOffice**

**Step-by-step:**

1. **Cari placeholder di markdown**
   - Example: "**INSERT FIGURE 6 HERE**"
   - Atau lihat section reference di Appendix JICEST Paper

2. **Insert Picture di Word**
   ```
   Insert → Pictures → Browse
   Navigate: luaran/figures/
   Pilih: figure6_architecture_diagram.png
   ```

3. **Resize Figure**
   ```
   Right-click → Size and Position
   Width: 6.5 inches (full-width) atau 3.25 inches (half-width)
   ✅ Lock aspect ratio
   Resolution: Keep original (300 DPI)
   ```

4. **Add Caption**
   ```
   Right-click figure → Insert Caption
   Format: "Figure 6. Option A Architecture Diagram"
   Position: Below image
   Numbering: Automatic
   ```

5. **Cross-Reference in Text**
   ```
   Di text yang mention figure:
   Insert → Cross-reference → Reference type: Figure
   Insert: "as shown in Figure 6"
   ```

### **Quick Reference: Which Figure Where**

**Laporan Kemajuan:**
- Section C.3 (Deteksi): figure4, figure5, figure7
- Section C.4 (Klasifikasi): figure2, figure3, figure8, figure9
- Section C.6 (Arsitektur): figure6
- Discussion: figure10 (Grad-CAM)

**JICEST Paper:**
- Section 2.1 (Datasets): figure1
- Section 2.2 (Architecture): figure6
- Section 3.1 (Detection): figure4, figure5, figure7
- Section 3.2 (Classification): figure2, figure3, figure8, figure9
- Section 4 (Discussion): figure10

---

## 🔧 GENERATE DOCX (Optional)

Bila ingin generate .docx dari markdown:

### **Step 1: Run Generator Script**

```bash
# Generate both Laporan Kemajuan & JICEST Paper
python generate_docx_from_markdown.py
```

**Output:**
- `luaran/Laporan_Kemajuan_Malaria_Detection_UPDATED.docx` (~68 KB)
- `luaran/JICEST_Paper_UPDATED.docx` (~45 KB)

### **Step 2: Open di Word**

```bash
# Open generated DOCX
start luaran/Laporan_Kemajuan_Malaria_Detection_UPDATED.docx
```

### **Step 3: Final Formatting**

**Di Word:**
- Verify semua tables ter-parse dengan benar
- Insert figures manually (script tidak auto-insert images)
- Add cover page (Insert → Cover Page)
- Add page numbers (Insert → Page Number → Bottom of Page)
- Adjust spacing/margins sesuai kebutuhan

**Note**: Script hanya convert text + markdown tables → Word tables. Figures harus di-insert manual.

---

## 📈 KEY METRICS SUMMARY

### **Datasets**
- **Total**: 731 images (IML: 313, MP-IDB Species: 209, MP-IDB Stages: 209)
- **Split**: 66% train / 17% val / 17% test (stratified)
- **Classes**: 12 total (4 species + 4 stages × 2 datasets)
- **Augmentation**: 4.4× detection, 3.5× classification

### **Detection (YOLO)**
- **Models**: YOLOv10, YOLOv11, YOLOv12
- **Best mAP@50**: 95.71% (YOLOv12 on IML Lifecycle)
- **Best Recall**: 94.98% (YOLOv11 on IML Lifecycle)
- **Training**: 6.3 hours total (9 models)
- **Inference**: 12-15 ms/image (66-81 FPS)

### **Classification (CNN)**
- **Models**: DenseNet121, EfficientNet-B0/B1/B2, ResNet50/101
- **Best Accuracy**: 98.8% (EfficientNet-B1 on MP-IDB Species)
- **Best Balanced Acc**: 93.18% (EfficientNet-B1 on MP-IDB Species)
- **Training**: 51.6 hours total (18 models)
- **Inference**: 8-10 ms/image

### **Computational Efficiency (Option A)**
- **Storage**: 70% reduction (45GB → 14GB)
- **Training Time**: 60% reduction (450h → 180h)
- **End-to-End**: <25ms/image (40+ FPS on RTX 3060)

### **Minority Class Challenge**
- **Imbalance**: 4-272 samples per class (68:1 ratio max)
- **F1-scores**: 51-77% on classes with <10 samples
- **Improvement**: +20-40% with Focal Loss (α=0.25, γ=2.0)

### **Key Findings**
✅ **Model Efficiency**: EfficientNet-B0/B1 (5.3-7.8M params) outperform ResNet50/101 (25.6-44.5M params) by 5-10%
✅ **Clinical Value**: 100% recall on P. ovale (5 samples) achieved by EfficientNet-B1
✅ **Real-Time**: 40+ FPS capable for point-of-care deployment

---

## 🔍 TROUBLESHOOTING

### **Problem: Tabel tidak terbaca dengan baik di Word**

**Solution 1: Re-export CSV dengan semicolon delimiter**
```bash
# Di Excel: File → Save As → CSV (Semicolon delimited) *.csv
# Then import to Word
```

**Solution 2: Manual import di Word**
```
Word → Insert → Table → Convert Text to Table
Browse ke CSV
Pilih delimiter: Comma atau Semicolon (test both)
```

**Solution 3: Convert to Markdown table first**
```bash
# Use online tool: https://www.convertcsv.com/csv-to-markdown.htm
# Or pandoc:
pandoc table.csv -o table.md
```

---

### **Problem: Figures terlalu besar/kecil di dokumen**

**Solution:**
```
Di Word:
1. Right-click figure → Size and Position
2. Width: 6.5 inches (full-width) atau 3.25 inches (half)
3. ✅ Lock aspect ratio
4. Resolution: Keep 300 DPI

Di LibreOffice:
1. Right-click → Properties → Type tab
2. Width: 16.5 cm (full) atau 8.25 cm (half)
3. ✅ Keep ratio
```

---

### **Problem: Markdown syntax error**

**Solution: Validate markdown online**
```
Tools:
- https://markdownlint.github.io/
- VSCode extension: "markdownlint"
- Or use: pandoc to test conversion
```

---

### **Problem: Ingin update data dengan experiment baru**

**Solution:**
```bash
# 1. Run new experiment
python run_multiple_models_pipeline_OPTION_A.py

# 2. Update CSV tables
# Extract new data dari comprehensive_summary.json

# 3. Update markdown files
# Find-replace old metrics with new ones

# 4. Verify consistency
# Check both Laporan + JICEST have same numbers
```

---

## 💡 TIPS & BEST PRACTICES

### **Markdown Editing**

1. **Use VSCode dengan preview**
   ```bash
   code luaran/JICEST_Paper_FINAL_WITH_TABLES.md
   # Ctrl+Shift+V untuk preview
   ```

2. **Track changes dengan Git**
   ```bash
   git add .
   git commit -m "Updated JICEST Paper Discussion section"
   ```

3. **Consistent formatting**
   - Bold: `**text**`
   - Italic: `*text*`
   - Headers: `## Header 2`, `### Header 3`
   - Lists: `- item` atau `1. item`

### **Table Formatting**

1. **Decimal places consistency**
   - Detection: 2 decimals (95.71%)
   - Classification: 2 decimals (98.8% atau 98.80%)
   - Time: 1 decimal (2.5 hours)

2. **Bold best values**
   - Highlight top performer per metric
   - Makes reader easy to identify

3. **Add notes/captions**
   - Explain abbreviations (mAP, F1, etc.)
   - Mention key findings

### **Figure Quality**

1. **Always 300 DPI for publication**
   - Sudah di-generate 300 DPI ✅
   - Jangan resize/compress

2. **Consistent style**
   - Detection: Blue tones
   - Classification: Green/Orange
   - Heatmaps: Red-Yellow-Green

3. **Clear labels**
   - Axis labels readable (font ≥10pt)
   - Legend positioned clearly

---

## ✅ CHECKLIST SUBMISSION

### **Laporan Kemajuan BISMA:**
- [✅] All sections C-H complete
- [✅] 9 tables integrated
- [✅] 24 references verified
- [✅] Phase 1 marked COMPLETED
- [ ] Tables di-insert dari CSV
- [ ] Figures di-insert
- [ ] Cover page added
- [ ] Page numbers added
- [ ] Final proofread

### **JICEST Paper:**
- [✅] IMRaD structure complete
- [✅] Bilingual abstracts
- [✅] 4 CSV tables referenced
- [✅] 10 main figures mapped
- [✅] 14 references cited
- [ ] Tables di-insert dari CSV
- [ ] Figures di-insert
- [ ] Format sesuai journal template
- [ ] Final proofread

---

## 📞 QUICK HELP

**Q: File mana yang digunakan untuk Laporan Kemajuan?**
A: `Laporan_Kemajuan_FINAL_WITH_TABLES.md` (87 KB)

**Q: File mana yang digunakan untuk JICEST Paper?**
A: `JICEST_Paper_FINAL_WITH_TABLES.md` (35 KB)

**Q: Di mana CSV tables?**
A: `luaran/tables/Table1-4_*_UPDATED.csv`

**Q: Di mana figures?**
A: `luaran/figures/figure1-10.png` dan `luaran/figures/supplementary/`

**Q: Apakah perlu generate .docx?**
A: Tidak wajib. Markdown sudah complete. Generate .docx hanya bila perlu preview Word atau submit format .docx/.pdf.

**Q: Bagaimana cara insert table?**
A: Buka CSV di Excel → Copy → Paste ke Word → Format table. Atau lihat section [Cara Insert Tables](#cara-insert-tables).

**Q: Bagaimana cara insert figure?**
A: Word: Insert → Pictures → Browse ke `luaran/figures/` → Pilih figure. Atau lihat section [Cara Insert Figures](#cara-insert-figures).

---

## 🎯 NEXT STEPS

**Immediate:**
1. Review markdown documents (Laporan + JICEST)
2. Verify CSV tables exist dan data correct
3. Check figures ada semua

**This Week:**
1. Insert tables dari CSV ke documents
2. Insert figures ke documents
3. Add cover page, page numbers
4. Final formatting pass

**Next Week:**
1. Submit Laporan Kemajuan to BISMA
2. Submit JICEST Paper to journal
3. Prepare presentation (bila diperlukan)

---

**Generated**: 2025-10-08
**Status**: ✅ Complete Guide
**Total Pages**: Comprehensive usage documentation

**Files to Use:**
- Laporan: `Laporan_Kemajuan_FINAL_WITH_TABLES.md`
- JICEST: `JICEST_Paper_FINAL_WITH_TABLES.md`
- Tables: `luaran/tables/*.csv`
- Figures: `luaran/figures/*.png`

**Good luck with your research! 🚀**
