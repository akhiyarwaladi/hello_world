
╔══════════════════════════════════════════════════════════════════════════════╗
║              CENTRALIZED VISUALIZATION OUTPUTS - ONE FOLDER!                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generated: 2025-12-13 12:56:10
Source Experiment: results\optA_20251207_233941
Output Directory: C:\Users\MyPC PRO\Documents\hello_world\visualization_outputs

═══════════════════════════════════════════════════════════════════════════════

📂 FOLDER STRUCTURE (EVERYTHING IN ONE PLACE!):

confusion_matrices/
├── individual/              0 per-model confusion matrices
└── consolidated_2x2.png     Publication 2x2 grid (best models)

training_curves/             8 accuracy curves
├── accuracy_iml_lifecycle.png
├── accuracy_md_2019_stages.png
├── accuracy_mp_idb_species.png
└── accuracy_mp_idb_stages.png

selected_cases/              Top 20 error cases each type
├── detection/               Detection errors (FP, FN, Mixed)
└── classification/          Classification errors

metadata/                    Analysis data & reports
├── selected_detection_errors.csv      (0 cases)
├── selected_classification_errors.csv (0 cases)
├── selected_error_images.csv          (combined)
└── visualization_report.md            (human-readable)

═══════════════════════════════════════════════════════════════════════════════

🎯 QUICK ACCESS FOR PAPER/LAPORAN:

📊 FIGURES:
   → confusion_matrices/consolidated_2x2.png    (Publication figure)
   → training_curves/*.png                      (4 accuracy curves)
   → selected_cases/                            (Error examples)

📋 DATA/ANALYSIS:
   → metadata/selected_*_errors.csv             (Sortable in Excel)
   → metadata/visualization_report.md           (Summary)

═══════════════════════════════════════════════════════════════════════════════

💡 CARA PAKAI:

1. UNTUK PAPER:
   - Ambil: confusion_matrices/consolidated_2x2.png
   - Ambil: training_curves/*.png (pilih yang perlu)
   - Ambil: selected_cases/ (error examples)

2. UNTUK ANALISIS:
   - Buka: metadata/selected_detection_errors.csv di Excel
   - Sort by: paper_score (descending) atau error_category
   - Lihat image_file column untuk lokasi gambar

3. UNTUK LAPORAN:
   - Baca: metadata/visualization_report.md
   - Quick summary semua results

═══════════════════════════════════════════════════════════════════════════════

🔄 TO REGENERATE:

python scripts/visualization/generate_all_centralized.py

═══════════════════════════════════════════════════════════════════════════════
