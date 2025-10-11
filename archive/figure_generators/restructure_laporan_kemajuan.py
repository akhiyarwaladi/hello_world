"""
Script to restructure Laporan_Kemajuan.md:
1. Split C.3 into C.3.1 (Quantitative) and C.3.2 (Qualitative)
2. Split C.4 into C.4.1 (Quantitative) and C.4.2 (Qualitative)
3. Remove redundant C.6 and integrate into C.3.2/C.4.2
4. Renumber C.7→C.6, C.8→C.7, C.9→C.8
5. Remove inline file paths from narrative text
"""

def main():
    input_file = 'luaran/Laporan_Kemajuan.md'
    output_file = 'luaran/Laporan_Kemajuan_RESTRUCTURED.md'

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by sections
    lines = content.split('\n')

    # Find section boundaries
    c3_start = None
    c3_end = None
    c4_start = None
    c4_end = None
    c5_start = None
    c6_start = None
    c7_start = None
    c8_start = None
    c9_start = None
    c9_end = None

    for i, line in enumerate(lines):
        if line.startswith('### C.3 Hasil Deteksi'):
            c3_start = i
        elif line.startswith('### C.4 Hasil Klasifikasi'):
            c4_start = i
            c3_end = i - 1
        elif line.startswith('### C.5 Analisis Efisiensi'):
            c5_start = i
            c4_end = i - 1
        elif line.startswith('### C.6 Validasi Kualitatif'):
            c6_start = i
        elif line.startswith('### C.7 Strategi Handling'):
            c7_start = i
        elif line.startswith('### C.8 Kelayakan Komputasi'):
            c8_start = i
        elif line.startswith('### C.9 Keterbatasan'):
            c9_start = i
        elif line.startswith('## D. DISKUSI') or line.startswith('## E.'):
            c9_end = i - 1

    print(f"C.3: {c3_start} to {c3_end}")
    print(f"C.4: {c4_start} to {c4_end}")
    print(f"C.5: {c5_start}")
    print(f"C.6: {c6_start}")
    print(f"C.7: {c7_start}")
    print(f"C.8: {c8_start}")
    print(f"C.9: {c9_start} to {c9_end}")

    # Build new content
    new_lines = []

    # Keep everything before C.3
    new_lines.extend(lines[:c3_start])

    # === C.3 RESTRUCTURED ===
    new_lines.append('### C.3 Hasil Deteksi Parasit Malaria')
    new_lines.append('')
    new_lines.append('#### C.3.1 Performa Kuantitatif')
    new_lines.append('')

    # Extract C.3 content WITHOUT Gambar 2A
    c3_content = []
    skip_gambar_2a = False
    for line in lines[c3_start+1:c3_end+1]:
        if '**[INSERT GAMBAR 2A:' in line:
            skip_gambar_2a = True
        if skip_gambar_2a:
            if line.startswith('Waktu pelatihan') or line.startswith('Pada dataset MP-IDB Stages'):
                skip_gambar_2a = False
            else:
                continue
        c3_content.append(line)

    new_lines.extend(c3_content)

    # Add C.3.2 Qualitative from C.6 (Detection part only)
    new_lines.append('')
    new_lines.append('#### C.3.2 Validasi Kualitatif: Visualisasi Deteksi')
    new_lines.append('')
    new_lines.append('Evaluasi kualitatif hasil deteksi dilakukan melalui perbandingan visual side-by-side antara ground truth annotations (blue bounding boxes) dengan automated predictions dari YOLOv11 model (green bounding boxes). Visualisasi ini memvalidasi temuan kuantitatif yang dilaporkan dalam Tabel 2, mendemonstrasikan bahwa sistem tidak hanya mencapai metrik performa tinggi secara statistik namun juga menghasilkan prediksi yang akurat secara visual pada beragam morfologi parasit dan kondisi kompleksitas gambar.')
    new_lines.append('')
    new_lines.append('**[INSERT GAMBAR 2A: Visualisasi Deteksi - Ground Truth vs Prediction]**')
    new_lines.append('Gambar 2A menampilkan contoh visualisasi hasil deteksi parasit malaria pada severe malaria case (test image: 1704282807-0012-R_T) yang mengandung 25+ parasites P. falciparum dengan estimated parasitemia >10%. Side-by-side comparison menunjukkan ground truth annotations (panel kiri, blue bounding boxes) versus YOLOv11 predicted detections (panel kanan, green bounding boxes), dimana semua 25+ parasites berhasil dideteksi dengan localization precision tinggi (IoU >0.8). Visualisasi ini memvalidasi YOLOv11 93.09% mAP@50 dan 92.26% recall pada MP-IDB Species dataset, demonstrating robust performance pada complex multi-parasite scenarios dengan varying morphologies (rings, trophozoites, gametocytes).')
    new_lines.append('**Files**: Ground Truth: `gt_detection/1704282807-0012-R_T.png` | Predicted: `pred_detection/1704282807-0012-R_T.png`')
    new_lines.append('')

    # === C.4 RESTRUCTURED ===
    new_lines.append('### C.4 Hasil Klasifikasi Spesies dan Tahapan Siklus Hidup')
    new_lines.append('')
    new_lines.append('#### C.4.1 Performa Kuantitatif')
    new_lines.append('')

    # Extract C.4 content WITHOUT Gambar 5A
    c4_content = []
    skip_gambar_5a = False
    for line in lines[c4_start+1:c4_end+1]:
        if '**[INSERT GAMBAR 5A:' in line:
            skip_gambar_5a = True
        if skip_gambar_5a:
            if line.startswith('Untuk tahap siklus hidup') or line.startswith('Skor F1 per-kelas'):
                skip_gambar_5a = False
            else:
                continue
        # Remove inline file paths from narrative
        if '`results/optA_' in line and not line.startswith('**File'):
            # This is an inline path in narrative - remove it
            continue
        c4_content.append(line)

    new_lines.extend(c4_content)

    # Add C.4.2 Qualitative from C.6 (Classification part only)
    new_lines.append('')
    new_lines.append('#### C.4.2 Validasi Kualitatif: Visualisasi Klasifikasi')
    new_lines.append('')
    new_lines.append('Evaluasi kualitatif hasil klasifikasi menyajikan visualisasi performa end-to-end dengan perbandingan side-by-side ground truth labels (blue boxes) versus automated predictions (color-coded boxes: green untuk correct, red untuk misclassifications). Visualisasi ini memberikan bukti visual yang mendukung metrik kuantitatif pada Tabel 3 dan Tabel 4.')
    new_lines.append('')
    new_lines.append('**[INSERT GAMBAR 5A: Visualisasi Klasifikasi Species - Success Case]**')
    new_lines.append('Gambar 5A menampilkan hasil klasifikasi spesies pada severe malaria case yang sama (1704282807-0012-R_T) dengan 25+ P. falciparum parasites. Ground truth classification (panel kiri, blue boxes dengan species labels) dibandingkan dengan EfficientNet-B1 predictions (panel kanan, color-coded boxes). Image ini mencapai remarkable 100% classification accuracy dengan semua 25 parasites correctly identified (all green boxes), providing compelling visual evidence bahwa classifier maintains high performance bahkan pada extreme parasite density dimana overlapping morphologies dan varying developmental stages present significant challenges. Contrast ini dengan stages classification (mixed green-red boxes) clearly demonstrates bahwa species discrimination task inherently easier dibandingkan lifecycle stages classification, reflecting bahwa morphological SIZE differences lebih mudah untuk deep learning models dibandingkan subtle chromatin PATTERN differences.')
    new_lines.append('**Files**: Ground Truth: `gt_classification/1704282807-0012-R_T.png` | Predicted: `pred_classification/1704282807-0012-R_T.png`')
    new_lines.append('')
    new_lines.append('**[INSERT GAMBAR 5B: Visualisasi Klasifikasi Stages - Minority Class Challenge]**')
    new_lines.append('Gambar 5B menampilkan hasil klasifikasi lifecycle stages pada complex multi-parasite image (1704282807-0021-T_G_R) dengan 17 parasites. Visualisasi ini mengungkap minority class challenge dimana approximately 65% classifications correct (green boxes) versus 35% misclassifications (red boxes), dengan errors concentrated pada Trophozoite class. Hal ini secara visual memvalidasi reported 46.7% F1-score untuk 15-sample minority Trophozoite class dan mendemonstrasikan bahwa extreme class imbalance (272 Ring vs 5 Gametocyte, ratio 54:1) tetap menyajikan significant classification difficulty terutama pada transitional morphologies antara lifecycle stages.')
    new_lines.append('**Files**: Ground Truth: `gt_classification/1704282807-0021-T_G_R.png` | Predicted: `pred_classification/1704282807-0021-T_G_R.png`')
    new_lines.append('')

    # === C.5 remains unchanged ===
    if c5_start and c6_start:
        new_lines.extend(lines[c5_start:c6_start])

    # === SKIP C.6 (already integrated) ===

    # === C.7 → C.6 (renumber) ===
    if c7_start and c8_start:
        new_lines.append('### C.6 Strategi Handling Class Imbalance dengan Focal Loss')
        new_lines.append('')
        # Remove inline paths from C.7 content
        for line in lines[c7_start+1:c8_start]:
            if '`results/optA_' in line and not line.startswith('**File'):
                continue
            new_lines.append(line)

    # === C.8 → C.7 (renumber) ===
    if c8_start and c9_start:
        new_lines.append('### C.7 Kelayakan Komputasi untuk Deployment Klinis')
        new_lines.append('')
        for line in lines[c8_start+1:c9_start]:
            new_lines.append(line)

    # === C.9 → C.8 (renumber) ===
    if c9_start and c9_end:
        new_lines.append('### C.8 Keterbatasan dan Arah Penelitian Masa Depan')
        new_lines.append('')
        for line in lines[c9_start+1:c9_end+1]:
            new_lines.append(line)

    # === Keep everything after C.9 ===
    if c9_end:
        new_lines.extend(lines[c9_end+1:])

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))

    print(f"\nSUCCESS: Restructured file saved to: {output_file}")
    print(f"   - C.3 split into C.3.1 (Kuantitatif) + C.3.2 (Kualitatif)")
    print(f"   - C.4 split into C.4.1 (Kuantitatif) + C.4.2 (Kualitatif)")
    print(f"   - C.6 removed (integrated into C.3.2 and C.4.2)")
    print(f"   - C.7->C.6, C.8->C.7, C.9->C.8 (renumbered)")
    print(f"   - Inline file paths removed from narrative text")

if __name__ == '__main__':
    main()
