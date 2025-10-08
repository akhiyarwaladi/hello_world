# 📄 ABSTRAK INDONESIA (INDONESIAN ABSTRACT)

**Untuk ditambahkan ke Research Paper sebelum submit ke jurnal SINTA 3**

---

## ABSTRAK (Bahasa Indonesia)

**Deteksi dan Klasifikasi Parasit Malaria Berbasis Deep Learning: Studi Komparatif Menggunakan Arsitektur YOLO dan CNN**

Malaria tetap menjadi tantangan kesehatan global yang signifikan dengan 263 juta kasus dan 597.000 kematian dilaporkan pada tahun 2023, terutama di wilayah Afrika. Diagnosis yang akurat dan cepat sangat penting untuk pengobatan efektif dan pengendalian penyakit. Penelitian ini mengembangkan sistem otomatis komprehensif untuk deteksi dan klasifikasi parasit malaria menggunakan arsitektur deep learning terkini. Kami mengimplementasikan pipeline dua tahap: (1) deteksi objek berbasis YOLO (YOLOv10, YOLOv11, YOLOv12) untuk lokalisasi parasit dalam citra apusan darah mikroskopis, dan (2) klasifikasi berbasis CNN (DenseNet121, EfficientNet-B0/B1/B2, ResNet50/101) untuk identifikasi spesies dan stadium hidup.

Pendekatan kami menggunakan dataset MP-IDB (Malaria Parasite Image Database) yang terdiri dari 209 citra untuk setiap tugas, mencakup empat spesies Plasmodium (P. falciparum, P. vivax, P. malariae, P. ovale) dan empat stadium hidup (ring, trophozoite, schizont, gametocyte). Model deteksi mencapai mean Average Precision (mAP@50) sebesar 93,10% untuk klasifikasi spesies dan 92,90% untuk klasifikasi stadium menggunakan YOLO11. Untuk klasifikasi, DenseNet121 dan EfficientNet-B1 mencapai akurasi luar biasa sebesar 98,80% untuk identifikasi spesies, sementara EfficientNet-B0 mencapai 94,31% untuk klasifikasi stadium hidup.

Kami menangani tantangan ketidakseimbangan kelas menggunakan Focal Loss (α=0,25, γ=2,0) dan mendemonstrasikan peningkatan signifikan dibandingkan pendekatan tradisional dengan peningkatan F1-score 20-40% pada minority classes. Arsitektur shared classification yang kami kembangkan mengurangi kebutuhan penyimpanan sekitar 70% dan waktu pelatihan sekitar 60% dibandingkan dengan metode konvensional. Hasil penelitian menunjukkan efektivitas kombinasi deteksi berbasis YOLO dengan klasifikasi CNN mendalam untuk diagnosis malaria otomatis, berpotensi mendukung pengambilan keputusan klinis di wilayah dengan sumber daya terbatas, terutama untuk screening berkala, pelatihan mikroskopis, quality assurance laboratorium diagnostik, dan surveilans epidemiologi.

**Kata kunci:** Deteksi malaria, Deep learning, YOLO, CNN, Deteksi objek, Analisis citra medis, Focal loss, Class imbalance, Diagnosa otomatis, Computer-aided diagnosis, Parasit Plasmodium, Klasifikasi spesies

---

## CARA PENGGUNAAN

### Di Microsoft Word (Research Paper):

1. **Posisi:** Tambahkan SETELAH English Abstract, SEBELUM Keywords

2. **Format:**
   ```
   ABSTRACT
   [English abstract text...]

   Keywords: [English keywords...]

   ─────────────────────────────────

   ABSTRAK  [← Add this heading]
   [Indonesian abstract text from above...]

   Kata kunci: [Indonesian keywords from above...]

   ─────────────────────────────────

   1. INTRODUCTION
   [Continue with main text...]
   ```

3. **Formatting:**
   - Font: Times New Roman, 12pt
   - Alignment: Justify
   - Spacing: 1.5 or Double (per journal template)
   - Bold: "ABSTRAK" and "Kata kunci"
   - Italic: Not needed for abstract

4. **Length Check:**
   - Count words (Word count tool)
   - Should be ~250 words
   - Current version: ~280 words (OK, within acceptable range 250-300)

---

## ALTERNATIF VERSI (Jika Perlu Lebih Singkat)

### ABSTRAK VERSI PENDEK (~200 kata):

Malaria tetap menjadi tantangan kesehatan global dengan 263 juta kasus dan 597.000 kematian pada 2023. Penelitian ini mengembangkan sistem otomatis untuk deteksi dan klasifikasi parasit malaria menggunakan deep learning. Kami mengimplementasikan pipeline dua tahap: (1) deteksi berbasis YOLO (v10, v11, v12) untuk lokalisasi parasit, dan (2) klasifikasi berbasis CNN (DenseNet121, EfficientNet-B0/B1/B2, ResNet50/101) untuk identifikasi spesies dan stadium hidup.

Dataset MP-IDB dengan 209 citra per tugas digunakan untuk empat spesies Plasmodium dan empat stadium hidup. Model deteksi mencapai mAP@50 sebesar 93,10% (YOLO11), sementara klasifikasi mencapai akurasi 98,80% untuk identifikasi spesies (EfficientNet-B1) dan 94,31% untuk stadium hidup (EfficientNet-B0). Focal Loss (α=0,25, γ=2,0) efektif menangani ketidakseimbangan kelas dengan peningkatan F1-score 20-40% pada minority classes.

Arsitektur shared classification mengurangi kebutuhan penyimpanan ~70% dan waktu pelatihan ~60%. Sistem ini menunjukkan potensi untuk mendukung screening klinis, pelatihan mikroskopis, dan surveilans epidemiologi di daerah endemis malaria.

**Kata kunci:** Deteksi malaria, Deep learning, YOLO, CNN, Focal loss, Class imbalance, Diagnosa otomatis, Medical imaging

---

## QUALITY CHECKLIST

Pastikan abstrak Indonesia memenuhi kriteria:

**Content:**
- [ ] Latar belakang masalah (malaria global challenge)
- [ ] Tujuan penelitian (sistem otomatis)
- [ ] Metode (YOLO + CNN, dataset MP-IDB)
- [ ] Hasil utama (mAP 93.1%, accuracy 98.8%)
- [ ] Temuan penting (Focal Loss, efficiency gains)
- [ ] Implikasi (clinical applications)

**Language:**
- [ ] Bahasa Indonesia baku dan formal
- [ ] Tidak ada typo atau grammatical errors
- [ ] Terminologi teknis konsisten
- [ ] Angka ditulis dengan benar (93,10% bukan 93.10%)
- [ ] Istilah asing di-italic jika belum diserap (optional)

**Format:**
- [ ] Panjang 250-300 kata
- [ ] Satu paragraf (continuous text, no line breaks)
- [ ] Font Times New Roman 12pt
- [ ] Justify alignment
- [ ] Heading "ABSTRAK" bold

**Keywords:**
- [ ] 8-12 kata kunci
- [ ] Kombinasi Indonesian + technical terms
- [ ] Dipisahkan koma
- [ ] Lowercase (kecuali nama proper: Plasmodium)
- [ ] Sesuai dengan isi penelitian

---

## CATATAN PENTING

### **Penggunaan Istilah Teknis:**

Beberapa istilah teknis dipertahankan dalam bahasa Inggris karena:
1. Sudah umum digunakan dalam literatur Indonesia
2. Tidak ada padanan bahasa Indonesia yang tepat
3. Konsistensi dengan literatur internasional

**Istilah yang dipertahankan:**
- Deep learning (bukan: pembelajaran mendalam)
- YOLO, CNN (nama algoritma)
- Focal Loss (nama metode)
- mAP, F1-score (metrik standard)
- Dataset (bisa: basis data, tapi dataset lebih umum)

**Istilah yang diterjemahkan:**
- Detection → Deteksi
- Classification → Klasifikasi
- Species identification → Identifikasi spesies
- Life stages → Stadium hidup
- Class imbalance → Ketidakseimbangan kelas
- Automated diagnosis → Diagnosa otomatis

### **Angka dan Persentase:**

Dalam bahasa Indonesia formal:
- Desimal menggunakan koma: 93,10% (bukan 93.10%)
- Ribuan tanpa titik untuk angka 4 digit: 2023 (bukan 2.023)
- Ribuan dengan titik untuk 5+ digit: 263.000 (atau 263 juta)

**Contoh:**
- ✅ 93,10%
- ❌ 93.10%
- ✅ 263 juta kasus
- ✅ 597.000 kematian
- ✅ α=0,25

---

## TEMPLATE INSERT (Copy-Paste Ready)

```markdown
ABSTRAK

Malaria tetap menjadi tantangan kesehatan global yang signifikan dengan 263 juta kasus dan 597.000 kematian dilaporkan pada tahun 2023, terutama di wilayah Afrika. Diagnosis yang akurat dan cepat sangat penting untuk pengobatan efektif dan pengendalian penyakit. Penelitian ini mengembangkan sistem otomatis komprehensif untuk deteksi dan klasifikasi parasit malaria menggunakan arsitektur deep learning terkini. Kami mengimplementasikan pipeline dua tahap: (1) deteksi objek berbasis YOLO (YOLOv10, YOLOv11, YOLOv12) untuk lokalisasi parasit dalam citra apusan darah mikroskopis, dan (2) klasifikasi berbasis CNN (DenseNet121, EfficientNet-B0/B1/B2, ResNet50/101) untuk identifikasi spesies dan stadium hidup. Pendekatan kami menggunakan dataset MP-IDB (Malaria Parasite Image Database) yang terdiri dari 209 citra untuk setiap tugas, mencakup empat spesies Plasmodium (P. falciparum, P. vivax, P. malariae, P. ovale) dan empat stadium hidup (ring, trophozoite, schizont, gametocyte). Model deteksi mencapai mean Average Precision (mAP@50) sebesar 93,10% untuk klasifikasi spesies dan 92,90% untuk klasifikasi stadium menggunakan YOLO11. Untuk klasifikasi, DenseNet121 dan EfficientNet-B1 mencapai akurasi luar biasa sebesar 98,80% untuk identifikasi spesies, sementara EfficientNet-B0 mencapai 94,31% untuk klasifikasi stadium hidup. Kami menangani tantangan ketidakseimbangan kelas menggunakan Focal Loss (α=0,25, γ=2,0) dan mendemonstrasikan peningkatan signifikan dibandingkan pendekatan tradisional dengan peningkatan F1-score 20-40% pada minority classes. Arsitektur shared classification yang kami kembangkan mengurangi kebutuhan penyimpanan sekitar 70% dan waktu pelatihan sekitar 60% dibandingkan dengan metode konvensional. Hasil penelitian menunjukkan efektivitas kombinasi deteksi berbasis YOLO dengan klasifikasi CNN mendalam untuk diagnosis malaria otomatis, berpotensi mendukung pengambilan keputusan klinis di wilayah dengan sumber daya terbatas, terutama untuk screening berkala, pelatihan mikroskopis, quality assurance laboratorium diagnostik, dan surveilans epidemiologi.

Kata kunci: Deteksi malaria, Deep learning, YOLO, CNN, Deteksi objek, Analisis citra medis, Focal loss, Class imbalance, Diagnosa otomatis, Computer-aided diagnosis, Parasit Plasmodium, Klasifikasi spesies
```

---

## ✅ FINAL CHECK BEFORE SUBMISSION

- [ ] Indonesian abstract added to manuscript
- [ ] Positioned correctly (after English abstract)
- [ ] Word count verified (250-300 words)
- [ ] Keywords added (8-12 terms)
- [ ] Formatting correct (Times New Roman 12pt, justify)
- [ ] No typos or errors
- [ ] Numbers formatted correctly (koma for decimal)
- [ ] Technical terms consistent
- [ ] Reads professionally in Indonesian

**After adding, total abstract section will be ~500-600 words (English + Indonesian combined) - THIS IS NORMAL for SINTA journals requiring bilingual abstracts.**

---

**Status:** ✅ READY TO INSERT INTO MANUSCRIPT

**Time Required:** 5 minutes (copy-paste + formatting)

**Impact:** ✅ Meets SINTA 3 bilingual requirements, increases acceptance probability
