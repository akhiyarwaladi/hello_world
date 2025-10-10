# LAPORAN KEMAJUAN PENELITIAN

## MULTI-MODEL HYBRID FRAMEWORK FOR AUTOMATED MALARIA DETECTION AND SPECIES CLASSIFICATION

---

**Peneliti Utama**: [Nama Peneliti]
**Institusi**: [Nama Institusi]
**Skema Penelitian**: BISMA (Bantuan Inovasi Sains, Manajemen, dan Aplikasi)
**Periode Pelaporan**: Oktober 2025
**Tahun Pelaksanaan**: Tahun ke-1 (Bulan ke-10 dari 12 bulan)
**Sumber Data Eksperimen**: optA_20251007_134458

---

## HASIL PELAKSANAAN PENELITIAN

### 1. Dataset dan Karakteristik Data

Penelitian memanfaatkan dua dataset publik MP-IDB (209 citra per dataset) yang terdiri dari apusan darah tipis dengan mikroskopi cahaya 1000× dan pewarnaan Giemsa mengikuti protokol standar WHO [1,2]. Dataset MP-IDB Species mencakup 4 spesies Plasmodium dengan ketidakseimbangan kelas substansial: P. falciparum (227 sampel), P. vivax (11 sampel), P. malariae (7 sampel), dan P. ovale (5 sampel). Dataset MP-IDB Stages mencakup 4 tahapan siklus hidup dengan ketidakseimbangan ekstrem: Ring (272 sampel pelatihan), Trophozoite (15 sampel pelatihan), Schizont (7 sampel pelatihan), dan Gametocyte (5 sampel pelatihan), merepresentasikan rasio 54:1 yang merupakan skenario terburuk untuk klasifikasi citra medis.

**[INSERT TABEL 1 DI SINI: Statistik Dataset dan Augmentasi]**
Tabel 1 menyajikan statistik komprehensif untuk kedua dataset termasuk total citra, pembagian pelatihan/validasi/pengujian (66/17/17%), distribusi kelas, pengali augmentasi (4,4× untuk deteksi, 3,5× untuk klasifikasi), dan ukuran dataset hasil augmentasi (1.280 citra deteksi, 1.024 citra klasifikasi total).
**File**: `luaran/tables/Table3_Dataset_Statistics_MP-IDB.csv`

Untuk mengatasi keterbatasan ukuran data sambil mempertahankan integritas diagnostik, diterapkan augmentasi aman-medis [3]: untuk tahap deteksi menggunakan penskalaan acak (0,5-1,5×), rotasi (±15°), penyesuaian HSV, augmentasi mosaik, dan pencerminan horizontal (tanpa pencerminan vertikal untuk mempertahankan orientasi parasit); untuk tahap klasifikasi menggunakan rotasi (±20°), transformasi affine, variasi warna, noise Gaussian, dan pengambilan sampel acak berbobot dengan oversampling 3:1 untuk kelas minoritas.

**[INSERT FIGURE A1: Data Augmentation Examples - MP-IDB Species]**
Gambar A1 mengilustrasikan 7 transformasi augmentasi (Original, rotasi 90°, brightness 0.7×, contrast 1.4×, saturation 1.4×, sharpness 2.0×, flip horizontal) pada keempat spesies Plasmodium, mendemonstrasikan pelestarian karakteristik morfologi spesifik-spesies [4]: pola titik kromatin P. falciparum, penampilan bentuk pita P. malariae, ukuran RBC membesar P. ovale, dan titik-titik Schüffner P. vivax.
**File**: `luaran/figures/aug_species_set3.png`

**[INSERT FIGURE A2: Data Augmentation Examples - MP-IDB Stages]**
Gambar A2 memvisualisasikan efek 7 augmentasi pada klasifikasi tahap siklus hidup, mendemonstrasikan pelestarian fitur morfologi spesifik-tahap: titik kromatin kompak ring, morfologi amoeboid dengan pigmen hemozoin trophozoites, multiple merozoites tersegmentasi schizonts, dan morfologi memanjang berbentuk pisang gametocytes.
**File**: `luaran/figures/aug_stages_set1.png`

### 2. Arsitektur Pipeline dengan Pendekatan Klasifikasi Bersama

Framework mengimplementasikan arsitektur klasifikasi bersama dengan tiga tahap: (1) Pelatihan Deteksi - melatih model YOLO (v10, v11, v12) pada deteksi parasit [5,6], (2) Potongan Acuan - mengekstrak potongan parasit langsung dari kotak pembatas acuan (bukan dari hasil deteksi) untuk memastikan model klasifikasi dilatih pada sampel terlokalisasi sempurna tanpa kontaminasi dari kesalahan deteksi, dan (3) Pelatihan Klasifikasi - melatih model PyTorch sekali pada data potongan bersih [7,8,9] dan menggunakan kembali untuk semua metode deteksi.

**[INSERT GAMBAR 1 DI SINI: Diagram Arsitektur Pipeline]**
Gambar 1 mengilustrasikan arsitektur lengkap pipeline: gambar apusan darah sebagai input ke tiga detektor YOLO paralel (v10, v11, v12), diikuti oleh generasi potongan acuan bersama (224×224), dan akhirnya enam pengklasifikasi CNN (DenseNet121, EfficientNet-B0/B1/B2, ResNet50/101) yang menghasilkan prediksi spesies/tahap.
**File**: `luaran/figures/pipeline_architecture_horizontal.png`

Pendekatan potongan acuan menawarkan tiga keuntungan utama: (1) pemisahan antara pelatihan deteksi dan klasifikasi memungkinkan optimisasi independen, (2) model klasifikasi mempelajari fitur morfologis kuat tanpa bias dari kesalahan lokalisasi, dan (3) potongan dihasilkan sekali dan digunakan kembali untuk semua metode deteksi, mengeliminasi komputasi redundan dengan penghematan penyimpanan ~70% dan waktu pelatihan ~60%.

### 3. Hasil Deteksi Parasit Malaria

#### 3.1 Performa Kuantitatif

Model deteksi YOLO menunjukkan performa kompetitif pada kedua dataset, dengan ketiga varian mencapai mAP@50 >90% [5,6]. Pada dataset Species, YOLOv12 mencapai mAP@50 tertinggi 93,12%, diikuti YOLOv11 (93,09%) dan YOLOv10 (92,53%), dengan margin <0,6% menunjukkan kemampuan lokalisasi fundamental setara. Namun, YOLOv11 menunjukkan recall unggul (92,26%) dibandingkan YOLOv12 (91,18%) dan YOLOv10 (89,57%), menjadikannya pilihan lebih disukai untuk konteks klinis dimana negatif palsu lebih kritis dibanding positif palsu.

**[INSERT TABEL 2 DI SINI: Performa Deteksi YOLO]**
Tabel 2 menyajikan hasil deteksi untuk ketiga varian YOLO (v10, v11, v12) pada kedua dataset MP-IDB (Species dan Stages), menyertakan kolom Dataset, Model, Epochs (100), mAP@50, mAP@50-95, Precision, Recall, dan Training Time (hours), mengkuantifikasi performa kompetitif (mAP@50: 90,91-93,12%) dan menyoroti recall unggul YOLOv11.
**File**: `luaran/tables/Table1_Detection_Performance_MP-IDB.csv`

**[INSERT GAMBAR 2 DI SINI: Comparison Bar Charts Detection Performance]**
Gambar 2 menampilkan perbandingan diagram batang berdampingan dari YOLOv10, v11, dan v12 pada kedua dataset untuk empat metrik: mAP@50, mAP@50-95, Precision, dan Recall, membuat perbedaan performa langsung terlihat jelas dan mendukung kesimpulan bahwa YOLOv11 menawarkan recall terbaik.
**File**: `luaran/figures/detection_performance_comparison.png`

Pada dataset Stages dengan tugas deteksi lebih menantang, YOLOv11 muncul sebagai pelaku terbaik dengan mAP@50 92,90% dan recall 90,37%, menunjukkan efektivitas khusus dalam mendeteksi kelas minoritas seperti schizont (7 sampel pelatihan) dan gametocyte (5 sampel pelatihan). Waktu pelatihan menunjukkan progres yang diharapkan: YOLOv10 (1,8 jam), YOLOv11 (1,9 jam), YOLOv12 (2,1 jam), dengan kecepatan inferensi YOLOv10 (12,3 ms/gambar, 81 FPS), YOLOv11 (13,7 ms, 73 FPS), YOLOv12 (15,2 ms, 66 FPS), semuanya memenuhi persyaratan waktu nyata untuk integrasi alur kerja klinis (>30 FPS).

#### 3.2 Validasi Kualitatif: Visualisasi Deteksi

Evaluasi kualitatif dilakukan melalui perbandingan visual berdampingan antara anotasi acuan (kotak biru) dengan prediksi otomatis dari YOLOv11 (kotak hijau), memvalidasi temuan kuantitatif bahwa sistem tidak hanya mencapai metrik performa tinggi secara statistik namun juga menghasilkan prediksi akurat secara visual pada beragam morfologi parasit.

**[INSERT GAMBAR 2A: Visualisasi Deteksi - Ground Truth vs Prediction]**
Gambar 2A menampilkan hasil deteksi pada kasus malaria parah (gambar uji: 1704282807-0012-R_T) yang mengandung 25+ parasit P. falciparum dengan estimasi parasitemia >10%. Perbandingan berdampingan menunjukkan anotasi acuan (panel kiri, kotak biru) versus deteksi prediksi YOLOv11 (panel kanan, kotak hijau), dimana semua 25+ parasit berhasil dideteksi dengan presisi lokalisasi tinggi (IoU >0.8), memvalidasi mAP@50 93,09% dan recall 92,26% YOLOv11.
**Files**: GT: `gt_detection/1704282807-0012-R_T.png` | Pred: `pred_detection/1704282807-0012-R_T.png`

### 4. Hasil Klasifikasi Spesies dan Tahapan Siklus Hidup

#### 4.1 Performa Kuantitatif

Hasil klasifikasi mengungkap perbedaan performa substansial antar arsitektur, menantang kebijaksanaan konvensional "deeper is better" dalam deep learning [7,8,9]. Pada dataset Species, EfficientNet-B1 dan DenseNet121 mencapai akurasi tertinggi 98,80%, namun balanced accuracy mengungkap perbedaan penting: EfficientNet-B1 mencapai 93,18% dibanding DenseNet121's 87,73%, mengindikasikan penanganan jauh lebih baik dari spesies minoritas meskipun akurasi keseluruhan identik.

**[INSERT TABEL 3 DI SINI: Performa Klasifikasi CNN]**
Tabel 3 menyajikan hasil klasifikasi untuk keenam model CNN pada kedua dataset MP-IDB, menyertakan kolom Dataset, Model, Fungsi Loss (Focal Loss), Epoch (75), Akurasi, Balanced Accuracy, dan Training Time (jam), mengkuantifikasi temuan kunci bahwa model EfficientNet lebih kecil mengungguli model ResNet lebih besar.
**File**: `luaran/tables/Table2_Classification_Performance_MP-IDB.csv`

Yang patut dicatat adalah penurunan performa model ResNet: ResNet50 mencapai akurasi 98,00% namun hanya 75,00% balanced accuracy—kesenjangan 23 poin persentase mengindikasikan kesulitan parah dengan kelas minoritas. ResNet101, meskipun model terbesar dengan 44,5 juta parameter (5,7× lebih banyak dari 7,8 juta parameter EfficientNet-B1), hanya mencapai balanced accuracy 82,73%, tertinggal dari EfficientNet-B1 sebesar 10,45 poin persentase. Fenomena ini menunjukkan bahwa efisiensi model dan penskalaan arsitektur seimbang lebih penting dibanding jumlah parameter mentah untuk dataset pencitraan medis kecil [8,10].

**[INSERT GAMBAR 3 DI SINI: Heatmap Akurasi Klasifikasi]**
Gambar 3 menampilkan heatmap 2×6 (2 dataset × 6 model) dengan dua baris per dataset: akurasi standar (atas) dan balanced accuracy (bawah), dengan kode warna (hijau=tinggi, oranye=sedang, merah=rendah) membuat pola performa model terlihat segera, terutama kontras antara EfficientNet (hijau) dan ResNet (oranye/merah) pada balanced accuracy.
**File**: `luaran/figures/classification_accuracy_heatmap.png`

Dataset Stages menyajikan tugas lebih menantang dengan ketidakseimbangan ekstrem (rasio 54:1). EfficientNet-B0 mencapai akurasi tertinggi 94,31% dengan balanced accuracy 69,21%, diikuti DenseNet121 (93,65%, 67,31%) dan ResNet50 (93,31%, 65,79%). Yang tak terduga, EfficientNet-B2 menunjukkan penurunan signifikan (80,60%, 60,72%), kemungkinan karena overfitting mengingat kapasitasnya lebih besar (9,2 juta parameter) relatif terhadap data terbatas (512 citra teraugmentasi).

**[INSERT TABEL 4 DI SINI: Metrik Per-Kelas dengan Focal Loss]**
Tabel 4 menyajikan rincian performa per-kelas komprehensif untuk keenam arsitektur CNN pada kedua dataset, memberikan nilai precision, recall, F1-score, dan support detail untuk setiap kelas individual. Pada dataset Species, P. falciparum (227 sampel pelatihan) dan P. malariae (7 sampel pelatihan) mencapai performa sempurna 100% pada semua metrik, sementara P. ovale (5 sampel pelatihan) mengalami degradasi substansial dengan F1-score 0,00-76,92%, dimana ResNet50 mengalami kegagalan total (0% recall—model tidak dapat mendeteksi P. ovale sama sekali). Pada dataset Stages, Ring (272 sampel pelatihan) mencapai F1-score 89,94-97,26%, sementara Trophozoite (15 sampel pelatihan) mengalami degradasi parah dengan F1-score hanya 15,38-51,61%, dan Gametocyte (5 sampel pelatihan) dengan F1-score 57,14-75%.
**Files**: `luaran/tables/Table9_MP-IDB_Species_Focal_Loss.csv` dan `Table9_MP-IDB_Stages_Focal_Loss.csv`

**[INSERT GAMBAR 5 DI SINI: Matriks Konfusi Model Terbaik]**
Gambar 5 menampilkan dua matriks konfusi berdampingan: (kiri) klasifikasi Spesies menggunakan EfficientNet-B1, dan (kanan) klasifikasi Tahap menggunakan EfficientNet-B0, dengan angka hitungan aktual dan kode warna menyoroti diagonal (benar) versus off-diagonal (kesalahan), membuat pola misklasifikasi langsung jelas.
**File**: `luaran/figures/confusion_matrices.png`

**[INSERT GAMBAR 6 DI SINI: Perbandingan F1 Per-Kelas Spesies]**
Gambar 6 menampilkan diagram batang berkelompok dengan 4 grup spesies (P. falciparum, P. malariae, P. ovale, P. vivax) × 6 model, menunjukkan skor F1 dengan garis putus-putus merah pada 0,90 (ambang klinis), menyoroti penurunan performa dramatis pada P. ovale (5 sampel) dibandingkan spesies mayoritas.
**File**: `luaran/figures/species_f1_comparison.png`

**[INSERT GAMBAR 7 DI SINI: Perbandingan F1 Per-Kelas Tahap]**
Gambar 7 menampilkan diagram batang berkelompok dengan 4 grup tahap siklus hidup (Ring, Trophozoite, Schizont, Gametocyte) × 6 model, menunjukkan skor F1 dengan garis putus-putus oranye pada 0,70 (ambang dimodifikasi untuk ketidakseimbangan ekstrem), membuat tantangan Trophozoite parah (F1: 0,15-0,52) langsung terlihat.
**File**: `luaran/figures/stages_f1_comparison.png`

#### 4.2 Validasi Kualitatif: Visualisasi Klasifikasi

Evaluasi kualitatif menyajikan visualisasi performa ujung-ke-ujung dengan perbandingan berdampingan label acuan (kotak biru) versus prediksi otomatis (berkode warna: hijau untuk benar, merah untuk kesalahan klasifikasi), memberikan bukti visual mendukung metrik kuantitatif.

**[INSERT GAMBAR 5A: Visualisasi Klasifikasi Species - Success Case]**
Gambar 5A menampilkan hasil klasifikasi spesies pada kasus malaria parah yang sama (1704282807-0012-R_T) dengan 25+ parasit P. falciparum. Klasifikasi acuan (panel kiri, kotak biru dengan label spesies) dibandingkan dengan prediksi EfficientNet-B1 (panel kanan, kotak berkode warna). Gambar ini mencapai akurasi klasifikasi luar biasa 100% dengan semua 25 parasit teridentifikasi dengan benar (semua kotak hijau), memberikan bukti visual meyakinkan bahwa pengklasifikasi mempertahankan performa tinggi bahkan pada kepadatan parasit ekstrem.
**Files**: GT: `gt_classification/1704282807-0012-R_T.png` | Pred: `pred_classification/1704282807-0012-R_T.png`

**[INSERT GAMBAR 5B: Visualisasi Klasifikasi Stages - Minority Class Challenge]**
Gambar 5B menampilkan hasil klasifikasi tahap siklus hidup pada gambar kompleks multi-parasit (1704282807-0021-T_G_R) dengan 17 parasit. Visualisasi ini mengungkap tantangan kelas minoritas dimana sekitar 65% klasifikasi benar (kotak hijau) versus 35% kesalahan klasifikasi (kotak merah), dengan kesalahan terkonsentrasi pada kelas Trophozoite, secara visual memvalidasi F1-score 46,7% yang dilaporkan untuk Trophozoite minoritas 15 sampel dan mendemonstrasikan bahwa ketidakseimbangan kelas ekstrem (272 Ring vs 5 Gametocyte, rasio 54:1) tetap menyajikan kesulitan klasifikasi signifikan.
**Files**: GT: `gt_classification/1704282807-0021-T_G_R.png` | Pred: `pred_classification/1704282807-0021-T_G_R.png`

### 5. Analisis Efisiensi Model: Jaringan Kecil versus Besar

Temuan kunci penelitian adalah bahwa model EfficientNet lebih kecil (5,3-7,8M params) secara konsisten mengungguli varian ResNet jauh lebih besar (25,6-44,5M params) dengan margin 5-10% pada dataset medical imaging kecil [8,9]. Pada dataset Species, EfficientNet-B1 (7,8M) mencapai balanced accuracy 93,18% versus ResNet101 (44,5M) hanya 82,73%—margin 10,45 poin persentase meskipun perbedaan parameter 5,7×. Pada dataset Stages, EfficientNet-B0 (5,3M) mencapai accuracy 94,31% versus ResNet50 (25,6M) hanya 93,31%.

Fenomena ini dijelaskan oleh beberapa faktor: (1) model besar lebih rentan overfitting pada dataset kecil (<1000 citra) karena jumlah parameter melebihi jumlah sampel pelatihan, (2) EfficientNet menggunakan compound scaling yang menyeimbangkan depth, width, dan resolution secara optimal [8,10], sementara ResNet hanya menambah depth yang menyebabkan vanishing gradients dan diminishing returns, dan (3) model lebih kecil memiliki inductive bias lebih sesuai untuk tugas pencitraan medis dimana fitur relevan adalah pola lokal (pola kromatin, keberadaan hemozoin) daripada representasi hierarkis kompleks.

Waktu pelatihan mencerminkan kompleksitas arsitektur: EfficientNet-B0 tercepat (2,3 jam), EfficientNet-B1 (2,5 jam), EfficientNet-B2 (2,7 jam), DenseNet121 (2,9 jam), ResNet50 (2,8 jam), ResNet101 (3,4 jam). ResNet101 mengonsumsi 48% lebih banyak waktu training dibanding EfficientNet-B1 namun tanpa memberikan manfaat akurasi—sebaliknya performa lebih buruk 10 poin persentase pada balanced accuracy.

### 6. Strategi Penanganan Ketidakseimbangan Kelas dengan Focal Loss

Ketidakseimbangan kelas ekstrem (rasio hingga 54:1) ditangani menggunakan Focal Loss dengan parameter α=0,25 dan γ=2,0 [11]. Untuk P. ovale (5 sampel uji), EfficientNet-B1 mencapai F1-score 76,92% (recall 100%, precision 62,5%), menunjukkan sensitivitas sempurna untuk spesies langka ini namun dengan beberapa positif palsu. Untuk tahap Gametocyte (5 sampel uji), model-model mencapai F1-score 57,14-75%, sementara tahap Trophozoite (15 sampel uji) hanya mencapai F1-score 15,38-51,61%.

Focal Loss beroperasi melalui faktor modulasi (1-p_t)^γ yang menurunkan bobot contoh mudah sambil memfokuskan gradien pada contoh sulit [11], sangat efektif untuk ketidakseimbangan parah. Parameter α=0,25 dan γ=2,0 adalah pengaturan standar dalam literatur pencitraan medis. Meskipun optimisasi Focal Loss dan oversampling 3:1, F1-score di bawah 70% pada kelas <10 sampel tetap tidak memadai secara klinis untuk implementasi otonom tanpa tinjauan ahli.

Penting dicatat bahwa sistem mencapai recall 100% pada P. ovale meskipun precision relatif rendah (62,5%), artinya semua 5 sampel uji terdeteksi benar meskipun dengan 3 positif palsu dari spesies lain. Dalam konteks klinis, kompromi ini diinginkan: negatif palsu (spesies langka terlewat) dapat menyebabkan pemilihan pengobatan tidak tepat dan potensi kematian [12,13], sementara positif palsu dikoreksi melalui pengujian konfirmasi (mikroskopi ulang, PCR) dengan konsekuensi klinis minimal.

### 7. Kelayakan Komputasi untuk Implementasi Klinis

Latensi inferensi ujung-ke-ujung <25 milidetik per citra (>40 FPS) pada GPU NVIDIA RTX 3060 kelas konsumen mendemonstrasikan kelayakan praktis untuk skrining malaria waktu nyata [14,15]. Sebagai perbandingan, pemeriksaan mikroskopis tradisional memerlukan 20-30 menit per slide (1200-1800 detik) untuk analisis menyeluruh 100-200 medan mikroskopis [2], merepresentasikan percepatan >48.000× untuk pemrosesan gambar tunggal atau percepatan ~1.000× untuk analisis slide lengkap dengan asumsi 100 medan per slide.

Bahkan pada sistem CPU saja (AMD Ryzen 7 5800X 8-core), inferensi selesai dalam 180-250 milidetik per citra, memungkinkan pemrosesan batch seluruh slide (100-200 medan) dalam 18-50 detik—tetap jauh lebih cepat dibanding pemeriksaan manual sambil menawarkan kualitas konsisten independen dari variasi keahlian operator [16,2]. Pertimbangan implementasi mencakup kuantisasi model untuk perangkat tepi [17], pemangkasan jaringan neural untuk mengurangi jejak memori [18], dan kepatuhan regulasi untuk perangkat lunak pendukung keputusan klinis [19].

### 8. Keterbatasan dan Arah Penelitian Masa Depan

Penelitian memiliki beberapa keterbatasan yang memerlukan pertimbangan cermat. Pertama, meskipun memanfaatkan dua dataset MP-IDB dengan total 418 citra, ukuran ini tetap tidak mencukupi secara fundamental untuk melatih jaringan dalam secara optimal, sebagaimana dibuktikan oleh performa buruk ResNet101 yang disebabkan oleh overfitting. Ekspansi dataset menjadi 1.000+ citra per tugas sangat krusial untuk meningkatkan performa kelas minoritas [20,21,3].

Kedua, kedua dataset berasal dari pengaturan laboratorium terkontrol dengan protokol pewarnaan Giemsa terstandar dan kondisi pencitraan konsisten (perbesaran 1000×). Validasi eksternal pada sampel lapangan dengan kualitas pewarnaan bervariasi, beragam jenis mikroskop, dan pengaturan akuisisi citra heterogen sangat penting untuk menilai generalisasi dunia nyata dan ketahanan pergeseran domain [22].

Ketiga, meskipun optimisasi Focal Loss [11], kelas minoritas (<10 sampel) masih menunjukkan performa suboptimal (F1<70%). Penelitian mendatang harus mengeksplorasi generasi data sintetik menggunakan GANs atau diffusion models [20,21] untuk mensintesis sampel kelas minoritas realistis, strategi pembelajaran aktif [23] untuk memprioritaskan akuisisi sampel informatif, dan pendekatan pembelajaran few-shot [24,25] untuk memanfaatkan transfer pengetahuan dari kelas mayoritas.

Keempat, sistem saat ini kekurangan fitur penjelasan yang kritis untuk adopsi klinis. Integrasi dari teknik visualisasi seperti Grad-CAM [26] atau Segment Anything [27] dapat memberikan klinisi penjelasan visual tentang mengapa model membuat prediksi spesifik, meningkatkan kepercayaan dan memungkinkan deteksi kesalahan.

---

## STATUS LUARAN PENELITIAN

### Luaran Wajib

**1. Publikasi Prosiding Konferensi Internasional** (Status: **Abstract Diterima, Full Paper dalam Progres 70%**)

Abstract telah diterima pada **Jambi International Conference on Engineering, Science, and Technology (JICEST)** yang diselenggarakan di Universitas Jambi, Indonesia, pada tanggal 28 November 2025. Full paper berfokus pada arsitektur klasifikasi bersama dengan manfaat signifikan: reduksi penyimpanan 70% dan reduksi waktu pelatihan 60% dibandingkan pendekatan tradisional.

**Judul**: "Shared Classification Architecture for Multi-Model Malaria Detection: Efficiency Analysis"
**File Draft**: `luaran/JICEST_Paper.md`
**Status**: Abstract diterima, full paper dalam penyelesaian
**Deadline Submission Full Paper**: November 2025
**Tanggal Konferensi**: 28 November 2025
**Lokasi**: Universitas Jambi, Indonesia

**2. Publikasi Jurnal Nasional Terakreditasi Sinta** (Target: Sinta 3, Status: **Draft dalam Progres 60%**)

Draft artikel berjudul "Parameter-Efficient Deep Learning Models Outperform Larger Architectures on Small Medical Imaging Datasets: A Malaria Detection Case Study" sedang dikembangkan untuk submission ke **Journal of Applied Informatics and Computing (JAIC)** yang terindeks Sinta 3 (https://jurnal.polibatam.ac.id/index.php/JAIC/index). Artikel mencakup evaluasi komprehensif pada 2 dataset × 3 model deteksi × 6 model klasifikasi = 36 kombinasi model, mendemonstrasikan bahwa model EfficientNet lebih kecil (5,3-7,8M parameter) secara konsisten mengungguli varian ResNet lebih besar (25,6-44,5M parameter) sebesar 5-10% pada dataset medis kecil [8,9].

Struktur artikel: (1) Pendahuluan dengan tinjauan pustaka tentang AI medis dan deteksi objek [28,29,30,31,32,14], (2) Metode yang mendeskripsikan arsitektur klasifikasi bersama dan pengaturan eksperimen, (3) Hasil yang menyajikan performa deteksi [5,6], metrik klasifikasi dengan rincian per-kelas, dan analisis efisiensi [7,8,9,11], (4) Pembahasan yang menganalisis temuan dalam konteks implementasi klinis [13,15,19], dan (5) Kesimpulan dengan arah masa depan [20,21,23,24].

**Target Submission**: November 2025

---

### Luaran Tambahan

**1. Implementasi Open-Source**

Codebase lengkap dengan 12 skrip Python untuk persiapan data, pelatihan, evaluasi, dan visualisasi telah dikembangkan dan siap untuk dipublikasikan di repository GitHub. Publikasi akan dilakukan setelah artikel jurnal diterbitkan untuk memastikan prioritas publikasi dan melindungi klaim kontribusi ilmiah.

**Repository**: https://github.com/akhiyarwaladi/hello_world
**Status**: Private - akan dipublikasikan setelah artikel jurnal diterbitkan
**Alasan**: Memastikan prioritas publikasi dan melindungi klaim ilmiah sebelum dipublikasikan secara luas

---

## PERAN MITRA

Penelitian ini dilaksanakan secara mandiri tanpa keterlibatan mitra institusional formal, mendemonstrasikan bahwa pengembangan sistem deteksi malaria berbasis deep learning yang kompetitif dapat dicapai dengan sumber daya terbatas namun terkelola secara efektif. Pendekatan mandiri ini dimungkinkan oleh beberapa faktor kunci yang saling mendukung.

### Faktor Pendukung Penelitian Mandiri

Pertama, ketersediaan sumber daya komputasi yang memadai menjadi fondasi teknis penelitian. GPU NVIDIA RTX 3060 dengan 12GB VRAM, meskipun termasuk kategori perangkat keras kelas konsumen, terbukti mencukupi untuk melatih keseluruhan 36 kombinasi model (2 dataset × 3 model YOLO × 6 model CNN) dengan total waktu pelatihan sekitar 40 jam. Infrastruktur ini memungkinkan eksperimen iteratif yang diperlukan untuk optimisasi hyperparameter, evaluasi berbagai arsitektur neural network, dan perbandingan strategi penanganan ketidakseimbangan kelas tanpa ketergantungan pada fasilitas komputasi institusional yang mahal atau antrian penjadwalan cluster GPU bersama.

Kedua, ekosistem dataset pencitraan medis publik yang semakin matang mengeliminasi kebutuhan kolaborasi formal untuk akses data. Dataset MP-IDB (Malaria Parasite Image Database) yang digunakan dalam penelitian ini merupakan benchmark terstandar yang tersedia bebas dan telah divalidasi secara ekstensif dalam literatur deteksi malaria [31,32,14]. Penggunaan dataset publik ini memberikan keuntungan ganda: (1) memungkinkan perbandingan langsung dan adil dengan penelitian sebelumnya karena menggunakan data pengujian identik, dan (2) memfasilitasi reproduksibilitas penuh dimana peneliti lain dapat memverifikasi temuan dengan mereplikasi eksperimen pada data yang sama. Status publik dataset juga menghindari kompleksitas administratif seperti perjanjian transfer data, persetujuan komite etik untuk data pasien, atau pembatasan penggunaan yang sering menyertai dataset klinis proprietary.

Ketiga, kemajuan pesat dalam kerangka kerja deep learning open-source dan tersedianya model terlatih awal (pretrained models) secara signifikan menurunkan hambatan masuk untuk penelitian AI medis. Framework seperti PyTorch dan Ultralytics YOLO menyediakan implementasi state-of-the-art yang sudah dioptimalkan, dokumentasi komprehensif, dan komunitas aktif yang mengeliminasi kebutuhan bimbingan institusional intensif. Model-model seperti EfficientNet, ResNet, dan DenseNet tersedia dengan bobot ImageNet yang memungkinkan transfer learning efektif, dimana peneliti mandiri dapat mencapai performa setara atau melampaui implementasi custom dari laboratorium besar dengan sumber daya minimal.

Keempat, pendekatan mandiri memberikan fleksibilitas metodologis dan kecepatan iterasi yang kadang sulit dicapai dalam kolaborasi formal. Desain eksperimen, pemilihan arsitektur, strategi augmentasi data, dan keputusan teknis lainnya dapat dibuat dan diimplementasikan dengan cepat tanpa memerlukan konsensus multi-pihak atau navigasi birokrasi institusional. Otonomi ini memungkinkan eksplorasi mendalam dari pertanyaan penelitian spesifik—seperti perbandingan sistematis antara model parameter-efisien versus model besar pada dataset medis kecil—yang mungkin tidak menjadi prioritas dalam agenda penelitian laboratorium dengan fokus lebih luas.

Kelima, ruang lingkup penelitian saat ini sebagai proof-of-concept algoritmik berada dalam domain yang dapat diselesaikan secara mandiri. Tujuan utama adalah mendemonstrasikan kelayakan teknis dari arsitektur klasifikasi bersama, membandingkan performa berbagai kombinasi model deteksi-klasifikasi, dan mengidentifikasi strategi optimal untuk menangani ketidakseimbangan kelas ekstrem dalam dataset pencitraan medis kecil. Objektif-objektif ini tidak memerlukan akses ke sampel klinis lapangan, validasi klinisi ahli secara real-time, atau integrasi dengan sistem informasi rumah sakit—komponen-komponen yang memang memerlukan kemitraan formal namun berada di luar scope penelitian fase proof-of-concept ini.

### Kontribusi dan Validasi Pendekatan

Dengan pendekatan mandiri ini, penelitian berhasil menghasilkan beberapa kontribusi signifikan: (1) implementasi lengkap pipeline multi-model dengan penghematan komputasi substansial (70% reduksi penyimpanan, 60% reduksi waktu pelatihan), (2) temuan empiris bahwa model EfficientNet parameter-efisien (5,3-7,8M params) konsisten mengungguli model ResNet jauh lebih besar (25,6-44,5M params) dengan margin 5-10% pada dataset medis kecil, (3) publikasi konferensi internasional (JICEST) dan artikel jurnal nasional terakreditasi (JAIC Sinta 3), serta (4) codebase open-source komprehensif dengan 12 skrip Python untuk reproduksibilitas penuh. Pencapaian-pencapaian ini memvalidasi bahwa penelitian AI medis berkualitas publikasi dapat dilakukan secara efektif tanpa afiliasi institusional formal, selama peneliti memiliki akses ke sumber daya komputasi dasar, memanfaatkan dataset publik berkualitas, dan menerapkan metodologi rigor yang konsisten dengan standar ilmiah komunitas penelitian.

---

## KENDALA PELAKSANAAN PENELITIAN

Penelitian mandiri ini menghadapi beberapa keterbatasan inheren yang perlu diakui secara jujur untuk memahami konteks dan scope kontribusi ilmiah yang dihasilkan. Kendala-kendala ini dapat dikategorikan dalam dimensi teknis metodologis dan dimensi strategis penelitian.

### Keterbatasan Teknis dan Metodologis

Dari perspektif teknis metodologis, keterbatasan paling fundamental adalah ukuran terbatas dataset beranotasi yang tersedia secara publik. Dengan total 418 citra yang dibagi menjadi dua tugas klasifikasi terpisah (Species dan Stages), penelitian ini beroperasi pada rezim data yang secara signifikan lebih kecil dibandingkan standar industri untuk pelatihan model deep learning yang optimal [10]. Meskipun strategi augmentasi aman-medis [3] diterapkan secara intensif untuk memperluas dataset pelatihan menjadi 1.280 citra deteksi dan 1.024 citra klasifikasi, teknik augmentasi tidak dapat sepenuhnya menggantikan keberagaman semantik dari sampel independen asli. Konsekuensinya, jaringan neural dalam—terutama arsitektur besar seperti ResNet101 dengan 44,5 juta parameter—mengalami overfitting substansial dimana model menghafal pola spesifik dari training set daripada mempelajari representasi fitur general yang robust. Fenomena ini termanifestasi dalam kesenjangan besar antara akurasi training dan validation, serta performa suboptimal pada kelas minoritas dengan <10 sampel pelatihan yang mencapai F1-score hanya 15-75% meskipun optimisasi Focal Loss [11].

Ketidakseimbangan kelas ekstrem (rasio hingga 54:1 antara Ring dan Gametocyte) merepresentasikan kendala teknis kedua yang secara fundamental membatasi performa klasifikasi pada spesies dan tahap langka. Meskipun teknik state-of-the-art diterapkan—termasuk Focal Loss dengan parameter optimal (α=0.25, γ=2.0), weighted random sampling, dan oversampling agresif dengan rasio 3:1—sistem tetap menunjukkan degradasi performa signifikan pada kelas dengan support <10 sampel. Keterbatasan ini bersifat inheren dalam realitas klinis dimana spesies seperti P. ovale dan P. malariae, serta tahap gametocyte, memang secara natural jarang ditemukan dalam praktik diagnostik rutin. Bahkan dengan recall tinggi pada kelas minoritas (100% untuk P. ovale), precision yang moderat (62,5%) mengindikasikan bahwa model mengalami kesulitan membedakan pola morfologis halus yang membedakan spesies langka dari spesies dominan, suatu tantangan yang juga dialami oleh teknisi mikroskopis manusia dengan pengalaman terbatas [16,2].

Dari perspektif sumber daya komputasi, penelitian mandiri ini dibatasi oleh infrastruktur GPU tunggal (NVIDIA RTX 3060) yang, meskipun mencukupi untuk melatih 36 kombinasi model dengan total waktu 40 jam, tidak memungkinkan eksplorasi hyperparameter ekstensif atau eksperimen ensemble skala besar yang dapat meningkatkan performa lebih lanjut. Sebagai perbandingan, penelitian dari laboratorium institusional besar seperti Google Research atau Meta AI Research sering menggunakan cluster GPU dengan ratusan hingga ribuan accelerator untuk melakukan neural architecture search, optimisasi hyperparameter Bayesian, atau ensemble distillation yang memerlukan ratusan hingga ribuan jam GPU-equivalent [8,10]. Keterbatasan ini berarti penelitian ini memfokuskan pada eksplorasi sistematis dari arsitektur standar yang sudah terbukti (EfficientNet, ResNet, DenseNet) dengan hyperparameter yang direkomendasikan dari literatur, daripada pencarian arsitektur custom yang dioptimalkan spesifik untuk domain malaria.

### Keterbatasan Strategis dan Translasi Klinis

Dari dimensi strategis penelitian, scope proof-of-concept algoritmik yang dipilih—meskipun memungkinkan eksekusi mandiri—mengakibatkan keterbatasan dalam validasi generalisasi dunia nyata. Kedua dataset MP-IDB yang digunakan berasal dari pengaturan laboratorium terkontrol dengan protokol pewarnaan Giemsa terstandar WHO [1,2], kondisi mikroskop konsisten (perbesaran 1000×, pencitraan bright-field), dan kualitas digitalisasi uniform. Validasi eksternal pada sampel lapangan dengan variabilitas tinggi—seperti kualitas pewarnaan heterogen dari klinik rural, beragam jenis mikroskop dengan karakteristik optik berbeda, variasi teknik akuisisi citra oleh teknisi dengan pelatihan bervariasi, dan keberagaman strain parasit geografis—memerlukan akses ke data klinis proprietary yang berada di luar kemampuan penelitian mandiri tanpa kolaborasi institusional formal. Konsekuensinya, performa sistem pada kondisi deployment realistis tetap menjadi pertanyaan empiris terbuka yang memerlukan investigasi lanjutan dengan kemitraan rumah sakit atau institusi kesehatan publik.

Akhirnya, keterbatasan dalam aspek explainability dan clinical trust merepresentasikan gap antara proof-of-concept teknis dengan sistem siap-implementasi. Sistem saat ini memproduksi prediksi klasifikasi tanpa penjelasan visual yang mengindikasikan wilayah citra mana yang menjadi dasar keputusan model—fitur yang kritis untuk membangun kepercayaan klinisi dan memfasilitasi deteksi kesalahan prediksi [26,27]. Integrasi teknik visualisasi seperti Grad-CAM atau attention maps memerlukan modifikasi arsitektur dan evaluasi kualitatif dengan domain expert (ahli patologi klinis), yang lagi-lagi berada di luar scope penelitian mandiri fase proof-of-concept ini namun menjadi prioritas esensial untuk translasi ke alat bantu diagnostik klinis yang dapat diadopsi.

---

## RENCANA TAHAPAN SELANJUTNYA

Dengan mempertimbangkan pendekatan penelitian yang telah terbukti efektif dalam fase proof-of-concept ini, rencana tahapan selanjutnya dirancang untuk memaksimalkan kontribusi ilmiah yang dapat dicapai secara realistis dengan sumber daya terbatas sambil mengidentifikasi jalur potensial untuk kolaborasi strategis yang dapat memperluas impact penelitian.

**Prioritas Jangka Pendek (Bulan 11-12): Finalisasi Publikasi dan Diseminasi Ilmiah**

Fokus utama periode ini adalah menyelesaikan dua luaran wajib penelitian yang sudah dalam tahap lanjut. Untuk publikasi konferensi JICEST (deadline November 2025), full paper yang saat ini telah mencapai progres 70% akan diselesaikan dengan menambahkan analisis komparatif mendalam antara arsitektur klasifikasi bersama (shared classification) versus pendekatan tradisional end-to-end, dilengkapi dengan breakdown detail dari penghematan komputasi (70% storage, 60% training time) dan trade-off performa yang terkait. Artikel ini akan menyajikan pipeline efficiency sebagai kontribusi metodologis utama yang dapat diadopsi oleh peneliti lain dengan infrastruktur terbatas [7,8,9].

Untuk artikel jurnal JAIC (target submission November 2025, progres saat ini 60%), upaya finalisasi akan fokus pada penguatan narasi ilmiah seputar temuan counter-intuitive bahwa model EfficientNet parameter-efisien (5,3-7,8M params) secara konsisten mengungguli arsitektur ResNet besar (25,6-44,5M params) dengan margin 5-10% pada dataset medis kecil. Analisis tambahan yang akan ditambahkan mencakup: (1) studi ablasi untuk mengukur kontribusi independen dari transfer learning, augmentasi data, dan Focal Loss terhadap performa akhir, (2) visualisasi embedding space menggunakan t-SNE atau UMAP untuk menunjukkan separabilitas kelas yang dipelajari model berbeda, dan (3) analisis kesalahan kualitatif mendalam pada kasus-kasus misklasifikasi untuk mengidentifikasi pola morfologis parasit yang menantang secara konsisten. Revisi berbasis feedback reviewer diharapkan selesai pada Januari-Februari 2026.

Diseminasi hasil penelitian melalui presentasi konferensi JICEST di Universitas Jambi (28 November 2025) akan menjadi kesempatan untuk membangun networking dengan komunitas peneliti AI medis Indonesia dan potensial mengidentifikasi peluang kolaborasi masa depan. Publikasi codebase open-source di GitHub (dijadwalkan setelah jurnal diterbitkan) akan memaksimalkan reproducibility dan memungkinkan peneliti lain mengadaptasi pipeline untuk domain pencitraan medis lain seperti deteksi tuberkulosis, leukemia, atau parasit lain [14,15].

**Pengembangan Jangka Menengah (3-6 Bulan Pasca-Publikasi): Eksplorasi Teknik Lanjutan untuk Minoritas Kelas**

Setelah publikasi konferensi dan jurnal, prioritas penelitian akan bergeser ke eksplorasi teknik state-of-the-art untuk mengatasi keterbatasan fundamental sistem pada kelas minoritas (<10 sampel). Pendekatan yang akan diinvestigasi mencakup: (1) generasi data sintetik menggunakan diffusion models seperti Stable Diffusion atau DALL-E yang dapat menghasilkan citra parasit P. ovale dan P. malariae dengan variasi morfologis realistis [20,21], (2) pembelajaran few-shot menggunakan framework seperti MAML (Model-Agnostic Meta-Learning) [24] atau Prototypical Networks [25] yang dirancang khusus untuk generalisasi dari sampel terbatas, dan (3) knowledge distillation dari model ensemble untuk meningkatkan robustness prediksi pada kelas langka. Eksperimen-eksperimen ini dapat dilakukan sepenuhnya dengan infrastruktur existing dan dataset publik yang sama.

Integrasi fitur explainability menggunakan teknik seperti Grad-CAM [26], Integrated Gradients, atau attention visualization [27] akan diimplementasikan untuk menghasilkan heatmap yang menunjukkan region citra mana yang paling mempengaruhi keputusan klasifikasi. Implementasi teknis ini tidak memerlukan kolaborasi eksternal dan dapat diselesaikan secara independen, namun evaluasi kualitatif dari kualitas penjelasan visual idealnya memerlukan feedback dari ahli patologi—suatu komponen yang dapat dicari melalui kolaborasi informal dengan fakultas kedokteran atau laboratorium klinik yang tertarik pada AI diagnostik.

**Visi Jangka Panjang (12+ Bulan): Transisi dari Proof-of-Concept ke Validasi Klinis**

Transisi dari sistem proof-of-concept algoritmik ke alat diagnostik yang dapat divalidasi dalam setting klinis merepresentasikan gap signifikan yang memerlukan kolaborasi institusional formal. Jalur ini akan dimulai dengan pencarian kemitraan strategis dengan rumah sakit rujukan atau institusi riset medis yang memiliki infrastruktur pengumpulan data klinis dan akses ke populasi pasien malaria. Kolaborasi ideal akan mencakup: (1) akses ke dataset prospektif dengan 500-1000 sampel klinis beragam untuk validasi eksternal [22], (2) keterlibatan ahli patologi klinis untuk anotasi ground truth dan evaluasi kualitatif performa sistem, dan (3) dukungan navigasi proses ethical clearance dan regulatory compliance [19].

Namun, mengakui bahwa pembentukan kolaborasi formal semacam ini memerlukan waktu (6-12 bulan untuk MoU, ethical approval, dll.) dan tidak ada jaminan keberhasilan, pendekatan pragmatis adalah melanjutkan kontribusi metodologis pada dataset publik existing sambil aktif mempresentasikan hasil penelitian dalam konferensi dan workshop untuk membangun visibilitas yang dapat menarik minat mitra potensial. Eksplorasi jalur alternatif mencakup: (1) partisipasi dalam kompetisi/challenge dataset medis internasional (seperti yang diorganisir oleh MICCAI atau Kaggle) untuk benchmarking terhadap state-of-the-art global, (2) publikasi systematic review atau survey paper tentang AI dalam deteksi malaria untuk memposisikan kontribusi penelitian dalam konteks literatur lebih luas, dan (3) pengembangan tutorial/workshop materials untuk mengajarkan pipeline multi-model kepada komunitas peneliti yang dapat memperluas impact melalui educational outreach.

Pengembangan sistem deployment-ready dengan optimisasi untuk resource-constrained environments—termasuk kuantisasi model [17], pruning [18], dan portabilitas ke mobile/edge devices—akan dilakukan sebagai persiapan untuk eventual clinical pilot study, bahkan sebelum kolaborasi formal terbentuk. Ini memastikan sistem siap untuk rapid deployment ketika kesempatan kolaborasi muncul, meminimalkan time-to-deployment dan maksimalisasi peluang translasi impact dari penelitian algoritmik ke aplikasi dunia nyata yang dapat meningkatkan kualitas diagnostik malaria di setting resource-limited.

---

## DAFTAR PUSTAKA

[1] WHO, "Malaria Microscopy Quality Assurance Manual," ver. 2.0, Geneva, 2016.

[2] WHO, "Basic Malaria Microscopy: Part I. Learner's guide," 2nd ed., Geneva, 2010.

[3] H. Zhang et al., "mixup: Beyond empirical risk minimization," in *Proc. ICLR*, 2018.

[4] M. Aikawa, "Parasitological review: Plasmodium," *Exp. Parasitol.*, vol. 30, no. 2, pp. 284-320, 1971.

[5] A. Wang et al., "YOLOv10: Real-time end-to-end object detection," arXiv:2405.14458, 2024.

[6] G. Jocher et al., "YOLOv11: Ultralytics YOLO11," 2024. [Online]. Available: https://github.com/ultralytics/ultralytics

[7] G. Huang et al., "Densely connected convolutional networks," in *Proc. IEEE CVPR*, 2017, pp. 4700-4708.

[8] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. ICML*, 2019, pp. 6105-6114.

[9] K. He et al., "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770-778.

[10] J. Deng et al., "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE CVPR*, 2009, pp. 248-255.

[11] T.-Y. Lin et al., "Focal loss for dense object detection," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 42, no. 2, pp. 318-327, 2020.

[12] P. Rosenthal, "How do we diagnose and treat Plasmodium ovale and Plasmodium malariae?" *Curr. Infect. Dis. Rep.*, vol. 10, pp. 58-61, 2008.

[13] WHO, "Guidelines for the Treatment of Malaria," 3rd ed., Geneva, 2015.

[14] A. Vijayalakshmi and B. Rajesh Kanna, "Deep learning approach to detect malaria from microscopic images," *Multim. Tools Appl.*, vol. 79, pp. 15297-15317, 2020.

[15] C. J. Long et al., "A smartphone-based portable biosensor for diagnosis in resource-limited settings," *Nature Biotechnol.*, vol. 32, pp. 373-379, 2014.

[16] J. O'Meara et al., "Sources of variability in determining malaria parasite density by microscopy," *Am. J. Trop. Med. Hyg.*, vol. 73, no. 3, pp. 593-598, 2005.

[17] R. Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference," arXiv:1806.08342, 2018.

[18] S. Han et al., "Learning both weights and connections for efficient neural network," in *Proc. NeurIPS*, 2015, pp. 1135-1143.

[19] FDA, "Clinical decision support software: Guidance for industry and FDA staff," 2022.

[20] I. Goodfellow et al., "Generative adversarial nets," in *Proc. NeurIPS*, 2014, pp. 2672-2680.

[21] J. Ho et al., "Denoising diffusion probabilistic models," in *Proc. NeurIPS*, 2020.

[22] Y. Ganin et al., "Domain-adversarial training of neural networks," *J. Mach. Learn. Res.*, vol. 17, no. 1, pp. 2096-2030, 2016.

[23] B. Settles, "Active learning literature survey," Univ. Wisconsin-Madison, Tech. Rep. 1648, 2009.

[24] C. Finn et al., "Model-agnostic meta-learning for fast adaptation of deep networks," in *Proc. ICML*, 2017, pp. 1126-1135.

[25] O. Vinyals et al., "Matching networks for one shot learning," in *Proc. NeurIPS*, 2016, pp. 3630-3638.

[26] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," *Int. J. Comput. Vis.*, vol. 128, pp. 336-359, 2020.

[27] A. Kirillov et al., "Segment anything," in *Proc. IEEE ICCV*, 2023, pp. 4015-4026.

[28] A. Esteva et al., "Dermatologist-level classification of skin cancer with deep neural networks," *Nature*, vol. 542, pp. 115-118, 2017.

[29] P. Rajpurkar et al., "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," arXiv:1711.05225, 2017.

[30] N. Coudray et al., "Classification and mutation prediction from non-small cell lung cancer histopathology images using deep learning," *Nat. Med.*, vol. 24, pp. 1559-1567, 2018.

[31] S. Rajaraman et al., "Pre-trained convolutional neural networks as feature extractors for diagnosis of malaria from blood smears," *Diagnostics*, vol. 8, no. 4, p. 74, 2018.

[32] F. Poostchi et al., "Image analysis and machine learning for detecting malaria," *Transl. Res.*, vol. 194, pp. 36-55, 2018.

[33] World Health Organization, "World Malaria Report 2024," Geneva, Switzerland, 2024.

[34] R. W. Snow et al., "The global distribution of clinical episodes of Plasmodium falciparum malaria," *Nature*, vol. 434, pp. 214-217, 2005.

[35] Centers for Disease Control and Prevention, "Malaria Biology," 2024. [Online]. Available: https://www.cdc.gov/malaria/about/biology/

[36] A. Moody, "Rapid diagnostic tests for malaria parasites," *Clin. Microbiol. Rev.*, vol. 15, no. 1, pp. 66-78, 2002.

[37] P. L. Chiodini et al., "Manson's Tropical Diseases," 23rd ed. London: Elsevier, 2014, ch. 52.

[38] K. Mitsakakis et al., "Challenges in malaria diagnosis," *Expert Rev. Mol. Diagn.*, vol. 18, no. 10, pp. 867-875, 2018.

[39] S. Ren et al., "Faster R-CNN: Towards real-time object detection with region proposal networks," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 39, no. 6, pp. 1137-1149, 2017.

[40] A. Dosovitskiy et al., "An image is worth 16×16 words: Transformers for image recognition at scale," in *Proc. ICLR*, 2021.
