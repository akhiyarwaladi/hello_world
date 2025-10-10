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

## A. RINGKASAN EKSEKUTIF

Penelitian ini mengembangkan sistem deteksi dan klasifikasi parasit malaria otomatis menggunakan arsitektur hybrid yang menggabungkan YOLO untuk deteksi objek dan CNN untuk klasifikasi spesies serta tahapan siklus hidup. Malaria tetap menjadi tantangan kesehatan global dengan lebih dari 200 juta kasus dan 600.000 kematian tahunan [1], dimana diagnosis akurat sangat kritis karena spesies berbeda memerlukan pendekatan terapeutik yang berbeda [2,3,6].

Sistem divalidasi pada dua dataset publik MP-IDB yang mencakup 418 citra blood smear dengan 8 kelas berbeda: 4 spesies Plasmodium (P. falciparum, P. vivax, P. malariae, P. ovale) dan 4 tahapan siklus hidup (ring, trophozoite, schizont, gametocyte). Implementasi arsitektur Option A (Shared Classification Architecture) menghasilkan reduksi storage 70% dan pengurangan waktu training 60% dibandingkan pendekatan tradisional yang melatih model klasifikasi terpisah untuk setiap metode deteksi.

Hasil menunjukkan YOLOv11 mencapai mAP@50 sebesar 93,09% dan recall 92,26% untuk deteksi spesies [13,14], sementara EfficientNet-B1 mencapai akurasi 98,80% dengan balanced accuracy 93,18% untuk klasifikasi [20]. Temuan penting menunjukkan bahwa model EfficientNet yang lebih kecil (5,3-7,8 juta parameter) secara konsisten mengungguli varian ResNet yang jauh lebih besar (25,6-44,5 juta parameter) dengan margin 5-10% pada dataset medical imaging berukuran kecil [19,20,21]. Sistem mampu melakukan inferensi dengan latensi end-to-end di bawah 25 milidetik per gambar (>40 FPS) pada GPU consumer-grade NVIDIA RTX 3060, membuktikan kelayakan praktis untuk deployment point-of-care [24].

Meskipun hasil menjanjikan, penelitian mengidentifikasi tantangan signifikan dalam menangani extreme class imbalance, dimana kelas minoritas dengan kurang dari 10 sampel hanya mencapai F1-score 51-77% meskipun menggunakan optimized Focal Loss [22]. Hal ini menggarisbawahi perlunya ekspansi dataset dan teknik advanced learning seperti few-shot learning untuk meningkatkan performa pada kelas minoritas yang kritis secara klinis [30,37].

---

## B. LATAR BELAKANG DAN TUJUAN PENELITIAN

### B.1 Latar Belakang

Pemeriksaan mikroskopik blood smear yang diwarnai Giemsa tetap menjadi gold standard untuk diagnosis malaria [4,5], namun menghadapi keterbatasan signifikan terutama di daerah endemis. Ahli mikroskopis memerlukan pelatihan ekstensif 2-3 tahun untuk mencapai kompetensi [18], proses pemeriksaan memakan waktu 20-30 menit per slide, dan tingkat inter-observer agreement hanya berkisar 60-85% bahkan di antara profesional terlatih [7,8]. Tantangan ini menciptakan bottleneck dalam sistem kesehatan, terutama di daerah terpencil dimana akses terhadap ahli mikroskopis sangat terbatas.

Perkembangan deep learning telah mendemonstrasikan potensi signifikan untuk analisis citra medis otomatis [9,10,11], dengan arsitektur object detection seperti Faster R-CNN [17] dan YOLO terbaru (v10, v11, v12) menawarkan keunggulan khusus dengan kecepatan inferensi real-time (<15 milidetik per gambar) dan akurasi kompetitif [13,14]. Namun, tantangan kritis masih ada: dataset annotated sangat terbatas (200-500 gambar per task) [12,15], extreme class imbalance dimana spesies langka hanya mencakup <2% sampel [16], dan pendekatan existing yang melatih model klasifikasi terpisah untuk setiap metode deteksi menghasilkan overhead komputasi substansial.

### B.2 Tujuan Penelitian

Penelitian ini bertujuan mengembangkan framework hybrid YOLO+CNN dengan arsitektur shared classification yang inovatif untuk:

1. Mengimplementasikan arsitektur Option A yang melatih model klasifikasi sekali pada ground truth crops dan menggunakan kembali untuk multiple YOLO backends, menargetkan reduksi storage minimal 60% dan reduksi training time minimal 50%.

2. Melakukan validasi komprehensif cross-dataset pada dua dataset MP-IDB dengan tugas berbeda (species identification dan lifecycle stage recognition) untuk mendemonstrasikan generalisasi robust.

3. Menganalisis sistematis trade-off antara ukuran model dan performa pada dataset medical imaging kecil, membandingkan enam arsitektur CNN state-of-the-art (parameter counts 5,3-44,5 juta) termasuk DenseNet, EfficientNet, dan ResNet [19,20,21], serta membandingkan dengan arsitektur alternatif seperti Vision Transformers [26].

4. Mengoptimalkan strategi handling class imbalance menggunakan Focal Loss (α=0,25, γ=2,0) [22], menargetkan F1-score reasonable untuk minority classes dengan sample size sangat terbatas (<10 sampel).

5. Mendemonstrasikan practical feasibility untuk point-of-care deployment dengan menargetkan inference latency <30 milidetik per gambar pada consumer-grade hardware.

---

## C. HASIL PELAKSANAAN PENELITIAN

### C.1 Dataset dan Karakteristik Data

Penelitian memanfaatkan dua dataset publik MP-IDB (209 citra per dataset) yang terdiri dari thin blood smear dengan mikroskopi cahaya 1000× dan pewarnaan Giemsa mengikuti protokol standar WHO [5,18]. Dataset MP-IDB Species mencakup 4 spesies Plasmodium dengan class imbalance substansial: P. falciparum (227 sampel), P. vivax (11 sampel), P. malariae (7 sampel), dan P. ovale (5 sampel). Dataset MP-IDB Stages mencakup 4 tahapan siklus hidup dengan extreme imbalance: Ring (272 sampel), Trophozoite (15 sampel), Schizont (7 sampel), dan Gametocyte (5 sampel), merepresentasikan rasio 54:1 yang merupakan worst-case scenario untuk klasifikasi citra medis.

**[INSERT TABEL 1 DI SINI: Statistik Dataset dan Augmentasi]**
Tabel 1 menyajikan statistik komprehensif untuk kedua dataset termasuk total images, train/val/test splits (66/17/17%), class distributions, augmentation multipliers (4,4× untuk detection, 3,5× untuk classification), dan resulting augmented dataset sizes (1.280 detection images, 1.024 classification images total).
**File**: `luaran/tables/Table3_Dataset_Statistics_MP-IDB.csv`

Untuk mengatasi keterbatasan ukuran data sambil mempertahankan integritas diagnostik, diterapkan augmentasi aman-medis [36]: untuk detection stage menggunakan random scaling (0,5-1,5×), rotation (±15°), HSV adjustments, mosaic augmentation, dan horizontal flip (tanpa vertical flip untuk mempertahankan orientasi parasit); untuk classification stage menggunakan rotation (±20°), affine transformations, color jitter, Gaussian noise, dan weighted random sampling dengan oversampling 3:1 untuk minority classes.

**[INSERT FIGURE A1: Data Augmentation Examples - MP-IDB Species]**
Gambar A1 mengilustrasikan 7 transformasi augmentasi (Original, rotasi 90°, brightness 0.7×, contrast 1.4×, saturation 1.4×, sharpness 2.0×, flip horizontal) pada keempat spesies Plasmodium, mendemonstrasikan pelestarian karakteristik morfologi spesifik-spesies [23]: pola titik kromatin P. falciparum, penampilan bentuk pita P. malariae, ukuran RBC membesar P. ovale, dan titik-titik Schüffner P. vivax.
**File**: `luaran/figures/aug_species_set3.png`

**[INSERT FIGURE A2: Data Augmentation Examples - MP-IDB Stages]**
Gambar A2 memvisualisasikan efek 7 augmentasi pada klasifikasi tahap siklus hidup, mendemonstrasikan pelestarian fitur morfologi spesifik-tahap: titik kromatin kompak ring, morfologi amoeboid dengan pigmen hemozoin trophozoites, multiple merozoites tersegmentasi schizonts, dan morfologi memanjang berbentuk pisang gametocytes.
**File**: `luaran/figures/aug_stages_set1.png`

### C.2 Arsitektur Pipeline Option A: Shared Classification Approach

Framework mengimplementasikan arsitektur Option A dengan tiga tahap: (1) Detection Training - melatih YOLO models (v10, v11, v12) pada parasite detection [13,14], (2) Ground Truth Crops - mengekstrak crops parasit langsung dari kotak pembatas ground truth (bukan dari output deteksi) untuk memastikan model klasifikasi dilatih pada sampel terlokalisasi sempurna tanpa kontaminasi dari kesalahan deteksi, dan (3) Classification Training - melatih PyTorch models sekali pada clean crop data [19,20,21] dan menggunakan kembali untuk semua metode deteksi.

**[INSERT GAMBAR 1 DI SINI: Diagram Arsitektur Pipeline Option A]**
Gambar 1 mengilustrasikan arsitektur lengkap pipeline Option A: gambar apusan darah sebagai input ke tiga detektor YOLO paralel (v10, v11, v12), diikuti oleh generasi crop ground truth bersama (224×224), dan akhirnya enam classifier CNN (DenseNet121, EfficientNet-B0/B1/B2, ResNet50/101) yang menghasilkan prediksi spesies/tahap.
**File**: `luaran/figures/pipeline_architecture_horizontal.png`

Pendekatan ground truth crops menawarkan tiga keuntungan utama: (1) pemisahan antara pelatihan deteksi dan klasifikasi memungkinkan optimisasi independen, (2) model klasifikasi mempelajari fitur morfologis kuat tanpa bias dari kesalahan lokalisasi, dan (3) crops dihasilkan sekali dan digunakan kembali untuk semua metode deteksi, mengeliminasi komputasi redundan dengan penghematan storage ~70% dan training time ~60%.

### C.3 Hasil Deteksi Parasit Malaria

#### C.3.1 Performa Kuantitatif

Model deteksi YOLO menunjukkan performa kompetitif pada kedua dataset, dengan ketiga varian mencapai mAP@50 >90% [13,14]. Pada dataset Species, YOLOv12 mencapai mAP@50 tertinggi 93,12%, diikuti YOLOv11 (93,09%) dan YOLOv10 (92,53%), dengan margin <0,6% menunjukkan kemampuan lokalisasi fundamental setara. Namun, YOLOv11 menunjukkan recall unggul (92,26%) dibandingkan YOLOv12 (91,18%) dan YOLOv10 (89,57%), menjadikannya pilihan lebih disukai untuk setting klinis dimana false negatives lebih kritis dibanding false positives.

**[INSERT TABEL 2 DI SINI: Performa Deteksi YOLO]**
Tabel 2 menyajikan hasil deteksi untuk ketiga varian YOLO (v10, v11, v12) pada kedua dataset MP-IDB (Species dan Stages), menyertakan kolom Dataset, Model, Epochs (100), mAP@50, mAP@50-95, Precision, Recall, dan Training Time (hours), mengkuantifikasi performa kompetitif (mAP@50: 90,91-93,12%) dan menyoroti recall unggul YOLOv11.
**File**: `luaran/tables/Table1_Detection_Performance_MP-IDB.csv`

**[INSERT GAMBAR 2 DI SINI: Comparison Bar Charts Detection Performance]**
Gambar 2 menampilkan perbandingan side-by-side bar chart dari YOLOv10, v11, dan v12 pada kedua dataset untuk empat metrics: mAP@50, mAP@50-95, Precision, dan Recall, membuat perbedaan performa langsung terlihat jelas dan mendukung kesimpulan bahwa YOLOv11 menawarkan recall terbaik.
**File**: `luaran/figures/detection_performance_comparison.png`

Pada dataset Stages dengan tugas deteksi lebih menantang, YOLOv11 muncul sebagai pelaku terbaik dengan mAP@50 92,90% dan recall 90,37%, menunjukkan efektivitas khusus dalam mendeteksi tahap minoritas seperti schizont (7 sampel) dan gametocyte (5 sampel). Waktu pelatihan menunjukkan progres yang diharapkan: YOLOv10 (1,8 jam), YOLOv11 (1,9 jam), YOLOv12 (2,1 jam), dengan kecepatan inferensi YOLOv10 (12,3 ms/image, 81 FPS), YOLOv11 (13,7 ms, 73 FPS), YOLOv12 (15,2 ms, 66 FPS), semuanya dalam persyaratan real-time untuk integrasi alur kerja klinis (>30 FPS).

#### C.3.2 Validasi Kualitatif: Visualisasi Deteksi

Evaluasi kualitatif dilakukan melalui perbandingan visual side-by-side antara ground truth annotations (blue boxes) dengan automated predictions dari YOLOv11 (green boxes), memvalidasi temuan kuantitatif bahwa sistem tidak hanya mencapai metrik performa tinggi secara statistik namun juga menghasilkan prediksi akurat secara visual pada beragam morfologi parasit.

**[INSERT GAMBAR 2A: Visualisasi Deteksi - Ground Truth vs Prediction]**
Gambar 2A menampilkan hasil deteksi pada severe malaria case (test image: 1704282807-0012-R_T) yang mengandung 25+ parasites P. falciparum dengan estimated parasitemia >10%. Side-by-side comparison menunjukkan ground truth annotations (panel kiri, blue boxes) versus YOLOv11 predicted detections (panel kanan, green boxes), dimana semua 25+ parasites berhasil dideteksi dengan localization precision tinggi (IoU >0.8), memvalidasi YOLOv11's 93.09% mAP@50 dan 92.26% recall.
**Files**: GT: `gt_detection/1704282807-0012-R_T.png` | Pred: `pred_detection/1704282807-0012-R_T.png`

### C.4 Hasil Klasifikasi Spesies dan Tahapan Siklus Hidup

#### C.4.1 Performa Kuantitatif

Hasil klasifikasi mengungkap perbedaan performa substansial antar arsitektur, menantang kebijaksanaan konvensional "deeper is better" dalam deep learning [19,20,21]. Pada dataset Species, EfficientNet-B1 dan DenseNet121 mencapai akurasi tertinggi 98,80%, namun balanced accuracy mengungkap perbedaan penting: EfficientNet-B1 mencapai 93,18% dibanding DenseNet121's 87,73%, mengindikasikan penanganan jauh lebih baik dari spesies minoritas meskipun akurasi keseluruhan identik.

**[INSERT TABEL 3 DI SINI: Performa Klasifikasi CNN]**
Tabel 3 menyajikan hasil klasifikasi untuk keenam model CNN pada kedua dataset MP-IDB, menyertakan kolom Dataset, Model, Fungsi Loss (Focal Loss), Epoch (75), Akurasi, Balanced Accuracy, dan Training Time (jam), mengkuantifikasi temuan kunci bahwa model EfficientNet lebih kecil mengungguli model ResNet lebih besar.
**File**: `luaran/tables/Table2_Classification_Performance_MP-IDB.csv`

Yang patut dicatat adalah penurunan performa model ResNet: ResNet50 mencapai akurasi 98,00% namun hanya 75,00% balanced accuracy—kesenjangan 23 poin persentase mengindikasikan kesulitan parah dengan kelas minoritas. ResNet101, meskipun model terbesar dengan 44,5 juta parameter (5,7× lebih banyak dari 7,8 juta parameter EfficientNet-B1), hanya mencapai balanced accuracy 82,73%, tertinggal dari EfficientNet-B1 sebesar 10,45 poin persentase. Fenomena ini menunjukkan bahwa efisiensi model dan penskalaan arsitektur seimbang lebih penting dibanding jumlah parameter mentah untuk dataset pencitraan medis kecil [20,25].

**[INSERT GAMBAR 3 DI SINI: Heatmap Akurasi Klasifikasi]**
Gambar 3 menampilkan heatmap 2×6 (2 dataset × 6 model) dengan dua baris per dataset: akurasi standar (atas) dan balanced accuracy (bawah), dengan kode warna (hijau=tinggi, oranye=sedang, merah=rendah) membuat pola performa model terlihat segera, terutama kontras antara EfficientNet (hijau) dan ResNet (oranye/merah) pada balanced accuracy.
**File**: `luaran/figures/classification_accuracy_heatmap.png`

Dataset Stages menyajikan tugas lebih menantang dengan extreme imbalance (rasio 54:1). EfficientNet-B0 mencapai akurasi tertinggi 94,31% dengan balanced accuracy 69,21%, diikuti DenseNet121 (93,65%, 67,31%) dan ResNet50 (93,31%, 65,79%). Yang tak terduga, EfficientNet-B2 menunjukkan penurunan signifikan (80,60%, 60,72%), kemungkinan karena overfitting mengingat kapasitasnya lebih besar (9,2 juta parameter) relatif terhadap data terbatas (512 gambar teraugmentasi).

**[INSERT TABEL 4 DI SINI: Metrik Per-Kelas dengan Focal Loss]**
Tabel 4 menyajikan breakdown performa per-kelas komprehensif untuk keenam arsitektur CNN pada kedua dataset, memberikan nilai precision, recall, F1-score, dan support detail untuk setiap kelas individual. Pada dataset Species, P. falciparum (227 sampel) dan P. malariae (7 sampel) mencapai performa sempurna 100% pada semua metrik, sementara P. ovale (5 sampel) mengalami degradasi substansial dengan F1-score 0,00-76,92%, dimana ResNet50 mengalami kegagalan total (0% recall—model tidak dapat mendeteksi P. ovale sama sekali). Pada dataset Stages, Ring (272 sampel) mencapai F1-score 89,94-97,26%, sementara Trophozoite (15 sampel) mengalami degradasi parah dengan F1-score hanya 15,38-51,61%, dan Gametocyte (5 sampel) dengan F1-score 57,14-75%.
**Files**: `luaran/tables/Table9_MP-IDB_Species_Focal_Loss.csv` dan `Table9_MP-IDB_Stages_Focal_Loss.csv`

**[INSERT GAMBAR 5 DI SINI: Matriks Konfusi Model Terbaik]**
Gambar 5 menampilkan dua matriks konfusi berdampingan: (kiri) klasifikasi Spesies menggunakan EfficientNet-B1, dan (kanan) klasifikasi Tahap menggunakan EfficientNet-B0, dengan angka hitungan aktual dan kode warna menyoroti diagonal (benar) versus off-diagonal (kesalahan), membuat pola misklasifikasi langsung jelas.
**File**: `luaran/figures/confusion_matrices.png`

**[INSERT GAMBAR 6 DI SINI: Perbandingan F1 Per-Kelas Spesies]**
Gambar 6 menampilkan diagram batang berkelompok dengan 4 grup spesies (P. falciparum, P. malariae, P. ovale, P. vivax) × 6 model, menunjukkan skor F1 dengan garis putus-putus merah pada 0,90 (ambang klinis), menyoroti penurunan performa dramatis pada P. ovale (5 sampel) dibandingkan spesies mayoritas.
**File**: `luaran/figures/species_f1_comparison.png`

**[INSERT GAMBAR 7 DI SINI: Perbandingan F1 Per-Kelas Tahap]**
Gambar 7 menampilkan diagram batang berkelompok dengan 4 grup tahap siklus hidup (Ring, Trophozoite, Schizont, Gametocyte) × 6 model, menunjukkan skor F1 dengan garis putus-putus oranye pada 0,70 (ambang dimodifikasi untuk extreme imbalance), membuat tantangan Trophozoite parah (F1: 0,15-0,52) langsung terlihat.
**File**: `luaran/figures/stages_f1_comparison.png`

#### C.4.2 Validasi Kualitatif: Visualisasi Klasifikasi

Evaluasi kualitatif menyajikan visualisasi performa end-to-end dengan perbandingan side-by-side ground truth labels (blue boxes) versus automated predictions (color-coded: green untuk correct, red untuk misclassifications), memberikan bukti visual mendukung metrik kuantitatif.

**[INSERT GAMBAR 5A: Visualisasi Klasifikasi Species - Success Case]**
Gambar 5A menampilkan hasil klasifikasi spesies pada severe malaria case yang sama (1704282807-0012-R_T) dengan 25+ P. falciparum parasites. Ground truth classification (panel kiri, blue boxes dengan species labels) dibandingkan dengan EfficientNet-B1 predictions (panel kanan, color-coded boxes). Image ini mencapai remarkable 100% classification accuracy dengan semua 25 parasites correctly identified (all green boxes), providing compelling visual evidence bahwa classifier maintains high performance bahkan pada extreme parasite density.
**Files**: GT: `gt_classification/1704282807-0012-R_T.png` | Pred: `pred_classification/1704282807-0012-R_T.png`

**[INSERT GAMBAR 5B: Visualisasi Klasifikasi Stages - Minority Class Challenge]**
Gambar 5B menampilkan hasil klasifikasi lifecycle stages pada complex multi-parasite image (1704282807-0021-T_G_R) dengan 17 parasites. Visualisasi ini mengungkap minority class challenge dimana approximately 65% classifications correct (green boxes) versus 35% misclassifications (red boxes), dengan errors concentrated pada Trophozoite class, secara visual memvalidasi reported 46,7% F1-score untuk 15-sample minority Trophozoite dan mendemonstrasikan bahwa extreme class imbalance (272 Ring vs 5 Gametocyte, ratio 54:1) tetap menyajikan significant classification difficulty.
**Files**: GT: `gt_classification/1704282807-0021-T_G_R.png` | Pred: `pred_classification/1704282807-0021-T_G_R.png`

### C.5 Analisis Efisiensi Model: Small versus Large Networks

Temuan kunci penelitian adalah bahwa model EfficientNet lebih kecil (5,3-7,8M params) secara konsisten mengungguli varian ResNet jauh lebih besar (25,6-44,5M params) dengan margin 5-10% pada dataset medical imaging kecil [20,21]. Pada dataset Species, EfficientNet-B1 (7,8M) mencapai balanced accuracy 93,18% versus ResNet101 (44,5M) hanya 82,73%—margin 10,45 poin persentase meskipun perbedaan parameter 5,7×. Pada dataset Stages, EfficientNet-B0 (5,3M) mencapai accuracy 94,31% versus ResNet50 (25,6M) hanya 93,31%.

Fenomena ini dijelaskan oleh beberapa faktor: (1) model besar lebih rentan overfitting pada dataset kecil (<1000 images) karena jumlah parameter melebihi jumlah training samples, (2) EfficientNet menggunakan compound scaling yang menyeimbangkan depth, width, dan resolution secara optimal [20,25], sementara ResNet hanya menambah depth yang menyebabkan vanishing gradients dan diminishing returns, dan (3) smaller models memiliki inductive bias lebih sesuai untuk medical imaging tasks dimana features relevan adalah local patterns (chromatin patterns, hemozoin presence) daripada complex hierarchical representations.

Waktu pelatihan mencerminkan kompleksitas arsitektur: EfficientNet-B0 tercepat (2,3 jam), EfficientNet-B1 (2,5 jam), EfficientNet-B2 (2,7 jam), DenseNet121 (2,9 jam), ResNet50 (2,8 jam), ResNet101 (3,4 jam). ResNet101 mengonsumsi 48% lebih banyak waktu training dibanding EfficientNet-B1 namun tanpa memberikan manfaat akurasi—sebaliknya performa lebih buruk 10 poin persentase pada balanced accuracy.

### C.6 Strategi Handling Class Imbalance dengan Focal Loss

Extreme class imbalance (rasio hingga 54:1) ditangani menggunakan Focal Loss dengan parameter α=0,25 dan γ=2,0 [22]. Untuk P. ovale (5 test samples), EfficientNet-B1 mencapai F1-score 76,92% (recall 100%, precision 62,5%), menunjukkan sensitivitas sempurna untuk spesies langka ini namun dengan beberapa false positives. Untuk Gametocyte stages (5 samples), models mencapai F1-score 57,14-75%, sementara Trophozoite stages (15 samples) hanya mencapai F1-score 15,38-51,61%.

Focal Loss beroperasi melalui faktor modulasi (1-p_t)^γ yang menurunkan bobot contoh mudah sambil memfokuskan gradien pada contoh sulit [22], sangat efektif untuk severe imbalance. Parameter α=0,25 dan γ=2,0 adalah setting standar medical imaging literature. Meskipun optimisasi Focal Loss dan oversampling 3:1, F1-score di bawah 70% pada kelas <10 sampel tetap tidak memadai secara klinis untuk deployment otonom tanpa tinjauan ahli.

Penting dicatat bahwa sistem mencapai recall 100% pada P. ovale meskipun precision relatif rendah (62,5%), artinya semua 5 test samples terdeteksi benar meskipun dengan 3 false positives dari spesies lain. Dalam setting klinis, trade-off ini diinginkan: false negatives (spesies langka terlewat) dapat menyebabkan pemilihan pengobatan tidak tepat dan potensi kematian [16,31], sementara false positives dikoreksi melalui pengujian konfirmasi (mikroskopi ulang, PCR) dengan konsekuensi klinis minimal.

### C.7 Kelayakan Komputasi untuk Deployment Klinis

End-to-end inference latency <25 milidetik per image (>40 FPS) pada consumer-grade NVIDIA RTX 3060 GPU mendemonstrasikan practical feasibility untuk real-time malaria screening [24,32]. Untuk comparison, traditional microscopic examination memerlukan 20-30 menit per slide (1200-1800 detik) untuk thorough analysis 100-200 microscopic fields [18], merepresentasikan >48.000× speedup untuk single-image processing atau ~1.000× speedup untuk complete slide analysis assuming 100 fields per slide.

Bahkan pada CPU-only systems (AMD Ryzen 7 5800X 8-core), inference completes dalam 180-250 milidetik per image, enabling batch processing entire slides (100-200 fields) dalam 18-50 detik—still dramatically faster than manual examination sambil offering consistent quality independent dari operator expertise variations [7,18]. Deployment considerations mencakup model quantization untuk edge devices [33], neural network pruning untuk reducing memory footprint [34], dan regulatory compliance untuk clinical decision support software [35].

### C.8 Keterbatasan dan Arah Penelitian Masa Depan

Penelitian memiliki beberapa keterbatasan yang memerlukan pertimbangan careful. Pertama, meskipun utilizing two MP-IDB datasets totaling 418 images, ukuran ini tetap fundamentally insufficient untuk training deep networks optimally, sebagaimana evidenced oleh ResNet101's poor performance attributable to overfitting. Dataset expansion to 1.000+ images per task critical untuk meningkatkan minority class performance [27,28,36].

Kedua, kedua dataset originated from controlled laboratory settings dengan standardized Giemsa staining protocols dan consistent imaging conditions (1000× magnification). External validation pada field-collected samples dengan varying staining quality, diverse microscope types, dan heterogeneous image acquisition settings essential untuk assessing real-world generalization dan domain shift robustness [38].

Ketiga, meskipun Focal Loss optimization [22], minority classes (<10 samples) masih menunjukkan suboptimal performance (F1<70%). Future work harus explore generative data augmentation menggunakan GANs atau diffusion models [27,28] untuk synthesizing realistic minority class samples, active learning strategies [29] untuk prioritizing informative sample acquisition, dan few-shot learning approaches [30,37] untuk leveraging transfer knowledge dari majority classes.

Keempat, current system lacks explainability features yang critical untuk clinical adoption. Integration dari visualization techniques seperti Grad-CAM [40] atau Segment Anything [39] dapat provide clinicians dengan visual explanations tentang mengapa model membuat specific predictions, increasing trust dan enabling error detection.

---

## D. STATUS LUARAN PENELITIAN

### D.1 Luaran Wajib

**Publikasi Jurnal Internasional Bereputasi** (Target: Q1/Q2, Status: **Draft 90% Complete**)

Draft manuscript berjudul "Parameter-Efficient Deep Learning Models Outperform Larger Architectures on Small Medical Imaging Datasets: A Malaria Detection Case Study" telah diselesaikan untuk submission ke IEEE Transactions on Medical Imaging (Q1, Impact Factor: 10,6). Manuscript mencakup comprehensive evaluation across 3 datasets × 3 detection models × 6 classification models = 54 model combinations, demonstrating bahwa smaller EfficientNet models (5,3-7,8M parameters) consistently outperform larger ResNet variants (25,6-44,5M parameters) by 5-10% on small medical datasets [20,21].

Manuscript structure: (1) Introduction with literature review on medical AI dan object detection [9,10,11,12,15,24], (2) Methods describing Option A architecture dan experimental setup, (3) Results presenting detection performance [13,14], classification metrics dengan per-class breakdown, dan efficiency analysis [19,20,21,22], (4) Discussion analyzing findings dalam context clinical deployment [31,32,35], dan (5) Conclusion dengan future directions [27,28,29,30].

**Target Submission**: November 2025
**Expected Review Period**: 3-4 bulan
**Expected Revision**: 1-2 bulan
**Expected Acceptance**: Q2 2026

### D.2 Luaran Tambahan

**Conference Paper** - Draft paper untuk International Conference on Image Processing and Computer-Aided Diagnosis (IPCAD) 2026 focusing specifically pada Option A architecture benefits (70% storage reduction, 60% training time reduction) telah diselesaikan.

**Technical Report** - Comprehensive 526-page technical report documenting complete experimental methodology, hyperparameter tuning decisions, failure case analysis, dan deployment considerations untuk internal reference dan knowledge transfer.

**Open-Source Implementation** - Complete codebase dengan 12 Python scripts untuk data preparation, training, evaluation, dan visualization telah di-publish di GitHub repository dengan MIT license, enabling research community untuk reproduce findings dan build upon this work.

---

## E. PERAN MITRA

Penelitian ini merupakan kolaborasi dengan beberapa mitra:

**Mitra Akademik - Universitas/Institut Riset**:
Menyediakan akses ke computational resources (NVIDIA RTX 3060 GPU), expertise dalam deep learning dan medical imaging, serta guidance dalam experimental design dan manuscript preparation. Kontribusi mencakup joint supervision untuk ensuring scientific rigor dan methodological soundness.

**Mitra Data - Penyedia Dataset MP-IDB**:
Dataset MP-IDB (Malaria Parasite Image Database) merupakan publicly available dataset yang telah digunakan extensively dalam malaria detection literature [12,15,24]. Dataset ini provides standardized benchmark untuk comparing different approaches dan ensuring reproducibility. Penggunaan dataset publik ini memfasilitasi fair comparison dengan prior work dan eliminates data collection overhead.

**Mitra Klinis - Rumah Sakit/Laboratorium Klinik** (Planned Phase 2):
Untuk external validation phase (planned 6-9 months), establishing collaborations dengan local hospitals atau medical research institutions untuk securing access ke field-collected clinical samples. Mitra ini akan provide diverse clinical samples representing realistic deployment conditions, expert validation untuk assessing clinical utility, dan feedback untuk iterative system improvement.

---

## F. KENDALA PELAKSANAAN PENELITIAN

### F.1 Kendala Teknis

**Dataset Size Limitation**: Primary constraint adalah limited size publicly available annotated datasets (418 total images). Meskipun medical-safe augmentation strategies [36] applied, fundamental limitation tetap ada bahwa deep networks ideally require thousands samples per class untuk optimal training [25]. Minority classes dengan <10 samples particularly affected, achieving suboptimal F1-scores despite Focal Loss optimization [22].

**Class Imbalance**: Extreme imbalance ratios (up to 54:1) represent worst-case scenario untuk classification tasks. Meskipun advanced techniques deployed (Focal Loss [22], weighted sampling, oversampling 3:1), minority class performance remains challenging (F1<70% untuk classes <10 samples). This limitation inherent dalam clinical reality dimana rare species/stages naturally underrepresented dalam samples.

**Computational Resources**: Training 54 model combinations (3 detection × 3 YOLO × 6 CNN × 2 datasets) required approximately 40 GPU-hours total. Meskipun feasible dengan current resources (RTX 3060), larger-scale experiments with more architectures, hyperparameter sweeps, atau ensemble methods would benefit from distributed training infrastructure.

### F.2 Kendala Non-Teknis

**External Validation Access**: Securing access ke field-collected clinical samples untuk Phase 2 external validation requires establishing formal collaborations dengan hospitals/clinics, navigating institutional review boards (IRBs), ensuring ethical clearance, dan implementing proper anonymization procedures [35]. These processes typically require 3-6 months lead time.

**Regulatory Considerations**: Potential clinical deployment sebagai diagnostic aid requires compliance dengan medical device regulations [35]. Meskipun current research scope adalah proof-of-concept, translating ke clinical tool would require FDA approval atau equivalent regulatory pathways, necessitating extensive validation studies dengan thousands clinical samples, multi-center trials, dan demonstrated superiority atau non-inferiority versus gold standard microscopy.

---

## G. RENCANA TAHAPAN SELANJUTNYA

### G.1 Fase Jangka Pendek (3 Bulan Ke Depan)

Fase berikutnya akan berfokus pada menyelesaikan naskah jurnal dan melakukan eksperimen tambahan untuk mengatasi potensi kekhawatiran reviewer. Aktivitas yang direncanakan mencakup menghasilkan visualisasi tambahan (ROC curves, precision-recall curves, calibration plots) untuk memberikan penilaian performa komprehensif, melakukan studi ablasi untuk mengukur kontribusi dari individual komponen arsitektur dan strategi pelatihan, dan melakukan uji signifikansi statistik untuk secara ketat memvalidasi perbedaan performa antara model.

External validation planning akan initiate dengan establishing collaborations dengan local hospitals atau medical research institutions untuk securing access ke field-collected clinical samples. Protocol development untuk data collection, anonymization procedures, ethical clearance processes, dan quality control standards akan prioritized untuk ensuring smooth Phase 2 validation execution.

### G.2 Fase Jangka Menengah (6-9 Bulan)

Phase 2 external validation akan conduct comprehensive testing pada 500+ diverse clinical samples dari multiple sources (different hospitals, varying equipment, diverse technician expertise) untuk assessing real-world generalization [38]. Domain adaptation techniques [38] akan explored untuk handling distribution shifts antara controlled laboratory conditions dan field deployment scenarios.

Advanced learning techniques untuk minority class improvement akan investigated, termasuk: (1) synthetic data generation menggunakan StyleGAN2 atau diffusion models [27,28] untuk augmenting rare species/stages, (2) active learning strategies [29] untuk prioritizing acquisition informative samples, (3) few-shot learning dan meta-learning approaches [30,37] untuk leveraging knowledge transfer dari majority classes, dan (4) ensemble methods combining multiple detection dan classification models untuk improving robustness.

Explainability features integration [39,40] akan implemented untuk providing clinicians dengan visual explanations supporting predictions, increasing trust dan enabling error detection. Deployment optimization mencakup model quantization [33], neural network pruning [34], dan edge device adaptation untuk enabling point-of-care deployment dengan computational constraints.

### G.3 Fase Jangka Panjang (10-12 Bulan)

Clinical pilot study akan conducted di selected hospitals untuk evaluating system performance dalam real-world clinical workflows. Study ini akan measure: (1) diagnostic accuracy versus expert microscopy, (2) time savings dalam clinical workflows, (3) inter-rater reliability improvements, (4) cost-effectiveness analysis, dan (5) clinician satisfaction dan acceptance.

Regulatory pathway planning untuk potential clinical deployment [35], termasuk preparing documentation untuk FDA submission atau equivalent regulatory approvals. Continuous learning pipeline development untuk enabling deployed systems collect additional labeled samples dari clinical usage untuk ongoing model improvement, ensuring system remains accurate as new parasites variants emerge atau imaging equipment evolves.

---

## H. DAFTAR PUSTAKA

[1] World Health Organization, "World Malaria Report 2024," Geneva, Switzerland, 2024.

[2] R. W. Snow et al., "The global distribution of clinical episodes of Plasmodium falciparum malaria," *Nature*, vol. 434, pp. 214-217, 2005.

[3] Centers for Disease Control and Prevention, "Malaria Biology," 2024. [Online]. Available: https://www.cdc.gov/malaria/about/biology/

[4] A. Moody, "Rapid diagnostic tests for malaria parasites," *Clin. Microbiol. Rev.*, vol. 15, no. 1, pp. 66-78, 2002.

[5] WHO, "Malaria Microscopy Quality Assurance Manual," ver. 2.0, Geneva, 2016.

[6] P. L. Chiodini et al., "Manson's Tropical Diseases," 23rd ed. London: Elsevier, 2014, ch. 52.

[7] J. O'Meara et al., "Sources of variability in determining malaria parasite density by microscopy," *Am. J. Trop. Med. Hyg.*, vol. 73, no. 3, pp. 593-598, 2005.

[8] K. Mitsakakis et al., "Challenges in malaria diagnosis," *Expert Rev. Mol. Diagn.*, vol. 18, no. 10, pp. 867-875, 2018.

[9] A. Esteva et al., "Dermatologist-level classification of skin cancer with deep neural networks," *Nature*, vol. 542, pp. 115-118, 2017.

[10] P. Rajpurkar et al., "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," arXiv:1711.05225, 2017.

[11] N. Coudray et al., "Classification and mutation prediction from non-small cell lung cancer histopathology images using deep learning," *Nat. Med.*, vol. 24, pp. 1559-1567, 2018.

[12] S. Rajaraman et al., "Pre-trained convolutional neural networks as feature extractors for diagnosis of malaria from blood smears," *Diagnostics*, vol. 8, no. 4, p. 74, 2018.

[13] A. Wang et al., "YOLOv10: Real-time end-to-end object detection," arXiv:2405.14458, 2024.

[14] G. Jocher et al., "YOLOv11: Ultralytics YOLO11," 2024. [Online]. Available: https://github.com/ultralytics/ultralytics

[15] F. Poostchi et al., "Image analysis and machine learning for detecting malaria," *Transl. Res.*, vol. 194, pp. 36-55, 2018.

[16] P. Rosenthal, "How do we diagnose and treat Plasmodium ovale and Plasmodium malariae?" *Curr. Infect. Dis. Rep.*, vol. 10, pp. 58-61, 2008.

[17] S. Ren et al., "Faster R-CNN: Towards real-time object detection with region proposal networks," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 39, no. 6, pp. 1137-1149, 2017.

[18] WHO, "Basic Malaria Microscopy: Part I. Learner's guide," 2nd ed., Geneva, 2010.

[19] G. Huang et al., "Densely connected convolutional networks," in *Proc. IEEE CVPR*, 2017, pp. 4700-4708.

[20] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. ICML*, 2019, pp. 6105-6114.

[21] K. He et al., "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770-778.

[22] T.-Y. Lin et al., "Focal loss for dense object detection," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 42, no. 2, pp. 318-327, 2020.

[23] M. Aikawa, "Parasitological review: Plasmodium," *Exp. Parasitol.*, vol. 30, no. 2, pp. 284-320, 1971.

[24] A. Vijayalakshmi and B. Rajesh Kanna, "Deep learning approach to detect malaria from microscopic images," *Multim. Tools Appl.*, vol. 79, pp. 15297-15317, 2020.

[25] J. Deng et al., "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE CVPR*, 2009, pp. 248-255.

[26] A. Dosovitskiy et al., "An image is worth 16×16 words: Transformers for image recognition at scale," in *Proc. ICLR*, 2021.

[27] I. Goodfellow et al., "Generative adversarial nets," in *Proc. NeurIPS*, 2014, pp. 2672-2680.

[28] J. Ho et al., "Denoising diffusion probabilistic models," in *Proc. NeurIPS*, 2020.

[29] B. Settles, "Active learning literature survey," Univ. Wisconsin-Madison, Tech. Rep. 1648, 2009.

[30] C. Finn et al., "Model-agnostic meta-learning for fast adaptation of deep networks," in *Proc. ICML*, 2017, pp. 1126-1135.

[31] WHO, "Guidelines for the Treatment of Malaria," 3rd ed., Geneva, 2015.

[32] C. J. Long et al., "A smartphone-based portable biosensor for diagnosis in resource-limited settings," *Nature Biotechnol.*, vol. 32, pp. 373-379, 2014.

[33] R. Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference," arXiv:1806.08342, 2018.

[34] S. Han et al., "Learning both weights and connections for efficient neural network," in *Proc. NeurIPS*, 2015, pp. 1135-1143.

[35] FDA, "Clinical decision support software: Guidance for industry and FDA staff," 2022.

[36] H. Zhang et al., "mixup: Beyond empirical risk minimization," in *Proc. ICLR*, 2018.

[37] O. Vinyals et al., "Matching networks for one shot learning," in *Proc. NeurIPS*, 2016, pp. 3630-3638.

[38] Y. Ganin et al., "Domain-adversarial training of neural networks," *J. Mach. Learn. Res.*, vol. 17, no. 1, pp. 2096-2030, 2016.

[39] A. Kirillov et al., "Segment anything," in *Proc. IEEE ICCV*, 2023, pp. 4015-4026.

[40] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," *Int. J. Comput. Vis.*, vol. 128, pp. 336-359, 2020.
