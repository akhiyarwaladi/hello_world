# Ringkasan dan Keyword Penelitian

## Ringkasan

Malaria tetap menjadi tantangan kesehatan global dengan sekitar 263 juta kasus dan 597.000 kematian pada tahun 2023. Diagnosis tradisional menggunakan mikroskop memerlukan pemeriksaan lebih dari 100 bidang mikroskopis oleh tenaga ahli terlatih, menciptakan hambatan serius di daerah endemis dengan keterbatasan sumber daya. Penelitian ini bertujuan mengembangkan kerangka kerja multi-model dengan arsitektur klasifikasi bersama (shared classification architecture) untuk deteksi dan klasifikasi parasit malaria secara otomatis menggunakan citra apusan darah skala kecil dengan ketidakseimbangan kelas yang ekstrem hingga rasio 54:1.

Tahapan metode penelitian meliputi: (1) tahap deteksi menggunakan tiga arsitektur YOLO Medium (YOLOv10, YOLOv11, YOLOv12) yang dilatih selama 100 epoch pada citra 640 piksel; (2) tahap ekstraksi crop 224 piksel dari anotasi ground truth; dan (3) tahap klasifikasi menggunakan enam arsitektur CNN (DenseNet121, EfficientNet-B0/B1/B2, ResNet50/101) dengan optimasi Focal Loss (alpha=1.0, gamma=1.5) untuk mengatasi ketidakseimbangan kelas.

Luaran yang ditargetkan berupa model deteksi dan klasifikasi parasit malaria yang efisien secara parameter dengan ukuran model kompak 46-89 MB, serta publikasi ilmiah pada jurnal terakreditasi. Penelitian ini berada pada TKT Level 1 yang berfokus pada penelitian dasar untuk mengidentifikasi prinsip-prinsip dasar dan memvalidasi konsep sistem deteksi malaria berbasis kecerdasan buatan.

Hasil penelitian menunjukkan deteksi mencapai mAP@50 sebesar 70,84%-96,27% dengan recall tinggi 71,05%-93,12% untuk meminimalkan parasit yang terlewat. Klasifikasi menunjukkan pemilihan model bergantung pada karakteristik dataset: EfficientNet-B1 mencapai akurasi 91,51% pada IML Lifecycle dan 98,28% pada MP-IDB Species, sedangkan ResNet50 mencapai 96,13% pada MP-IDB Stages yang memiliki ketidakseimbangan ekstrem. Model EfficientNet dengan parameter efisien (5,3M-9,2M parameter) secara konsisten mengungguli varian ResNet yang lebih besar (hingga 44,5M parameter). Optimasi Focal Loss berhasil menangani kelas minoritas dengan F1-score 0,44-1,00 pada kelas ultra-minoritas, termasuk F1-score sempurna 1,00 pada kelas schizont dengan hanya 4 sampel uji.

---

Keyword

Diagnosis malaria; Multi-model framework; Klasifikasi parasit; Transfer learning; Class imbalance

---

## Informasi Tambahan

- **TKT Level:** 1 (Penelitian Dasar)
- **Sumber Paper:** KINETIK Paper - Parameter Efficient Models for Malaria Detection and Classification Using Small-Scale Imbalanced Blood Smear Images
- **Total Dataset:** 1.544 citra mikroskopi dari 4 dataset (IML Lifecycle, MP-IDB Species, MP-IDB Stages, MD-2019)
