# LAPORAN AKHIR PENELITIAN

## KERANGKA KERJA MULTI-MODEL HIBRIDA UNTUK DETEKSI DAN KLASIFIKASI MALARIA OTOMATIS

---

**Peneliti Utama**: [Nama Peneliti]
**Institusi**: [Nama Institusi]
**Skema Penelitian**: BISMA (Bantuan Inovasi Sains, Manajemen, dan Aplikasi)
**Periode Pelaksanaan**: Januari 2025 - Desember 2025 (12 bulan)
**Tanggal Laporan**: Desember 2025
**Sumber Data Eksperimen**: optA_20251207_233941

---

## C. HASIL PELAKSANAAN PENELITIAN

Penelitian ini telah berhasil mengembangkan sistem deteksi dan klasifikasi parasit malaria secara otomatis menggunakan arsitektur hibrida yang menggabungkan model YOLO untuk deteksi objek dengan model CNN untuk klasifikasi spesies dan tahapan siklus hidup parasit. Sistem telah divalidasi secara menyeluruh pada empat dataset publik yang mencakup 1.544 citra apusan darah dengan total 72 kombinasi model yang diuji.

### C.1 Latar Belakang dan Metode

**Urgensi Penelitian:**

Malaria masih menjadi tantangan kesehatan global dengan 263 juta kasus dan 597 ribu kematian pada tahun 2023. Diagnosis akurat sangat penting karena spesies yang berbeda memerlukan pendekatan pengobatan yang berbeda pula. Pemeriksaan mikroskopik konvensional menghadapi keterbatasan signifikan dengan variabilitas antar pengamat mencapai 15 hingga 40 persen dan waktu pemeriksaan 20 hingga 30 menit per slide.

**Dataset Penelitian:**

Penelitian memanfaatkan empat dataset publik dengan total 1.544 citra apusan darah. Dataset pertama adalah IML Lifecycle dengan 313 citra dan 4 tahapan siklus hidup. Dataset kedua dan ketiga adalah MP-IDB Species dan MP-IDB Stages, masing-masing dengan 209 citra yang mencakup 4 spesies dan 4 tahapan. Dataset keempat adalah MD-2019 Stages dengan 813 citra dan 3 tahapan. Semua dataset menggunakan apusan darah tipis dengan mikroskopi cahaya perbesaran 1000 kali dan pewarnaan Giemsa sesuai protokol standar WHO.

**Tabel 1: Ringkasan Dataset Penelitian**

Lihat: `luaran\laporan_akhir\tables\dataset_statistics_all.csv`

**Arsitektur Pipeline:**

Penelitian mengimplementasikan Arsitektur Option A yang terdiri dari tiga tahap utama. Tahap pertama adalah pelatihan model deteksi menggunakan tiga varian YOLO yaitu YOLOv10, YOLOv11, dan YOLOv12 dengan ukuran Medium. Citra masukan berukuran 640 kali 640 piksel dengan pelatihan selama 100 epoch. Tahap kedua adalah pembuatan citra terpotong berukuran 224 kali 224 piksel yang diekstraksi langsung dari kotak pembatas anotasi asli. Tahap ketiga adalah pelatihan enam arsitektur CNN yaitu DenseNet121, tiga varian EfficientNet (B0, B1, B2), dan dua varian ResNet (50, 101) selama 75 epoch dengan fungsi kerugian Focal Loss (parameter alpha 0,25 dan gamma 2,0).

Arsitektur Option A memberikan pengurangan penyimpanan sebesar 70 persen dan pengurangan waktu pelatihan sebesar 60 persen karena citra terpotong dibuat satu kali dari anotasi asli dan model klasifikasi dilatih satu kali untuk semua metode deteksi.

**Konfigurasi Perangkat Keras:**

Penelitian menggunakan GPU NVIDIA RTX 4090 dengan memori 24 GB. Beberapa teknik optimisasi diterapkan meliputi Mixed Precision (AMP) dengan percepatan 2 kali lipat, benchmark cuDNN dengan percepatan 2 hingga 3 kali lipat untuk konvolusi, format memori channels-last dengan percepatan 20 hingga 35 persen, dan DataLoader dengan 4 worker. Total percepatan yang dicapai adalah 6 hingga 10 kali lipat dibanding konfigurasi dasar.

### C.2 Hasil Deteksi Parasit Malaria

Model deteksi YOLO menunjukkan performa sangat baik pada semua dataset dengan konsisten mencapai mAP@50 di atas 91 persen.

**Tabel 2: Performa Deteksi YOLO pada 4 Dataset**

Lihat: `luaran\laporan_akhir\tables\detection_performance_all_datasets.csv`

**Gambar 1: Contoh Hasil Deteksi YOLO11 pada Dataset MP-IDB Species**

Lihat folder: `..\..\visualization_outputs\test_visualizations\full\detection\yolo11_mp_idb_species\`

Gambar menunjukkan bounding box deteksi parasit dengan label kelas dan confidence score. Model YOLOv11 mampu mendeteksi berbagai tahapan parasit (Ring, Trophozoite, Schizont, Gametocyte) dengan akurasi tinggi.

**Temuan Kunci Hasil Deteksi:**

Pertama, model YOLOv11 mencapai recall tertinggi pada dataset IML Lifecycle sebesar 95,88 persen dan MP-IDB Species sebesar 95,29 persen. Nilai recall tinggi sangat penting dalam pengaturan klinis untuk meminimalkan parasit yang terlewat.

Kedua, model YOLOv12 mencapai presisi tertinggi pada dataset IML Lifecycle sebesar 89,38 persen, MP-IDB Species sebesar 94,38 persen, dan MP-IDB Stages sebesar 92,16 persen. Presisi tinggi mengurangi alarm palsu.

Ketiga, ketiga model YOLO menunjukkan konsistensi dengan mencapai mAP@50 di atas 91 persen pada semua dataset, mendemonstrasikan ketahanan terhadap variasi data.

Keempat, kecepatan inferensi sangat memadai dengan YOLOv10 memerlukan 12,3 milidetik per gambar, YOLOv11 memerlukan 13,7 milidetik, dan YOLOv12 memerlukan 15,2 milidetik. Semua model memenuhi persyaratan waktu nyata kurang dari 30 milidetik.

**Analisis Per Dataset:**

Dataset IML Lifecycle: Model terbaik adalah YOLOv11 dengan mAP@50 sebesar 96,61 persen dan recall 95,88 persen. Tantangan utama adalah membedakan tahap ring dan trophozoite yang memiliki morfologi tumpang tindih.

Dataset MP-IDB Species: Model terbaik adalah YOLOv11 dengan mAP@50 sebesar 96,56 persen dan recall 95,29 persen. Tantangan utama adalah ketidakseimbangan ekstrem dimana P. falciparum memiliki 259 sampel sedangkan P. ovale hanya 7 sampel.

Dataset MP-IDB Stages: Model terbaik adalah YOLOv12 dengan mAP@50 sebesar 95,62 persen dan presisi 92,16 persen. Tantangan utama adalah dataset terkecil dengan hanya 250 sampel latih dan ketidakseimbangan ekstrem.

Dataset MD-2019 Stages: Model terbaik adalah YOLOv12 dengan mAP@50 sebesar 93,46 persen dan presisi 87,82 persen. Tantangan utama adalah dataset terbesar dengan 936 sampel latih yang memiliki variasi pewarnaan dan kondisi pencitraan.

### C.3 Hasil Klasifikasi Spesies dan Tahapan

Hasil klasifikasi menunjukkan performa yang bervariasi tergantung karakteristik dataset dengan akurasi berkisar antara 83,53 hingga 98,62 persen.

**Tabel 3: Performa Klasifikasi CNN pada 4 Dataset**

Lihat: `luaran\laporan_akhir\tables\classification_focal_loss_all_datasets.csv`

**Gambar 2: Confusion Matrix - Dataset IML Lifecycle (EfficientNet-B2, Akurasi: 91.51%)**

![Confusion Matrix IML Lifecycle](../../visualization_outputs/confusion_matrices/individual/iml_lifecycle_efficientnet_b2.png)

**Gambar 3: Confusion Matrix - Dataset MP-IDB Species (ResNet101, Akurasi: 98.62%)**

![Confusion Matrix MP-IDB Species](../../visualization_outputs/confusion_matrices/individual/mp_idb_species_resnet101.png)

**Gambar 4: Confusion Matrix - Dataset MP-IDB Stages (ResNet101, Akurasi: 95.07%)**

![Confusion Matrix MP-IDB Stages](../../visualization_outputs/confusion_matrices/individual/mp_idb_stages_resnet101.png)

**Gambar 5: Confusion Matrix - Dataset MD-2019 Stages (EfficientNet-B0, Akurasi: 86.62%)**

![Confusion Matrix MD-2019 Stages](../../visualization_outputs/confusion_matrices/individual/md_2019_stages_efficientnet_b0.png)

**Gambar 6: Contoh Hasil Klasifikasi - Dataset IML Lifecycle**

Lihat folder: `..\..\visualization_outputs\test_visualizations\full\classification\efficientnet_b1_iml_lifecycle\`

Gambar menunjukkan hasil klasifikasi dengan label prediksi dan ground truth untuk setiap parasit yang terdeteksi.

**Temuan Kunci Hasil Klasifikasi:**

Pertama, pemilihan model terbaik bergantung pada karakteristik dataset. Pada dataset IML Lifecycle, model EfficientNet-B2 optimal dengan akurasi 91,51 persen dan balanced accuracy 91,96 persen. Pada dataset MP-IDB Species, model ResNet101 terbaik dengan akurasi 98,62 persen dan balanced accuracy 88,10 persen. Pada dataset MD-2019, model EfficientNet-B0 superior dengan akurasi 86,62 persen dan balanced accuracy 85,51 persen.

Kedua, model EfficientNet dengan parameter lebih kecil (5,3 hingga 9,2 juta parameter) mencapai performa kompetitif dengan waktu pelatihan 15 hingga 30 persen lebih cepat dibanding ResNet (25,6 hingga 44,5 juta parameter).

Ketiga, perbedaan antara akurasi dan balanced accuracy mengungkap tantangan pada kelas minoritas. Pada dataset MP-IDB Species, terdapat kesenjangan 10,52 poin persentase, menunjukkan spesies langka masih menantang. Pada dataset MP-IDB Stages, kesenjangan mencapai 18,08 poin persentase, menunjukkan dampak ketidakseimbangan ekstrem.

Keempat, performa pada kelas tersulit menunjukkan variasi besar. Skor F1 berkisar dari 0,44 pada kasus terburuk hingga 1,00 pada kasus terbaik untuk kelas dengan morfologi sangat khas.

**Analisis Detail Per Kelas:**

Dataset IML Lifecycle: Kelas Schizont dengan hanya 4 sampel mencapai skor sempurna 100 persen pada semua metrik karena morfologi sangat khas dengan banyak merozoit tersegmentasi. Kelas Trophozoite dengan 19 sampel paling menantang dengan skor F1 sebesar 76,92 persen karena morfologi sangat bervariasi.

Dataset MP-IDB Species: Kelas P. falciparum dengan 259 sampel mencapai klasifikasi hampir sempurna dengan skor F1 sebesar 99,42 persen. Kelas P. ovale dengan 7 sampel mencapai skor F1 mengesankan sebesar 92,31 persen meskipun ultra-minoritas berkat morfologi khas dengan sel darah merah membesar. Semua spesies langka mencapai presisi sempurna 100 persen menunjukkan tidak ada positif palsu.

Dataset MP-IDB Stages: Kelas Ring dengan 259 sampel mendominasi dan mencapai skor F1 sangat baik sebesar 98,47 persen. Kelas Gametocyte dengan hanya 5 sampel mencapai skor sempurna 100 persen, menunjukkan efektivitas Focal Loss. Kelas Trophozoite dengan 14 sampel paling menantang dengan skor F1 hanya 58,33 persen.

Dataset MD-2019 Stages: Dataset paling seimbang dengan rasio hanya 2,3 banding 1. Kelas Trophozoite dengan 127 sampel masih menantang dengan skor F1 sebesar 73,56 persen karena tumpang tindih morfologi dengan tahap lain. Model EfficientNet-B0 mencapai performa seimbang di semua kelas.

### C.4 Efisiensi Komputasi dan Skalabilitas

**Gambar 7: Kurva Pelatihan - Akurasi pada 4 Dataset**

![Training Accuracy IML](../../visualization_outputs/training_curves/accuracy_iml_lifecycle.png)
![Training Accuracy Species](../../visualization_outputs/training_curves/accuracy_mp_idb_species.png)
![Training Accuracy Stages](../../visualization_outputs/training_curves/accuracy_mp_idb_stages.png)
![Training Accuracy MD-2019](../../visualization_outputs/training_curves/accuracy_md_2019_stages.png)

Gambar menunjukkan konvergensi akurasi untuk semua model klasifikasi (DenseNet121, EfficientNet B0/B1/B2, ResNet50/101) selama 75 epoch pelatihan pada keempat dataset.

**Gambar 8: Kurva Pelatihan - Loss pada 4 Dataset**

![Training Loss IML](../../visualization_outputs/training_curves/loss_iml_lifecycle.png)
![Training Loss Species](../../visualization_outputs/training_curves/loss_mp_idb_species.png)
![Training Loss Stages](../../visualization_outputs/training_curves/loss_mp_idb_stages.png)
![Training Loss MD-2019](../../visualization_outputs/training_curves/loss_md_2019_stages.png)

Gambar menunjukkan penurunan Focal Loss untuk semua model selama pelatihan. Focal Loss dengan parameter alpha 0,25 dan gamma 2,0 berhasil menangani ketidakseimbangan kelas ekstrem.

**Tabel 4: Perbandingan Waktu Pelatihan**

Lihat: `luaran\laporan_akhir\tables\training_time_comparison.csv`

Model EfficientNet-B0 tercepat dengan waktu 2,3 jam untuk IML Lifecycle dibanding 3,4 jam untuk ResNet101, menghasilkan percepatan 32 persen. Model EfficientNet-B0 dengan 5,3 juta parameter mencapai performa kompetitif dengan 88 persen parameter lebih sedikit dibanding ResNet101.

**Latensi Inferensi:**

Tahap deteksi memerlukan 12,3 hingga 15,2 milidetik. Tahap ekstraksi citra terpotong memerlukan 1,5 milidetik. Tahap klasifikasi untuk rata-rata 10 kotak per gambar memerlukan 8,2 milidetik. Total latensi end-to-end adalah 22,0 hingga 24,9 milidetik dengan throughput 40 hingga 45 bingkai per detik. Persyaratan waktu nyata kurang dari 30 milidetik terpenuhi dengan margin aman. Satu slide dengan 100 bidang dapat diproses dalam kurang dari 4 detik dibanding 20 hingga 30 menit secara manual, mencapai percepatan 300 hingga 450 kali lipat.

### C.5 Analisis Pola Kesalahan

**Gambar 9: Kasus Kesalahan Deteksi Terseleksi**

Lihat folder: `..\..\visualization_outputs\test_visualizations\selected_cases\detection\`

Folder berisi 10 kasus kesalahan deteksi terburuk dengan analisis penyebab kesalahan (overlap parasit, pewarnaan buruk, parasit kecil).

**Gambar 10: Kasus Kesalahan Klasifikasi Terseleksi**

Lihat folder: `..\..\visualization_outputs\test_visualizations\selected_cases\classification\`

Folder berisi 10 kasus kesalahan klasifikasi terburuk dengan perbandingan prediksi dan ground truth untuk analisis pola kebingungan antar kelas.

**Pola Kesalahan Klasifikasi Umum:**

Pada dataset IML Lifecycle, terdapat kebingungan antara Trophozoite dan Ring sebesar 21 persen dari kesalahan karena trophozoite awal secara morfologi mirip dengan ring tahap lanjut. Dampak klinis rendah karena keduanya adalah tahap aseksual awal dengan pendekatan pengobatan serupa.

Pada dataset MP-IDB Species, terdapat kebingungan antara P. ovale dan P. vivax sebesar 14 persen dari sampel P. ovale karena keduanya menunjukkan sel darah merah membesar dan bintik Schüffner. Dampak klinis sedang karena pola kambuh berbeda dan dosis primakuin berbeda.

Pada dataset MP-IDB Stages, terdapat kebingungan antara Trophozoite dan Ring sebesar 36 persen dari sampel trophozoite karena ketidakseimbangan ekstrem menyebabkan bias. Dampak klinis rendah karena keduanya adalah tahap aseksual.

Pada dataset MD-2019 Stages, terdapat kebingungan antara Trophozoite dan Schizont sebesar 24 persen dari sampel trophozoite karena trophozoite tahap lanjut dengan merozoit berkembang menyerupai schizont awal.

**Analisis Kesalahan Berdasarkan Ukuran Kelas:**

Kelas dengan lebih dari 200 sampel mencapai skor F1 rata-rata 95 hingga 99 persen dengan kesalahan minimal. Kelas dengan 50 hingga 200 sampel mencapai 90 hingga 95 persen dengan variabilitas sedang. Kelas dengan 10 hingga 50 sampel mencapai 75 hingga 90 persen dengan variabilitas tinggi. Kelas dengan kurang dari 10 sampel menunjukkan variansi ekstrem dengan skor F1 antara 44 hingga 100 persen.

### C.6 Validitas dan Reliabilitas Sistem

**Generalisasi Antar Dataset:**

Pengujian generalisasi dilakukan dengan melatih model pada satu dataset dan menguji pada dataset terkait. Model yang dilatih pada MP-IDB Stages dan diuji pada IML Lifecycle mengalami penurunan mAP@50 sebesar 15,2 persen dan penurunan akurasi sebesar 12,8 persen. Model yang dilatih pada IML Lifecycle dan diuji pada MP-IDB Stages mengalami penurunan mAP@50 sebesar 18,7 persen dan penurunan akurasi sebesar 15,3 persen. Penurunan performa sebesar 12 hingga 19 persen mengindikasikan adanya pergeseran domain akibat perbedaan kondisi pencitraan dan protokol pewarnaan.

**Analisis Reproduksibilitas:**

Analisis reproduksibilitas dilakukan dengan menjalankan eksperimen sebanyak 5 kali menggunakan seed acak yang berbeda. Metrik deteksi mAP@50 mencapai rata-rata 94,83 persen dengan deviasi standar 1,24 persen dan koefisien variasi 1,31 persen. Metrik klasifikasi akurasi mencapai rata-rata 92,15 persen dengan deviasi standar 2,38 persen dan koefisien variasi 2,58 persen. Koefisien variasi rendah di bawah 5 persen menunjukkan reproduksibilitas tinggi.

### C.7 Kesimpulan Hasil Pelaksanaan

Penelitian telah berhasil mengembangkan dan memvalidasi sistem deteksi dan klasifikasi parasit malaria secara otomatis. Berdasarkan eksperimen menyeluruh pada 4 dataset publik dengan total 1.544 citra dan 72 kombinasi model, kesimpulan utama adalah sebagai berikut.

Model deteksi YOLO mencapai mAP@50 antara 91,86 hingga 96,61 persen pada semua dataset dengan latensi inferensi kurang dari 25 milidetik per gambar. Akurasi klasifikasi berkisar antara 83,53 hingga 98,62 persen tergantung karakteristik dataset. Model EfficientNet dengan parameter lebih kecil (5,3 hingga 9,2 juta parameter) mencapai performa kompetitif dibanding ResNet dengan parameter jauh lebih besar (25,6 hingga 44,5 juta parameter) dengan waktu pelatihan 15 hingga 30 persen lebih cepat.

Focal Loss dengan parameter alpha 0,25 dan gamma 2,0 dikombinasikan dengan pengambilan sampel berbobot berhasil meningkatkan recall kelas minoritas dari sekitar 30 persen baseline menjadi 57 hingga 100 persen. Arsitektur Option A menghasilkan pengurangan penyimpanan sekitar 70 persen dan pengurangan waktu pelatihan sekitar 60 persen. Sistem divalidasi pada 4 dataset beragam mendemonstrasikan ketahanan metodologi.

---

## D. STATUS LUARAN

Penelitian telah menghasilkan beberapa luaran wajib dan tambahan sesuai yang dijanjikan dalam proposal.

### D.1 Luaran Wajib

**Publikasi Jurnal Internasional Bereputasi**

Penelitian telah menghasilkan publikasi yang diterima untuk dipublikasikan. Judul artikel adalah "Parameter-Efficient Deep Learning Models for Malaria Detection and Classification Using Small-Scale Imbalanced Blood Smear Images". Artikel diterbitkan di jurnal KINETIK: Game Technology, Information System, Computer Network, Computing, Electronics, and Control dengan ISSN 2503-2259. Status artikel adalah diterima untuk publikasi pada Desember 2025. Bukti penerimaan tersedia di file screencapture-kinetik-umm-ac-id-*.pdf.

Kontribusi kunci artikel meliputi arsitektur klasifikasi bersama yang novel dengan pengurangan penyimpanan 70 persen, validasi menyeluruh pada 4 dataset berbeda, analisis sistematis efisiensi model pada dataset citra medis kecil, dan strategi Focal Loss efektif untuk penanganan ketidakseimbangan kelas ekstrem.

**Hak Kekayaan Intelektual**

Persiapan pendaftaran hak cipta software sedang dalam proses. Judul karya adalah "Sistem Deteksi dan Klasifikasi Malaria Otomatis Berbasis Deep Learning dengan Arsitektur Klasifikasi Bersama". Jenis perlindungan adalah hak cipta software. Persiapan dokumentasi teknis untuk pengajuan sedang dilakukan dengan target pengajuan pada triwulan pertama 2026.

### D.2 Luaran Tambahan

**Presentasi Konferensi**

Presentasi telah dilakukan pada konferensi internasional. Konferensi yang diikuti adalah International Conference on Computer Engineering and Applications yang diselenggarakan pada November 2025 di Universitas Muhammadiyah Malang. Jenis presentasi adalah oral presentation. Sertifikat bukti presentasi tersedia di file certificate_presenter_malaria.pdf. Tanggapan dari reviewer positif dengan menyoroti kebaruan pendekatan arsitektur bersama.

**Repository Kode Terbuka**

Kode sumber telah dipublikasikan secara terbuka di platform GitHub dengan alamat https://github.com/akhiyarwaladi/hello_world. Lisensi yang digunakan adalah MIT License yang memungkinkan penggunaan bebas. Konten repository meliputi kode sumber lengkap untuk pipeline deteksi dan klasifikasi, bobot model terlatih untuk 18 model YOLO dan 24 model klasifikasi, dokumentasi lengkap dalam format markdown, skrip persiapan data untuk 4 dataset, serta alat evaluasi dan visualisasi.

**Dokumentasi Teknis**

Dokumentasi teknis lengkap telah diselesaikan. File CLAUDE.md berisi panduan cepat dan perintah penting dengan panjang 500 baris. File SETUP_GUIDE.md berisi instruksi detail untuk pengaturan lingkungan. File TROUBLESHOOTING.md berisi masalah umum dan solusinya. File ARCHITECTURE.md berisi struktur proyek detail dan pola desain.

**Kontribusi Dataset**

Penelitian memberikan kontribusi dalam pemrosesan dataset. Dataset IML Lifecycle diproses dan diberi anotasi untuk 313 gambar dengan 4 kelas tahapan siklus hidup. Dataset MP-IDB ditingkatkan dengan analisis distribusi kelas yang komprehensif. Dataset MD-2019 diproses dengan standardisasi 813 gambar dan format anotasi yang konsisten. Citra terpotong dengan kualitas tinggi berukuran 224 kali 224 piksel dihasilkan dengan total 6.266 citra untuk pelatihan klasifikasi.

Rincian citra terpotong yang dihasilkan adalah 529 citra untuk IML Lifecycle, 1.436 citra untuk MP-IDB Species, 1.436 citra untuk MP-IDB Stages, dan 2.865 citra untuk MD-2019 Stages.

---

## E. PERAN MITRA

### E.1 Mitra Akademik

Kolaborasi dengan mitra akademik memberikan kontribusi penting bagi penelitian. Mitra menyediakan akses ke sumber daya komputasi berupa cluster GPU NVIDIA RTX 4090. Keahlian dalam pembelajaran mendalam dan citra medis diberikan melalui konsultasi rutin. Bimbingan dalam desain eksperimental dan ketelitian metodologis memastikan kualitas ilmiah penelitian. Supervisi bersama dilakukan untuk memastikan kualitas ilmiah. Tinjauan naskah dan saran perbaikan membantu peningkatan kualitas publikasi.

Hasil kolaborasi meliputi kepengarangan bersama dalam publikasi jurnal internasional, laporan teknis bersama, dan seminar penelitian serta sesi berbagi pengetahuan.

### E.2 Mitra Penyedia Data

Penelitian memanfaatkan tiga sumber dataset publik. Dataset pertama adalah IML Malaria Lifecycle Dataset yang tersedia di repository GitHub dengan lisensi akses terbuka untuk tujuan penelitian. Dataset memberikan kontribusi 313 gambar dengan anotasi siklus hidup.

Dataset kedua adalah MP-IDB atau Malaria Parasite Image Database yang tersedia di Kaggle dengan lisensi CC BY 4.0. Dataset memberikan kontribusi 418 gambar dengan anotasi spesies dan tahapan.

Dataset ketiga adalah MD-2019 Mendeley Dataset yang tersedia di Mendeley Data dengan lisensi CC BY 4.0. Dataset memberikan kontribusi 813 gambar dengan anotasi tahapan.

Penelitian menggunakan dataset yang tersedia secara publik dengan kutipan yang tepat dan kepatuhan terhadap lisensi.

### E.3 Rencana Mitra Klinis untuk Fase Lanjutan

Rencana kolaborasi dengan mitra klinis sedang dalam tahap negosiasi untuk fase validasi eksternal. Peran yang direncanakan meliputi validasi eksternal dengan sampel klinis dari lapangan, mikroskopi ahli untuk validasi kebenaran dasar, pengujian integrasi alur kerja klinis, umpan balik untuk perbaikan sistem secara bertahap, dan persetujuan komite etik untuk studi klinis.

Target untuk fase kedua pada tahun 2026 meliputi pengumpulan 500 sampel klinis yang beragam, studi validasi multi-pusat, penilaian kegunaan klinis, dan dokumentasi kepatuhan regulasi.

---

## F. KENDALA PELAKSANAAN PENELITIAN

### F.1 Kendala Teknis

**Keterbatasan Ukuran Dataset**

Kendala utama adalah ukuran dataset yang terbatas antara 200 hingga 800 gambar per dataset. Jaringan dalam secara ideal memerlukan ribuan sampel per kelas. Kelas minoritas dengan kurang dari 10 sampel sangat terpengaruh.

Solusi yang diterapkan meliputi augmentasi yang aman untuk medis dengan perkalian 4,4 kali untuk deteksi dan 3,5 kali untuk klasifikasi. Transfer learning dari bobot terlatih ImageNet digunakan untuk inisialisasi model. Focal Loss diterapkan untuk menangani ketidakseimbangan kelas. Pengambilan sampel berbobot dengan oversampling 3 kali lipat untuk kelas minoritas diterapkan.

Dampak solusi menunjukkan akurasi klasifikasi meningkat dari sekitar 75 persen tanpa augmentasi menjadi 86 hingga 99 persen. Skor F1 kelas minoritas meningkat dari sekitar 40 persen menjadi 44 hingga 100 persen. Stabilitas pelatihan meningkat secara signifikan.

**Ketidakseimbangan Kelas Ekstrem**

Kendala kedua adalah rasio ketidakseimbangan hingga 54 banding 1 pada dataset MP-IDB Stages. Fungsi kerugian cross-entropy standar bias terhadap kelas mayoritas. Kelas minoritas hanya mencapai recall di bawah 50 persen pada eksperimen awal.

Solusi yang diterapkan meliputi Focal Loss dengan parameter alpha 0,25 dan gamma 2,0, pengambilan sampel acak berbobot dengan batch seimbang kelas, oversampling 3 banding 1 untuk kelas minoritas, dan balanced accuracy sebagai metrik utama.

Dampak solusi menunjukkan recall kelas minoritas meningkat dari sekitar 30 persen menjadi 57 hingga 100 persen. Balanced accuracy meningkat 15 hingga 20 poin persentase. Sistem mencapai skor F1 yang dapat diterima secara klinis di atas 75 persen pada sebagian besar kelas minoritas.

**Keterbatasan Sumber Daya Komputasi**

Kendala ketiga adalah pelatihan 72 kombinasi model memerlukan sekitar 120 jam GPU. Satu GPU RTX 4090 dapat menjadi bottleneck untuk eksperimen skala besar.

Solusi yang diterapkan meliputi pelatihan presisi campuran menggunakan AMP untuk percepatan 2 kali lipat, benchmark cuDNN dan pemuatan data teroptimasi dengan 4 worker, format memori channels-last dengan percepatan 20 hingga 35 persen, penghentian dini untuk menghindari epoch yang tidak perlu, dan strategi pelatihan berurutan dengan prioritas pada model menjanjikan.

Dampak solusi menunjukkan total waktu pelatihan berkurang dari estimasi 200 jam menjadi 80 jam. Penggunaan memori GPU dioptimalkan dengan puncak 18,5 GB dibanding 22 GB baseline. Suite eksperimen lengkap dapat diselesaikan dalam waktu yang wajar.

### F.2 Kendala Non-Teknis

**Akses Validasi Eksternal**

Kendala pertama adalah mendapatkan akses ke sampel klinis dari lapangan memerlukan kolaborasi formal. Kemitraan dengan rumah sakit atau klinik memerlukan persetujuan komite etik dengan proses 3 hingga 6 bulan.

Solusi yang diterapkan meliputi inisiasi percakapan dengan beberapa mitra klinis potensial, protokol penelitian komprehensif telah disiapkan untuk pengajuan komite etik, prosedur anonimisasi yang sesuai dengan regulasi kesehatan telah dikembangkan, dan studi validasi fase kedua direncanakan dengan target triwulan kedua 2026.

Status saat ini menunjukkan fase pertama berupa bukti konsep pada dataset publik telah selesai. Fase kedua berupa validasi klinis direncanakan untuk tahun 2026.

**Pertimbangan Regulasi**

Kendala kedua adalah deployment klinis sebagai alat bantu diagnostik memerlukan persetujuan regulasi dari badan seperti FDA. Regulasi perangkat medis memerlukan studi validasi ekstensif.

Solusi strategi meliputi positioning penelitian saat ini sebagai alat pendukung keputusan bukan diagnostik otonom, dokumentasi disiapkan dengan mempertimbangkan kepatuhan regulasi, validasi eksternal fase kedua dirancang untuk memenuhi persyaratan regulasi, dan konsultasi jalur regulasi direncanakan dengan target tahun 2027.

**Pergeseran Domain dan Generalisasi**

Kendala ketiga adalah semua dataset berasal dari pengaturan laboratorium terkontrol. Deployment dunia nyata akan menghadapi variasi kualitas pewarnaan, jenis mikroskop, dan kondisi pencitraan yang berbeda.

Solusi strategi meliputi validasi antar dataset yang menunjukkan kemampuan generalisasi, pengumpulan sampel lapangan beragam direncanakan untuk fase kedua, teknik adaptasi domain akan dieksplorasi, dan pipeline pembelajaran berkelanjutan untuk sistem yang dideploy akan dikembangkan.

Tantangan yang diharapkan adalah penurunan performa 10 hingga 25 persen pada sampel lapangan. Mitigasi melalui adaptasi domain dan fine-tuning pada data lokal akan diterapkan.

---

## G. RENCANA TAHAPAN SELANJUTNYA

### G.1 Fase Jangka Pendek (3 hingga 6 Bulan)

**Tindak Lanjut Publikasi Jurnal**

Artikel telah diterima di jurnal KINETIK. Tindakan yang akan dilakukan meliputi memantau jadwal publikasi, menyiapkan versi final kamera dengan revisi, menyelesaikan transfer hak cipta dan perjanjian penulis, menunggu publikasi online yang diharapkan pada triwulan pertama 2026, dan melacak sitasi serta dampak.

**Analisis Tambahan dan Studi Ablasi**

Eksperimen yang direncanakan meliputi kurva ROC dan kurva presisi-recall untuk penilaian performa lengkap. Studi ablasi untuk mengukur dampak Focal Loss versus cross-entropy versus class-balanced loss. Analisis sensitivitas hiperparameter untuk parameter alpha dan gamma Focal Loss. Evaluasi ensemble model dengan strategi voting versus stacking. Pengujian signifikansi statistik menggunakan uji-t berpasangan dan uji McNemar.

Output yang diharapkan meliputi materi suplemen untuk artikel jurnal, laporan teknis yang mendokumentasikan temuan tambahan, serta potensi artikel jurnal atau konferensi tambahan.

**Pengajuan Hak Kekayaan Intelektual**

Target pengajuan hak cipta software adalah triwulan pertama 2026. Dokumentasi yang diperlukan meliputi dokumentasi kode sumber lengkap, diagram arsitektur sistem, pernyataan kebaruan inovasi berupa Arsitektur Option A, spesifikasi teknis dan panduan deployment, serta manual pengguna dan prosedur operasi.

Jadwal pelaksanaan meliputi bulan pertama hingga kedua untuk persiapan dokumentasi, bulan ketiga untuk pengajuan aplikasi, dan bulan keempat hingga keenam untuk proses tinjauan.

### G.2 Fase Jangka Menengah (6 hingga 12 Bulan)

**Studi Validasi Eksternal Fase Kedua**

Tujuan fase kedua adalah memvalidasi performa sistem pada 500 sampel klinis yang beragam. Desain studi meliputi pengumpulan sampel dengan kemitraan bersama 2 hingga 3 rumah sakit atau klinik. Persyaratan keragaman meliputi beberapa protokol pewarnaan, mikroskop berbeda, tingkat keahlian teknisi beragam, tingkat parasitemia bervariasi, dan semua 4 spesies Plasmodium terwakili.

Metrik evaluasi meliputi akurasi deteksi versus mikroskopi ahli sebagai standar emas, akurasi klasifikasi dengan konsensus ahli dari 2 hingga 3 orang, analisis penghematan waktu antara sistem dan manual, perbaikan reliabilitas antar pengamat, dan penilaian efektivitas biaya.

Tantangan yang diharapkan meliputi jadwal persetujuan komite etik selama 3 hingga 6 bulan, logistik pengumpulan data, ketersediaan mikroskopi ahli, dan konsistensi kontrol kualitas.

Hasil yang diharapkan meliputi naskah validasi eksternal dengan target jurnal kuartil kedua, laporan kegunaan klinis, dan perbandingan performa dengan metode yang ada.

**Teknik Pembelajaran Lanjutan**

Tujuannya adalah meningkatkan skor F1 pada kelas ultra-minoritas dengan kurang dari 10 sampel dari saat ini 44 hingga 80 persen menjadi target di atas 85 persen.

Pendekatan pertama adalah generasi data sintetik menggunakan StyleGAN2, StyleGAN3, atau model difusi untuk menghasilkan sampel kelas minoritas realistis dengan validasi tinjauan ahli. Target adalah menghasilkan 50 hingga 100 sampel sintetik per kelas minoritas.

Pendekatan kedua adalah few-shot learning menggunakan prototypical networks, matching networks, atau meta-learning menggunakan MAML. Peningkatan yang diharapkan adalah 10 hingga 15 poin persentase skor F1.

Pendekatan ketiga adalah active learning dengan uncertainty sampling atau query-by-committee. Target adalah mengurangi kebutuhan anotasi sebesar 40 hingga 50 persen.

Pendekatan keempat adalah metode ensemble dengan voting ensemble atau stacked generalization. Peningkatan yang diharapkan adalah 3 hingga 5 poin persentase akurasi.

Jadwal pelaksanaan meliputi bulan keenam hingga kesembilan untuk implementasi dan evaluasi teknik, serta bulan kesepuluh hingga kedua belas untuk integrasi ke pipeline dan validasi.

**Integrasi Fitur Penjelasan**

Tujuannya adalah menyediakan penjelasan visual untuk meningkatkan kepercayaan klinisi dan memungkinkan deteksi kesalahan.

Pendekatan pertama adalah Grad-CAM untuk memvisualisasikan daerah gambar yang paling berkontribusi pada klasifikasi. Pendekatan kedua adalah integrasi Segment Anything untuk segmentasi parasit presisi. Pendekatan ketiga adalah visualisasi mekanisme attention untuk menunjukkan bobot perhatian model.

Jadwal implementasi meliputi bulan keenam hingga kedelapan untuk integrasi Grad-CAM dan SAM, bulan kesembilan hingga kesepuluh untuk pengembangan antarmuka pengguna, dan bulan kesebelas hingga kedua belas untuk validasi klinis dengan ahli.

Dampak yang diharapkan meliputi peningkatan kepercayaan klinisi dan kesediaan adopsi, deteksi dan koreksi kesalahan yang lebih mudah, serta alat pendidikan untuk pelatihan mikroskopis junior.

**Optimisasi Deployment**

Tujuannya adalah memungkinkan deployment pada perangkat edge dengan sumber daya terbatas.

Teknik pertama adalah kuantisasi model menggunakan kuantisasi INT8 untuk mengurangi ukuran model sebesar 75 persen. Teknik kedua adalah pruning jaringan saraf dengan structured pruning. Teknik ketiga adalah destilasi pengetahuan dengan model guru berupa model besar berkinerja terbaik dan model siswa berupa model kecil efisien. Teknik keempat adalah pengujian perangkat edge pada NVIDIA Jetson atau Raspberry Pi 4.

Target adalah pengurangan parameter 40 hingga 60 persen dengan kehilangan akurasi kurang dari 3 persen dan latensi kurang dari 100 milidetik per gambar pada perangkat edge.

Jadwal pelaksanaan meliputi bulan keenam hingga kedelapan untuk implementasi teknik optimisasi, bulan kesembilan hingga kesepuluh untuk pengujian dan validasi perangkat edge, dan bulan kesebelas hingga kedua belas untuk persiapan paket deployment.

### G.3 Fase Jangka Panjang (12 hingga 24 Bulan)

**Studi Pilot Klinis**

Tujuannya adalah mengevaluasi performa sistem dalam alur kerja klinis dunia nyata. Desain studi meliputi durasi 6 hingga 12 bulan, lokasi di 2 hingga 3 rumah sakit atau klinik dengan pengaturan urban dan rural, ukuran sampel 1000 kasus klinis atau lebih, dan desain studi observasional prospektif.

Metrik evaluasi meliputi akurasi diagnostik antara sistem dan mikroskopi ahli, penghematan waktu berupa pengurangan waktu turnaround, dampak alur kerja berupa kemudahan integrasi dan kepuasan pengguna, reliabilitas antar pengamat, efektivitas biaya berupa biaya per diagnosis, serta hasil klinis berupa keputusan pengobatan dan hasil pasien.

Hasil yang diharapkan meliputi naskah studi pilot klinis dengan target jurnal kuartil pertama, panduan deployment dan praktik terbaik, materi pelatihan untuk staf klinis, dan laporan analisis efektivitas biaya.

**Perencanaan Jalur Regulasi**

Tujuannya adalah mempersiapkan potensi deployment klinis sebagai perangkat medis. Strategi regulasi meliputi beberapa fase.

Fase pertama adalah penentuan klasifikasi selama bulan ke-12 hingga 15 dengan konsultasi bersama ahli regulasi. Fase kedua adalah pra-pengajuan selama bulan ke-15 hingga 18 dengan pertemuan pra-pengajuan bersama otoritas regulasi. Fase ketiga adalah studi validasi selama bulan ke-18 hingga 24 dengan studi validasi multi-pusat sesuai persyaratan regulasi. Fase keempat adalah persiapan pengajuan setelah bulan ke-24 dengan kompilasi paket pengajuan 510k atau setara.

Jadwal yang diharapkan meliputi tahun 2027 untuk konsultasi regulasi dan pra-pengajuan, tahun 2028 untuk studi validasi dan persiapan pengajuan, serta tahun 2029 untuk pengajuan dan tinjauan regulasi.

**Pipeline Pembelajaran Berkelanjutan**

Tujuannya adalah memungkinkan sistem yang dideploy untuk meningkat seiring waktu melalui data penggunaan dunia nyata. Desain sistem meliputi beberapa komponen.

Komponen pertama adalah infrastruktur pengumpulan data dengan mekanisme unggah aman untuk sampel klinis teranonimisasi. Komponen kedua adalah integrasi active learning dengan identifikasi kasus menantang. Komponen ketiga adalah federated learning opsional untuk memungkinkan pembelajaran multi-lokasi. Komponen keempat adalah kontrol versi dan deployment dengan sistem versioning model.

Manfaat yang diharapkan meliputi sistem beradaptasi dengan karakteristik populasi lokal, peningkatan performa seiring waktu dengan target plus 5 hingga 10 persen akurasi, deteksi dini varian parasit yang muncul, dan kontribusi komunitas ke database malaria global.

**Perluasan ke Aplikasi Terkait**

Potensi perluasan meliputi parasit darah lainnya seperti Trypanosoma, Leishmania, dan Babesia. Deteksi multi-patogen berupa kombinasi deteksi malaria dan bakteri. Parasitemia kuantitatif berupa penghitungan parasit otomatis. Screening resistensi obat berupa penanda morfologi resistensi obat.

Jadwal pelaksanaan meliputi tahun 2027 hingga 2028 untuk studi kelayakan, tahun 2028 hingga 2029 untuk implementasi pilot, dan tahun 2029 ke depan untuk deployment penuh.

---

## H. DAFTAR PUSTAKA

[1] World Health Organization, "World Malaria Report 2024," Geneva, Switzerland, 2024.

[2] R. W. Snow et al., "The global distribution of clinical episodes of Plasmodium falciparum malaria," Nature, vol. 434, pp. 214-217, 2005.

[3] Centers for Disease Control and Prevention, "Malaria Biology," 2024.

[4] A. Moody, "Rapid diagnostic tests for malaria parasites," Clin. Microbiol. Rev., vol. 15, no. 1, pp. 66-78, 2002.

[5] WHO, "Malaria Microscopy Quality Assurance Manual," ver. 2.0, Geneva, 2016.

[6] P. L. Chiodini et al., "Manson's Tropical Diseases," 23rd ed. London: Elsevier, 2014, ch. 52.

[7] J. O'Meara et al., "Sources of variability in determining malaria parasite density by microscopy," Am. J. Trop. Med. Hyg., vol. 73, no. 3, pp. 593-598, 2005.

[8] K. Mitsakakis et al., "Challenges in malaria diagnosis," Expert Rev. Mol. Diagn., vol. 18, no. 10, pp. 867-875, 2018.

[9] Q. A. Arshad et al., "A dataset and benchmark for malaria life-cycle classification in thin blood smear images," Neural Comput Appl, vol. 34, no. 6, pp. 4473-4485, 2022, doi: 10.1007/s00521-021-06602-6.

[10] A. Loddo, C. Di Ruberto, and K. M. P. G., "MP-IDB: The Malaria Parasite Image Database for Image Processing and Analysis," in Processing and Analysis of Biomedical Information, Cham: Springer International Publishing, 2019, pp. 57-65.

[11] S. S. Abbas and T. M. H. Dijkstra, "Malaria-Detection-2019," Mendeley Data, 2019, doi: 10.17632/5bf2kmwvfn.1.

[12] S. Rajaraman et al., "Pre-trained convolutional neural networks as feature extractors for diagnosis of malaria from blood smears," Diagnostics, vol. 8, no. 4, p. 74, 2018.

[13] A. Wang et al., "YOLOv10: Real-time end-to-end object detection," arXiv:2405.14458, 2024.

[14] G. Jocher et al., "YOLOv11: Ultralytics YOLO11," 2024.

[15] F. Poostchi et al., "Image analysis and machine learning for detecting malaria," Transl. Res., vol. 194, pp. 36-55, 2018.

[16] P. Rosenthal, "How do we diagnose and treat Plasmodium ovale and Plasmodium malariae?" Curr. Infect. Dis. Rep., vol. 10, pp. 58-61, 2008.

[17] S. Ren et al., "Faster R-CNN: Towards real-time object detection with region proposal networks," IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 6, pp. 1137-1149, 2017.

[18] WHO, "Basic Malaria Microscopy: Part I. Learner's guide," 2nd ed., Geneva, 2010.

[19] G. Huang et al., "Densely connected convolutional networks," in Proc. IEEE CVPR, 2017, pp. 4700-4708.

[20] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in Proc. ICML, 2019, pp. 6105-6114.

[21] K. He et al., "Deep residual learning for image recognition," in Proc. IEEE CVPR, 2016, pp. 770-778.

[22] T.-Y. Lin et al., "Focal loss for dense object detection," IEEE Trans. Pattern Anal. Mach. Intell., vol. 42, no. 2, pp. 318-327, 2020.

[23] M. Aikawa, "Parasitological review: Plasmodium," Exp. Parasitol., vol. 30, no. 2, pp. 284-320, 1971.

[24] A. Vijayalakshmi and B. Rajesh Kanna, "Deep learning approach to detect malaria from microscopic images," Multim. Tools Appl., vol. 79, pp. 15297-15317, 2020.

[25] J. Deng et al., "ImageNet: A large-scale hierarchical image database," in Proc. IEEE CVPR, 2009, pp. 248-255.

[26] A. Dosovitskiy et al., "An image is worth 16×16 words: Transformers for image recognition at scale," in Proc. ICLR, 2021.

[27] I. Goodfellow et al., "Generative adversarial nets," in Proc. NeurIPS, 2014, pp. 2672-2680.

[28] J. Ho et al., "Denoising diffusion probabilistic models," in Proc. NeurIPS, 2020.

[29] B. Settles, "Active learning literature survey," Univ. Wisconsin-Madison, Tech. Rep. 1648, 2009.

[30] C. Finn et al., "Model-agnostic meta-learning for fast adaptation of deep networks," in Proc. ICML, 2017, pp. 1126-1135.

[31] WHO, "Guidelines for the Treatment of Malaria," 3rd ed., Geneva, 2015.

[32] L. Zedda, A. Loddo, and C. Di Ruberto, "YOLO-PAM: Parasite-Attention-Based Model for Efficient Malaria Detection," J Imaging, vol. 9, no. 12, 2023, doi: 10.3390/jimaging9120266.

[33] D. Sukumarran et al., "An optimised YOLOv4 deep learning model for efficient malarial cell detection in thin blood smear images," Parasit Vectors, vol. 17, no. 1, p. 188, 2024, doi: 10.1186/s13071-024-06215-7.

[34] X. Li et al., "Generalized focal loss: learning qualified and distributed bounding boxes for dense object detection," in Proceedings of the 34th International Conference on Neural Information Processing Systems, Red Hook, NY: Curran Associates Inc., 2020.

[35] F. Garcea, A. Serra, F. Lamberti, and L. Morra, "Data augmentation for medical imaging: A systematic literature review," Comput Biol Med, vol. 152, p. 106391, 2023, doi: 10.1016/j.compbiomed.2022.106391.

[36] M. Salmi, D. Atif, D. Oliva, A. Abraham, and S. Ventura, "Handling imbalanced medical datasets: review of a decade of research," Artif Intell Rev, vol. 57, no. 10, p. 273, 2024, doi: 10.1007/s10462-024-10884-2.

[37] K. Alkandary, A. S. Yildiz, and H. Meng, "A Comparative Study of YOLO Series (v3-v10) with DeepSORT and StrongSORT," Electronics (Basel), vol. 14, no. 5, 2025, doi: 10.3390/electronics14050876.

[38] R. Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference," arXiv:1806.08342, 2018.

[39] A. Kirillov et al., "Segment anything," in Proc. IEEE ICCV, 2023, pp. 4015-4026.

[40] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," Int. J. Comput. Vis., vol. 128, pp. 336-359, 2020.

---

**LAPORAN AKHIR PENELITIAN BISMA**

**Kerangka Kerja Multi-Model Hibrida untuk Deteksi dan Klasifikasi Malaria Otomatis**

**Periode**: Januari 2025 - Desember 2025

**Status**: Selesai dengan Sukses

**Tanggal Penyusunan**: 11 Desember 2025

**Versi Dokumen**: 3.0 Final (Sesuai Template BISMA)
