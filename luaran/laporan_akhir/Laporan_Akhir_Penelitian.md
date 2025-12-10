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

## A. RINGKASAN EKSEKUTIF

Penelitian ini telah berhasil mengembangkan sistem deteksi dan klasifikasi parasit malaria secara otomatis. Sistem menggunakan arsitektur hibrida yang menggabungkan model YOLO untuk deteksi objek dengan model CNN untuk klasifikasi spesies dan tahapan siklus hidup parasit. Malaria masih menjadi tantangan kesehatan global dengan lebih dari 200 juta kasus dan 600 ribu kematian setiap tahun. Diagnosis akurat sangat penting karena spesies yang berbeda memerlukan pendekatan pengobatan yang berbeda pula.

Sistem telah divalidasi secara menyeluruh pada empat dataset publik yang mencakup 1.544 citra apusan darah. Dataset tersebut meliputi IML Lifecycle dengan 313 citra dan 4 tahapan, MP-IDB Species dengan 209 citra dan 4 spesies, MP-IDB Stages dengan 209 citra dan 4 tahapan, serta MD-2019 Stages dengan 813 citra dan 3 tahapan. Implementasi menggunakan Arsitektur Option A yang menghasilkan pengurangan penyimpanan sebesar 70 persen dan pengurangan waktu pelatihan sebesar 60 persen. Pendekatan ini jauh lebih efisien dibanding metode tradisional yang melatih model klasifikasi terpisah untuk setiap metode deteksi.

### Hasil Utama Penelitian

**Performa Deteksi Parasit:**

Hasil deteksi menggunakan metrik mAP@50 menunjukkan performa sangat baik pada keempat dataset. Pada dataset IML Lifecycle, model YOLOv11 mencapai akurasi 96,61 persen dengan latensi hanya 13,7 milidetik per gambar. Dataset MP-IDB Species mencapai akurasi 96,56 persen dengan YOLOv11 yang memiliki nilai recall tertinggi sebesar 95,29 persen. Pada dataset MP-IDB Stages, model YOLOv12 mencapai 95,62 persen dengan presisi tertinggi 92,16 persen. Dataset terbesar MD-2019 Stages mencapai akurasi 93,46 persen menggunakan YOLOv12.

**Performa Klasifikasi Spesies:**

Hasil klasifikasi menunjukkan tingkat akurasi yang bervariasi tergantung karakteristik dataset. Dataset IML Lifecycle mencapai akurasi 91,51 persen dan balanced accuracy 91,96 persen menggunakan model EfficientNet-B2. Dataset MP-IDB Species mencapai akurasi tertinggi sebesar 98,62 persen dengan balanced accuracy 88,10 persen menggunakan ResNet101. Dataset MP-IDB Stages dengan ketidakseimbangan kelas ekstrem mencapai 95,42 persen akurasi dan 76,99 persen balanced accuracy. Dataset MD-2019 Stages yang paling menantang mencapai 86,62 persen akurasi dan 85,51 persen balanced accuracy menggunakan EfficientNet-B0.

**Temuan Penting:**

Penelitian ini menghasilkan beberapa temuan penting. Pertama, model EfficientNet dengan parameter lebih kecil (5,3 hingga 9,2 juta parameter) secara konsisten mencapai performa setara atau lebih baik dibanding model ResNet yang jauh lebih besar (25,6 hingga 44,5 juta parameter). Kedua, model YOLOv11 menunjukkan keunggulan dalam nilai recall (92 hingga 96 persen) yang sangat penting untuk meminimalkan parasit yang terlewat dalam pengaturan klinis. Ketiga, optimisasi menggunakan Focal Loss dengan parameter alpha 0,25 dan gamma 2,0 berhasil menangani ketidakseimbangan kelas ekstrem dengan rasio hingga 54 banding 1. Keempat, sistem mampu melakukan inferensi kurang dari 25 milidetik per gambar atau lebih dari 40 bingkai per detik pada GPU NVIDIA RTX 4090, membuktikan kelayakan untuk penerapan praktis.

**Kontribusi Ilmiah:**

Penelitian ini memberikan beberapa kontribusi ilmiah. Arsitektur Option A yang inovatif mengurangi kompleksitas komputasi secara signifikan dengan pendekatan klasifikasi bersama. Validasi pada empat dataset berbeda menunjukkan kemampuan generalisasi sistem yang baik. Analisis sistematis mengungkap hubungan antara ukuran model dan performa pada dataset citra medis berskala kecil. Strategi penanganan ketidakseimbangan kelas terbukti efektif menggunakan Focal Loss dan pengambilan sampel berbobot.

Meskipun hasil sangat menjanjikan, penelitian mengidentifikasi tantangan pada kelas minoritas dengan kurang dari 10 sampel. Kelas minoritas hanya mencapai skor F1 antara 44 hingga 80 persen pada beberapa kasus. Hal ini menunjukkan perlunya perluasan dataset dan teknik pembelajaran lanjutan seperti few-shot learning untuk meningkatkan performa pada kelas minoritas yang penting secara klinis.

---

## B. LATAR BELAKANG DAN TUJUAN PENELITIAN

### B.1 Latar Belakang

Pemeriksaan mikroskopik apusan darah dengan pewarnaan Giemsa tetap menjadi standar emas untuk diagnosis malaria. Namun metode ini menghadapi keterbatasan signifikan terutama di daerah endemis. Ahli mikroskopis memerlukan pelatihan intensif selama 2 hingga 3 tahun untuk mencapai kompetensi yang memadai. Proses pemeriksaan memakan waktu 20 hingga 30 menit per slide dan memerlukan konsentrasi tinggi. Tingkat kesepakatan antar pengamat hanya berkisar 60 hingga 85 persen bahkan di antara profesional terlatih. Tantangan ini menciptakan hambatan dalam sistem kesehatan, terutama di daerah terpencil dimana akses terhadap ahli mikroskopis sangat terbatas.

Perkembangan pembelajaran mendalam telah menunjukkan potensi besar untuk analisis citra medis secara otomatis. Arsitektur deteksi objek seperti Faster R-CNN dan YOLO versi terbaru menawarkan keunggulan dengan kecepatan inferensi waktu nyata kurang dari 15 milidetik per gambar dan akurasi yang kompetitif. Namun tantangan kritis masih ada dalam penerapannya. Dataset beranotasi sangat terbatas dengan hanya 200 hingga 500 gambar per tugas. Ketidakseimbangan kelas ekstrem dimana spesies langka hanya mencakup kurang dari 2 persen sampel menjadi masalah serius. Pendekatan yang ada melatih model klasifikasi terpisah untuk setiap metode deteksi sehingga menghasilkan beban komputasi yang besar.

### B.2 Tujuan Penelitian

Penelitian ini bertujuan mengembangkan kerangka kerja hibrida dengan arsitektur klasifikasi bersama yang inovatif. Tujuan spesifik penelitian adalah sebagai berikut.

Pertama, mengimplementasikan Arsitektur Option A yang melatih model klasifikasi satu kali pada citra terpotong dari anotasi asli. Model yang sama kemudian digunakan kembali untuk beberapa model YOLO yang berbeda. Target penelitian adalah mencapai pengurangan penyimpanan minimal 60 persen dan pengurangan waktu pelatihan minimal 50 persen.

Kedua, melakukan validasi menyeluruh pada empat dataset berbeda untuk menunjukkan kemampuan generalisasi sistem. Dataset yang digunakan meliputi IML Lifecycle, MP-IDB Species, MP-IDB Stages, dan MD-2019 Stages.

Ketiga, menganalisis secara sistematis hubungan antara ukuran model dan performa pada dataset citra medis berskala kecil. Penelitian membandingkan enam arsitektur CNN dengan jumlah parameter antara 5,3 hingga 44,5 juta. Arsitektur yang dibandingkan meliputi DenseNet121, tiga varian EfficientNet, dan dua varian ResNet.

Keempat, mengoptimalkan strategi penanganan ketidakseimbangan kelas menggunakan Focal Loss dengan parameter alpha 0,25 dan gamma 2,0. Target penelitian adalah mencapai skor F1 yang memadai untuk kelas minoritas dengan ukuran sampel sangat terbatas kurang dari 10 sampel.

Kelima, mendemonstrasikan kelayakan praktis untuk penerapan di tempat layanan dengan menargetkan latensi inferensi kurang dari 30 milidetik per gambar pada perangkat keras konsumen.

### B.3 Urgensi dan Manfaat

**Urgensi Penelitian:**

Organisasi Kesehatan Dunia melaporkan 263 juta kasus malaria dan 597 ribu kematian pada tahun 2023. Sebanyak 85 persen kematian terjadi di Afrika Sub-Sahara dimana akses ke ahli mikroskopis sangat terbatas. Keterlambatan diagnosis dapat meningkatkan kematian hingga 10 kali lipat pada kasus malaria berat. Variabilitas antar pengamat mencapai 15 hingga 40 persen bahkan di antara ahli, menunjukkan kebutuhan akan sistem objektif.

**Manfaat Penelitian:**

Penelitian ini memberikan beberapa manfaat penting. Dari aspek kesehatan publik, sistem deteksi otomatis dapat mengurangi beban kerja mikroskopis hingga 70 persen sehingga memungkinkan penyaringan lebih cepat di daerah endemis. Dari aspek efisiensi biaya, pengurangan waktu diagnosis dari 20-30 menit menjadi kurang dari 1 menit per slide menghemat biaya operasional secara signifikan. Dari aspek konsistensi diagnostik, sistem mengeliminasi variabilitas antar pengamat melalui diagnosis otomatis. Dari aspek aksesibilitas, penerapan pada perangkat keras konsumen memungkinkan implementasi di fasilitas kesehatan dengan sumber daya terbatas. Dari aspek pengembangan ilmu, penelitian memberikan kontribusi metodologi untuk citra medis dengan dataset kecil dan tidak seimbang.

---

## C. METODE PENELITIAN

### C.1 Dataset dan Karakteristik Data

Penelitian memanfaatkan empat dataset publik dengan total 1.544 citra apusan darah. Dataset pertama adalah IML Lifecycle [9] dengan 313 citra yang mencakup 4 tahapan siklus hidup parasit. Dataset kedua dan ketiga adalah MP-IDB Species dan MP-IDB Stages [10], masing-masing dengan 209 citra yang mencakup 4 spesies dan 4 tahapan. Dataset keempat adalah MD-2019 Stages [11] dengan 813 citra yang mencakup 3 tahapan. Ringkasan lengkap karakteristik keempat dataset dapat dilihat pada Tabel 1.

**Tabel 1: Ringkasan Dataset Penelitian**

Lihat: `luaran\laporan_akhir\tables\dataset_statistics_all.csv`

Semua dataset menggunakan apusan darah tipis dengan mikroskopi cahaya perbesaran 1000 kali dan pewarnaan Giemsa sesuai protokol standar WHO. Pembagian data menggunakan rasio 60:20:20 untuk data latih, validasi, dan uji dengan pengambilan sampel berlapis untuk mempertahankan distribusi kelas.

**Karakteristik Ketidakseimbangan Kelas:**

Dataset IML Lifecycle memiliki rasio antara kelas Gametocyte dengan 49 sampel dan Schizont dengan 4 sampel sebesar 12 banding 1. Dataset MP-IDB Species memiliki rasio ekstrem antara P. falciparum dengan 259 sampel dan P. ovale dengan 7 sampel sebesar 37 banding 1. Dataset MP-IDB Stages memiliki ketidakseimbangan paling ekstrem dengan rasio antara Ring dengan 259 sampel dan Gametocyte dengan 5 sampel sebesar 52 banding 1. Dataset MD-2019 Stages paling seimbang dengan rasio antara Schizont dengan 286 sampel dan Trophozoite dengan 127 sampel sebesar 2,3 banding 1.

### C.2 Augmentasi Data yang Aman untuk Medis

Penelitian menerapkan augmentasi data yang aman untuk citra medis [35] guna mengatasi keterbatasan ukuran data sambil mempertahankan integritas diagnostik. Augmentasi dibedakan menjadi dua tahap yaitu tahap deteksi dan tahap klasifikasi.

Pada tahap deteksi, augmentasi menghasilkan perkalian 4,4 kali lipat jumlah data. Teknik yang digunakan meliputi penskalaan acak antara 0,5 hingga 1,5 kali, rotasi hingga plus minus 15 derajat, penyesuaian nilai HSV, augmentasi mosaik, dan pencerminan horizontal. Pencerminan vertikal tidak digunakan untuk mempertahankan orientasi parasit yang benar.

Pada tahap klasifikasi, augmentasi menghasilkan perkalian 3,5 kali lipat jumlah data. Teknik yang digunakan meliputi rotasi hingga plus minus 20 derajat, transformasi affine, variasi warna, penambahan derau Gaussian, dan pengambilan sampel acak berbobot. Kelas minoritas diberi pembobotan lebih besar dengan rasio oversampling 3 banding 1 untuk memastikan representasi yang lebih baik.

### C.3 Arsitektur Pipeline dengan Pendekatan Klasifikasi Bersama

Penelitian mengimplementasikan Arsitektur Option A yang terdiri dari tiga tahap utama. Tahap pertama adalah pelatihan model deteksi. Tahap kedua adalah pembuatan citra terpotong dari anotasi asli. Tahap ketiga adalah pelatihan model klasifikasi yang digunakan bersama untuk semua metode deteksi.

**Tahap Pertama - Pelatihan Model Deteksi:**

Penelitian menggunakan tiga varian model YOLO yaitu YOLOv10, YOLOv11, dan YOLOv12 dengan ukuran Medium. Citra masukan berukuran 640 kali 640 piksel. Pelatihan dilakukan selama 100 epoch dengan mekanisme penghentian dini. Optimisasi menggunakan algoritma AdamW dengan laju pembelajaran 0,001. Fungsi kerugian gabungan mencakup kerugian untuk kotak pembatas, tingkat objektivitas, dan klasifikasi.

**Tahap Kedua - Pembuatan Citra Terpotong:**

Citra terpotong berukuran 224 kali 224 piksel diekstraksi langsung dari kotak pembatas anotasi asli. Prapemrosesan meliputi normalisasi menggunakan statistik ImageNet. Keuntungan pendekatan ini adalah citra terpotong dibuat satu kali dari anotasi yang sempurna, bukan dari output deteksi yang mungkin mengandung kesalahan.

**Tahap Ketiga - Pelatihan Model Klasifikasi:**

Penelitian melatih enam arsitektur CNN yaitu DenseNet121, tiga varian EfficientNet (B0, B1, B2), dan dua varian ResNet (50, 101). Pelatihan dilakukan selama 75 epoch dengan penjadwalan laju pembelajaran. Fungsi kerugian menggunakan Focal Loss dengan parameter alpha 0,25 dan gamma 2,0. Optimisasi menggunakan algoritma AdamW dengan peluruhan bobot 0,0001. Laju pembelajaran awal sebesar 0,0001 dengan peluruhan 0,5 setiap 10 epoch.

Focal Loss dirancang khusus untuk menangani ketidakseimbangan kelas dengan rumus FL sama dengan minus alpha dikali satu minus probabilitas prediksi pangkat gamma dikali logaritma probabilitas prediksi. Parameter alpha 0,25 menyeimbangkan kepentingan antara sampel positif dan negatif. Parameter gamma 2,0 memfokuskan pembelajaran pada contoh yang sulit.

**Manfaat Arsitektur Option A:**

Pendekatan ini memberikan empat manfaat utama. Pertama, pengurangan penyimpanan sekitar 70 persen karena citra terpotong dibuat satu kali, bukan untuk setiap model YOLO. Kedua, pengurangan waktu pelatihan sekitar 60 persen karena model klasifikasi dilatih satu kali untuk semua metode deteksi. Ketiga, evaluasi konsisten karena model klasifikasi yang sama digunakan untuk membandingkan metode deteksi yang berbeda. Keempat, pemisahan bersih antara tahap deteksi dan klasifikasi memungkinkan optimisasi independen.

### C.4 Konfigurasi Perangkat Keras dan Optimisasi

Penelitian menggunakan GPU NVIDIA RTX 4090 dengan memori 24 GB untuk pelatihan model. Lingkungan perangkat lunak meliputi CUDA 12.8, PyTorch 2.8.0 dengan dukungan CUDA, dan Ultralytics 8.3.202 untuk framework YOLO.

Beberapa teknik optimisasi komputasi diterapkan untuk mempercepat pelatihan. Teknik Mixed Precision menggunakan AMP memberikan percepatan 2 kali lipat. Benchmark cuDNN memberikan percepatan 2 hingga 3 kali lipat untuk operasi konvolusi. Format memori channels-last memberikan percepatan 20 hingga 35 persen untuk operasi tensor. DataLoader dengan 4 worker memberikan startup cepat dan throughput tinggi. Total percepatan yang dicapai adalah 6 hingga 10 kali lipat dibanding konfigurasi dasar.

### C.5 Metrik Evaluasi

Penelitian menggunakan beberapa metrik untuk mengevaluasi performa sistem. Untuk tahap deteksi, metrik yang digunakan meliputi mAP@50 yang mengukur presisi rata-rata pada ambang batas IoU 0,5, mAP@50-95 yang mengukur mAP rata-rata pada rentang ambang batas 0,5 hingga 0,95, presisi yang menghitung rasio prediksi benar terhadap total prediksi positif, dan recall yang menghitung rasio parasit terdeteksi terhadap total parasit sebenarnya.

Untuk tahap klasifikasi, metrik yang digunakan meliputi akurasi keseluruhan, balanced accuracy yang menghitung rata-rata recall per kelas untuk menangani ketidakseimbangan, precision recall dan skor F1 untuk setiap kelas, serta matriks konfusi untuk analisis pola kesalahan.

---

## D. HASIL DAN PEMBAHASAN

### D.1 Hasil Deteksi Parasit Malaria

#### D.1.1 Performa Kuantitatif

Model deteksi YOLO menunjukkan performa sangat baik pada semua dataset dengan konsisten mencapai mAP@50 di atas 91 persen. Hasil lengkap untuk keempat dataset dapat dilihat pada Tabel 2.

**Tabel 2: Performa Deteksi YOLO pada 4 Dataset**

Lihat: `luaran\laporan_akhir\tables\detection_performance_all_datasets.csv`

**Temuan Kunci dari Hasil Deteksi:**

Beberapa temuan penting dapat disimpulkan dari hasil deteksi. Pertama, model YOLOv11 mencapai recall tertinggi pada dataset IML Lifecycle sebesar 95,88 persen dan MP-IDB Species sebesar 95,29 persen. Nilai recall tinggi sangat penting dalam pengaturan klinis untuk meminimalkan parasit yang terlewat karena dapat berakibat fatal.

Kedua, model YOLOv12 mencapai presisi tertinggi pada dataset IML Lifecycle sebesar 89,38 persen, MP-IDB Species sebesar 94,38 persen, dan MP-IDB Stages sebesar 92,16 persen. Presisi tinggi mengurangi alarm palsu yang dapat menyebabkan pengobatan yang tidak perlu.

Ketiga, ketiga model YOLO menunjukkan konsistensi dengan mencapai mAP@50 di atas 91 persen pada semua dataset. Hal ini mendemonstrasikan ketahanan terhadap variasi data yang berbeda.

Keempat, terdapat pertukaran antara presisi dan recall dimana YOLOv11 mengoptimalkan recall untuk mendeteksi semua parasit, sedangkan YOLOv12 mengoptimalkan presisi untuk deteksi yang lebih akurat.

Kelima, kecepatan inferensi sangat memadai dengan YOLOv10 memerlukan 12,3 milidetik per gambar, YOLOv11 memerlukan 13,7 milidetik, dan YOLOv12 memerlukan 15,2 milidetik. Semua model memenuhi persyaratan waktu nyata kurang dari 30 milidetik.

#### D.1.2 Analisis Per Dataset

**Dataset IML Lifecycle:**

Dataset ini mencakup 4 tahapan siklus hidup P. falciparum. Model terbaik adalah YOLOv11 dengan mAP@50 sebesar 96,61 persen dan recall 95,88 persen. Tantangan utama adalah membedakan tahap ring dan trophozoite yang memiliki morfologi tumpang tindih. Nilai recall 95,88 persen menunjukkan kemampuan deteksi sangat baik pada semua tahap.

**Dataset MP-IDB Species:**

Dataset ini mencakup 4 spesies Plasmodium. Model terbaik adalah YOLOv11 dengan mAP@50 sebesar 96,56 persen dan recall 95,29 persen. Tantangan utama adalah ketidakseimbangan ekstrem dimana P. falciparum memiliki 259 sampel sedangkan P. ovale hanya 7 sampel. Nilai recall tinggi menunjukkan deteksi efektif bahkan untuk spesies langka.

**Dataset MP-IDB Stages:**

Dataset ini mencakup 4 tahapan generik. Model terbaik adalah YOLOv12 dengan mAP@50 sebesar 95,62 persen dan presisi 92,16 persen. Tantangan utama adalah dataset terkecil dengan hanya 250 sampel latih dan ketidakseimbangan ekstrem. Hasil mAP@50 sebesar 95,62 persen menunjukkan pembelajaran efektif meskipun data terbatas.

**Dataset MD-2019 Stages:**

Dataset ini mencakup 3 tahapan P. vivax. Model terbaik adalah YOLOv12 dengan mAP@50 sebesar 93,46 persen dan presisi 87,82 persen. Tantangan utama adalah dataset terbesar dengan 936 sampel latih yang memiliki variasi pewarnaan dan kondisi pencitraan. Konsistensi di atas 92 persen mAP@50 menunjukkan generalisasi baik pada data beragam.

### D.2 Hasil Klasifikasi Spesies dan Tahapan

#### D.2.1 Performa Kuantitatif

Hasil klasifikasi menunjukkan performa yang bervariasi tergantung karakteristik dataset. Performa model terbaik untuk setiap dataset dapat dilihat pada Tabel 3.

**Tabel 3: Performa Klasifikasi CNN pada 4 Dataset**

Lihat: `luaran\laporan_akhir\tables\classification_focal_loss_all_datasets.csv`

**Temuan Kunci dari Hasil Klasifikasi:**

Beberapa temuan penting dapat disimpulkan dari hasil klasifikasi. Pertama, pemilihan model terbaik bergantung pada karakteristik dataset. Pada dataset IML Lifecycle, model EfficientNet-B2 optimal dengan akurasi 91,51 persen dan balanced accuracy 91,96 persen. Pada dataset MP-IDB Species, model ResNet101 terbaik dengan akurasi 98,62 persen dan balanced accuracy 88,10 persen. Pada dataset MP-IDB Stages, model EfficientNet-B1 dan ResNet101 kompetitif dengan akurasi di atas 95 persen. Pada dataset MD-2019, model EfficientNet-B0 superior dengan akurasi 86,62 persen dan balanced accuracy 85,51 persen.

Kedua, model EfficientNet dengan parameter lebih kecil (5,3 hingga 9,2 juta parameter) mencapai performa kompetitif dengan waktu pelatihan 15 hingga 30 persen lebih cepat dibanding ResNet (25,6 hingga 44,5 juta parameter).

Ketiga, perbedaan antara akurasi dan balanced accuracy mengungkap tantangan pada kelas minoritas. Pada dataset MP-IDB Species, terdapat kesenjangan 10,52 poin persentase antara akurasi 98,62 persen dan balanced accuracy 88,10 persen, menunjukkan spesies langka masih menantang. Pada dataset MP-IDB Stages, kesenjangan mencapai 18,08 poin persentase dari akurasi 95,07 persen ke balanced accuracy 76,99 persen, menunjukkan dampak ketidakseimbangan ekstrem.

Keempat, performa pada kelas tersulit yang umumnya adalah Trophozoite atau P. ovale menunjukkan variasi besar. Skor F1 terbaik mencapai 0,92 untuk P. ovale menggunakan ResNet101 pada dataset Species. Skor F1 tersulit berada di rentang 0,48 hingga 0,74 untuk Trophozoite pada dataset Stages, menunjukkan kebutuhan strategi lebih baik untuk kelas ultra-minoritas dengan kurang dari 10 sampel.

#### D.2.2 Analisis Detail Per Kelas

**Dataset IML Lifecycle:**

Dataset ini mencakup 4 tahapan P. falciparum. Kelas Gametocyte dengan 49 sampel mencapai presisi 95,74 persen, recall 91,84 persen, dan skor F1 sebesar 93,75 persen menggunakan model EfficientNet-B2. Kelas Ring dengan 34 sampel mencapai presisi 94,29 persen, recall 97,06 persen, dan skor F1 sebesar 95,65 persen. Kelas Schizont dengan hanya 4 sampel mencapai skor sempurna 100 persen pada semua metrik menggunakan model DenseNet121 dan EfficientNet-B2 karena morfologi yang sangat khas dengan banyak merozoit tersegmentasi. Kelas Trophozoite dengan 19 sampel paling menantang dengan presisi 75 persen, recall 78,95 persen, dan skor F1 sebesar 76,92 persen karena morfologi yang sangat bervariasi.

**Dataset MP-IDB Species:**

Dataset ini mencakup 4 spesies Plasmodium. Kelas P. falciparum dengan 259 sampel mencapai klasifikasi hampir sempurna dengan presisi 98,85 persen, recall 100 persen, dan skor F1 sebesar 99,42 persen menggunakan ResNet101. Kelas P. malariae dengan 9 sampel mencapai presisi sempurna 100 persen namun recall hanya 66,67 persen sehingga skor F1 sebesar 80 persen. Kelas P. ovale dengan 7 sampel mencapai presisi sempurna 100 persen, recall 85,71 persen, dan skor F1 mengesankan sebesar 92,31 persen meskipun ultra-minoritas berkat morfologi khas dengan sel darah merah membesar. Kelas P. vivax dengan 15 sampel mencapai presisi 93,75 persen, recall sempurna 100 persen, dan skor F1 sebesar 96,77 persen. Semua spesies langka mencapai presisi sempurna 100 persen menunjukkan tidak ada positif palsu.

**Dataset MP-IDB Stages:**

Dataset ini memiliki ketidakseimbangan paling ekstrem. Kelas Ring dengan 259 sampel mendominasi dan mencapai presisi 97,36 persen, recall 99,61 persen, dan skor F1 sangat baik sebesar 98,47 persen menggunakan EfficientNet-B1. Kelas Gametocyte dengan hanya 5 sampel mencapai skor sempurna 100 persen pada semua metrik menggunakan ResNet101, menunjukkan efektivitas Focal Loss. Kelas Schizont dengan 6 sampel mencapai presisi 50 persen, recall 83,33 persen, dan skor F1 sebesar 62,50 persen menggunakan DenseNet121. Kelas Trophozoite dengan 14 sampel paling menantang dengan presisi 70 persen, recall 50 persen, dan skor F1 hanya 58,33 persen, mengindikasikan pola kebingungan antara tahap minoritas.

**Dataset MD-2019 Stages:**

Dataset ini paling seimbang dengan rasio hanya 2,3 banding 1. Kelas Ring dengan 170 sampel mencapai presisi 86,41 persen, recall 93,53 persen, dan skor F1 sebesar 89,83 persen menggunakan EfficientNet-B0. Kelas Schizont dengan 286 sampel mencapai presisi 94,34 persen, recall 87,41 persen, dan skor F1 sebesar 90,74 persen. Kelas Trophozoite dengan 127 sampel masih menantang meskipun ukuran sampel memadai, dengan presisi 71,64 persen, recall 75,59 persen, dan skor F1 sebesar 73,56 persen karena tumpang tindih morfologi dengan tahap lain. Model EfficientNet-B0 mencapai performa seimbang di semua kelas.

### D.3 Efisiensi Komputasi dan Skalabilitas

#### D.3.1 Analisis Waktu Pelatihan

Perbandingan waktu pelatihan menunjukkan efisiensi model yang berbeda. Waktu pelatihan untuk 75 epoch pada tahap klasifikasi dapat dilihat pada Tabel 4.

**Tabel 4: Perbandingan Waktu Pelatihan**

Lihat: `luaran\laporan_akhir\tables\training_time_comparison.csv`

Model EfficientNet-B0 tercepat dengan waktu 2,3 jam untuk IML Lifecycle dibanding 3,4 jam untuk ResNet101, menghasilkan percepatan 32 persen. Model EfficientNet-B0 dengan 5,3 juta parameter mencapai performa kompetitif dengan 88 persen parameter lebih sedikit dibanding ResNet101 dengan 44,5 juta parameter. Perilaku penskalaan menunjukkan dataset MD-2019 dengan 936 sampel latih memerlukan waktu 1,7 kali lebih lama dibanding IML dengan 372 sampel.

#### D.3.2 Latensi Inferensi

Analisis latensi untuk satu gambar menunjukkan sistem memenuhi persyaratan waktu nyata. Tahap deteksi memerlukan 12,3 milidetik untuk YOLO10, 13,7 milidetik untuk YOLO11, dan 15,2 milidetik untuk YOLO12. Tahap ekstraksi citra terpotong memerlukan 1,5 milidetik. Tahap klasifikasi untuk rata-rata 10 kotak per gambar memerlukan 8,2 milidetik. Total latensi end-to-end adalah 22,0 milidetik untuk YOLO10, 23,4 milidetik untuk YOLO11, dan 24,9 milidetik untuk YOLO12. Throughput mencapai 45 bingkai per detik untuk YOLO10, 43 untuk YOLO11, dan 40 untuk YOLO12.

Pengujian dilakukan pada GPU NVIDIA RTX 4090 dengan presisi campuran FP16 dan ukuran batch 1. Semua konfigurasi mencapai latensi kurang dari 25 milidetik atau lebih dari 40 bingkai per detik. Persyaratan waktu nyata kurang dari 30 milidetik terpenuhi dengan margin aman. Satu slide dengan 100 bidang dapat diproses dalam kurang dari 4 detik dibanding 20 hingga 30 menit secara manual. Percepatan yang dicapai adalah 300 hingga 450 kali lipat dibanding mikroskopi tradisional.

#### D.3.3 Penggunaan Memori

Penggunaan memori GPU puncak untuk pelatihan deteksi adalah 18,5 GB untuk YOLO11 dengan ukuran batch 16. Untuk pelatihan klasifikasi adalah 12,3 GB untuk ResNet101 dengan ukuran batch 64. Untuk inferensi adalah 4,2 GB untuk pipeline lengkap dengan ukuran batch 1.

Ukuran model untuk deployment menunjukkan efisiensi penyimpanan. Model YOLO berukuran antara 46 hingga 52 MB masing-masing. Model klasifikasi berukuran antara 20 hingga 170 MB masing-masing. Total sistem lengkap dengan YOLO11 dan EfficientNet-B1 hanya 98 MB.

Kelayakan deployment menunjukkan sistem dapat berjalan pada GPU 24 GB seperti RTX 4090 atau 3090. Inferensi dimungkinkan pada GPU 8 GB seperti RTX 3060 atau 4060. Pada sistem CPU saja, latensi inferensi meningkat menjadi 180 hingga 250 milidetik namun masih dapat digunakan.

### D.4 Analisis Pola Kesalahan

#### D.4.1 Pola Kesalahan Klasifikasi Umum

Analisis pola kesalahan mengungkap beberapa kebingungan klasifikasi yang umum terjadi. Pada dataset IML Lifecycle, terdapat kebingungan antara Trophozoite dan Ring sebesar 21 persen dari kesalahan. Penyebabnya adalah trophozoite awal secara morfologi mirip dengan ring tahap lanjut. Dampak klinis rendah karena keduanya adalah tahap aseksual awal dengan pendekatan pengobatan serupa.

Pada dataset MP-IDB Species, terdapat kebingungan antara P. ovale dan P. vivax sebesar 14 persen dari sampel P. ovale. Penyebabnya adalah keduanya menunjukkan sel darah merah membesar dan bintik Schüffner. Dampak klinis sedang karena pola kambuh berbeda dan dosis primakuin berbeda untuk pengobatan radikal.

Pada dataset MP-IDB Stages, terdapat kebingungan antara Trophozoite dan Ring sebesar 36 persen dari sampel trophozoite. Penyebabnya adalah ketidakseimbangan ekstrem dengan 259 sampel Ring versus 14 sampel Trophozoite menyebabkan bias. Dampak klinis rendah karena keduanya adalah tahap aseksual dengan pendekatan pengobatan serupa.

Pada dataset MD-2019 Stages, terdapat kebingungan antara Trophozoite dan Schizont sebesar 24 persen dari sampel trophozoite. Penyebabnya adalah trophozoite tahap lanjut dengan merozoit berkembang menyerupai schizont awal. Dampak klinis rendah karena merupakan tahap perkembangan berurutan.

#### D.4.2 Analisis Kesalahan Berdasarkan Ukuran Kelas

Analisis berdasarkan rentang ukuran sampel menunjukkan pola yang jelas. Kelas dengan lebih dari 200 sampel mencapai skor F1 rata-rata 95 hingga 99 persen dengan kesalahan minimal dan pelatihan standar sudah cukup. Kelas dengan 50 hingga 200 sampel mencapai 90 hingga 95 persen dengan variabilitas sedang dan memerlukan Focal Loss serta augmentasi. Kelas dengan 10 hingga 50 sampel mencapai 75 hingga 90 persen dengan variabilitas tinggi dan memerlukan pengambilan sampel berbobot serta oversampling 3 kali lipat. Kelas dengan kurang dari 10 sampel menunjukkan variansi ekstrem dengan skor F1 antara 44 hingga 100 persen dan memerlukan teknik few-shot learning.

Temuan penting adalah penurunan performa menjadi parah di bawah 10 sampel. Skor F1 berkisar dari 44 persen pada kasus terburuk yaitu Trophozoite pada dataset MP-IDB Stages hingga 100 persen pada kasus terbaik yaitu Schizont pada dataset IML Lifecycle dengan morfologi sangat khas.

### D.5 Validitas dan Reliabilitas Sistem

#### D.5.1 Generalisasi Antar Dataset

Pengujian generalisasi dilakukan dengan melatih model pada satu dataset dan menguji pada dataset terkait. Model yang dilatih pada MP-IDB Stages dan diuji pada IML Lifecycle mengalami penurunan mAP@50 sebesar 15,2 persen dan penurunan akurasi sebesar 12,8 persen. Model yang dilatih pada IML Lifecycle dan diuji pada MP-IDB Stages mengalami penurunan mAP@50 sebesar 18,7 persen dan penurunan akurasi sebesar 15,3 persen.

Penurunan performa sebesar 12 hingga 19 persen mengindikasikan adanya pergeseran domain akibat perbedaan kondisi pencitraan dan protokol pewarnaan. Validasi eksternal dan adaptasi domain sangat penting untuk deployment klinis yang sesungguhnya.

#### D.5.2 Analisis Reproduksibilitas

Analisis reproduksibilitas dilakukan dengan menjalankan eksperimen sebanyak 5 kali menggunakan seed acak yang berbeda. Metrik deteksi mAP@50 mencapai rata-rata 94,83 persen dengan deviasi standar 1,24 persen dan koefisien variasi 1,31 persen. Metrik klasifikasi akurasi mencapai rata-rata 92,15 persen dengan deviasi standar 2,38 persen dan koefisien variasi 2,58 persen. Metrik balanced accuracy mencapai rata-rata 85,67 persen dengan deviasi standar 3,92 persen dan koefisien variasi 4,57 persen.

Koefisien variasi rendah di bawah 5 persen menunjukkan reproduksibilitas tinggi. Hal ini penting untuk validitas ilmiah dan keandalan klinis sistem.

---

## E. PENCAPAIAN LUARAN PENELITIAN

### E.1 Luaran Wajib

**Publikasi Jurnal Internasional Bereputasi**

Penelitian telah menghasilkan publikasi yang diterima untuk dipublikasikan. Judul artikel adalah "Parameter-Efficient Deep Learning Models for Malaria Detection and Classification Using Small-Scale Imbalanced Blood Smear Images". Artikel diterbitkan di jurnal KINETIK: Game Technology, Information System, Computer Network, Computing, Electronics, and Control dengan ISSN 2503-2259. Status artikel adalah diterima untuk publikasi pada Desember 2025. Bukti penerimaan tersedia di file screencapture-kinetik-umm-ac-id-*.pdf.

Kontribusi kunci artikel meliputi arsitektur klasifikasi bersama yang novel dengan pengurangan penyimpanan 70 persen, validasi menyeluruh pada 4 dataset berbeda, analisis sistematis efisiensi model pada dataset citra medis kecil, dan strategi Focal Loss efektif untuk penanganan ketidakseimbangan kelas ekstrem.

**Hak Kekayaan Intelektual**

Persiapan pendaftaran hak cipta software sedang dalam proses. Judul karya adalah "Sistem Deteksi dan Klasifikasi Malaria Otomatis Berbasis Deep Learning dengan Arsitektur Klasifikasi Bersama". Jenis perlindungan adalah hak cipta software. Persiapan dokumentasi teknis untuk pengajuan sedang dilakukan dengan target pengajuan pada triwulan pertama 2026.

### E.2 Luaran Tambahan

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

## F. PERAN MITRA DAN KOLABORASI

### F.1 Mitra Akademik

Kolaborasi dengan mitra akademik memberikan kontribusi penting bagi penelitian. Mitra menyediakan akses ke sumber daya komputasi berupa cluster GPU NVIDIA RTX 4090. Keahlian dalam pembelajaran mendalam dan citra medis diberikan melalui konsultasi rutin. Bimbingan dalam desain eksperimental dan ketelitian metodologis memastikan kualitas ilmiah penelitian. Supervisi bersama dilakukan untuk memastikan kualitas ilmiah. Tinjauan naskah dan saran perbaikan membantu peningkatan kualitas publikasi.

Hasil kolaborasi meliputi kepengarangan bersama dalam publikasi jurnal internasional, laporan teknis bersama, dan seminar penelitian serta sesi berbagi pengetahuan.

### F.2 Mitra Penyedia Data

Penelitian memanfaatkan tiga sumber dataset publik. Dataset pertama adalah IML Malaria Lifecycle Dataset yang tersedia di repository GitHub dengan lisensi akses terbuka untuk tujuan penelitian. Dataset memberikan kontribusi 313 gambar dengan anotasi siklus hidup.

Dataset kedua adalah MP-IDB atau Malaria Parasite Image Database yang tersedia di Kaggle dengan lisensi CC BY 4.0. Dataset memberikan kontribusi 418 gambar dengan anotasi spesies dan tahapan.

Dataset ketiga adalah MD-2019 Mendeley Dataset yang tersedia di Mendeley Data dengan lisensi CC BY 4.0. Dataset memberikan kontribusi 813 gambar dengan anotasi tahapan.

Penelitian menggunakan dataset yang tersedia secara publik dengan kutipan yang tepat dan kepatuhan terhadap lisensi.

### F.3 Rencana Mitra Klinis untuk Fase Lanjutan

Rencana kolaborasi dengan mitra klinis sedang dalam tahap negosiasi untuk fase validasi eksternal. Peran yang direncanakan meliputi validasi eksternal dengan sampel klinis dari lapangan, mikroskopi ahli untuk validasi kebenaran dasar, pengujian integrasi alur kerja klinis, umpan balik untuk perbaikan sistem secara bertahap, dan persetujuan komite etik untuk studi klinis.

Target untuk fase kedua pada tahun 2026 meliputi pengumpulan 500 sampel klinis yang beragam, studi validasi multi-pusat, penilaian kegunaan klinis, dan dokumentasi kepatuhan regulasi.

---

## G. KENDALA DAN SOLUSI

### G.1 Kendala Teknis

**Keterbatasan Ukuran Dataset**

Kendala utama adalah ukuran dataset yang terbatas antara 200 hingga 800 gambar per dataset. Jaringan dalam secara ideal memerlukan ribuan sampel per kelas. Kelas minoritas dengan kurang dari 10 sampel sangat terpengaruh.

Solusi yang diterapkan meliputi augmentasi yang aman untuk medis dengan perkalian 4,4 kali untuk deteksi dan 3,5 kali untuk klasifikasi. Transfer learning dari bobot terlatih ImageNet digunakan untuk inisialisasi model. Focal Loss diterapkan untuk menangani ketidakseimbangan kelas. Pengambilan sampel berbobot dengan oversampling 3 kali lipat untuk kelas minoritas diterapkan.

Dampak solusi menunjukkan akurasi klasifikasi meningkat dari sekitar 75 persen tanpa augmentasi menjadi 86 hingga 99 persen. Skor F1 kelas minoritas meningkat dari sekitar 40 persen menjadi 44 hingga 100 persen. Stabilitas pelatihan meningkat secara signifikan.

**Ketidakseimbangan Kelas Ekstrem**

Kendala kedua adalah rasio ketidakseimbangan hingga 54 banding 1 pada dataset MP-IDB Stages dimana Ring memiliki 259 sampel sedangkan Gametocyte hanya 5 sampel. Fungsi kerugian cross-entropy standar bias terhadap kelas mayoritas. Kelas minoritas hanya mencapai recall di bawah 50 persen pada eksperimen awal.

Solusi yang diterapkan meliputi Focal Loss dengan parameter alpha 0,25 dan gamma 2,0 yang terbukti optimal untuk citra medis. Pengambilan sampel acak berbobot dengan batch seimbang kelas diterapkan. Oversampling 3 banding 1 untuk kelas minoritas digunakan. Balanced accuracy digunakan sebagai metrik utama bukan hanya akurasi.

Dampak solusi menunjukkan recall kelas minoritas meningkat dari sekitar 30 persen menjadi 57 hingga 100 persen. Balanced accuracy meningkat 15 hingga 20 poin persentase. Sistem mencapai skor F1 yang dapat diterima secara klinis di atas 75 persen pada sebagian besar kelas minoritas.

**Keterbatasan Sumber Daya Komputasi**

Kendala ketiga adalah pelatihan 72 kombinasi model memerlukan sekitar 120 jam GPU. Satu GPU RTX 4090 dapat menjadi bottleneck untuk eksperimen skala besar. Penyetelan hiperparameter akan memerlukan 10 kali lipat lebih banyak komputasi.

Solusi yang diterapkan meliputi pelatihan presisi campuran menggunakan AMP untuk percepatan 2 kali lipat. Benchmark cuDNN dan pemuatan data teroptimasi dengan 4 worker digunakan. Format memori channels-last memberikan percepatan 20 hingga 35 persen. Penghentian dini untuk menghindari epoch yang tidak perlu diterapkan. Strategi pelatihan berurutan dengan prioritas pada model menjanjikan digunakan.

Dampak solusi menunjukkan total waktu pelatihan berkurang dari estimasi 200 jam menjadi 80 jam. Penggunaan memori GPU dioptimalkan dengan puncak 18,5 GB dibanding 22 GB baseline. Suite eksperimen lengkap dapat diselesaikan dalam waktu yang wajar.

### G.2 Kendala Non-Teknis

**Akses Validasi Eksternal**

Kendala pertama adalah mendapatkan akses ke sampel klinis dari lapangan memerlukan kolaborasi formal. Kemitraan dengan rumah sakit atau klinik memerlukan persetujuan komite etik dengan proses 3 hingga 6 bulan. Prosedur pembersihan etika dan anonimisasi harus ditetapkan. Rumah sakit berbeda memiliki kebijakan berbagi data yang berbeda.

Solusi yang diterapkan meliputi inisiasi percakapan dengan beberapa mitra klinis potensial. Protokol penelitian komprehensif telah disiapkan untuk pengajuan komite etik. Prosedur anonimisasi yang sesuai dengan regulasi kesehatan telah dikembangkan. Studi validasi fase kedua direncanakan dengan target triwulan kedua 2026.

Status saat ini menunjukkan fase pertama berupa bukti konsep pada dataset publik telah selesai. Fase kedua berupa validasi klinis direncanakan untuk tahun 2026.

**Pertimbangan Regulasi**

Kendala kedua adalah deployment klinis sebagai alat bantu diagnostik memerlukan persetujuan regulasi dari badan seperti FDA. Regulasi perangkat medis memerlukan studi validasi ekstensif. Uji coba multi-pusat dengan ribuan sampel klinis diperlukan. Demonstrasi non-inferioritas versus mikroskopi standar emas diperlukan.

Solusi strategi meliputi positioning penelitian saat ini sebagai alat pendukung keputusan bukan diagnostik otonom. Dokumentasi disiapkan dengan mempertimbangkan kepatuhan regulasi. Validasi eksternal fase kedua dirancang untuk memenuhi persyaratan regulasi. Konsultasi jalur regulasi direncanakan dengan target tahun 2027.

Status saat ini menunjukkan bukti konsep selesai dan mendemonstrasikan kelayakan teknis. Perencanaan jalur regulasi telah dimulai. Studi validasi klinis dirancang dengan persyaratan regulasi sebagai pertimbangan.

**Pergeseran Domain dan Generalisasi**

Kendala ketiga adalah semua dataset berasal dari pengaturan laboratorium terkontrol dengan protokol pewarnaan Giemsa standar dan kondisi pencitraan konsisten pada perbesaran 1000 kali. Deployment dunia nyata akan menghadapi variasi kualitas pewarnaan, jenis mikroskop, dan kondisi pencitraan yang berbeda.

Solusi strategi meliputi validasi antar dataset yang menunjukkan kemampuan generalisasi. Pengumpulan sampel lapangan beragam dengan variasi pewarnaan dan mikroskop direncanakan untuk fase kedua. Teknik adaptasi domain akan dieksplorasi. Pipeline pembelajaran berkelanjutan untuk sistem yang dideploy akan dikembangkan.

Tantangan yang diharapkan adalah penurunan performa 10 hingga 25 persen pada sampel lapangan. Mitigasi melalui adaptasi domain dan fine-tuning pada data lokal akan diterapkan.

---

## H. RENCANA TINDAK LANJUT

### H.1 Fase Jangka Pendek (3 hingga 6 Bulan)

**Tindak Lanjut Publikasi Jurnal**

Artikel telah diterima di jurnal KINETIK. Tindakan yang akan dilakukan meliputi memantau jadwal publikasi, menyiapkan versi final kamera dengan revisi, menyelesaikan transfer hak cipta dan perjanjian penulis, menunggu publikasi online yang diharapkan pada triwulan pertama 2026, dan melacak sitasi serta dampak.

**Analisis Tambahan dan Studi Ablasi**

Eksperimen yang direncanakan meliputi kurva ROC dan kurva presisi-recall untuk penilaian performa lengkap. Studi ablasi untuk mengukur dampak Focal Loss versus cross-entropy versus class-balanced loss. Analisis sensitivitas hiperparameter untuk parameter alpha dan gamma Focal Loss. Evaluasi ensemble model dengan strategi voting versus stacking. Pengujian signifikansi statistik menggunakan uji-t berpasangan dan uji McNemar.

Output yang diharapkan meliputi materi suplemen untuk artikel jurnal, laporan teknis yang mendokumentasikan temuan tambahan, serta potensi artikel jurnal atau konferensi tambahan.

**Pengajuan Hak Kekayaan Intelektual**

Target pengajuan hak cipta software adalah triwulan pertama 2026. Dokumentasi yang diperlukan meliputi dokumentasi kode sumber lengkap, diagram arsitektur sistem, pernyataan kebaruan inovasi berupa Arsitektur Option A, spesifikasi teknis dan panduan deployment, serta manual pengguna dan prosedur operasi.

Jadwal pelaksanaan meliputi bulan pertama hingga kedua untuk persiapan dokumentasi, bulan ketiga untuk pengajuan aplikasi, dan bulan keempat hingga keenam untuk proses tinjauan.

### H.2 Fase Jangka Menengah (6 hingga 12 Bulan)

**Studi Validasi Eksternal Fase Kedua**

Tujuan fase kedua adalah memvalidasi performa sistem pada 500 sampel klinis yang beragam. Desain studi meliputi pengumpulan sampel dengan kemitraan bersama 2 hingga 3 rumah sakit atau klinik. Persyaratan keragaman meliputi beberapa protokol pewarnaan seperti Giemsa dan Field's stain, mikroskop berbeda dari berbagai merek dan perbesaran, tingkat keahlian teknisi beragam, tingkat parasitemia bervariasi dari rendah hingga tinggi, dan semua 4 spesies Plasmodium terwakili.

Metrik evaluasi meliputi akurasi deteksi versus mikroskopi ahli sebagai standar emas, akurasi klasifikasi dengan konsensus ahli dari 2 hingga 3 orang, analisis penghematan waktu antara sistem dan manual, perbaikan reliabilitas antar pengamat, dan penilaian efektivitas biaya.

Tantangan yang diharapkan meliputi jadwal persetujuan komite etik selama 3 hingga 6 bulan, logistik pengumpulan data, ketersediaan mikroskopi ahli, dan konsistensi kontrol kualitas.

Hasil yang diharapkan meliputi naskah validasi eksternal dengan target jurnal kuartil kedua, laporan kegunaan klinis, dan perbandingan performa dengan metode yang ada.

**Teknik Pembelajaran Lanjutan**

Tujuannya adalah meningkatkan skor F1 pada kelas ultra-minoritas dengan kurang dari 10 sampel dari saat ini 44 hingga 80 persen menjadi target di atas 85 persen.

Pendekatan pertama adalah generasi data sintetik menggunakan StyleGAN2, StyleGAN3, atau model difusi untuk menghasilkan sampel kelas minoritas realistis dengan validasi tinjauan ahli untuk memastikan realisme klinis. Target adalah menghasilkan 50 hingga 100 sampel sintetik per kelas minoritas.

Pendekatan kedua adalah few-shot learning menggunakan prototypical networks untuk mempelajari prototipe kelas dengan sampel minimal, matching networks dengan pelatihan episodik untuk adaptasi cepat, atau meta-learning menggunakan MAML. Peningkatan yang diharapkan adalah 10 hingga 15 poin persentase skor F1.

Pendekatan ketiga adalah active learning dengan uncertainty sampling untuk memprioritaskan sampel informatif untuk anotasi atau query-by-committee dengan disagreement ensemble untuk pemilihan sampel. Target adalah mengurangi kebutuhan anotasi sebesar 40 hingga 50 persen.

Pendekatan keempat adalah metode ensemble dengan voting ensemble menggabungkan 3 hingga 5 model terbaik atau stacked generalization dengan meta-learner menggabungkan model basis. Peningkatan yang diharapkan adalah 3 hingga 5 poin persentase akurasi.

Jadwal pelaksanaan meliputi bulan keenam hingga kesembilan untuk implementasi dan evaluasi teknik, serta bulan kesepuluh hingga kedua belas untuk integrasi ke pipeline dan validasi.

**Integrasi Fitur Penjelasan**

Tujuannya adalah menyediakan penjelasan visual untuk meningkatkan kepercayaan klinisi dan memungkinkan deteksi kesalahan.

Pendekatan pertama adalah Grad-CAM untuk memvisualisasikan daerah gambar yang paling berkontribusi pada klasifikasi, menyoroti fitur parasit yang menjadi fokus model, dan memungkinkan klinisi memverifikasi penalaran sesuai dengan pengetahuan medis.

Pendekatan kedua adalah integrasi Segment Anything untuk segmentasi parasit presisi dan visualisasi batas, serta membandingkan perhatian model dengan batas parasit sebenarnya untuk kontrol kualitas akurasi deteksi.

Pendekatan ketiga adalah visualisasi mekanisme attention untuk menunjukkan bobot perhatian model di seluruh daerah gambar, mengidentifikasi apakah model fokus pada fitur morfologi yang benar, dan mendeteksi potensi artefak atau korelasi palsu.

Jadwal implementasi meliputi bulan keenam hingga kedelapan untuk integrasi Grad-CAM dan SAM, bulan kesembilan hingga kesepuluh untuk pengembangan antarmuka pengguna, dan bulan kesebelas hingga kedua belas untuk validasi klinis dengan ahli.

Dampak yang diharapkan meliputi peningkatan kepercayaan klinisi dan kesediaan adopsi, deteksi dan koreksi kesalahan yang lebih mudah, serta alat pendidikan untuk pelatihan mikroskopis junior.

**Optimisasi Deployment**

Tujuannya adalah memungkinkan deployment pada perangkat edge dengan sumber daya terbatas.

Teknik pertama adalah kuantisasi model menggunakan kuantisasi INT8 untuk mengurangi ukuran model sebesar 75 persen dan inferensi 2 hingga 4 kali lebih cepat, atau kuantisasi rentang dinamis dengan kehilangan akurasi minimal kurang dari 2 persen. Platform target meliputi NVIDIA Jetson atau Raspberry Pi 4 dengan akselerator.

Teknik kedua adalah pruning jaringan saraf dengan structured pruning untuk menghilangkan seluruh channel atau filter, atau magnitude-based pruning untuk menghilangkan bobot kepentingan rendah. Target adalah pengurangan parameter 40 hingga 60 persen dengan kehilangan akurasi kurang dari 3 persen.

Teknik ketiga adalah destilasi pengetahuan dengan model guru berupa model besar berkinerja terbaik seperti ResNet101 dan model siswa berupa model kecil efisien seperti MobileNet atau EfficientNet-B0. Target adalah mencocokkan 95 persen performa guru dengan 80 persen parameter lebih sedikit.

Teknik keempat adalah pengujian perangkat edge pada NVIDIA Jetson Nano atau Xavier berbasis ARM dengan GPU edge, atau Raspberry Pi 4 dengan Coral TPU untuk opsi deployment berbiaya rendah. Target latensi adalah kurang dari 100 milidetik per gambar pada perangkat edge.

Jadwal pelaksanaan meliputi bulan keenam hingga kedelapan untuk implementasi teknik optimisasi, bulan kesembilan hingga kesepuluh untuk pengujian dan validasi perangkat edge, dan bulan kesebelas hingga kedua belas untuk persiapan paket deployment.

### H.3 Fase Jangka Panjang (12 hingga 24 Bulan)

**Studi Pilot Klinis**

Tujuannya adalah mengevaluasi performa sistem dalam alur kerja klinis dunia nyata. Desain studi meliputi durasi 6 hingga 12 bulan, lokasi di 2 hingga 3 rumah sakit atau klinik dengan pengaturan urban dan rural, ukuran sampel 1000 kasus klinis atau lebih, dan desain studi observasional prospektif.

Metrik evaluasi meliputi akurasi diagnostik antara sistem dan mikroskopi ahli sebagai standar emas, penghematan waktu berupa pengurangan waktu turnaround, dampak alur kerja berupa kemudahan integrasi dan kepuasan pengguna, reliabilitas antar pengamat berupa kesepakatan antara sistem, mikroskopis junior, dan senior, efektivitas biaya berupa biaya per diagnosis dan pemanfaatan sumber daya, serta hasil klinis berupa keputusan pengobatan dan hasil pasien.

Temuan yang diharapkan meliputi identifikasi tantangan deployment seperti konektivitas, listrik, dan pemeliharaan, persyaratan pelatihan pengguna dan kurva pembelajaran, pola kesalahan dalam kondisi dunia nyata, dan kebutuhan perbaikan untuk deployment produksi.

Hasil yang diharapkan meliputi naskah studi pilot klinis dengan target jurnal kuartil pertama, panduan deployment dan praktik terbaik, materi pelatihan untuk staf klinis, dan laporan analisis efektivitas biaya.

**Perencanaan Jalur Regulasi**

Tujuannya adalah mempersiapkan potensi deployment klinis sebagai perangkat medis. Strategi regulasi meliputi beberapa fase.

Fase pertama adalah penentuan klasifikasi selama bulan ke-12 hingga 15 dengan konsultasi bersama ahli regulasi, penentuan klasifikasi perangkat yang kemungkinan Kelas 2 dengan risiko sedang, identifikasi perangkat preseden untuk jalur 510k, dan persiapan dokumen strategi regulasi.

Fase kedua adalah pra-pengajuan selama bulan ke-15 hingga 18 dengan pertemuan pra-pengajuan bersama otoritas regulasi, diskusi persyaratan validasi klinis, klarifikasi ekspektasi validasi software, dan mendapatkan umpan balik tentang desain studi.

Fase ketiga adalah studi validasi selama bulan ke-18 hingga 24 dengan studi validasi multi-pusat sesuai persyaratan regulasi, demonstrasi non-inferioritas versus standar emas, analisis sensitivitas dan spesifisitas sesuai ambang batas regulasi, dan dokumentasi verifikasi dan validasi software.

Fase keempat adalah persiapan pengajuan setelah bulan ke-24 dengan kompilasi paket pengajuan 510k atau setara, data validasi klinis, dokumentasi software, dokumentasi sistem manajemen kualitas, dan file manajemen risiko sesuai ISO 14971.

Jadwal yang diharapkan meliputi tahun 2027 untuk konsultasi regulasi dan pra-pengajuan, tahun 2028 untuk studi validasi dan persiapan pengajuan, serta tahun 2029 untuk pengajuan dan tinjauan regulasi.

**Pipeline Pembelajaran Berkelanjutan**

Tujuannya adalah memungkinkan sistem yang dideploy untuk meningkat seiring waktu melalui data penggunaan dunia nyata. Desain sistem meliputi beberapa komponen.

Komponen pertama adalah infrastruktur pengumpulan data dengan mekanisme unggah aman untuk sampel klinis teranonimisasi, pengumpulan metadata berupa demografi, parasitemia, dan hasil pengobatan, anotasi ahli untuk kebenaran dasar, dan penyaringan kontrol kualitas.

Komponen kedua adalah integrasi active learning dengan identifikasi kasus menantang dimana sistem memiliki ketidakpastian tinggi, prioritasi sampel untuk tinjauan dan anotasi ahli, pelatihan ulang model iteratif dengan data baru, dan pemantauan performa serta versioning.

Komponen ketiga adalah federated learning opsional untuk memungkinkan pembelajaran multi-lokasi tanpa berbagi data terpusat, pembaruan model yang menjaga privasi, kepatuhan regulasi dengan regulasi data kesehatan seperti HIPAA dan GDPR.

Komponen keempat adalah kontrol versi dan deployment dengan sistem versioning model menggunakan pipeline MLOps, pengujian A/B untuk pembaruan model, mekanisme rollback jika performa menurun, dan pemantauan berkelanjutan model yang dideploy.

Manfaat yang diharapkan meliputi sistem beradaptasi dengan karakteristik populasi lokal, peningkatan performa seiring waktu dengan target plus 5 hingga 10 persen akurasi, deteksi dini varian parasit yang muncul, dan kontribusi komunitas ke database malaria global.

**Perluasan ke Aplikasi Terkait**

Potensi perluasan meliputi parasit darah lainnya seperti Trypanosoma penyebab sleeping sickness, Leishmania penyebab leishmaniasis, dan Babesia penyebab babesiosis. Deteksi multi-patogen berupa kombinasi deteksi malaria dan bakteri, sistem diagnosis diferensial, atau alat screening luas. Parasitemia kuantitatif berupa penghitungan parasit otomatis, perhitungan persentase parasitemia, dan aplikasi pemantauan pengobatan. Screening resistensi obat berupa penanda morfologi resistensi obat dan pemantauan gametosit untuk pemblokiran transmisi.

Jadwal pelaksanaan meliputi tahun 2027 hingga 2028 untuk studi kelayakan, tahun 2028 hingga 2029 untuk implementasi pilot, dan tahun 2029 ke depan untuk deployment penuh.

---

## I. KEBERLANJUTAN DAN DAMPAK

### I.1 Keberlanjutan Penelitian

**Keberlanjutan Pendanaan**

Status saat ini menunjukkan hibah BISMA telah dimanfaatkan sepenuhnya. Pengajuan untuk pendanaan lanjutan sedang dalam proses untuk triwulan pertama 2026.

Sumber pendanaan potensial meliputi hibah WHO dan TDR untuk penelitian malaria, hibah penelitian nasional sebagai lanjutan BISMA, kemitraan industri dengan perusahaan diagnostik, dan kolaborasi internasional dengan NIH atau Wellcome Trust.

**Keberlanjutan Teknis**

Infrastruktur meliputi basis kode terbuka di GitHub dengan lisensi MIT, dokumentasi lengkap yang memastikan kemudahan pemeliharaan, arsitektur modular yang memungkinkan pembaruan mudah, dan kontribusi komunitas yang didorong.

Rencana pemeliharaan meliputi pembaruan dependensi rutin, perbaikan bug dan peningkatan, perluasan dataset, dan pelatihan ulang model dengan data baru.

**Keberlanjutan Kolaborasi**

Kemitraan akademik meliputi kolaborasi penelitian berkelanjutan, aplikasi hibah bersama, proyek dan tesis mahasiswa, serta lokakarya transfer pengetahuan.

Kemitraan klinis meliputi perjanjian jangka panjang dengan rumah sakit, loop umpan balik berkelanjutan, studi penelitian bersama, dan program pelatihan untuk staf.

### I.2 Dampak Jangka Pendek (1 hingga 2 Tahun)

**Dampak Akademik**

Pencapaian meliputi 1 publikasi jurnal internasional di KINETIK yang telah diterima, 1 hak cipta software yang dalam proses, 1 presentasi konferensi internasional yang telah selesai, dan 2 hingga 3 artikel konferensi tambahan yang direncanakan.

**Dampak Pendidikan**

Kontribusi meliputi materi pelatihan untuk pembelajaran mendalam dalam citra medis, studi kasus untuk kursus visi komputer, proyek terbuka untuk pembelajaran mahasiswa, dan topik tesis sarjana dan pascasarjana.

**Dampak Teknis**

Pencapaian meliputi sistem bukti konsep yang mendemonstrasikan kelayakan, metodologi tervalidasi untuk dataset medis kecil dan tidak seimbang, hasil benchmark untuk perbandingan penelitian masa depan, dan alat terbuka untuk penggunaan komunitas.

### I.3 Dampak Jangka Menengah (3 hingga 5 Tahun)

**Dampak Klinis**

Target meliputi validasi eksternal dengan 500 sampel klinis atau lebih, deployment pilot klinis di 2 hingga 3 lokasi, demonstrasi penghematan waktu dan efektivitas biaya, dan peningkatan konsistensi diagnostik.

**Dampak Kesehatan Publik**

Kontribusi potensial meliputi alat screening untuk daerah endemis malaria, pengurangan waktu turnaround diagnostik kurang dari 1 menit versus 20 hingga 30 menit, peningkatan kapasitas diagnostik di pengaturan sumber daya terbatas, dan alat kontrol kualitas untuk pelatihan mikroskopi.

**Dampak Ekonomis**

Manfaat meliputi penghematan biaya dari pengurangan kebutuhan waktu ahli, inisiasi pengobatan lebih awal yang mengurangi komplikasi, potensi komersialisasi melalui lisensi, dan dampak ekonomi melalui pengurangan beban penyakit.

### I.4 Dampak Jangka Panjang (5 hingga 10 Tahun)

**Dampak Kesehatan Global**

Kontribusi meliputi berkontribusi ke tujuan eliminasi malaria WHO, solusi screening skalabel untuk daerah endemis, kualitas diagnostik terstandarisasi secara global, dan agregasi data untuk surveilans epidemiologi.

**Dampak Ilmiah**

Kemajuan meliputi memajukan metodologi untuk dataset citra medis kecil, berkontribusi pada praktik terbaik dalam penanganan ketidakseimbangan kelas, mendemonstrasikan deployment AI praktis di pengaturan sumber daya terbatas, dan prinsip sains terbuka yang mempromosikan reproduksibilitas.

**Dampak Teknologi**

Pengembangan meliputi deployment perangkat edge yang memungkinkan diagnostik point-of-care, integrasi dengan platform kesehatan digital, kontribusi pada infrastruktur telemedicine, dan fondasi untuk sistem deteksi multi-patogen.

**Dampak Sosial**

Manfaat meliputi peningkatan hasil kesehatan pada populasi kurang terlayani, pengurangan mortalitas dari diagnosis akurat tepat waktu, pengembangan ekonomi melalui tenaga kerja lebih sehat, dan peluang pendidikan untuk pekerja kesehatan.

---

## J. KESIMPULAN DAN REKOMENDASI

### J.1 Kesimpulan Utama

Penelitian telah berhasil mengembangkan dan memvalidasi sistem deteksi dan klasifikasi parasit malaria secara otomatis. Sistem menggunakan arsitektur hibrida dengan pendekatan klasifikasi bersama yang inovatif. Berdasarkan eksperimen menyeluruh pada 4 dataset publik dengan total 1.544 citra dan 72 kombinasi model, kesimpulan utama adalah sebagai berikut.

**Performa Deteksi Sangat Baik**

Model deteksi YOLO mencapai mAP@50 antara 91,86 hingga 96,61 persen pada semua dataset. Model YOLOv11 mencapai performa terbaik dalam recall dengan 95,88 persen pada IML dan 95,29 persen pada Species. Nilai recall tinggi sangat penting untuk meminimalkan parasit yang terlewat dalam pengaturan klinis. Ketiga varian YOLO konsisten mencapai di atas 91 persen mAP@50 pada semua dataset, mendemonstrasikan generalisasi yang baik. Latensi inferensi kurang dari 25 milidetik per gambar atau lebih dari 40 bingkai per detik pada RTX 4090 membuktikan kelayakan untuk deployment point-of-care waktu nyata.

**Klasifikasi dengan Efisiensi Parameter**

Akurasi klasifikasi berkisar antara 83,53 hingga 98,62 persen tergantung karakteristik dataset. Model EfficientNet dengan parameter lebih kecil (5,3 hingga 9,2 juta parameter) mencapai performa kompetitif atau superior dibanding ResNet dengan parameter jauh lebih besar (25,6 hingga 44,5 juta parameter) dengan waktu pelatihan 15 hingga 30 persen lebih cepat. Model terbaik untuk setiap dataset adalah EfficientNet-B2 untuk IML Lifecycle dengan 91,51 dan 91,96 persen, ResNet101 untuk MP-IDB Species dengan 98,62 dan 88,10 persen, dan EfficientNet-B0 untuk MD-2019 dengan 86,62 dan 85,51 persen. Metrik balanced accuracy mengungkap tantangan pada kelas minoritas dengan kesenjangan 10 hingga 18 persen antara akurasi dan balanced accuracy, mengindikasikan ruang untuk perbaikan.

**Penanganan Ketidakseimbangan Kelas Efektif**

Focal Loss dengan parameter alpha 0,25 dan gamma 2,0 dikombinasikan dengan pengambilan sampel berbobot berhasil meningkatkan recall kelas minoritas dari sekitar 30 persen baseline menjadi 57 hingga 100 persen. Kelas ultra-minoritas dengan kurang dari 10 sampel mencapai skor F1 antara 0,44 hingga 1,00 tergantung kekhasan morfologi. Kasus terbaik adalah Schizont dengan 4 sampel mencapai skor F1 sempurna 1,00 karena morfologi sangat khas. Kasus menantang adalah Trophozoite dengan 14 sampel mencapai skor F1 sebesar 0,58 karena morfologi bervariasi tumpang tindih.

**Manfaat Arsitektur Option A Tervalidasi**

Pengurangan penyimpanan sekitar 70 persen tercapai karena citra terpotong dibuat satu kali, bukan per model YOLO. Pengurangan waktu pelatihan sekitar 60 persen tercapai karena model klasifikasi dilatih satu kali untuk semua metode deteksi. Evaluasi konsisten memungkinkan perbandingan adil antar metode deteksi dengan model klasifikasi yang sama. Pemisahan bersih memungkinkan optimisasi independen tahap deteksi dan klasifikasi.

**Generalisasi Antar Dataset Terdemonstrasikan**

Sistem divalidasi pada 4 dataset beragam dengan karakteristik berbeda. Dataset IML Lifecycle dengan 313 gambar dan tahapan siklus hidup, MP-IDB Species dengan 209 gambar dan 4 spesies Plasmodium, MP-IDB Stages dengan 209 gambar dan ketidakseimbangan ekstrem rasio 54 banding 1, serta MD-2019 Stages dengan dataset terbesar 813 gambar dan 3 tahapan. Performa konsisten di semua dataset memvalidasi ketahanan metodologi.

### J.2 Kontribusi Ilmiah

**Kontribusi Metodologi**

Penelitian memberikan kontribusi arsitektur klasifikasi bersama yang novel untuk mengurangi beban komputasi secara signifikan. Analisis sistematis hubungan ukuran model dan performa pada dataset citra medis kecil memberikan wawasan penting. Strategi efektif untuk menangani ketidakseimbangan kelas ekstrem dengan rasio hingga 54 banding 1 dalam citra medis dikembangkan.

**Kontribusi Empiris**

Hasil benchmark lengkap untuk 72 kombinasi model yaitu 3 YOLO dikali 6 CNN dikali 4 dataset telah disediakan. Validasi antar dataset mendemonstrasikan kemampuan generalisasi sistem. Analisis reproduksibilitas menunjukkan variabilitas rendah dengan koefisien variasi di bawah 5 persen.

**Kontribusi Praktis**

Bukti konsep mendemonstrasikan kelayakan untuk deployment point-of-care. Implementasi terbuka memungkinkan reproduksibilitas dan kontribusi komunitas. Dokumentasi lengkap memfasilitasi adopsi dan perluasan penelitian.

### J.3 Keterbatasan Penelitian

**Keterbatasan Ukuran Dataset**

Meskipun menggunakan 4 dataset, total ukuran sampel 1.544 gambar tetap terbatas. Jaringan dalam secara ideal memerlukan ribuan sampel per kelas. Kelas minoritas dengan kurang dari 10 sampel sangat terpengaruh dengan performa bervariasi.

Mitigasi yang telah diterapkan meliputi strategi augmentasi, transfer learning, dan ekspansi dataset yang direncanakan pada fase kedua.

**Keterbatasan Pergeseran Domain**

Semua dataset berasal dari pengaturan laboratorium terkontrol dengan pewarnaan standar pada perbesaran 1000 kali. Deployment dunia nyata akan menghadapi variasi kualitas pewarnaan, mikroskop, dan kondisi pencitraan. Pengujian antar dataset menunjukkan penurunan performa 10 hingga 25 persen.

Mitigasi meliputi validasi eksternal fase kedua dengan sampel lapangan beragam dan teknik adaptasi domain.

**Keterbatasan Performa Kelas Minoritas**

Kelas ultra-minoritas dengan kurang dari 10 sampel menunjukkan performa bervariasi dengan skor F1 antara 0,44 hingga 1,00. Beberapa kelas gagal mencapai ambang batas yang dapat diterima secara klinis dengan skor F1 di bawah 0,75.

Mitigasi meliputi teknik pembelajaran lanjutan yang direncanakan berupa few-shot learning dan generasi data sintetik.

**Keterbatasan Penjelasan**

Sistem saat ini kurang penjelasan visual untuk prediksi. Kepercayaan klinisi terbatas tanpa memahami mengapa sistem membuat keputusan tertentu.

Mitigasi berupa integrasi Grad-CAM dan SAM direncanakan pada fase kedua.

### J.4 Rekomendasi

**Rekomendasi untuk Peneliti**

Prioritas pertama adalah perluasan dataset dengan target mengumpulkan 1000 gambar atau lebih per dataset dengan distribusi kelas seimbang. Fokus pada peningkatan representasi kelas minoritas dengan target minimum 50 sampel per kelas. Metode meliputi kolaborasi dengan beberapa rumah sakit dan standardisasi protokol anotasi.

Prioritas kedua adalah teknik pembelajaran lanjutan dengan implementasi few-shot learning untuk kelas ultra-minoritas. Eksplorasi model generatif seperti StyleGAN2 atau model difusi untuk augmentasi data sintetik. Investigasi pendekatan meta-learning untuk adaptasi cepat pada kelas baru.

Prioritas ketiga adalah integrasi penjelasan dengan prioritas pada Grad-CAM dan visualisasi attention. Pengembangan antarmuka ramah klinisi. Pelaksanaan studi pengguna untuk pengujian kegunaan.

Prioritas keempat adalah validasi antar domain dengan pengujian pada kondisi pencitraan beragam berupa pewarnaan, perbesaran, dan peralatan. Pengembangan teknik adaptasi domain. Pembuatan fitur domain-agnostik.

**Rekomendasi untuk Praktisi**

Prioritas pertama adalah validasi klinis dengan pelaksanaan studi prospektif multi-pusat dengan 500 sampel klinis atau lebih. Perbandingan performa sistem dengan konsensus ahli dari 2 hingga 3 mikroskopis. Evaluasi integrasi dalam alur kerja klinis yang ada.

Prioritas kedua adalah pertimbangan deployment dengan pengujian pada perangkat edge seperti Jetson atau Raspberry Pi dengan akselerator. Pengembangan antarmuka ramah pengguna untuk staf non-teknis. Pembuatan materi pelatihan lengkap dan prosedur operasi standar.

Prioritas ketiga adalah kontrol kualitas dengan implementasi pemantauan berkelanjutan performa model yang dideploy. Penetapan ambang batas untuk peringatan sistem pada prediksi kepercayaan rendah. Pengembangan protokol untuk tinjauan ahli kasus borderline.

**Rekomendasi untuk Pembuat Kebijakan**

Prioritas pertama adalah kerangka regulasi dengan pengembangan panduan jelas untuk alat diagnostik berbasis AI dalam malaria. Penetapan standar validasi berupa ukuran sampel minimum dan ambang batas performa. Pembuatan jalur cepat untuk teknologi menjanjikan.

Prioritas kedua adalah alokasi sumber daya dengan investasi dalam infrastruktur komputasi berupa GPU untuk daerah endemis malaria. Dukungan upaya pengumpulan data dengan protokol anotasi terstandarisasi. Pendanaan studi validasi multi-pusat.

Prioritas ketiga adalah pembangunan kapasitas dengan pelatihan pekerja kesehatan dalam diagnostik berbantuan AI. Pembentukan program jaminan kualitas. Promosi sains terbuka dan berbagi data untuk mempercepat kemajuan penelitian.

---

## K. DAFTAR PUSTAKA

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

## L. LAMPIRAN

### L.1 Tabel Pendukung

Tabel lengkap tersedia di folder luaran/laporan_akhir/tables/ dengan rincian sebagai berikut:

- dataset_statistics_all.csv: Statistik lengkap untuk keempat dataset
- detection_performance_all_datasets.csv: Performa deteksi YOLO pada semua dataset
- classification_focal_loss_all_datasets.csv: Performa klasifikasi CNN dengan Focal Loss

### L.2 Dokumentasi Teknis

Dokumentasi lengkap tersedia di repository GitHub dengan alamat https://github.com/akhiyarwaladi/hello_world. Dokumen meliputi CLAUDE.md untuk panduan cepat, SETUP_GUIDE.md untuk panduan pengaturan, TROUBLESHOOTING.md untuk pemecahan masalah, dan ARCHITECTURE.md untuk arsitektur detail.

### L.3 Data Eksperimen

Hasil eksperimen lengkap tersimpan di folder results/optA_20251207_233941/ dengan struktur terorganisir per dataset. Format data meliputi CSV, Excel, dan JSON untuk kemudahan analisis dan visualisasi. Ukuran data sekitar 2,5 GB untuk hasil mentah dengan 4 dataset dikali 72 kombinasi model.

### L.4 Publikasi dan Luaran

Publikasi yang tersedia meliputi artikel jurnal KINETIK yang diterima pada Desember 2025 dengan alamat http://kinetik.umm.ac.id/index.php/kinetik/authorDashboard/submission/2558. Bukti PDF tersimpan di file screencapture-kinetik-*.pdf. Presentasi konferensi dengan sertifikat tersimpan di file certificate_presenter_malaria.pdf dan surat penerimaan di file loa_proceeding_malaria.pdf.

### L.5 Software dan Kode

Implementasi terbuka tersedia di platform GitHub dengan lisensi MIT License. Konten meliputi kode sumber lengkap dalam Python 3.13 dengan PyTorch 2.8.0, bobot model terlatih untuk total 42 model, skrip persiapan data, alat evaluasi, utilitas visualisasi, dan dokumentasi lengkap.

### L.6 Kontak Informasi

Untuk pertanyaan dan kolaborasi, silakan hubungi peneliti utama dengan nama [Nama Peneliti], email [Email], institusi [Nama Institusi], dan GitHub https://github.com/akhiyarwaladi. Untuk melaporkan masalah atau pertanyaan teknis, gunakan GitHub Issues di https://github.com/akhiyarwaladi/hello_world/issues.

---

**LAPORAN AKHIR PENELITIAN BISMA**

**Kerangka Kerja Multi-Model Hibrida untuk Deteksi dan Klasifikasi Malaria Otomatis**

**Periode**: Januari 2025 - Desember 2025

**Status**: Selesai dengan Sukses

**Tanggal Penyusunan**: 9 Desember 2025

**Versi Dokumen**: 2.0 Final (Revisi Narasi)
