# 📁 STRUKTUR FOLDER HASIL PIPELINE - PENJELASAN LENGKAP

## 🎯 KONSEP DASAR
Pipeline menggunakan 3 folder berdasarkan **NAMA EXPERIMENT**:

### 1️⃣ Folder `training/`
- **Kapan**: Nama experiment biasa (tanpa kata kunci khusus)
- **Contoh nama**: `my_experiment`, `yolo8_run1`, `pipeline_test123`
- **Isi**: Hasil training untuk development/eksperimen

### 2️⃣ Folder `validation/`
- **Kapan**: Nama mengandung kata "test" atau "validation"
- **Contoh nama**: `my_TEST_run`, `validation_check`, `test_pipeline`
- **Isi**: Hasil validasi atau test mode

### 3️⃣ Folder `production/`
- **Kapan**: Nama mengandung kata "production" atau "final"
- **Contoh nama**: `production_v1`, `final_model`, `production_ready`
- **Isi**: Model final siap deploy

## 📊 STRUKTUR LENGKAP

```
results/
└── current_experiments/
    ├── training/          # Eksperimen biasa
    │   ├── detection/
    │   │   ├── yolov8_detection/
    │   │   │   └── experiment_name_yolo8_det/
    │   │   │       ├── weights/best.pt
    │   │   │       └── results.csv
    │   │   ├── yolov11_detection/
    │   │   ├── yolo12_detection/
    │   │   └── rtdetr_detection/
    │   ├── classification/
    │   │   ├── yolov8_classification/
    │   │   └── yolov11_classification/
    │   └── pytorch_classification/
    │       ├── resnet18/
    │       ├── efficientnet_b0/
    │       ├── densenet121/
    │       └── mobilenet_v2/
    │
    ├── validation/        # Test mode / validasi
    │   ├── detection/     # Sama struktur dengan training
    │   └── classification/
    │
    └── production/        # Model final
        ├── detection/     # Sama struktur dengan training
        └── classification/
```

## 🔄 FLOW PIPELINE

### NORMAL MODE (tanpa --test-mode):
```
python run_multiple_models_pipeline.py --epochs-det 30 --epochs-cls 20
```

**Nama yang dibuat**: `pipeline_20250920_123456_yolo8_det`
- ✅ Tidak ada kata "test" → masuk folder `training/`

**Hasil**:
- Detection: `results/current_experiments/training/detection/.../`
- Classification: `results/current_experiments/training/classification/.../`

### TEST MODE (dengan --test-mode):
```
python run_multiple_models_pipeline.py --test-mode --epochs-det 2 --epochs-cls 2
```

**Nama yang dibuat**: `pipeline_20250920_123456_TEST_yolo8_det`
- ⚠️ Ada kata "TEST" → masuk folder `validation/`

**Hasil**:
- Detection: `results/current_experiments/validation/detection/.../`
- Classification: `results/current_experiments/validation/classification/.../`

## ❓ KENAPA TERPISAH-PISAH?

### Alasan Design:
1. **Pemisahan Environment**: Development vs Testing vs Production
2. **Tracking Eksperimen**: Mudah lihat mana yang test, mana yang serius
3. **Keamanan**: Production model terpisah dari eksperimen

### Masalah yang Timbul:
1. **Bingung cari hasil**: Harus ingat nama experiment
2. **Path berbeda**: Pipeline harus cek multiple folders
3. **Inkonsistensi**: PyTorch models selalu di `training/`

## 🛠️ SOLUSI YANG SUDAH DILAKUKAN

### 1. Pipeline Diperbaiki
File `run_multiple_models_pipeline.py` sudah diperbaiki untuk:
- Otomatis cek folder yang tepat berdasarkan nama
- Handle perbedaan YOLO vs PyTorch models

### 2. File Clean Version
File `run_multiple_models_pipeline_clean.py` yang baru:
- Lebih sederhana dan konsisten
- Fungsi `get_experiment_folder()` untuk tentukan folder
- Fungsi `get_paths()` untuk semua path

## 💡 REKOMENDASI PENGGUNAAN

### Untuk Testing Cepat:
```bash
python run_multiple_models_pipeline_clean.py \
    --test-mode \
    --epochs-det 2 \
    --epochs-cls 2
```
→ Hasil di `validation/`

### Untuk Training Serius:
```bash
python run_multiple_models_pipeline_clean.py \
    --experiment-name "my_research" \
    --epochs-det 50 \
    --epochs-cls 30
```
→ Hasil di `training/`

### Untuk Model Final:
```bash
python run_multiple_models_pipeline_clean.py \
    --experiment-name "production_v1" \
    --epochs-det 100 \
    --epochs-cls 50
```
→ Hasil di `production/`

## 📝 TIPS

1. **Gunakan nama yang jelas**:
   - Testing: `test_xxx` atau gunakan `--test-mode`
   - Development: nama biasa tanpa kata kunci
   - Final: `production_xxx` atau `final_xxx`

2. **Cari hasil dengan pattern**:
   ```bash
   # Cari semua hasil YOLO8
   ls results/current_experiments/*/detection/yolov8_detection/
   ```

3. **Gunakan clean version**:
   File `run_multiple_models_pipeline_clean.py` lebih rapi dan mudah dipahami

## ✅ KESIMPULAN

Struktur folder memang terpisah berdasarkan tujuan experiment:
- `training/` → Development
- `validation/` → Testing
- `production/` → Final models

Pipeline sudah diperbaiki untuk handle semua kasus ini secara otomatis.