# 🔍 PENJELASAN FLOW PIPELINE DAN MASALAH FOLDER

## 🚨 MASALAH UTAMA
Anda menemukan bahwa weight detection model tidak ditemukan di folder `training`, tetapi ada di folder `validation`. Ini terjadi karena:

## 📁 LOGIKA FOLDER BERDASARKAN NAMA EXPERIMENT

Script training detection (`scripts/training/07_train_yolo_detection.py`) memiliki logika:

```python
# Baris 48-53 dari 07_train_yolo_detection.py
if "production" in args.name.lower() or "final" in args.name.lower():
    experiment_type = "production"
elif "validation" in args.name.lower() or "test" in args.name.lower():
    experiment_type = "validation"  # <-- INILAH PENYEBABNYA!
else:
    experiment_type = "training"
```

### Ketika Anda menjalankan dengan `--test-mode`:
1. Pipeline menambahkan sufiks `_TEST` ke nama experiment
2. Nama jadi: `multi_pipeline_20250920_181505_TEST_yolo8_det`
3. Karena ada kata "TEST", script training menyimpan ke `validation/` bukan `training/`

## 🔄 FLOW LENGKAP PIPELINE

### STAGE 1: Detection Training
```
Command: python pipeline.py train yolov8_detection --name xxx_TEST_yyy --epochs 2
              ↓
Script melihat "TEST" dalam nama
              ↓
Menyimpan ke: results/current_experiments/validation/detection/...
```

### STAGE 2: Crop Generation
```
Pipeline mencari weights di:
- Test Mode: validation/detection/.../weights/best.pt ✅
- Normal Mode: training/detection/.../weights/best.pt
              ↓
Gunakan weights untuk generate crops dengan confidence:
- Test Mode: 0.05 (rendah, untuk garantikan crops)
- Normal Mode: 0.25 (normal)
```

### STAGE 3: Classification Training
```
Gunakan crops yang sudah digenerate untuk training classification
```

## 📊 KENAPA ADA 4 KELAS DI DETECTION?

Detection model ditraining untuk mendeteksi 4 kelas parasit malaria:
- P_falciparum
- P_malariae
- P_ovale
- P_vivax

Output yang Anda lihat setelah detection training:
```
all         28         28    0.00381          1      0.253      0.194
P_falciparum 15         15    0.00944          1       0.57      0.448
P_malariae   4          4    0.00154          1     0.0758     0.0637
P_ovale      5          5    0.00234          1      0.218      0.196
P_vivax      4          4    0.00193          1      0.147     0.0661
```

Ini adalah hasil validasi detection model terhadap 4 kelas parasit, BUKAN classification model.

## 🛠️ PERBAIKAN YANG SUDAH DILAKUKAN

File `run_multiple_models_pipeline.py` sudah diperbaiki untuk:

1. **Cek weights di folder yang benar berdasarkan test mode:**
```python
if args.test_mode:
    model_path = f"results/.../validation/detection/..."
else:
    model_path = f"results/.../training/detection/..."
```

2. **Cari results CSV di folder yang benar:**
```python
if "TEST" in det_exp_name:
    det_results_path = f"results/.../validation/detection/..."
else:
    det_results_path = f"results/.../training/detection/..."
```

## ✅ KESIMPULAN

1. **Folder weights tergantung nama experiment:**
   - Nama mengandung "test"/"validation" → folder `validation/`
   - Nama mengandung "production"/"final" → folder `production/`
   - Lainnya → folder `training/`

2. **Test mode bekerja dengan baik:**
   - Confidence threshold rendah (0.05) untuk garantikan crops
   - Weights disimpan di folder validation (expected behavior)
   - Pipeline sudah diperbaiki untuk mencari di folder yang benar

3. **4 Kelas yang Anda lihat:**
   - Itu adalah output dari detection model validation
   - Classification model terpisah dan ditraining setelah crop generation