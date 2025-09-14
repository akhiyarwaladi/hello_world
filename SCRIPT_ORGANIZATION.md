# Script Organization - Malaria Detection Pipeline

## Perubahan Organisasi Script

Sebelumnya script menggunakan nomor urut (01_, 02_, dst.) yang tidak mencerminkan urutan eksekusi sesungguhnya. Sekarang sudah dirapikan dengan nama yang lebih jelas sesuai fungsi.

## Struktur Script Baru

### Pipeline Utama
- `malaria_pipeline.py` - Pipeline sederhana dengan nama script yang jelas
- `pipeline_enhanced.py` - Pipeline lengkap dengan fitur advanced (sudah ada & berfungsi)

### Script Proses Data
- `download_datasets.py` (dari 01_download_datasets.py)
- `preprocess_images.py` (dari 02_preprocess_data.py)
- `integrate_datasets.py` (dari 03_integrate_datasets.py)
- `convert_to_yolo.py` (dari 04_convert_to_yolo.py)
- `augment_data.py` (dari 05_augment_data.py)
- `split_dataset.py` (dari 06_split_dataset.py)

### Script Training
- `train_yolo_detection.py` (dari 10_train_yolo_detection.py) ✅ SUDAH DIRENAME
- `train_classification_crops.py` (dari 11_train_classification_crops.py) - belum bisa rename karena masih berjalan
- `train_yolo11_detection.py` (dari 12_train_yolo11_detection.py)
- `train_rtdetr_detection.py` (dari 13_train_rtdetr_detection.py)

### Script Utilitas
- `crop_detections.py` (dari 10_crop_detections.py)
- `compare_model_performance.py` (dari 14_compare_models_performance.py)
- `parse_mpid_annotations.py` (dari 08_parse_mpid_detection.py)

## Status Reorganisasi

### ✅ Selesai
1. ✅ Satu script training berhasil direname: `train_yolo_detection.py`
2. ✅ Pipeline enhanced berjalan dengan baik
3. ✅ Copy script dengan nama baru di direktori root

### 🔄 Sedang Berlangsung
- Background processes masih menggunakan script lama
- Pipeline enhanced sedang berjalan dan berhasil menyelesaikan beberapa tahap

### ⏳ Menunggu
- Rename script lainnya setelah background processes selesai
- Cleanup script lama dengan nomor urut

## Manfaat Reorganisasi

1. **Nama yang Jelas**: Script name langsung menjelaskan fungsinya
2. **Tidak Ada Kebingungan Urutan**: Tidak ada nomor yang menyesatkan
3. **Mudah Dipahami**: Developer baru bisa langsung paham struktur
4. **Organisasi yang Baik**: Script dikelompokkan berdasarkan fungsi

## Penggunaan

### Pipeline Lengkap (Recommended)
```bash
python pipeline_enhanced.py --continue  # melanjutkan dari checkpoint terakhir
python pipeline_enhanced.py --restart   # mulai dari awal
python pipeline_enhanced.py --repair    # repair mode untuk fix errors
```

### Pipeline Sederhana
```bash
python malaria_pipeline.py --stage 1    # jalankan tahap tertentu
python malaria_pipeline.py              # jalankan semua tahap
```

### Script Individual
```bash
python train_yolo_detection.py --data data/detection_fixed/dataset.yaml
python train_classification_crops.py --data data/classification_crops
python compare_model_performance.py --results-dir results
```

## Hasil Testing

✅ Pipeline enhanced berhasil berjalan dan menyelesaikan:
- Environment & Dependencies Check
- Dataset Download (1649 images)
- Detection Dataset Preparation (sedang berjalan)

Pipeline dengan nama script yang baru terbukti lebih mudah dimengerti dan digunakan.