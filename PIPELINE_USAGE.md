# Pipeline Manager Usage Guide

## 🎯 Quick Start

**Cara termudah (recommended):**
```bash
python run_pipeline.py
```

## 🔧 Advanced Usage

**Full pipeline manager dengan berbagai options:**

```bash
# Lihat status pipeline
python pipeline_manager.py --status

# Jalankan pipeline (otomatis skip yang sudah selesai)
python pipeline_manager.py

# Restart dari awal (hapus semua checkpoint)
python pipeline_manager.py --force-restart

# Mulai dari stage tertentu
python pipeline_manager.py --start-from detection_preparation

# Reset stage tertentu
python pipeline_manager.py --reset-stage parasite_cropping
```

## 📋 Pipeline Stages

1. **environment_check** - Check Python environment & dependencies
2. **dataset_download** - Download MP-IDB dataset
3. **detection_preparation** - Parse detection dataset (103 images)
4. **parasite_cropping** - Crop parasites for classification (1,242 crops)
5. **training_verification** - Test training system

## 🔄 Checkpoint System

- **Automatic Resume**: Pipeline otomatis melanjutkan dari tahap yang belum selesai
- **Smart Skip**: Jika data sudah ada, tahap akan di-skip
- **Status Tracking**: File `.pipeline_checkpoint.json` menyimpan progress
- **Error Handling**: Tahap yang gagal bisa dijalankan ulang tanpa mengulang yang sukses

## 📝 Log Files

- `pipeline.log` - Detailed execution log
- `.pipeline_checkpoint.json` - Checkpoint status (jangan dihapus!)

## ⚡ Keunggulan vs Bash Script

| Bash Script | Python Pipeline Manager |
|-------------|-------------------------|
| ❌ Restart dari awal setiap kali | ✅ Resume dari tahap yang belum selesai |
| ❌ Tidak ada status tracking | ✅ Status detail setiap tahap |
| ❌ Sulit debug error | ✅ Log detail dan error handling |
| ❌ Tidak fleksibel | ✅ Bisa start dari tahap manapun |

## 🚀 After Pipeline Success

Setelah pipeline selesai, jalankan training:

```bash
# Detection training
python scripts/10_train_yolo_detection.py --epochs 30 --name production_detection

# Classification training
python scripts/11_train_classification_crops.py --epochs 25 --name production_classification
```