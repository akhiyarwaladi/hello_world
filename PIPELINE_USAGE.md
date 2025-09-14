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

---

# 🚀 NEW: Unified Training Pipeline

## 🎯 Mengatasi Redundansi Testing

**MASALAH SEBELUMNYA:**
- ❌ 2 proses `train_multispecies.py test` berjalan redundant
- ❌ `test_models_comprehensive.py` manual dan terpisah
- ❌ Multiple scattered training scripts

**SOLUSI BARU:**
```bash
# Single unified interface - RECOMMENDED
python pipeline.py validate          # Replace redundant testing
python pipeline.py train MODEL_NAME  # Unified training
python pipeline.py evaluate --comprehensive  # Replace comprehensive testing
```

## 🔧 Unified Pipeline Commands

### Status & Information
```bash
python pipeline.py status                    # Check current state
python pipeline.py list --models            # Available models
python pipeline.py list --datasets          # Available datasets
```

### Quick Validation (Menggantikan train_multispecies.py test)
```bash
python pipeline.py validate                 # Test all models (2 epochs)
python pipeline.py validate --models yolov8_detection  # Test specific model
```

### Training (Unified Interface)
```bash
python pipeline.py train yolov8_detection --epochs 30
python pipeline.py train yolov8_classification --batch 8 --background
```

### Evaluation (Menggantikan test_models_comprehensive.py)
```bash
python pipeline.py evaluate --comprehensive  # Full comprehensive testing
python pipeline.py evaluate --models yolov8_detection --comprehensive
```

### Export Results
```bash
python pipeline.py export --format journal  # Ready for publication
```

## ✅ Migration Path

### Stop Redundant Processes
```bash
# Kill redundant testing (if still running)
pkill -f "train_multispecies.py test"
```

### Use New Unified Pipeline
```bash
# OLD way (redundant):
python train_multispecies.py test

# NEW way (unified):
python pipeline.py validate
```

## 🛡️ Safety Features

- ✅ Prerequisites validation before execution
- ✅ Configuration-driven (config/models.yaml, config/datasets.yaml)
- ✅ Integrated with ExperimentLogger
- ✅ Background training support
- ✅ Timeout protection