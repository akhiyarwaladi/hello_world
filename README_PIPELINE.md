# Comprehensive Malaria Detection Pipeline

Pipeline Python yang menggantikan `quick_setup_new_machine.sh` dengan sistem validasi data yang menyeluruh dan kemampuan perbaikan otomatis.

## ✨ Features

### 🔍 Comprehensive Data Validation
- **Raw Dataset Validation**: Memvalidasi struktur dan konten dataset MP-IDB
- **Detection Dataset Validation**: Mengecek format YOLO, pasangan image-label, dan integritas file
- **Cropped Dataset Validation**: Memverifikasi kualitas crops, distribusi train/val/test, dan metadata

### 🔧 Intelligent Repair System
- **Automatic Detection**: Mendeteksi data yang rusak atau tidak lengkap
- **Smart Recovery**: Memberikan informasi detail tentang masalah dan saran perbaikan
- **Repair Mode**: Mode khusus untuk validasi ulang dan perbaikan semua tahap

### 📊 Detailed Monitoring
- **Real-time Progress**: Progress tracking dengan detail statistik
- **Comprehensive Logging**: Log file terstruktur dengan timestamp
- **Status Dashboard**: Overview lengkap status setiap tahap pipeline

### 🚀 User-Friendly Interface
- **Interactive Mode**: Menu interaktif yang mudah digunakan
- **CLI Mode**: Command-line interface untuk automation
- **Checkpoint System**: Resume otomatis dari tahap terakhir yang berhasil

## 🎯 Pipeline Stages

| Stage | Description | Validation |
|-------|-------------|------------|
| 1️⃣ **Environment Check** | Validasi Python, conda, dan dependencies | ✅ Package imports |
| 2️⃣ **Dataset Download** | Download MP-IDB dataset | ✅ Structure, image count, species |
| 3️⃣ **Detection Preparation** | Convert ke format YOLO detection | ✅ Images, labels, YAML, corruption |
| 4️⃣ **Parasite Cropping** | Extract individual parasites | ✅ Crop count, quality, metadata |
| 5️⃣ **Training Verification** | Test training system | ✅ Training execution |

## 💻 Usage

### Interactive Mode (Recommended)
```bash
python pipeline.py
```

Menu yang akan muncul:
```
🔬 COMPREHENSIVE MALARIA DETECTION PIPELINE
==================================================

🔍 PIPELINE STATUS
==================================================
1. Environment & Dependencies Check: ✅ COMPLETED
2. Dataset Download: ✅ COMPLETED
3. Detection Dataset Preparation: ✅ COMPLETED
4. Parasite Cropping: ✅ COMPLETED
5. Training System Verification: ✅ COMPLETED

Completed: 5/5 stages

Pilihan:
1. Lanjutkan pipeline (skip completed stages)
2. Restart dari awal (hapus semua checkpoint)
3. Repair mode (validasi ulang dan perbaiki semua stage)
4. Lihat status detail saja
5. Keluar
```

### CLI Mode
```bash
# Lanjutkan pipeline
python pipeline.py --continue

# Restart dari awal
python pipeline.py --restart

# Repair mode (validasi ulang semua data)
python pipeline.py --repair

# Lihat status saja
python pipeline.py --status
```

## 🔍 Data Validation Examples

### Raw Dataset Validation
```
✅ Dataset Download validation: PASSED
📊 Data Statistics:
   • total_images: 288
   • species_counts: {'Falciparum': 208, 'Vivax': 80}
   • dataset_type: classification
⚠️  Warnings:
   • This is a classification dataset - no bounding box annotations expected
```

### Detection Dataset Validation
```
✅ Detection Dataset Preparation validation: PASSED
📊 Data Statistics:
   • total_images: 103
   • total_labels: 103
   • corrupt_images: 0
```

### Cropped Dataset Validation
```
✅ Parasite Cropping validation: PASSED
📊 Data Statistics:
   • train_crops: 869
   • val_crops: 186
   • test_crops: 187
   • total_crops: 1242
   • corrupt_crops: 0
   • small_crops: 0
```

## 🔧 Repair Mode

Repair mode akan:

1. **Re-validate semua data** dari setiap tahap
2. **Detect masalah** seperti:
   - File corrupt atau hilang
   - Format data tidak sesuai
   - Jumlah data di bawah threshold
   - Metadata tidak konsisten

3. **Provide detailed feedback**:
   ```
   ❌ Stage 'Detection Dataset Preparation' validation: FAILED
   📊 Data Statistics:
      • total_images: 50
      • total_labels: 45
   ❌ Errors found:
      • Insufficient images: 50 < 100
      • Missing labels for 5 images
   🔧 Stage needs repair
   ```

4. **Auto-repair** tahap yang bermasalah

## 📁 File Structure

```
malaria_detection/
├── pipeline.py              # Main pipeline (NEW!)
├── pipeline_simple.py       # Backup simple version
├── pipeline_manager_old.py  # Backup old manager
├── run_pipeline_old.py      # Backup old runner
├── test_validation.py       # Validation test script
├── logs/                    # Pipeline logs
│   ├── pipeline.log
│   └── pipeline_20240914_*.log
├── .pipeline_checkpoint.json # Checkpoint state
└── data/
    ├── raw/mp_idb/          # Original dataset
    ├── detection_fixed/     # YOLO detection format
    └── classification_crops/ # Cropped parasites
```

## 🎉 Benefits

### vs. Bash Script (`quick_setup_new_machine.sh`)
- ✅ **Resume capability**: Tidak restart dari awal jika ada tahap yang gagal
- ✅ **Data validation**: Memastikan setiap tahap menghasilkan data berkualitas
- ✅ **Error recovery**: Smart error handling dan recovery suggestions
- ✅ **Progress tracking**: Real-time status dan logging
- ✅ **User-friendly**: Interface yang mudah dipahami

### Smart Features
- **Checkpoint system**: JSON-based state management
- **Parallel validation**: Efficient multi-stage validation
- **Repair detection**: Automatic detection of data corruption
- **Statistics tracking**: Detailed metrics untuk setiap tahap
- **Log management**: Structured logging dengan retention

## 🚀 Training Ready

Setelah pipeline selesai:

```bash
# Detection Training
python scripts/10_train_yolo_detection.py --epochs 30

# Classification Training
python scripts/11_train_classification_crops.py --epochs 25
```

## 🔮 Advanced Usage

### Custom Validation
```python
from pipeline import DataValidator

validator = DataValidator()
is_valid, result = validator.validate_detection_dataset("path/to/data")
print(f"Valid: {is_valid}")
print(f"Stats: {result['stats']}")
```

### Programmatic Pipeline
```python
from pipeline import MalariaPipeline

pipeline = MalariaPipeline()
success = pipeline.run_pipeline(repair_mode=True)
```

---

**🎯 Result**: Pipeline komprehensif yang reliable, user-friendly, dan production-ready untuk sistem deteksi malaria!