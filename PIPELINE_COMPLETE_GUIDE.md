# Malaria Detection Pipeline - Complete Guide

## 🚀 Pipeline dari Awal Sampai Akhir

Pipeline ini mendukung **SELURUH PROSES** mulai dari download data hingga training model dengan struktur folder terorganisir otomatis.

## ⚡ Quick Start - Pipeline Lengkap

### 1. Setup Awal (Sekali Saja)
```bash
# Clone repository
git clone [repository-url]
cd malaria_detection

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Jalankan Pipeline Lengkap dari Awal
```bash
# Aktifkan environment
source venv/bin/activate

# Opsi 1: Pipeline Otomatis (RECOMMENDED)
python scripts/run_full_pipeline.py

# Opsi 2: Step-by-step Manual
python scripts/01_download_datasets.py
python scripts/02_preprocess_data.py
python scripts/03_integrate_datasets.py
python scripts/04_convert_to_yolo.py
python scripts/05_augment_data.py
python scripts/06_split_dataset.py

# Opsi 3: Pipeline Unified (Baru!)
python pipeline.py validate --models all
python pipeline.py train yolov8_detection --epochs 50 --background
python pipeline.py train yolov8_classification --epochs 50 --background
```

## 📁 Struktur Folder Terorganisir Otomatis

Pipeline secara otomatis mengatur hasil dalam struktur terorganisir:

```
results/
├── current_experiments/    # Eksperimen aktif
│   ├── validation/        # Testing & validation runs
│   ├── training/          # Training eksperimen
│   └── comparison/        # Model comparison
├── completed_models/      # Model production-ready
├── publications/         # Hasil siap publikasi
├── archive/             # Eksperimen historis
└── experiment_logs/     # Log semua eksperimen
```

## 🔄 Mengulangi dari Awal - Data Download

### Jika Anda Ingin Mengulang dari Awal:

**✅ AMAN untuk diulang:**
```bash
# Hapus data lama (opsional)
rm -rf data/raw data/processed data/integrated data/augmented data/splits

# Download ulang datasets
python scripts/01_download_datasets.py

# Lanjutkan pipeline
python scripts/02_preprocess_data.py
# ... dst
```

**⚠️ Yang Terjadi Saat Download Ulang:**
- **Data lama akan ditimpa** jika ada
- **Download resume otomatis** jika terputus
- **Validasi checksum** untuk memastikan data utuh
- **~15GB space dibutuhkan** untuk semua datasets

### Datasets yang Akan Didownload:
1. **NIH Malaria Dataset** - 27,558 images
2. **MP-IDB Dataset** - Detection annotations
3. **BBBC041 Dataset** - Microscopy images
4. **PlasmoID Dataset** - Multi-species data
5. **IML Dataset** - Additional training data
6. **Uganda Dataset** - Field data

## 🎯 Pipeline Commands - Unified Interface

### Status & Info
```bash
python pipeline.py status                    # Status sistem
python pipeline.py list                      # List models & datasets
```

### Validation
```bash
python pipeline.py validate --models all     # Validate semua models
python pipeline.py validate --models yolov8_detection
```

### Training
```bash
# Detection Models
python pipeline.py train yolov8_detection --epochs 50 --batch 16 --background
python pipeline.py train yolov11_detection --epochs 50 --batch 8 --background
python pipeline.py train rtdetr_detection --epochs 50 --batch 4 --background

# Classification Models
python pipeline.py train yolov8_classification --epochs 50 --batch 16 --background
python pipeline.py train yolov11_classification --epochs 50 --batch 8 --background
```

### Evaluation
```bash
python pipeline.py evaluate --models all --comprehensive
python pipeline.py export --format journal    # Export hasil publikasi
```

## 🔧 Konfigurasi Pipeline

### Model Configuration (`config/models.yaml`)
```yaml
models:
  yolov8_detection:
    script: "python scripts/07_train_yolo_detection.py"
    type: "detection"
    args:
      epochs: 50
      batch: 16
      device: "cpu"
    validation_args:
      epochs: 2
      batch: 4
```

### Dataset Configuration (`config/datasets.yaml`)
```yaml
processed_datasets:
  detection_multispecies:
    path: "data/detection_multispecies"
    type: "detection"
    classes: 4
    class_names: ["falciparum", "malariae", "ovale", "vivax"]
```

## 📊 Monitoring & Results

### Real-time Monitoring
```bash
# Check training progress
python pipeline.py status

# Monitor background processes
ps aux | grep python
watch -n 30 'python pipeline.py status'
```

### Results Location
- **Training logs**: `results/experiment_logs/`
- **Model weights**: `results/current_experiments/[type]/[model]/[name]/weights/`
- **CSV results**: `results/current_experiments/[type]/[model]/[name]/results.csv`
- **Publication ready**: `results/publications/`

## 🚨 Troubleshooting

### Common Issues:

**1. Download Gagal:**
```bash
# Hapus file corrupted dan coba lagi
rm -rf data/raw/[dataset_name]
python scripts/01_download_datasets.py
```

**2. Memory Issues:**
```bash
# Kurangi batch size
python pipeline.py train yolov8_detection --batch 4 --epochs 10
```

**3. Permission Errors:**
```bash
# Fix permissions
chmod +x scripts/*.py
```

**4. Import Errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📈 Performance Optimization

### Untuk CPU Training:
```bash
# Disable NNPACK warnings
export NNPACK_DISABLE=1

# Optimal settings
python pipeline.py train yolov8_detection --batch 4 --epochs 50 --device cpu
```

### Untuk GPU Training:
```bash
# Check GPU
nvidia-smi

# GPU settings
python pipeline.py train yolov8_detection --batch 32 --epochs 100 --device cuda
```

## 🎯 Workflow Recommendations

### Development Workflow:
1. **Quick validation**: `python pipeline.py validate --models all`
2. **Small test run**: `python pipeline.py train yolov8_detection --epochs 5 --batch 2`
3. **Full training**: `python pipeline.py train yolov8_detection --epochs 50 --background`
4. **Compare models**: `python pipeline.py evaluate --comprehensive`

### Production Workflow:
1. **Full pipeline**: `python scripts/run_full_pipeline.py`
2. **All models**: Train semua model parallel dengan `--background`
3. **Evaluation**: `python pipeline.py evaluate --comprehensive`
4. **Export**: `python pipeline.py export --format journal`

## ⚙️ Advanced Features

### Background Training:
```bash
# Start multiple training parallel
python pipeline.py train yolov8_detection --background --name exp1 &
python pipeline.py train yolov11_detection --background --name exp2 &
python pipeline.py train yolov8_classification --background --name exp3 &
```

### Custom Experiments:
```bash
# Custom hyperparameters
python pipeline.py train yolov8_detection \
  --epochs 100 \
  --batch 8 \
  --device cpu \
  --name "custom_experiment_v1" \
  --background
```

### Export & Publication:
```bash
# Generate journal-ready results
python pipeline.py export --format journal

# Results akan tersimpan di:
# results/publications/journal_export_[timestamp]/
```

## 📚 File Structure

```
malaria_detection/
├── pipeline.py              # Main unified interface
├── config/                 # Configuration files
│   ├── models.yaml        # Model definitions
│   ├── datasets.yaml      # Dataset configurations
│   └── results_structure.yaml
├── scripts/               # Individual pipeline steps
│   ├── 01_download_datasets.py
│   ├── 02_preprocess_data.py
│   ├── ...
│   └── utils/
├── data/                  # All data (gitignored)
│   ├── raw/              # Original downloads
│   ├── processed/        # Preprocessed data
│   ├── integrated/       # Unified datasets
│   └── splits/           # Train/val/test splits
├── results/              # All results (organized)
└── utils/                # Utilities
    └── results_manager.py # Auto organization
```

---

## 🎉 Summary

Pipeline ini **COMPLETAMENTE AUTOMÁTICO**:
- ✅ Download datasets otomatis
- ✅ Preprocessing dan augmentasi
- ✅ Training dengan organized folder structure
- ✅ Background parallel training
- ✅ Auto-monitoring dan logging
- ✅ Export hasil publikasi

**Untuk menjalankan dari awal:** Cukup jalankan `python scripts/run_full_pipeline.py` dan pipeline akan mengurus semuanya!
