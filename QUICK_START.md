# 🚀 QUICK START GUIDE
**Untuk yang bingung sama codebase - panduan super simple!**

## 📋 Yang Penting Doang

### 1. Download Dataset
```bash
source venv/bin/activate
python scripts/data_setup/01_download_datasets.py --dataset mp_idb
```

### 2. Jalankan Full Pipeline
```bash
python run_complete_pipeline.py --detection yolo8 --epochs-det 50 --epochs-cls 30
```

**Selesai! 🎉**

## 🗂️ File yang Penting

**Yang Harus Kamu Tau:**
```
📦 hello_world/
├── 🎮 pipeline.py                    # Interface utama
├── 🚀 run_complete_pipeline.py       # Automation lengkap
├── 📂 scripts/
│   ├── 📁 data_setup/                # Setup data
│   │   └── 01_download_datasets.py   # Download dataset
│   └── 📁 training/                  # Training models
│       └── 10_crop_detections.py     # Crop generation
├── 📂 config/                        # Semua configurasi
└── 📂 data/raw/mp_idb/               # Dataset MP-IDB
```

**Yang Bisa Diabaikan:**
- `archive_unused/` - File complex yang ga kepake
- Semua script analysis yang ribet

## ⚡ Commands Cepat

**Lihat model tersedia:**
```bash
python pipeline.py list
```

**Training manual:**
```bash
# Step 1: Train detection
python pipeline.py train yolov8_detection --name test_det --epochs 30

# Step 2: Generate crops
python scripts/training/10_crop_detections.py --model yolo8 --experiment test_det

# Step 3: Train classification
python pipeline.py train yolov8_classification --name test_cls --epochs 20
```

**Check hasil:**
```bash
ls results/current_experiments/training/
```

## 🎯 Workflow Sederhana

1. **Dataset ready** ✅ - MP-IDB sudah didownload
2. **Train detection** 🎯 - Detect parasite di gambar
3. **Generate crops** ✂️ - Potong parasite jadi gambar kecil
4. **Train classification** 🔍 - Klasifikasi jenis parasite

## 💡 Tips

- **Pertama kali**: Pake `run_complete_pipeline.py` aja
- **Kalo error**: Cek di `results/` ada log error nya
- **Kalo mau experiment**: Ganti nama experiment beda-beda
- **Kalo pindah computer**: Tinggal git clone + download dataset lagi

---
**Sesimple itu! 🎉**