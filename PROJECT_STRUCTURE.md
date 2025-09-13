# Malaria Detection Project - Folder Structure

## 📁 Data Structure (Rapi & Terorganisir)

### Classification Pipeline
```
data/classification/           # ✅ DATASET UTAMA KLASIFIKASI
├── train/                    # Data training (70%)
│   ├── P_falciparum/         # 6 kelas malaria
│   ├── P_vivax/
│   ├── P_malariae/
│   ├── P_ovale/
│   ├── Mixed_infection/
│   └── Uninfected/
├── val/                      # Data validation (15%)
└── test/                     # Data testing (15%)
```

### Detection Pipeline
```
data/detection_fixed/         # ✅ DATASET UTAMA DETECTION (BARU)
├── images/                   # 103 gambar mikroskopi
├── labels/                   # YOLO format (.txt)
├── annotations/              # Metadata
└── dataset.yaml             # Config YOLO
```

### Raw Data (Source)
```
data/raw/                     # Data mentah (tidak diubah)
├── mp_idb/                   # MP-IDB dataset
├── nih/                      # NIH dataset
├── bbbc041/                  # BBBC041 dataset
├── plasmodium_id/            # PlasmoID dataset
├── iml/                      # IML dataset
└── uganda/                   # Uganda dataset
```

## 🚀 Training Results

### Classification Results
```
results/classification/       # Hasil training klasifikasi
├── quick_test/              # Quick training results
├── ultra_fast/              # Ultra fast training results
└── experiments/             # Eksperimen lainnya
```

### Detection Results (Coming Soon)
```
results/detection/           # Hasil training detection
├── yolov8/                  # YOLOv8 results
├── yolov11/                 # YOLOv11 results
├── yolov12/                 # YOLOv12 results
└── rtdetr/                  # RT-DETR results
```

## 📊 Current Status

### ✅ SELESAI
1. **Classification Dataset**: `data/classification/` (56,754 images)
2. **Detection Dataset**: `data/detection_fixed/` (103 images, 1,242 parasites)
3. **Quick Training**: Results in `results/classification/`

### 🔄 NEXT STEPS
1. **Train YOLO Detection**: YOLOv8, YOLOv11, YOLOv12
2. **Train RT-DETR**: Detection comparison
3. **Paper Comparison**: YOLOv11 vs YOLOv12 vs RT-DETR

## 🧹 Folder Yang Akan Dihapus (Cleanup)

### Folder Duplikat/Lama (Akan Dihapus)
- `data/detection/` → Diganti dengan `data/detection_fixed/`
- `data/classification_crops*` → Test folder, sudah tidak perlu
- `data/augmented_quick/` → Test folder
- `data/integrated_v2/` → Raw processing folder
- `data/final_v2/` → Old version

### Debug Folders (Optional Keep)
- `results/debug_boxes_fixed/` → Bukti perbaikan bounding box
- `results/debug_boxes/` → Bukti error lama

## 🎯 Path untuk Setiap Task

### Classification Training
```bash
# Data: data/classification/
# Command: python scripts/07_train_yolo_quick.py --data data/classification
# Results: results/classification/
```

### Detection Training (Next)
```bash
# Data: data/detection_fixed/
# Command: python scripts/09_train_detection.py --data data/detection_fixed/dataset.yaml
# Results: results/detection/
```

### Paper Research
```bash
# Compare: YOLOv11 vs YOLOv12 vs RT-DETR detection performance
# Dataset: data/detection_fixed/ (1,242 parasites, 103 images)
# Focus: Detection accuracy, speed, model size
```