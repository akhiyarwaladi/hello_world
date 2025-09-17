# 🔬 CORRECT Detection + Classification Workflow

## ⚠️ PERBAIKAN WORKFLOW YANG BENAR

### ❌ Yang Salah (Yang Tadi Saya Lakukan):
```
Detection Training + Classification Training (bersamaan) ← SALAH!
```

### ✅ Yang Benar - Sequential Workflow:

## 🎯 Stage 1: Detection Model Training (SEDANG BERJALAN)

**✅ Currently Active:**
- YOLOv8 Detection (`auto_yolov8_det`) - 50 epochs
- YOLOv11 Detection (`auto_yolov11_det`) - 50 epochs
- RT-DETR Detection (`auto_rtdetr_det`) - 50 epochs

**❌ Stopped (Premature):**
- YOLOv8 Classification (dihentikan)
- YOLOv11 Classification (dihentikan)
- ResNet18 Classification (dihentikan)

---

## 🔄 Stage 2: Generate Crops (SETELAH DETECTION SELESAI)

**Menunggu detection models selesai, lalu:**

```bash
# Pilih detection model terbaik berdasarkan mAP
best_detection_model="results/.../auto_yolov11_det/weights/best.pt"

# Generate crops dari detection model terbaik
python scripts/10_crop_detections.py \
    --model $best_detection_model \
    --input data/detection_multispecies/train/images \
    --output data/crops_from_detection/ \
    --confidence 0.5 \
    --crop_size 128 \
    --create_yolo_structure
```

**Output Expected:**
```
data/crops_from_detection/
├── crops/
│   ├── train/
│   ├── val/
│   └── test/
├── yolo_classification/
│   ├── train/
│   │   ├── falciparum/
│   │   ├── vivax/
│   │   ├── ovale/
│   │   └── malariae/
│   └── val/...
└── crop_metadata.csv
```

---

## 🏷️ Stage 3: Classification Training (SETELAH CROPS READY)

**Baru mulai classification training:**

```bash
# YOLOv8 Classification pada crops
python pipeline.py train yolov8_classification \
    --name crops_yolov8_cls \
    --data data/crops_from_detection/yolo_classification \
    --epochs 50 \
    --background

# YOLOv11 Classification pada crops
python pipeline.py train yolov11_classification \
    --name crops_yolov11_cls \
    --data data/crops_from_detection/yolo_classification \
    --epochs 50 \
    --background

# PyTorch ResNet18 pada crops
python pipeline.py train pytorch_resnet18_classification \
    --name crops_resnet18_cls \
    --data data/crops_from_detection/ \
    --epochs 50 \
    --background
```

---

## 🎯 Stage 4: Complete Pipeline (FINAL)

**End-to-end inference:**

```bash
# Full detection → classification pipeline
python scripts/13_full_detection_classification_pipeline.py \
    --detection_model results/.../auto_yolov11_det/weights/best.pt \
    --classification_model results/.../crops_yolov8_cls/weights/best.pt \
    --input_images test_data/ \
    --output results/complete_analysis/ \
    --confidence 0.5
```

---

## 📊 Expected Timeline

### ⏰ Current Status (Stage 1):
- **Detection Training**: 2-4 jam (sedang berjalan)
- **Progress**: Dapat dimonitor dengan `python monitor_training.py`

### ⏰ Stage 2 (Generate Crops):
- **Duration**: 30-60 menit
- **Depends on**: Jumlah gambar dan detections per gambar
- **Manual Step**: Perlu pilih best detection model berdasarkan mAP

### ⏰ Stage 3 (Classification Training):
- **Duration**: 1-2 jam per model
- **Can Run Parallel**: 3 classification models bersamaan
- **Total**: ~2 jam (jika parallel)

### ⏰ Stage 4 (Final Pipeline):
- **Duration**: 5-10 menit per test batch
- **Output**: Production-ready detection + classification

---

## 🎯 Mengapa Sequential?

### 1. **Data Dependency**
```
Raw Images → Detection Model → Crops → Classification Model
```

### 2. **Quality Control**
- Detection model harus divalidasi dulu (mAP, precision, recall)
- Crops quality tergantung detection accuracy
- Classification hanya sebaik crops yang di-generate

### 3. **Efficiency**
- Classification training di data yang salah = waste time
- Better: train detection dulu, validate, generate crops, then classify

---

## 🔍 Monitoring Current Status

```bash
# Check detection training progress
python monitor_training.py

# Check specific model progress
ls results/current_experiments/training/detection/*/auto_*/weights/

# When detection completes, best.pt akan muncul di folder weights/
```

---

## ✅ Action Plan

1. **NOW**: Let detection training complete (2-4 hours)
2. **NEXT**: Choose best detection model based on validation mAP
3. **THEN**: Generate crops using `scripts/10_crop_detections.py`
4. **FINALLY**: Train classification models on generated crops

**Workflow sekarang sudah benar! Detection models sedang training.** 🚀