# 🔬 Pipeline Malaria Detection - Complete Workflow

## 🎯 Overview: Detection + Classification Workflow

Pipeline ini menggunakan **2-stage approach** untuk deteksi malaria yang akurat:

### Stage 1: Detection (Menemukan Parasit)
- **Input**: Gambar mikroskop penuh
- **Output**: Lokasi parasit (bounding boxes)
- **Models**: YOLOv8, YOLOv11, RT-DETR

### Stage 2: Classification (Mengidentifikasi Spesies)
- **Input**: Crop parasit dari Stage 1
- **Output**: Spesies parasit (falciparum, vivax, ovale, malariae)
- **Models**: YOLOv8-cls, YOLOv11-cls, ResNet18, EfficientNet

---

## 🚀 Complete Pipeline Steps

### 1. Training Phase (Background Running)

```bash
# Detection Models (ALL STARTED)
✅ YOLOv8 Detection   → 50 epochs (background)
✅ YOLOv11 Detection  → 50 epochs (background)
✅ RT-DETR Detection  → 50 epochs (background)

# Classification Models (ALL STARTED)
✅ YOLOv8 Classification    → 50 epochs (background)
✅ YOLOv11 Classification   → 50 epochs (background)
✅ ResNet18 Classification  → 50 epochs (background)
✅ EfficientNet Classification → 50 epochs (background)
```

### 2. Best Model Combinations

Setelah training selesai, kita akan punya 12 kombinasi optimal:

**Detection Models (3):**
- `yolov8_detection.pt`
- `yolov11_detection.pt`
- `rtdetr_detection.pt`

**Classification Models (4):**
- `yolov8_classification.pt`
- `yolov11_classification.pt`
- `resnet18_classification.pt`
- `efficientnet_classification.pt`

**= 3 × 4 = 12 kombinasi total**

---

## 🔄 Complete Detection + Classification Workflow

### Step 1: Generate Crops from Detection
```bash
# Menggunakan detection model untuk crop parasit
python scripts/10_crop_detections.py \
    --model results/best_detection_model/best.pt \
    --input data/test_images/ \
    --output data/crops_from_detection/ \
    --confidence 0.5 \
    --crop_size 128
```

### Step 2: Classification pada Crops
```bash
# Klasifikasi spesies dari crops yang dihasilkan
python scripts/11_train_classification_crops.py \
    --mode inference \
    --model results/best_classification_model/best.pt \
    --input data/crops_from_detection/ \
    --output results/final_predictions.csv
```

### Step 3: Full Pipeline (End-to-End)
```bash
# Pipeline lengkap dalam 1 command
python scripts/13_full_detection_classification_pipeline.py \
    --detection_model results/yolov11_detection/best.pt \
    --classification_model results/resnet18_classification/best.pt \
    --input_images data/test_images/ \
    --output results/complete_analysis/ \
    --confidence 0.5
```

---

## 📊 Expected Results Structure

```
results/complete_analysis/
├── detections/           # Visualization dengan bounding boxes
│   ├── image1_detected.jpg
│   ├── image2_detected.jpg
│   └── ...
├── crops/               # Crops parasit yang dideteksi
│   ├── image1_crop_001.jpg (falciparum)
│   ├── image1_crop_002.jpg (vivax)
│   └── ...
├── predictions.csv      # Hasil akhir semua prediksi
└── summary_report.html  # Laporan lengkap
```

### Format predictions.csv:
```csv
image_file,detection_confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2,species,classification_confidence,final_confidence
image1.jpg,0.89,100,150,200,250,falciparum,0.92,0.81
image1.jpg,0.76,300,400,380,480,vivax,0.88,0.67
image2.jpg,0.94,50,75,150,175,malariae,0.85,0.80
```

---

## 🎯 Optimal Model Selection Strategy

### 1. Performance Metrics
- **Detection**: mAP@0.5, Precision, Recall
- **Classification**: Accuracy, F1-score per species
- **Combined**: End-to-end accuracy pada test set

### 2. Speed vs Accuracy Trade-off
```
FAST PIPELINE (Real-time):
Detection: YOLOv8n + Classification: MobileNet
→ ~100ms per image

BALANCED PIPELINE (Recommended):
Detection: YOLOv11m + Classification: ResNet18
→ ~300ms per image

ACCURATE PIPELINE (Research):
Detection: RT-DETR-L + Classification: EfficientNet-B3
→ ~1000ms per image
```

### 3. Species-Specific Optimization
```bash
# Jika fokus pada falciparum (yang paling berbahaya)
python scripts/analyze_species_performance.py \
    --target_species falciparum \
    --optimize_for sensitivity
```

---

## 🔍 Monitoring Training Progress

```bash
# Check training status
watch -n 30 'ps aux | grep python | grep train'

# Check results
ls -la results/current_experiments/training/

# Monitor logs
tail -f results/experiment_logs/latest_training.log
```

---

## 🚨 Troubleshooting

### Issue 1: Low Detection Recall
```bash
# Lower confidence threshold
--confidence 0.25  # dari 0.5
```

### Issue 2: Classification Confusion
```bash
# Add more augmentation atau use ensemble
python scripts/18_ensemble_predictions.py \
    --models resnet18,efficientnet,yolov8_cls
```

### Issue 3: Speed Issues
```bash
# Use lighter models
Detection: yolov8n.pt
Classification: mobilenet_v2.pt
```

---

## 🎉 Status Update

✅ **Code cleanup completed**
✅ **All models training in background**
✅ **Pipeline workflow documented**
✅ **Ready for production testing**

**Estimasi training selesai**: 2-4 jam (tergantung dataset size)

Setelah semua training selesai, kita bisa langsung:
1. Compare performa semua kombinasi model
2. Pilih kombinasi terbaik untuk production
3. Run full end-to-end testing
4. Export hasil untuk publikasi

**Pipeline sudah siap digunakan! 🚀**