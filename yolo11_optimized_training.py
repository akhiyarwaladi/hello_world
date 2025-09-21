#!/usr/bin/env python3
"""
YOLOv11 optimized training script
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def train_yolo11_optimized():
    """Train YOLOv11 with optimized parameters"""

    print("🎯 YOLO11-OPTIMIZED TRAINING")
    print("Using YOLOv11 with full augmentation")

    # Use Kaggle dataset
    data_yaml = "data/kaggle_pipeline_ready/data.yaml"

    if not Path(data_yaml).exists():
        print(f"❌ Dataset not found: {data_yaml}")
        return False

    # YOLOv11 configuration
    MODEL_NAME = "yolo11n"  # YOLOv11 nano
    EPOCHS = 50
    IMGSZ = 640
    BATCH_SIZE = 16

    # Use centralized pipeline structure
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/exp_yolo11_optimized_{timestamp}")
    detection_dir = output_dir / "detection" / "yolo11_detection"
    detection_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(f"{MODEL_NAME}.pt")

    print(f"📊 Training parameters:")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Image size: {IMGSZ}")
    print(f"   Dataset: {data_yaml}")
    print(f"   Output: {output_dir}")

    # Train with full augmentation
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        patience=15,
        save=True,
        save_period=10,
        device='cpu',
        workers=4,
        exist_ok=True,
        optimizer='AdamW',
        lr0=0.001,

        # FULL AUGMENTATION
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=45,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.2,

        # Output settings
        project=str(detection_dir),
        name=f'yolo11_optimized_{timestamp}',
        plots=True,
        val=True,
        verbose=True
    )

    print("\n🎉 YOLOv11 Training completed!")

    # Validation
    print("\n📊 Running validation...")
    metrics = model.val(
        data=data_yaml,
        batch=BATCH_SIZE,
        imgsz=IMGSZ,
        conf=0.25,
        iou=0.45,
        device='cpu',
        split='val'
    )

    print(f"\n✅ FINAL YOLO11 RESULTS:")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP75: {metrics.box.map75:.4f}")

    if metrics.box.map50 > 0.5:
        print(f"🎯 SUCCESS! mAP50 = {metrics.box.map50:.4f} > 0.5")
    else:
        print(f"⚠️  mAP50 = {metrics.box.map50:.4f} < 0.5, may need more epochs")

    return results

if __name__ == "__main__":
    train_yolo11_optimized()