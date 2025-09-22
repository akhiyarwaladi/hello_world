#!/usr/bin/env python3
"""
Kaggle-optimized training script with full augmentation
Reproduces the exact parameters from kaggle_yolo_example1.py
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def train_kaggle_optimized(model_type="yolo12"):
    """Train with exact Kaggle script parameters for optimal mAP"""

    print("🎯 KAGGLE-OPTIMIZED TRAINING")
    print("Using exact parameters from successful Kaggle script (mAP 0.8)")

    # Use Kaggle dataset
    data_yaml = "data/kaggle_pipeline_ready/data.yaml"

    if not Path(data_yaml).exists():
        print(f"❌ Dataset not found: {data_yaml}")
        print("Run setup_kaggle_for_pipeline.py first")
        return False

    # Support both YOLOv11 and YOLOv12
    if model_type == "yolo11":
        MODEL_NAME = "yolo11m"  # YOLOv11 medium (higher accuracy, slower training)
    else:
        MODEL_NAME = "yolo12n"  # YOLOv12 - latest and best
    EPOCHS = 100
    IMGSZ = 640
    BATCH_SIZE = 16  # CPU training can handle larger batch size

    # Use centralized pipeline structure
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/exp_kaggle_optimized_{timestamp}")
    detection_dir = output_dir / "detection" / "yolo12_detection"
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

    # Train with EXACT Kaggle script parameters
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        patience=50,              # Increased from 15 to allow training until epoch 49+ (historical best: 89.15% at epoch 49)
        save=True,
        save_period=10,
        device='cpu',
        workers=4,
        exist_ok=True,
        optimizer='AdamW',        # Same as Kaggle
        lr0=0.001,               # Same as Kaggle
        warmup_epochs=5,         # Warmup for training stability
        weight_decay=0.0005,     # L2 regularization for better generalization

        # FULL AUGMENTATION - Exact same as Kaggle script
        augment=True,            # ← Key difference!
        hsv_h=0.015,            # HSV color augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=45,             # Rotation (vs 0.0 in pipeline)
        scale=0.5,              # Scaling
        flipud=0.5,             # Vertical flip (vs 0.0 in pipeline)
        fliplr=0.5,             # Horizontal flip
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.2,              # Mixup (vs 0.0 in pipeline)
        copy_paste=0.2,         # Copy-paste (vs 0.0 in pipeline)

        # Output settings - centralized structure
        project=str(detection_dir),
        name=f'kaggle_optimized_yolo12_{timestamp}',
        plots=True,
        val=True,
        verbose=True
    )

    print("\n🎉 Training completed!")

    # Validation
    print("\n📊 Running validation...")
    metrics = model.val(
        data=data_yaml,
        batch=BATCH_SIZE,
        imgsz=IMGSZ,
        conf=0.25,              # Same as Kaggle validation
        iou=0.45,
        device='cpu',
        split='val'
    )

    print(f"\n✅ FINAL RESULTS:")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP75: {metrics.box.map75:.4f}")

    if metrics.box.map50 > 0.5:
        print(f"🎯 SUCCESS! mAP50 = {metrics.box.map50:.4f} > 0.5")
    else:
        print(f"⚠️  mAP50 = {metrics.box.map50:.4f} < 0.5, may need more epochs or tuning")

    return results

if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "yolo12"
    train_kaggle_optimized(model_type)