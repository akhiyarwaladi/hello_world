#!/usr/bin/env python3
"""
Quick YOLOv8 Classification Training - Optimized for <30 minutes
"""

import os
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser(description="Quick YOLOv8 Classification Training")
    ap.add_argument("--data", default="data/classification", help="Classification dataset root")
    ap.add_argument("--epochs", type=int, default=10, help="Epochs (default 10)")
    ap.add_argument("--imgsz", type=int, default=64, help="Image size (default 64)")
    ap.add_argument("--batch", type=int, default=32, help="Batch size (default 32)")
    ap.add_argument("--device", default="cpu", help="Device, e.g., 'cpu' or 'cuda:0'")
    ap.add_argument("--name", default="quick_test", help="Run name")
    args = ap.parse_args()
    print("=" * 60)
    print("QUICK YOLOV8 MALARIA CLASSIFICATION TRAINING")
    print("=" * 60)

    # Configuration for quick training
    data_path = args.data

    # Check if data exists
    if not Path(data_path).exists():
        print(f"❌ Data path {data_path} not found!")
        return

    print(f"📁 Using data from: {data_path}")

    # Count images per class
    train_path = Path(data_path) / "train"
    for class_dir in train_path.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            print(f"   📊 {class_dir.name}: {count} images")

    # Initialize model with nano version for speed
    print("\n🚀 Loading YOLOv8n-cls (nano) model...")
    model = YOLO('yolov8n-cls.pt')  # nano version for speed

    # Quick training configuration
    print("\n⏱️  Starting QUICK training (optimized for <30 minutes)...")
    start_time = time.time()

    # Train with minimal settings for speed
    results = model.train(
        data=data_path,
        epochs=args.epochs,       # configurable
        imgsz=args.imgsz,
        batch=args.batch,
        workers=4,
        device=args.device,
        patience=3,
        save_period=-1,
        cache=True,
        project='results/classification',
        name=args.name,
        exist_ok=True,
        verbose=True,
        plots=True
    )

    end_time = time.time()
    training_time = end_time - start_time

    print("\n" + "=" * 60)
    print("🎉 QUICK TRAINING COMPLETED!")
    print("=" * 60)
    print(f"⏱️  Total training time: {training_time/60:.1f} minutes")
    print(f"📊 Best accuracy: {results.best_fitness:.3f}")
    print(f"📂 Results saved to: results/classification/{args.name}")

    # Show some results
    print("\n📈 Training Summary:")
    if hasattr(results, 'results_dict'):
        for key, value in results.results_dict.items():
            if 'accuracy' in key.lower():
                print(f"   {key}: {value:.3f}")

    print("\n✅ Quick training test completed successfully!")
    print("💡 You can now check the results and decide on longer training.")

if __name__ == "__main__":
    main()
