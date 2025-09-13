#!/usr/bin/env python3
"""
Train YOLOv11 Detection Model for Malaria Parasites
"""

import os
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 Detection for Malaria")
    parser.add_argument("--data", default="data/detection_fixed/dataset.yaml",
                       help="Dataset YAML file")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size for training")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--device", default="cpu",
                       help="Device (cpu or cuda)")
    parser.add_argument("--model", default="yolo11n.pt",
                       help="YOLOv11 model (yolo11n.pt, yolo11s.pt, yolo11m.pt)")
    parser.add_argument("--name", default="yolo11_malaria_detection",
                       help="Experiment name")

    args = parser.parse_args()

    print("=" * 60)
    print("YOLOV11 MALARIA PARASITE DETECTION TRAINING")
    print("=" * 60)

    # Check if data exists
    if not Path(args.data).exists():
        print(f"❌ Dataset file not found: {args.data}")
        return

    print(f"📁 Using dataset: {args.data}")
    print(f"🎯 Model: {args.model}")
    print(f"📊 Epochs: {args.epochs}")
    print(f"🖼️  Image size: {args.imgsz}")
    print(f"📦 Batch size: {args.batch}")
    print(f"💻 Device: {args.device}")

    # Load model
    print(f"\\n🚀 Loading {args.model} model...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"❌ Error loading YOLOv11 model: {e}")
        print("💡 YOLOv11 might not be available yet. Using YOLOv8 instead...")
        model_fallback = args.model.replace("yolo11", "yolov8")
        print(f"🔄 Loading fallback model: {model_fallback}")
        model = YOLO(model_fallback)

    # Start training
    print("\\n⏱️  Starting YOLOv11 detection training...")
    start_time = time.time()

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        patience=10,
        save_period=10,  # Save every 10 epochs
        cache=True,
        project='results/detection',
        name=args.name,
        exist_ok=True,
        verbose=True,
        plots=True,
        val=True
    )

    end_time = time.time()
    training_time = end_time - start_time

    print("\\n" + "=" * 60)
    print("🎉 YOLOV11 DETECTION TRAINING COMPLETED!")
    print("=" * 60)
    print(f"⏱️  Training time: {training_time/60:.1f} minutes")
    print(f"📂 Results saved to: results/detection/{args.name}")

    # Show training results
    if hasattr(results, 'results_dict'):
        print("\\n📈 Training Results:")
        for key, value in results.results_dict.items():
            if any(metric in key.lower() for metric in ['map', 'precision', 'recall']):
                print(f"   {key}: {value:.4f}")

    # Get best model path
    best_model = Path(f"results/detection/{args.name}/weights/best.pt")
    if best_model.exists():
        print(f"\\n✅ Best model saved: {best_model}")

        # Validate on test set if available
        print("\\n🧪 Running validation...")
        val_results = model.val(data=args.data)

        print("\\n📊 Validation Metrics:")
        print(f"   mAP50: {val_results.box.map50:.4f}")
        print(f"   mAP50-95: {val_results.box.map:.4f}")
        print(f"   Precision: {val_results.box.mp:.4f}")
        print(f"   Recall: {val_results.box.mr:.4f}")

    print("\\n✅ YOLOv11 detection training completed successfully!")

if __name__ == "__main__":
    main()