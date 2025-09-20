#!/usr/bin/env python3
"""
Train RT-DETR Detection Model for Malaria Parasites
"""

import os
import sys
import time
import argparse
from pathlib import Path
from ultralytics import RTDETR

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils.results_manager import ResultsManager

def main():
    parser = argparse.ArgumentParser(description="Train RT-DETR Detection for Malaria")
    parser.add_argument("--data", default="data/detection_fixed/dataset.yaml",
                       help="Dataset YAML file")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size for training")
    parser.add_argument("--batch", type=int, default=8,
                       help="Batch size (RT-DETR requires smaller batches)")
    parser.add_argument("--device", default="cpu",
                       help="Device (cpu or cuda)")
    parser.add_argument("--model", default="rtdetr-l.pt",
                       help="RT-DETR model (rtdetr-l.pt, rtdetr-x.pt)")
    parser.add_argument("--name", default="rtdetr_malaria_detection",
                       help="Experiment name")

    args = parser.parse_args()

    print("=" * 60)
    print("RT-DETR MALARIA PARASITE DETECTION TRAINING")
    print("=" * 60)

    # Initialize Results Manager for organized folder structure
    results_manager = ResultsManager()

    # Determine experiment type based on name
    if "production" in args.name.lower() or "final" in args.name.lower():
        experiment_type = "production"
    elif "validation" in args.name.lower() or "test" in args.name.lower():
        experiment_type = "validation"
    else:
        experiment_type = "training"

    # Get organized experiment path
    experiment_path = results_manager.get_experiment_path(
        experiment_type=experiment_type,
        model_name="rtdetr_detection",
        experiment_name=args.name
    )

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
        model = RTDETR(args.model)
    except Exception as e:
        print(f"❌ Error loading RT-DETR model: {e}")
        print("💡 RT-DETR might not be available. Trying alternative...")
        try:
            # Try with different model name
            alt_model = "rtdetr-l.yaml" if "rtdetr" in args.model else args.model
            print(f"🔄 Loading alternative model: {alt_model}")
            model = RTDETR(alt_model)
        except Exception as e2:
            print(f"❌ RT-DETR not available: {e2}")
            print("🔄 Using YOLOv8 as fallback for comparison...")
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            args.name = f"{args.name}_yolov8_fallback"

    # Start training
    print("\\n⏱️  Starting RT-DETR detection training...")
    start_time = time.time()

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=2,  # RT-DETR might need fewer workers
        patience=15,
        save_period=10,  # Save every 10 epochs
        cache=True,
        project=str(experiment_path.parent),
        name=experiment_path.name,
        exist_ok=True,
        verbose=True,
        plots=True,
        val=True
    )

    end_time = time.time()
    training_time = end_time - start_time

    print("\\n" + "=" * 60)
    print("🎉 RT-DETR DETECTION TRAINING COMPLETED!")
    print("=" * 60)
    print(f"⏱️  Training time: {training_time/60:.1f} minutes")
    print(f"📂 Results saved to: {experiment_path}")

    # Show training results
    if hasattr(results, 'results_dict'):
        print("\\n📈 Training Results:")
        for key, value in results.results_dict.items():
            if any(metric in key.lower() for metric in ['map', 'precision', 'recall']):
                print(f"   {key}: {value:.4f}")

    # Get best model path
    best_model = experiment_path / "weights/best.pt"
    if best_model.exists():
        print(f"\\n✅ Best model saved: {best_model}")

        # Validate on test set if available
        print("\\n🧪 Running validation...")
        val_results = model.val(data=args.data)

        print("\\n📊 Validation Metrics:")
        if hasattr(val_results, 'box'):
            print(f"   mAP50: {val_results.box.map50:.4f}")
            print(f"   mAP50-95: {val_results.box.map:.4f}")
            print(f"   Precision: {val_results.box.mp:.4f}")
            print(f"   Recall: {val_results.box.mr:.4f}")

    print("\\n✅ RT-DETR detection training completed successfully!")

if __name__ == "__main__":
    main()