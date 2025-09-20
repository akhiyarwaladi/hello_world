#!/usr/bin/env python3
"""
Train Classification Model on Cropped Parasites
"""

import os
import sys
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils.results_manager import ResultsManager

def main():
    parser = argparse.ArgumentParser(description="Train Classification on Cropped Parasites")
    parser.add_argument("--data", default="data/classification_crops",
                       help="Classification dataset root")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=128,
                       help="Image size for training")
    parser.add_argument("--batch", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--device", default="cpu",
                       help="Device (cpu or cuda)")
    parser.add_argument("--model", default="yolov8n-cls.pt",
                       help="YOLOv8 classification model")
    parser.add_argument("--name", default="yolov8_parasite_classification",
                       help="Experiment name")

    args = parser.parse_args()

    print("=" * 60)
    print("YOLOV8 PARASITE CLASSIFICATION TRAINING")
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
        model_name="yolov8_classification",
        experiment_name=args.name
    )

    # Check if data exists
    if not Path(args.data).exists():
        print(f"❌ Dataset directory not found: {args.data}")
        return

    print(f"📁 Using dataset: {args.data}")
    print(f"🎯 Model: {args.model}")
    print(f"📊 Epochs: {args.epochs}")
    print(f"🖼️  Image size: {args.imgsz}")
    print(f"📦 Batch size: {args.batch}")
    print(f"💻 Device: {args.device}")

    # Count images per split
    train_path = Path(args.data) / "train" / "parasite"
    val_path = Path(args.data) / "val" / "parasite"
    test_path = Path(args.data) / "test" / "parasite"

    def count_images(path: Path) -> int:
        supported = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        return sum(len(list(path.glob(pattern))) for pattern in supported)

    train_count = count_images(train_path)
    val_count = count_images(val_path)
    test_count = count_images(test_path)

    print(f"\\n📊 Dataset composition:")
    print(f"   Train: {train_count} images")
    print(f"   Val: {val_count} images")
    print(f"   Test: {test_count} images")

    # Load model
    print(f"\\n🚀 Loading {args.model} model...")
    model = YOLO(args.model)

    # Start training
    print("\\n⏱️  Starting classification training...")
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
    print("🎉 PARASITE CLASSIFICATION TRAINING COMPLETED!")
    print("=" * 60)
    print(f"⏱️  Training time: {training_time/60:.1f} minutes")
    print(f"📂 Results saved to: {experiment_path}")

    # Show training results
    if hasattr(results, 'results_dict'):
        print("\\n📈 Training Results:")
        for key, value in results.results_dict.items():
            if any(metric in key.lower() for metric in ['accuracy', 'loss']):
                print(f"   {key}: {value:.4f}")

    # Get best model path
    best_model = experiment_path / "weights/best.pt"
    if best_model.exists():
        print(f"\\n✅ Best model saved: {best_model}")

        # Test on test set
        print("\\n🧪 Running test evaluation...")
        test_results = model.val(data=args.data, split='test')

        print("\\n📊 Test Metrics:")
        if hasattr(test_results, 'top1acc'):
            print(f"   Top-1 Accuracy: {test_results.top1acc:.4f}")
        if hasattr(test_results, 'top5acc'):
            print(f"   Top-5 Accuracy: {test_results.top5acc:.4f}")

    print("\\n✅ Parasite classification training completed successfully!")
    print("\\n🎯 Results:")
    print(f"   - Trained on {train_count} cropped parasites")
    print(f"   - Validated on {val_count} cropped parasites")
    print(f"   - Tested on {test_count} cropped parasites")

if __name__ == "__main__":
    main()
