#!/usr/bin/env python3
"""
Ultra Fast YOLOv8 Training - Optimized for <10 minutes
Uses subset of data for rapid prototyping and testing
"""

import os
import time
import shutil
from pathlib import Path
from ultralytics import YOLO

def create_mini_dataset(source_path, target_path, samples_per_class=50):
    """Create a mini dataset with limited samples per class"""
    print(f"🔄 Creating mini dataset with {samples_per_class} samples per class...")

    source_path = Path(source_path)
    target_path = Path(target_path)

    # Create target directories
    for split in ['train', 'val']:
        split_source = source_path / split
        split_target = target_path / split
        split_target.mkdir(parents=True, exist_ok=True)

        if not split_source.exists():
            continue

        for class_dir in split_source.iterdir():
            if not class_dir.is_dir():
                continue

            class_target = split_target / class_dir.name
            class_target.mkdir(exist_ok=True)

            # Get limited number of images
            images = list(class_dir.glob("*.jpg"))[:samples_per_class]

            for img in images:
                shutil.copy2(img, class_target / img.name)

            print(f"   📊 {class_dir.name}: copied {len(images)} images")

def main():
    print("=" * 60)
    print("ULTRA FAST YOLOV8 MALARIA CLASSIFICATION TRAINING")
    print("=" * 60)

    # Configuration for ultra-fast training
    source_data = "data/classification"
    mini_data = "data/mini_classification"

    # Check if source data exists
    if not Path(source_data).exists():
        print(f"❌ Source data {source_data} not found!")
        return

    # Create mini dataset (much smaller)
    create_mini_dataset(source_data, mini_data, samples_per_class=25)

    print(f"📁 Using mini dataset: {mini_data}")

    # Count mini dataset
    train_path = Path(mini_data) / "train"
    total_images = 0
    for class_dir in train_path.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            total_images += count
            print(f"   📊 {class_dir.name}: {count} images")

    print(f"   🔢 Total training images: {total_images}")

    # Initialize model with nano version for speed
    print("\n🚀 Loading YOLOv8n-cls (nano) model...")
    model = YOLO('yolov8n-cls.pt')

    # Ultra-fast training configuration
    print("\n⚡ Starting ULTRA-FAST training (optimized for <10 minutes)...")
    start_time = time.time()

    # Train with minimal settings for maximum speed
    results = model.train(
        data=mini_data,
        epochs=5,              # Very few epochs
        imgsz=32,              # Tiny image size
        batch=16,              # Smaller batch for speed
        workers=2,             # Fewer workers
        device='cpu',          # Use CPU
        patience=2,            # Early stopping
        save_period=-1,        # Don't save intermediate models
        cache=False,           # No caching to avoid memory issues
        project='results/classification',
        name='ultra_fast_test',
        exist_ok=True,
        verbose=True,
        plots=False,           # Skip plots for speed
        val=False,             # Skip validation during training
        lr0=0.01,              # Higher learning rate for faster convergence
        warmup_epochs=0        # No warmup
    )

    end_time = time.time()
    training_time = end_time - start_time

    print("\n" + "=" * 60)
    print("🎉 ULTRA-FAST TRAINING COMPLETED!")
    print("=" * 60)
    print(f"⏱️  Total training time: {training_time/60:.1f} minutes")
    print(f"📊 Final loss: {results.trainer.loss:.3f}" if hasattr(results, 'trainer') else "")
    print(f"📂 Results saved to: results/classification/ultra_fast_test")

    # Quick validation on a few samples
    print("\n🧪 Quick validation test...")
    try:
        val_results = model.val(data=mini_data, split='val', verbose=False)
        if hasattr(val_results, 'top1'):
            print(f"📈 Top-1 Accuracy: {val_results.top1:.3f}")
        if hasattr(val_results, 'top5'):
            print(f"📈 Top-5 Accuracy: {val_results.top5:.3f}")
    except Exception as e:
        print(f"⚠️ Validation failed: {e}")

    print("\n✅ Ultra-fast training completed successfully!")
    print(f"💡 Training time: {training_time/60:.1f} minutes with {total_images} images")
    print("🚀 Ready for quick results analysis!")

    # Cleanup mini dataset to save space
    print("\n🧹 Cleaning up mini dataset...")
    try:
        shutil.rmtree(mini_data)
        print("✅ Mini dataset cleaned up")
    except:
        print("⚠️ Could not clean up mini dataset")

if __name__ == "__main__":
    main()