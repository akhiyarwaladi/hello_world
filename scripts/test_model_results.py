#!/usr/bin/env python3
"""
Test trained model on test dataset
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch

def test_model():
    print("=" * 60)
    print("TESTING ULTRA-FAST TRAINED MODEL")
    print("=" * 60)

    # Path to the trained model
    model_path = "results/classification/ultra_fast_test/weights/best.pt"

    if not Path(model_path).exists():
        print(f"❌ Model not found at: {model_path}")
        return

    print(f"📁 Loading model from: {model_path}")

    # Load the trained model
    model = YOLO(model_path)

    # Use validation data for testing since no separate test set
    val_data_path = "data/classification/val"

    if not Path(val_data_path).exists():
        print(f"❌ Validation data not found at: {val_data_path}")
        return

    print(f"📊 Using validation dataset for testing: {val_data_path}")

    # Count validation images
    total_val_images = 0
    print("\n📈 Validation dataset distribution:")
    for class_dir in Path(val_data_path).iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            total_val_images += count
            print(f"   📊 {class_dir.name}: {count} images")

    print(f"   🔢 Total validation images: {total_val_images}")

    if total_val_images == 0:
        print("❌ No validation images found!")
        return

    print(f"\n🧪 Running model validation on validation dataset...")

    # Run validation on validation set
    results = model.val(
        data="data/classification",
        split="val",
        imgsz=32,  # Same size as training
        batch=16,
        verbose=True
    )

    print(f"\n🔍 Testing individual predictions on sample images...")

    # Test individual predictions
    sample_predictions = []
    for class_dir in Path(val_data_path).iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))[:3]  # Take 3 samples per class
            for img in images:
                try:
                    pred_results = model.predict(str(img), imgsz=32, verbose=False)
                    if pred_results:
                        pred_result = pred_results[0]
                        if hasattr(pred_result, 'probs') and pred_result.probs is not None:
                            top1_conf = float(pred_result.probs.top1conf)
                            top1_class = int(pred_result.probs.top1)
                            class_names = pred_result.names
                            predicted_class = class_names[top1_class]
                            actual_class = class_dir.name

                            sample_predictions.append({
                                'image': img.name,
                                'actual': actual_class,
                                'predicted': predicted_class,
                                'confidence': top1_conf,
                                'correct': actual_class == predicted_class
                            })
                except Exception as e:
                    print(f"⚠️ Error predicting {img.name}: {e}")
                    continue

    # Show sample predictions
    print(f"\n📋 Sample Predictions (showing first 15):")
    print("=" * 80)
    correct_predictions = 0
    for i, pred in enumerate(sample_predictions[:15]):
        status = "✅" if pred['correct'] else "❌"
        print(f"{status} {pred['image']:<20} | Actual: {pred['actual']:<15} | Predicted: {pred['predicted']:<15} | Conf: {pred['confidence']:.3f}")
        if pred['correct']:
            correct_predictions += 1

    if sample_predictions:
        sample_accuracy = correct_predictions / len(sample_predictions[:15])
        print(f"\n📊 Sample accuracy: {sample_accuracy:.3f} ({sample_accuracy*100:.1f}%)")
        print(f"🔢 Correct predictions: {correct_predictions}/{len(sample_predictions[:15])}")

    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS")
    print("=" * 60)

    # Print detailed results
    if hasattr(results, 'top1'):
        print(f"🏆 Top-1 Accuracy: {results.top1:.3f} ({results.top1*100:.1f}%)")

    if hasattr(results, 'top5'):
        print(f"🥇 Top-5 Accuracy: {results.top5:.3f} ({results.top5*100:.1f}%)")

    # Print per-class results if available
    if hasattr(results, 'results_dict'):
        print(f"\n📊 Detailed metrics:")
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.3f}")

    print(f"\n⚡ Model info:")
    print(f"   📱 Model size: {Path(model_path).stat().st_size / (1024*1024):.1f} MB")
    print(f"   🔧 Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"   🎯 Image size: 32x32 pixels")

    print(f"\n✅ Model testing completed!")
    return results

if __name__ == "__main__":
    test_model()