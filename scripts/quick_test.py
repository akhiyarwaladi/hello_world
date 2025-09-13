#!/usr/bin/env python3
"""
Quick test of trained model - focusing on sample predictions
"""

import os
from pathlib import Path
from ultralytics import YOLO
import random

def quick_test():
    print("=" * 60)
    print("QUICK TEST OF ULTRA-FAST TRAINED MODEL")
    print("=" * 60)

    # Path to the trained model
    model_path = "results/classification/ultra_fast_test/weights/best.pt"

    if not Path(model_path).exists():
        print(f"❌ Model not found at: {model_path}")
        return

    print(f"📁 Loading model from: {model_path}")

    # Load the trained model
    model = YOLO(model_path)

    # Test on validation data
    val_data_path = "data/classification/val"

    print(f"📊 Using validation dataset for testing: {val_data_path}")

    # Count validation images per class
    class_counts = {}
    for class_dir in Path(val_data_path).iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            class_counts[class_dir.name] = count
            print(f"   📊 {class_dir.name}: {count} images")

    print(f"   🔢 Total validation images: {sum(class_counts.values())}")

    print(f"\n🔍 Testing individual predictions on sample images...")

    # Test individual predictions (sample from each class)
    sample_predictions = []
    total_samples = 0

    for class_dir in Path(val_data_path).iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))
            # Take 5 random samples per class, or all if less than 5
            sample_count = min(5, len(images))
            samples = random.sample(images, sample_count) if len(images) > sample_count else images

            print(f"   🧪 Testing {len(samples)} samples from {class_dir.name}")

            for img in samples:
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
                            total_samples += 1

                except Exception as e:
                    print(f"⚠️ Error predicting {img.name}: {e}")
                    continue

    # Show sample predictions
    print(f"\n📋 Individual Predictions (Total: {len(sample_predictions)}):")
    print("=" * 90)
    correct_predictions = 0

    # Group by actual class for better overview
    by_class = {}
    for pred in sample_predictions:
        actual = pred['actual']
        if actual not in by_class:
            by_class[actual] = []
        by_class[actual].append(pred)

    for class_name, preds in by_class.items():
        print(f"\n🏷️  Class: {class_name}")
        print("-" * 50)
        class_correct = 0
        for pred in preds:
            status = "✅" if pred['correct'] else "❌"
            print(f"{status} {pred['image']:<20} | Predicted: {pred['predicted']:<15} | Conf: {pred['confidence']:.3f}")
            if pred['correct']:
                correct_predictions += 1
                class_correct += 1

        class_accuracy = class_correct / len(preds) if preds else 0
        print(f"   📊 Class accuracy: {class_accuracy:.3f} ({class_accuracy*100:.1f}%) - {class_correct}/{len(preds)}")

    print(f"\n" + "=" * 60)
    print("🎯 QUICK TEST RESULTS")
    print("=" * 60)

    if sample_predictions:
        overall_accuracy = correct_predictions / len(sample_predictions)
        print(f"🏆 Overall Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        print(f"🔢 Correct predictions: {correct_predictions}/{len(sample_predictions)}")
        print(f"📊 Test samples per class: ~5 images")

        # Show confusion patterns
        confusion = {}
        for pred in sample_predictions:
            if not pred['correct']:
                key = f"{pred['actual']} → {pred['predicted']}"
                confusion[key] = confusion.get(key, 0) + 1

        if confusion:
            print(f"\n🔄 Most common misclassifications:")
            sorted_confusion = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
            for mistake, count in sorted_confusion[:5]:
                print(f"   {mistake}: {count} times")

    print(f"\n⚡ Model Info:")
    print(f"   📱 Model size: {Path(model_path).stat().st_size / (1024*1024):.1f} MB")
    print(f"   🎯 Training: Ultra-fast (1.7 minutes)")
    print(f"   🔧 Image size: 32x32 pixels")
    print(f"   📚 Classes: 6 malaria species + uninfected")

    print(f"\n✅ Quick test completed!")

if __name__ == "__main__":
    quick_test()