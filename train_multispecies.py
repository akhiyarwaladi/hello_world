#!/usr/bin/env python3
"""
Master Training Script for Multi-Species Malaria Detection
Clean, organized training with minimal warnings
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Disable NNPACK warnings and other noise
os.environ['NNPACK_DISABLE'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:torch'
os.environ['OMP_NUM_THREADS'] = '4'

def run_yolov8_detection_training():
    """Train YOLOv8 detection model with multi-species dataset"""
    print("🎯 Starting YOLOv8 Multi-Species Detection Training")
    print("=" * 50)

    cmd = [
        'python', 'scripts/train_yolo_detection.py',
        '--data', 'data/detection_multispecies/dataset.yaml',
        '--epochs', '30',
        '--batch', '8',
        '--device', 'cpu',
        '--project', 'results/pipeline_final',
        '--name', 'yolov8_detection'
    ]

    subprocess.run(cmd, env=os.environ)

def run_yolov11_detection_training():
    """Train YOLOv11 detection model with multi-species dataset"""
    print("🎯 Starting YOLOv11 Multi-Species Detection Training")
    print("=" * 50)

    cmd = [
        'yolo', 'detect', 'train',
        'data=data/detection_multispecies/dataset.yaml',
        'model=yolo11n.pt',
        'epochs=30',
        'batch=8', 
        'device=cpu',
        'project=results/pipeline_final',
        'name=yolov11_detection'
    ]

    subprocess.run(cmd, env=os.environ)

def run_rtdetr_detection_training():
    """Train RT-DETR detection model with multi-species dataset"""
    print("🎯 Starting RT-DETR Multi-Species Detection Training")
    print("=" * 50)

    cmd = [
        'yolo', 'detect', 'train',
        'data=data/detection_multispecies/dataset.yaml', 
        'model=rtdetr-l.pt',
        'epochs=30',
        'batch=4',
        'device=cpu',
        'project=results/pipeline_final',
        'name=rtdetr_detection'
    ]

    subprocess.run(cmd, env=os.environ)

def run_yolov8_classification_training():
    """Train YOLOv8 classification model with multi-species dataset"""
    print("📊 Starting YOLOv8 Multi-Species Classification Training")
    print("=" * 50)

    cmd = [
        'python', 'scripts/train_classification_crops.py',
        '--data', 'data/classification_multispecies',
        '--epochs', '25',
        '--batch', '8',
        '--device', 'cpu',
        '--project', 'results/pipeline_final',
        '--name', 'yolov8_classification'
    ]

    subprocess.run(cmd, env=os.environ)

def run_yolov11_classification_training():
    """Train YOLOv11 classification model with multi-species dataset"""
    print("📊 Starting YOLOv11 Multi-Species Classification Training")
    print("=" * 50)

    cmd = [
        'yolo', 'classify', 'train',
        'data=data/classification_multispecies',
        'model=yolo11n-cls.pt',
        'epochs=25',
        'batch=8',
        'device=cpu',
        'project=results/pipeline_final',
        'name=yolov11_classification'
    ]

    subprocess.run(cmd, env=os.environ)

def test_model_quick(model_name, command, test_epochs=2, timeout=300):
    """Quick test of a model with minimal epochs"""
    print(f"\n🧪 Testing {model_name} (2 epochs)...")

    # Modify command for testing
    test_cmd = []
    for param in command:
        if param.startswith('epochs='):
            test_cmd.append(f'epochs={test_epochs}')
        elif param == '--epochs':
            test_cmd.append(param)
        elif param in ['30', '25'] and test_cmd and test_cmd[-1] == '--epochs':
            test_cmd.append(str(test_epochs))
        elif param.startswith('name='):
            test_cmd.append(f"name=test_{param.split('=')[1]}")
        elif test_cmd and test_cmd[-1] == '--name':
            test_cmd.append(f"test_{param}")
        else:
            test_cmd.append(param)

    start_time = time.time()
    try:
        result = subprocess.run(test_cmd, env=os.environ, timeout=timeout,
                              capture_output=True, text=True)
        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {model_name} - SUCCESS ({duration:.1f}s)")
            return True
        else:
            print(f"❌ {model_name} - FAILED ({duration:.1f}s)")
            print(f"Error: {result.stderr[-200:] if result.stderr else 'No error details'}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} - TIMEOUT ({timeout}s)")
        return False
    except Exception as e:
        print(f"💥 {model_name} - ERROR: {str(e)}")
        return False

def run_model_validation():
    """Test all models with quick validation"""
    print("🧪 MODEL VALIDATION TEST")
    print("=" * 50)
    print("Testing all models with 2 epochs to verify functionality")
    print("This will take approximately 10-15 minutes")
    print("=" * 50)

    # Check prerequisites
    detection_yaml = Path("data/detection_multispecies/dataset.yaml")
    classification_dir = Path("data/classification_multispecies")

    if not detection_yaml.exists():
        print(f"❌ Detection dataset not found: {detection_yaml}")
        return False
    if not classification_dir.exists():
        print(f"❌ Classification dataset not found: {classification_dir}")
        return False

    print("✅ Prerequisites check passed")

    # Auto-proceed in test mode - no interactive input needed
    print("\n🚀 Starting automatic model validation...")

    # Test configurations with organized output
    models_to_test = [
        ("YOLOv8 Detection", [
            'python', 'scripts/train_yolo_detection.py',
            '--data', 'data/detection_multispecies/dataset.yaml',
            '--epochs', '2', '--batch', '4', '--device', 'cpu', 
            '--project', 'results/pipeline_final/validation',
            '--name', 'test_yolov8_detection'
        ]),
        ("YOLOv11 Detection", [
            'yolo', 'detect', 'train', 'data=data/detection_multispecies/dataset.yaml',
            'model=yolo11n.pt', 'epochs=2', 'batch=4', 'device=cpu',
            'project=results/pipeline_final/validation',
            'name=test_yolov11_detection'
        ]),
        ("RT-DETR Detection", [
            'yolo', 'detect', 'train', 'data=data/detection_multispecies/dataset.yaml',
            'model=rtdetr-l.pt', 'epochs=2', 'batch=2', 'device=cpu',
            'project=results/pipeline_final/validation',
            'name=test_rtdetr_detection'
        ]),
        ("YOLOv8 Classification", [
            'python', 'scripts/train_classification_crops.py',
            '--data', 'data/classification_multispecies', '--epochs', '2',
            '--batch', '4', '--device', 'cpu',
            '--project', 'results/pipeline_final/validation',
            '--name', 'test_yolov8_classification'
        ]),
        ("YOLOv11 Classification", [
            'yolo', 'classify', 'train', 'data=data/classification_multispecies',
            'model=yolo11n-cls.pt', 'epochs=2', 'batch=4', 'device=cpu',
            'project=results/pipeline_final/validation',
            'name=test_yolov11_classification'
        ])
    ]

    # Run tests
    results = []
    print(f"\n🚀 Testing {len(models_to_test)} models...")

    for model_name, command in models_to_test:
        success = test_model_quick(model_name, command)
        results.append((model_name, success))

    # Summary
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"\n📊 VALIDATION RESULTS:")
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}")

    if successful:
        print("\n🎉 WORKING MODELS:")
        for name, _ in successful:
            print(f"   ✅ {name}")

    if failed:
        print("\n💥 FAILED MODELS:")
        for name, _ in failed:
            print(f"   ❌ {name}")

    if len(successful) == len(results):
        print("\n🎉 All models working! Pipeline is ready for any model selection.")
        print("📁 Results saved to: results/pipeline_final/validation/")
    else:
        print("\n⚠️  Some models failed. Check error messages above.")

    return len(failed) == 0

def main():
    print("🚀 MULTI-SPECIES MALARIA DETECTION TRAINING")
    print("=" * 60)
    print("Available models: YOLOv8, YOLOv11, RT-DETR")
    print("Dataset: 4 species (falciparum, vivax, malariae, ovale)")
    print("Classification: 97.4% accuracy ✅ COMPLETED")
    print("=" * 60)

    if len(sys.argv) > 1:
        if sys.argv[1] == 'yolov8_detection':
            run_yolov8_detection_training()
        elif sys.argv[1] == 'yolov11_detection':
            run_yolov11_detection_training()
        elif sys.argv[1] == 'rtdetr_detection':
            run_rtdetr_detection_training()
        elif sys.argv[1] == 'yolov8_classification':
            run_yolov8_classification_training()
        elif sys.argv[1] == 'yolov11_classification':
            run_yolov11_classification_training()
        elif sys.argv[1] == 'test' or sys.argv[1] == 'validate':
            run_model_validation()
        else:
            print("Usage: python train_multispecies.py [yolov8_detection|yolov11_detection|rtdetr_detection|yolov8_classification|yolov11_classification|test]")
    else:
        print("Select training mode:")
        print("=" * 40)
        print("🎯 DETECTION MODELS:")
        print("1. YOLOv8 Detection (current: running)")
        print("2. YOLOv11 Detection (NEW)")
        print("3. RT-DETR Detection (NEW)")
        print("")
        print("📊 CLASSIFICATION MODELS:")
        print("4. YOLOv8 Classification (97.4% ✅ COMPLETED)")
        print("5. YOLOv11 Classification (NEW)")
        print("")
        print("🔥 COMPARISON:")
        print("6. All Detection Models (v8 + v11 + RT-DETR)")
        print("7. All Classification Models (v8 + v11)")
        print("")
        print("🧪 TESTING:")
        print("8. Test All Models (quick validation)")
        print("=" * 40)

        choice = input("Enter choice (1-8): ").strip()

        if choice == '1':
            run_yolov8_detection_training()
        elif choice == '2':
            run_yolov11_detection_training()
        elif choice == '3':
            run_rtdetr_detection_training()
        elif choice == '4':
            print("✅ YOLOv8 Classification already completed with 97.4% accuracy!")
            print("Results available at: results/pipeline_final/multispecies_classification/")
        elif choice == '5':
            run_yolov11_classification_training()
        elif choice == '6':
            print("🔥 Training all detection models...")
            run_yolov11_detection_training()  # YOLOv8 already running
            print("\n" + "="*60)
            run_rtdetr_detection_training()
        elif choice == '7':
            print("📊 Training all classification models...")
            print("✅ YOLOv8 already completed (97.4% accuracy)")
            run_yolov11_classification_training()
        elif choice == '8':
            run_model_validation()
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()