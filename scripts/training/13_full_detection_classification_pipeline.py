#!/usr/bin/env python3
"""
Full Detection → Classification Pipeline
Automatically generates crops from detection models and trains classification models
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import glob
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils.results_manager import ResultsManager

def find_detection_model(detection_type, experiment_pattern="*"):
    """Find trained detection model weights"""
    base_path = Path("results/current_experiments/training/detection")

    # Detection type mapping
    detection_dirs = {
        "yolo8": "yolov8_detection",
        "yolo10": "yolov10_detection",
        "yolo11": "yolo11_detection",
        "rtdetr": "rtdetr_detection"
    }

    if detection_type not in detection_dirs:
        return None

    detection_dir = detection_dirs[detection_type]
    pattern = base_path / detection_dir / experiment_pattern / "weights/best.pt"

    matches = glob.glob(str(pattern))
    if matches:
        return matches[0]  # Return first match
    return None

def generate_crops_from_detection(model_path, detection_type, output_dir):
    """Generate crops from detection model"""
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False

    cmd = [
        "python3", "scripts/10_crop_detections.py",
        "--model", model_path,
        "--input", "data/detection_multispecies",
        "--output", output_dir,
        "--confidence", "0.25",
        "--crop_size", "128",
        "--create_yolo_structure"
    ]

    print(f"🔄 Generating crops from {detection_type} detection...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Crops generated successfully to {output_dir}")
        return True
    else:
        print(f"❌ Failed to generate crops: {result.stderr}")
        return False

def train_classification_models(crop_data_path, detection_type, classification_models):
    """Train multiple classification models on generated crops"""
    results = {}

    for model_config in classification_models:
        model_type = model_config["type"]
        model_name = model_config["model"]
        script = model_config["script"]
        epochs = model_config.get("epochs", 10)
        batch = model_config.get("batch", 4)

        experiment_name = f"{detection_type}_det_to_{model_name.replace('-', '_')}_cls"

        cmd = [
            "NNPACK_DISABLE=1", "python3", script,
            "--data", crop_data_path,
            "--model", model_name,
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--device", "cpu",
            "--name", experiment_name
        ]

        print(f"🚀 Starting training: {experiment_name}")

        # Run in background
        process = subprocess.Popen(" ".join(cmd), shell=True)
        results[experiment_name] = {
            "process": process,
            "model_type": model_type,
            "detection_type": detection_type
        }

        time.sleep(5)  # Small delay between starts

    return results

def monitor_training(processes):
    """Monitor training processes"""
    print(f"\n📊 Monitoring {len(processes)} training processes...")

    while any(proc["process"].poll() is None for proc in processes.values()):
        time.sleep(30)  # Check every 30 seconds

        running = sum(1 for proc in processes.values() if proc["process"].poll() is None)
        completed = len(processes) - running

        print(f"🔄 Training status: {running} running, {completed} completed")

    print("✅ All training processes completed!")

def main():
    parser = argparse.ArgumentParser(description="Full Detection → Classification Pipeline")
    parser.add_argument("--detection_models", nargs="+",
                       choices=["yolo8", "yolo10", "yolo11", "rtdetr", "all"],
                       default=["all"],
                       help="Detection models to process")
    parser.add_argument("--classification_models", nargs="+",
                       choices=[
                           "yolo8",
                           "yolo11",
                           "resnet18",
                           "efficientnet",
                           "densenet121",
                           "mobilenet_v2",
                           "all"
                       ],
                       default=["all"],
                       help="Classification models to train")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor training processes")
    parser.add_argument("--skip_crop_generation", action="store_true",
                       help="Skip crop generation (assume already done)")

    args = parser.parse_args()

    print("=" * 80)
    print("FULL DETECTION → CLASSIFICATION PIPELINE")
    print("=" * 80)

    # Define classification models
    classification_configs = {
        "yolo8": {
            "type": "yolo",
            "script": "scripts/11_train_classification_crops.py",
            "model": "yolov8n-cls.pt",
            "epochs": 10,
            "batch": 4
        },
        "yolo11": {
            "type": "yolo",
            "script": "scripts/11_train_classification_crops.py",
            "model": "yolo11n-cls.pt",
            "epochs": 10,
            "batch": 4
        },
        "resnet18": {
            "type": "pytorch",
            "script": "scripts/11b_train_pytorch_classification.py",
            "model": "resnet18",
            "epochs": 10,
            "batch": 8
        },
        "efficientnet": {
            "type": "pytorch",
            "script": "scripts/11b_train_pytorch_classification.py",
            "model": "efficientnet_b0",
            "epochs": 10,
            "batch": 8
        },
        "densenet121": {
            "type": "pytorch",
            "script": "scripts/11b_train_pytorch_classification.py",
            "model": "densenet121",
            "epochs": 10,
            "batch": 8
        },
        "mobilenet_v2": {
            "type": "pytorch",
            "script": "scripts/11b_train_pytorch_classification.py",
            "model": "mobilenet_v2",
            "epochs": 10,
            "batch": 8
        }
    }

    # Expand "all" selections
    if "all" in args.detection_models:
        detection_models = ["yolo8", "yolo10", "yolo11", "rtdetr"]
    else:
        detection_models = args.detection_models

    if "all" in args.classification_models:
        selected_classification = list(classification_configs.keys())
    else:
        selected_classification = args.classification_models

    all_training_processes = {}

    # Process each detection model
    for detection_type in detection_models:
        print(f"\n🎯 Processing {detection_type.upper()} detection model...")

        # Find detection model
        model_path = find_detection_model(detection_type)
        if not model_path:
            print(f"❌ No trained {detection_type} detection model found")
            continue

        print(f"✅ Found model: {model_path}")

        # Generate crops
        crop_output_dir = f"data/crops_from_{detection_type}_detection"

        if not args.skip_crop_generation:
            success = generate_crops_from_detection(model_path, detection_type, crop_output_dir)
            if not success:
                continue
        else:
            print(f"⏭️  Skipping crop generation for {detection_type}")

        # Train classification models
        crop_data_path = f"{crop_output_dir}/yolo_classification"

        if Path(crop_data_path).exists():
            selected_configs = [classification_configs[name] for name in selected_classification]
            processes = train_classification_models(crop_data_path, detection_type, selected_configs)
            all_training_processes.update(processes)
        else:
            print(f"❌ Crop data not found: {crop_data_path}")

    # Monitor training if requested
    if args.monitor and all_training_processes:
        monitor_training(all_training_processes)
    elif all_training_processes:
        print(f"\n🚀 Started {len(all_training_processes)} training processes in background")
        print("Use --monitor flag to monitor training progress")

    print("\n🎉 Pipeline execution completed!")

    # Summary
    if all_training_processes:
        print(f"\n📋 Training Summary:")
        for name, info in all_training_processes.items():
            status = "✅ Completed" if info["process"].poll() is not None else "🔄 Running"
            print(f"   {status} {name}")

if __name__ == "__main__":
    main()
