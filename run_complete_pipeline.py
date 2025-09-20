#!/usr/bin/env python3
"""
COMPLETE PIPELINE: Detection Training → Crop Generation → Classification Training
This script ACTUALLY trains detection models first, then processes crops and classification
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run command with logging"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"✅ {description} - COMPLETED")
        return True
    else:
        print(f"❌ {description} - FAILED")
        return False

def main():
    parser = argparse.ArgumentParser(description="Complete Pipeline: Train Detection → Generate Crops → Train Classification")
    parser.add_argument("--detection", choices=["yolo8", "yolo11", "yolo12", "rtdetr"], required=True,
                       help="Detection model to train")
    parser.add_argument("--epochs-det", type=int, default=50,
                       help="Epochs for detection training")
    parser.add_argument("--epochs-cls", type=int, default=30,
                       help="Epochs for classification training")
    parser.add_argument("--experiment-name", default="auto_pipeline",
                       help="Base name for experiments")

    args = parser.parse_args()

    # Add timestamp to experiment name for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_exp_name = f"{args.experiment_name}_{timestamp}"

    print("🎯 COMPLETE DETECTION → CLASSIFICATION PIPELINE")
    print(f"Detection Model: {args.detection}")
    print(f"Detection Epochs: {args.epochs_det}")
    print(f"Classification Epochs: {args.epochs_cls}")
    print(f"Experiment Name: {base_exp_name}")

    # Model mapping
    detection_models = {
        "yolo8": "yolov8_detection",
        "yolo11": "yolov11_detection",
        "yolo12": "yolov12_detection",
        "rtdetr": "rtdetr_detection"
    }

    detection_model = detection_models[args.detection]
    det_exp_name = f"{base_exp_name}_{args.detection}_det"
    cls_exp_name = f"{base_exp_name}_{args.detection}_cls"

    # STAGE 1: Train Detection Model
    cmd1 = [
        "python", "pipeline.py", "train", detection_model,
        "--name", det_exp_name,
        "--epochs", str(args.epochs_det)
    ]

    if not run_command(cmd1, f"STAGE 1: Training {detection_model}"):
        return

    print(f"\n⏳ Waiting 5 seconds for model to be saved...")
    time.sleep(5)

    # STAGE 2: Generate Crops + Auto Fix Classification Structure
    model_path = f"results/current_experiments/training/detection/{detection_model}/{det_exp_name}/weights/best.pt"
    input_path = "data/integrated/yolo"
    output_path = f"data/crops_from_{args.detection}_{det_exp_name}"

    cmd2 = [
        "python", "scripts/training/10_crop_detections.py",
        "--model", model_path,
        "--input", input_path,
        "--output", output_path,
        "--confidence", "0.25",
        "--crop_size", "128",
        "--create_yolo_structure",
        "--fix_classification_structure"  # Auto fix 4-class structure
    ]

    if not run_command(cmd2, f"STAGE 2: Generating crops + fixing classification structure"):
        return

    # STAGE 3: Train Classification
    crop_data_path = f"data/crops_from_{args.detection}_{det_exp_name}/yolo_classification"

    cmd3 = [
        "python", "pipeline.py", "train", "yolov8_classification",
        "--name", cls_exp_name,
        "--epochs", str(args.epochs_cls),
        "--data", crop_data_path
    ]

    if not run_command(cmd3, f"STAGE 3: Training classification with 4-class crops"):
        return

    print(f"\n🎉 COMPLETE PIPELINE FINISHED!")
    print(f"✅ Detection Model: results/current_experiments/training/detection/{detection_model}/{det_exp_name}/")
    print(f"✅ Crop Data: {crop_data_path}/")
    print(f"✅ Classification Model: results/current_experiments/training/classification/yolov8_classification/{cls_exp_name}/")

if __name__ == "__main__":
    main()