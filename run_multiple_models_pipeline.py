#!/usr/bin/env python3
"""
MULTIPLE MODELS PIPELINE: Train multiple detection models (with exclusions)
Runs complete pipeline for selected detection models, excluding specified ones
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
    parser = argparse.ArgumentParser(description="Multiple Models Pipeline: Train multiple detection models → Generate Crops → Train Classification")
    parser.add_argument("--include", nargs="+",
                       choices=["yolo8", "yolo11", "yolo12", "rtdetr"],
                       help="Detection models to include (if not specified, includes all)")
    parser.add_argument("--exclude", nargs="+",
                       choices=["yolo8", "yolo11", "yolo12", "rtdetr"],
                       default=[],
                       help="Detection models to exclude")
    parser.add_argument("--epochs-det", type=int, default=50,
                       help="Epochs for detection training")
    parser.add_argument("--epochs-cls", type=int, default=30,
                       help="Epochs for classification training")
    parser.add_argument("--experiment-name", default="multi_pipeline",
                       help="Base name for experiments")

    args = parser.parse_args()

    # Determine which models to run
    all_models = ["yolo8", "yolo11", "yolo12", "rtdetr"]

    if args.include:
        models_to_run = args.include
    else:
        models_to_run = all_models

    # Remove excluded models
    models_to_run = [model for model in models_to_run if model not in args.exclude]

    if not models_to_run:
        print("❌ No models to run after exclusions!")
        return

    # Add timestamp to experiment name for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_exp_name = f"{args.experiment_name}_{timestamp}"

    print("🎯 MULTIPLE MODELS DETECTION → CLASSIFICATION PIPELINE")
    print(f"Models to run: {', '.join(models_to_run)}")
    print(f"Excluded models: {', '.join(args.exclude) if args.exclude else 'None'}")
    print(f"Detection Epochs: {args.epochs_det}")
    print(f"Classification Epochs: {args.epochs_cls}")
    print(f"Experiment Base Name: {base_exp_name}")

    # Model mapping
    detection_models = {
        "yolo8": "yolov8_detection",
        "yolo11": "yolov11_detection",
        "yolo12": "yolov12_detection",
        "rtdetr": "rtdetr_detection"
    }

    successful_models = []
    failed_models = []

    for model_key in models_to_run:
        print(f"\n{'🎯'*20}")
        print(f"🎯 STARTING PIPELINE FOR {model_key.upper()}")
        print(f"{'🎯'*20}")

        detection_model = detection_models[model_key]
        det_exp_name = f"{base_exp_name}_{model_key}_det"
        cls_exp_name = f"{base_exp_name}_{model_key}_cls"

        # STAGE 1: Train Detection Model
        print(f"\n📊 STAGE 1: Training {detection_model}")
        cmd1 = [
            "python3", "pipeline.py", "train", detection_model,
            "--name", det_exp_name,
            "--epochs", str(args.epochs_det)
        ]

        if not run_command(cmd1, f"STAGE 1: Training {detection_model}"):
            print(f"❌ Detection training failed for {model_key}")
            failed_models.append(f"{model_key} (detection)")
            continue

        print(f"\n⏳ Waiting 5 seconds for model to be saved...")
        time.sleep(5)

        # Check if weights were saved
        model_path = f"results/current_experiments/training/detection/{detection_model}/{det_exp_name}/weights/best.pt"
        if not Path(model_path).exists():
            print(f"❌ Weights not found for {model_key}: {model_path}")
            failed_models.append(f"{model_key} (weights missing)")
            continue

        # STAGE 2: Generate Crops + Auto Fix Classification Structure
        print(f"\n🔄 STAGE 2: Generating crops for {model_key}")
        input_path = "data/integrated/yolo"
        output_path = f"data/crops_from_{model_key}_{det_exp_name}"

        cmd2 = [
            "python3", "scripts/training/10_crop_detections.py",
            "--model", model_path,
            "--input", input_path,
            "--output", output_path,
            "--confidence", "0.25",
            "--crop_size", "128",
            "--create_yolo_structure",
            "--fix_classification_structure"
        ]

        if not run_command(cmd2, f"STAGE 2: Generating crops for {model_key}"):
            print(f"❌ Crop generation failed for {model_key}")
            failed_models.append(f"{model_key} (crops)")
            continue

        # STAGE 3: Train Classification
        print(f"\n📈 STAGE 3: Training classification for {model_key}")
        crop_data_path = f"data/crops_from_{model_key}_{det_exp_name}/yolo_classification"

        cmd3 = [
            "python3", "pipeline.py", "train", "yolov8_classification",
            "--name", cls_exp_name,
            "--epochs", str(args.epochs_cls),
            "--data", crop_data_path
        ]

        if not run_command(cmd3, f"STAGE 3: Training classification for {model_key}"):
            print(f"❌ Classification training failed for {model_key}")
            failed_models.append(f"{model_key} (classification)")
            continue

        successful_models.append(model_key)
        print(f"\n🎉 {model_key.upper()} PIPELINE COMPLETED SUCCESSFULLY!")

    # Final summary
    print(f"\n{'🎉'*30}")
    print(f"🎉 MULTIPLE MODELS PIPELINE FINISHED!")
    print(f"{'🎉'*30}")

    if successful_models:
        print(f"\n✅ SUCCESSFUL MODELS ({len(successful_models)}):")
        for model in successful_models:
            detection_model = detection_models[model]
            det_exp_name = f"{base_exp_name}_{model}_det"
            cls_exp_name = f"{base_exp_name}_{model}_cls"
            print(f"   🎯 {model.upper()}:")
            print(f"      Detection: results/current_experiments/training/detection/{detection_model}/{det_exp_name}/")
            print(f"      Crops: data/crops_from_{model}_{det_exp_name}/yolo_classification/")
            print(f"      Classification: results/current_experiments/training/classification/yolov8_classification/{cls_exp_name}/")

    if failed_models:
        print(f"\n❌ FAILED MODELS ({len(failed_models)}):")
        for model in failed_models:
            print(f"   ❌ {model}")

    print(f"\nTotal Success Rate: {len(successful_models)}/{len(models_to_run)} models")

if __name__ == "__main__":
    main()