#!/usr/bin/env python3
"""
RUN ALL MODELS PIPELINE: Sequential execution of all detection models
Runs YOLO8, YOLO11, YOLO12, and RT-DETR sequentially with classification for each
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Run complete pipeline for ALL detection models sequentially")
    parser.add_argument("--epochs-det", type=int, default=50,
                       help="Epochs for detection training")
    parser.add_argument("--epochs-cls", type=int, default=30,
                       help="Epochs for classification training")
    parser.add_argument("--experiment-name", default="all_models",
                       help="Base name for experiments")

    args = parser.parse_args()

    print("🎯 COMPLETE ALL MODELS PIPELINE - SEQUENTIAL EXECUTION")
    print(f"Detection Epochs: {args.epochs_det}")
    print(f"Classification Epochs: {args.epochs_cls}")
    print(f"Experiment Name: {args.experiment_name}")

    # All detection models to run
    detection_models = ["yolo8", "yolo11", "yolo12", "rtdetr"]

    successful_models = []
    failed_models = []

    # Run pipeline for each model sequentially
    for i, model in enumerate(detection_models, 1):
        print(f"\n🔄 STARTING MODEL {i}/{len(detection_models)}: {model.upper()}")
        print(f"{'🌟'*60}")

        # Run complete pipeline for this model
        cmd = [
            "python", "run_complete_pipeline.py",
            "--detection", model,
            "--epochs-det", str(args.epochs_det),
            "--epochs-cls", str(args.epochs_cls),
            "--experiment-name", f"{args.experiment_name}_{model}"
        ]

        success = run_command(cmd, f"COMPLETE PIPELINE FOR {model.upper()}")

        if success:
            successful_models.append(model)
            print(f"✅ {model.upper()} pipeline completed successfully!")
        else:
            failed_models.append(model)
            print(f"❌ {model.upper()} pipeline failed!")
            print(f"⚠️  Continuing with next model...")

        # Small break between models
        if i < len(detection_models):
            print(f"\n⏳ Waiting 10 seconds before next model...")
            time.sleep(10)

    # Final summary
    print(f"\n{'🎉'*60}")
    print(f"ALL MODELS PIPELINE COMPLETED!")
    print(f"{'🎉'*60}")

    print(f"\n📊 SUMMARY:")
    print(f"✅ Successful models ({len(successful_models)}): {', '.join(successful_models)}")
    if failed_models:
        print(f"❌ Failed models ({len(failed_models)}): {', '.join(failed_models)}")

    print(f"\n📂 RESULTS LOCATIONS:")
    for model in successful_models:
        model_mapping = {
            "yolo8": "yolov8_detection",
            "yolo11": "yolov11_detection",
            "yolo12": "yolov12_detection",
            "rtdetr": "rtdetr_detection"
        }
        detection_model = model_mapping[model]
        det_exp_name = f"{args.experiment_name}_{model}_{model}_det"
        cls_exp_name = f"{args.experiment_name}_{model}_{model}_cls"

        print(f"\n{model.upper()}:")
        print(f"  Detection: results/current_experiments/training/detection/{detection_model}/{det_exp_name}/")
        print(f"  Crops: data/crops_from_{model}_{det_exp_name}/")
        print(f"  Classification: results/current_experiments/training/classification/yolov8_classification/{cls_exp_name}/")

    print(f"\n🏁 Total execution completed: {len(successful_models)}/{len(detection_models)} models successful")

if __name__ == "__main__":
    main()