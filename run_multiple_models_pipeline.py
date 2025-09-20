#!/usr/bin/env python3
"""
Multiple Models Pipeline: Train detection models → Generate crops → Train classification
Automatically handles folder routing based on experiment names
"""

import os
import sys
import argparse
import subprocess
import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def get_experiment_folder(experiment_name):
    """Determine folder based on experiment name keywords"""
    name_lower = experiment_name.lower()
    if "production" in name_lower or "final" in name_lower:
        return "production"
    elif "validation" in name_lower or "test" in name_lower:
        return "validation"
    else:
        return "training"

def run_command(cmd, description):
    """Run command with logging"""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0

def wait_for_file(file_path, max_wait_seconds=60, check_interval=2):
    """Wait for file to exist with size stability check"""
    print(f"⏳ Waiting for file: {file_path}")

    for attempt in range(max_wait_seconds // check_interval):
        if Path(file_path).exists():
            try:
                initial_size = Path(file_path).stat().st_size
                time.sleep(1)
                final_size = Path(file_path).stat().st_size

                if initial_size == final_size and final_size > 0:
                    print(f"✅ File ready: {file_path}")
                    return True
            except Exception:
                pass

        time.sleep(check_interval)

    print(f"❌ Timeout waiting for file: {file_path}")
    return False

def create_experiment_summary(exp_dir, model_key, det_exp_name, cls_exp_name, detection_model, cls_model_name="yolo8"):
    """Create experiment summary"""
    try:
        summary_data = {
            "experiment_info": {
                "detection_model": model_key.upper(),
                "classification_model": cls_model_name.upper(),
                "detection_experiment": det_exp_name,
                "classification_experiment": cls_exp_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        # Get detection results
        det_folder = get_experiment_folder(det_exp_name)
        det_results_path = f"results/current_experiments/{det_folder}/detection/{detection_model}/{det_exp_name}/results.csv"
        if Path(det_results_path).exists():
            det_df = pd.read_csv(det_results_path)
            final_det = det_df.iloc[-1]
            summary_data["detection"] = {
                "epochs": len(det_df),
                "mAP50": float(final_det.get('metrics/mAP50(B)', 0)),
                "mAP50_95": float(final_det.get('metrics/mAP50-95(B)', 0)),
                "precision": float(final_det.get('metrics/precision(B)', 0)),
                "recall": float(final_det.get('metrics/recall(B)', 0))
            }

        # Get classification results
        cls_folder = get_experiment_folder(cls_exp_name)
        if cls_model_name in ["yolo8", "yolo11"]:
            cls_config_name = "yolov8_classification" if cls_model_name == "yolo8" else "yolov11_classification"
            cls_results_path = f"results/current_experiments/{cls_folder}/classification/{cls_config_name}/{cls_exp_name}/results.csv"
        else:
            cls_results_path = f"results/current_experiments/training/pytorch_classification/{cls_model_name}/{cls_exp_name}/results.csv"

        if Path(cls_results_path).exists():
            cls_df = pd.read_csv(cls_results_path)
            final_cls = cls_df.iloc[-1]
            summary_data["classification"] = {
                "model_type": cls_model_name,
                "epochs": len(cls_df),
                "top1_accuracy": float(final_cls.get('metrics/accuracy_top1', final_cls.get('accuracy', 0))),
                "top5_accuracy": float(final_cls.get('metrics/accuracy_top5', 0))
            }

        # Get IoU analysis results
        iou_results_file = f"{exp_dir}/analysis/iou_variation/iou_variation_results.json"
        if Path(iou_results_file).exists():
            try:
                with open(iou_results_file, 'r') as f:
                    iou_data = json.load(f)
                    summary_data["iou_analysis"] = iou_data
            except Exception:
                pass

        # Save JSON summary
        with open(f"{exp_dir}/experiment_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2)

        # Create simple markdown summary
        md_content = f"""# {model_key.upper()} → {cls_model_name.upper()} Pipeline Results

**Generated**: {summary_data['experiment_info']['timestamp']}

## Detection Performance
- **mAP50**: {summary_data.get('detection', {}).get('mAP50', 0):.3f}
- **mAP50-95**: {summary_data.get('detection', {}).get('mAP50_95', 0):.3f}
- **Precision**: {summary_data.get('detection', {}).get('precision', 0):.3f}
- **Recall**: {summary_data.get('detection', {}).get('recall', 0):.3f}

"""

        # Add IoU Analysis if available
        if 'iou_analysis' in summary_data:
            best_iou = max(summary_data['iou_analysis'].values(), key=lambda x: x['map50'])
            md_content += f"""## IoU Analysis (TEST SET)
**Best Performance**: mAP@0.5={best_iou['map50']:.3f} at IoU={best_iou['iou_threshold']}

"""

        md_content += f"""## Classification Performance ({cls_model_name.upper()})
- **Top-1 Accuracy**: {summary_data.get('classification', {}).get('top1_accuracy', 0):.3f}
- **Top-5 Accuracy**: {summary_data.get('classification', {}).get('top5_accuracy', 0):.3f}

## Results Locations
- **Detection**: results/current_experiments/{"validation" if "TEST" in det_exp_name else "training"}/detection/{detection_model}/{det_exp_name}/
- **Crops**: data/crops_from_{model_key}_{det_exp_name}/
"""

        with open(f"{exp_dir}/experiment_summary.md", 'w') as f:
            f.write(md_content)

    except Exception as e:
        print(f"⚠️ Could not create experiment summary: {e}")

def main():
    parser = argparse.ArgumentParser(description="Multiple Models Pipeline: Train multiple detection models → Generate Crops → Train Classification")
    parser.add_argument("--include", nargs="+",
                       choices=["yolo8", "yolo11", "yolo12", "rtdetr"],
                       help="Detection models to include (if not specified, includes all)")
    parser.add_argument("--exclude-detection", nargs="+",
                       choices=["yolo8", "yolo11", "yolo12", "rtdetr"],
                       default=[],
                       help="Detection models to exclude")
    parser.add_argument("--epochs-det", type=int, default=50,
                       help="Epochs for detection training")
    parser.add_argument("--epochs-cls", type=int, default=30,
                       help="Epochs for classification training")
    parser.add_argument("--experiment-name", default="multi_pipeline",
                       help="Base name for experiments")
    parser.add_argument("--classification-models", nargs="+",
                       choices=["yolo8", "yolo11", "resnet18", "efficientnet", "densenet121", "mobilenet_v2", "all"],
                       default=["yolo8"],
                       help="Classification models to train (default: yolo8)")
    parser.add_argument("--exclude-classification", nargs="+",
                       choices=["yolo8", "yolo11", "resnet18", "efficientnet", "densenet121", "mobilenet_v2"],
                       default=[],
                       help="Classification models to exclude")
    parser.add_argument("--test-mode", action="store_true",
                       help="Enable test mode: lower confidence threshold for crops and faster settings")

    args = parser.parse_args()

    # Determine which detection models to run
    all_detection_models = ["yolo8", "yolo11", "yolo12", "rtdetr"]

    if args.include:
        models_to_run = args.include
    else:
        models_to_run = all_detection_models

    # Remove excluded detection models
    models_to_run = [model for model in models_to_run if model not in args.exclude_detection]

    if not models_to_run:
        print("❌ No detection models to run after exclusions!")
        return

    # Define classification models
    classification_configs = {
        "yolo8": {
            "type": "yolo",
            "script": "pipeline.py",
            "model": "yolov8_classification",
            "epochs": 30,
            "batch": 4
        },
        "yolo11": {
            "type": "yolo", 
            "script": "pipeline.py",
            "model": "yolov11_classification",
            "epochs": 30,
            "batch": 4
        },
        "resnet18": {
            "type": "pytorch",
            "script": "scripts/training/11b_train_pytorch_classification.py",
            "model": "resnet18",
            "epochs": 30,
            "batch": 8
        },
        "efficientnet": {
            "type": "pytorch",
            "script": "scripts/training/11b_train_pytorch_classification.py",
            "model": "efficientnet_b0",
            "epochs": 30,
            "batch": 8
        },
        "densenet121": {
            "type": "pytorch",
            "script": "scripts/training/11b_train_pytorch_classification.py",
            "model": "densenet121",
            "epochs": 30,
            "batch": 8
        },
        "mobilenet_v2": {
            "type": "pytorch",
            "script": "scripts/training/11b_train_pytorch_classification.py",
            "model": "mobilenet_v2",
            "epochs": 30,
            "batch": 8
        }
    }

    # Determine which classification models to run
    all_classification_models = list(classification_configs.keys())

    # Expand "all" selections for classification models
    if "all" in args.classification_models:
        selected_classification = all_classification_models
    else:
        selected_classification = args.classification_models

    # Remove excluded classification models
    selected_classification = [model for model in selected_classification if model not in args.exclude_classification]

    if not selected_classification:
        print("❌ No classification models to run after exclusions!")
        return

    # Set test mode parameters
    if args.test_mode:
        confidence_threshold = "0.01"  # Lower threshold for test mode with few epochs
        print("🧪 TEST MODE ENABLED")
    else:
        confidence_threshold = "0.25"

    # Add timestamp to experiment name for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_exp_name = f"{args.experiment_name}_{timestamp}"
    
    if args.test_mode:
        base_exp_name += "_TEST"

    print("🎯 MULTIPLE MODELS PIPELINE")
    print(f"Detection models: {', '.join(models_to_run)}")
    print(f"Classification models: {', '.join(selected_classification)}")
    print(f"Epochs: {args.epochs_det} det, {args.epochs_cls} cls")
    print(f"Confidence: {confidence_threshold}")

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
        print(f"\n🎯 STARTING {model_key.upper()} PIPELINE")

        detection_model = detection_models[model_key]
        det_exp_name = f"{base_exp_name}_{model_key}_det"

        # STAGE 1: Train Detection Model
        print(f"\n📊 STAGE 1: Training {detection_model}")
        cmd1 = [
            "python3", "pipeline.py", "train", detection_model,
            "--name", det_exp_name,
            "--epochs", str(args.epochs_det)
        ]

        if not run_command(cmd1, f"Training {detection_model}"):
            failed_models.append(f"{model_key} (detection)")
            continue

        # Wait for weights
        det_folder = get_experiment_folder(det_exp_name)
        model_path = f"results/current_experiments/{det_folder}/detection/{detection_model}/{det_exp_name}/weights/best.pt"

        if not wait_for_file(model_path, max_wait_seconds=120, check_interval=3):
            failed_models.append(f"{model_key} (weights missing)")
            continue

        # STAGE 2: Generate Crops
        print(f"\n🔄 STAGE 2: Generating crops for {model_key}")
        input_path = "data/integrated/yolo"
        output_path = f"data/crops_from_{model_key}_{det_exp_name}"

        cmd2 = [
            "python3", "scripts/training/10_crop_detections.py",
            "--model", model_path,
            "--input", input_path,
            "--output", output_path,
            "--confidence", confidence_threshold,  # Use variable confidence
            "--crop_size", "128",
            "--create_yolo_structure",
            "--fix_classification_structure"
        ]

        if not run_command(cmd2, f"Generating crops for {model_key}"):
            failed_models.append(f"{model_key} (crops)")
            continue

        # Verify crop data
        crop_data_path = f"data/crops_from_{model_key}_{det_exp_name}/yolo_classification"
        if not Path(crop_data_path).exists():
            failed_models.append(f"{model_key} (crop data missing)")
            continue

        # Count crops
        total_crops = 0
        for split in ['train', 'val', 'test']:
            split_path = Path(crop_data_path) / split
            if split_path.exists():
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        total_crops += len(list(class_dir.glob("*.jpg")))

        print(f"   📊 Generated {total_crops} crops")
        if total_crops == 0:
            failed_models.append(f"{model_key} (no crops generated)")
            continue

        # STAGE 3: Train Classification Models
        print(f"\n📈 STAGE 3: Training classification for {model_key}")
        classification_success = []
        classification_failed = []

        for cls_model_name in selected_classification:
            if cls_model_name not in classification_configs:
                continue

            cls_config = classification_configs[cls_model_name]
            cls_exp_name = f"{base_exp_name}_{model_key}_{cls_model_name}_cls"

            print(f"   🚀 Training {cls_model_name.upper()}")

            if cls_config["type"] == "yolo":
                # YOLO classification using pipeline.py
                cmd3 = [
                    "python3", "pipeline.py", "train", cls_config["model"],
                    "--name", cls_exp_name,
                    "--epochs", str(args.epochs_cls),
                    "--data", crop_data_path
                ]
            else:
                # PyTorch classification using specialized script
                cmd3 = [
                    "python3", cls_config["script"],
                    "--data", crop_data_path,
                    "--model", cls_config["model"],
                    "--epochs", str(args.epochs_cls),
                    "--batch", str(cls_config["batch"]),
                    "--device", "cpu",
                    "--name", cls_exp_name
                ]

            if run_command(cmd3, f"Training {cls_model_name.upper()}"):
                classification_success.append(cls_model_name)
            else:
                classification_failed.append(cls_model_name)

        if not classification_success:
            failed_models.append(f"{model_key} (all classification)")
            continue

        # STAGE 4: Create Organized Analysis for each successful classification model
        print(f"\n🔬 STAGE 4: Creating organized analysis for {model_key}")

        for cls_model_name in classification_success:
            cls_exp_name = f"{base_exp_name}_{model_key}_{cls_model_name}_cls"

            # Create experiment directory structure
            exp_dir = f"experiments/{base_exp_name}_{model_key}_{cls_model_name}_complete"
            analysis_dir = f"{exp_dir}/analysis"

            # Create directories
            for dir_path in [exp_dir, analysis_dir]:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

            # Run unified analysis for this specific experiment
            cls_folder = get_experiment_folder(cls_exp_name)

            if cls_model_name in ["yolo8", "yolo11"]:
                cls_config_name = "yolov8_classification" if cls_model_name == "yolo8" else "yolov11_classification"
                classification_model = f"results/current_experiments/{cls_folder}/classification/{cls_config_name}/{cls_exp_name}/weights/best.pt"
            else:
                # PyTorch models always in training folder (hardcoded in script)
                classification_model = f"results/current_experiments/training/pytorch_classification/{cls_model_name}/{cls_exp_name}/best_model.pth"

            test_data = f"data/crops_from_{model_key}_{det_exp_name}/yolo_classification/test"

            # Check if paths exist before running analysis
            if Path(classification_model).exists() and Path(test_data).exists():
                print(f"   📊 Running analysis for {cls_model_name.upper()}")

                # Use standalone classification analysis script
                analysis_cmd = [
                    "python3", "scripts/analysis/classification_deep_analysis.py",
                    "--model", classification_model,
                    "--test-data", test_data,
                    "--output", analysis_dir
                ]

                run_command(analysis_cmd, f"Analysis for {cls_model_name.upper()}")

        # STAGE 4B: IoU Variation Analysis (on TEST SET) - once per detection model
        # Run IoU analysis in all modes to ensure complete pipeline validation
        print(f"   📊 Running IoU variation analysis")
        det_folder = get_experiment_folder(det_exp_name)
        detection_model_path = f"results/current_experiments/{det_folder}/detection/{detection_model}/{det_exp_name}/weights/best.pt"

        if Path(detection_model_path).exists() and classification_success:
            first_cls = classification_success[0]
            exp_dir_for_iou = f"experiments/{base_exp_name}_{model_key}_{first_cls}_complete"
            iou_analysis_dir = f"{exp_dir_for_iou}/analysis/iou_variation"
            Path(iou_analysis_dir).mkdir(parents=True, exist_ok=True)

            # Use standalone IoU analysis script
            iou_cmd = [
                "python3", "scripts/analysis/14_compare_models_performance.py",
                "--iou-analysis",
                "--model", detection_model_path,
                "--output", iou_analysis_dir
            ]

            if args.test_mode:
                print(f"   🧪 Test mode: Running quick IoU analysis")

            run_command(iou_cmd, f"IoU Analysis for {model_key}")
        else:
            print(f"   ⚠️ Skipping IoU analysis - detection model not found or no classification success")

        # Create experiment summaries
        for cls_model_name in classification_success:
            cls_exp_name = f"{base_exp_name}_{model_key}_{cls_model_name}_cls"
            exp_dir = f"experiments/{base_exp_name}_{model_key}_{cls_model_name}_complete"
            create_experiment_summary(exp_dir, model_key, det_exp_name, cls_exp_name, detection_model, cls_model_name)

        successful_models.append(f"{model_key} ({', '.join(classification_success)})")
        print(f"\n✅ {model_key.upper()} PIPELINE COMPLETED")
        if classification_failed:
            print(f"❌ Failed: {', '.join(classification_failed)}")

    # Final summary
    print(f"\n🎉 PIPELINE FINISHED")
    if successful_models:
        print(f"\n✅ SUCCESSFUL ({len(successful_models)}):")
        for model in successful_models:
            print(f"   {model}")
    if failed_models:
        print(f"\n❌ FAILED ({len(failed_models)}):")
        for model in failed_models:
            print(f"   {model}")
    print(f"\nSuccess Rate: {len(successful_models)}/{len(models_to_run)}")

if __name__ == "__main__":
    main()