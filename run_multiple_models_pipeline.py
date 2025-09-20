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
import json
import pandas as pd
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

def create_experiment_summary(exp_dir, model_key, det_exp_name, cls_exp_name, detection_model):
    """Create comprehensive experiment summary"""
    try:
        summary_data = {
            "experiment_info": {
                "model": model_key.upper(),
                "detection_experiment": det_exp_name,
                "classification_experiment": cls_exp_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        # Get detection results
        det_results_path = f"results/current_experiments/training/detection/{detection_model}/{det_exp_name}/results.csv"
        if Path(det_results_path).exists():
            det_df = pd.read_csv(det_results_path)
            final_det = det_df.iloc[-1]
            summary_data["detection"] = {
                "epochs": len(det_df),
                "mAP50": float(final_det.get('metrics/mAP50(B)', 0)),
                "mAP50_95": float(final_det.get('metrics/mAP50-95(B)', 0)),
                "precision": float(final_det.get('metrics/precision(B)', 0)),
                "recall": float(final_det.get('metrics/recall(B)', 0)),
                "training_time_sec": float(final_det.get('time', 0))
            }

        # Get classification results
        cls_results_path = f"results/current_experiments/training/classification/yolov8_classification/{cls_exp_name}/results.csv"
        if Path(cls_results_path).exists():
            cls_df = pd.read_csv(cls_results_path)
            final_cls = cls_df.iloc[-1]
            summary_data["classification"] = {
                "epochs": len(cls_df),
                "top1_accuracy": float(final_cls.get('metrics/accuracy_top1', 0)),
                "top5_accuracy": float(final_cls.get('metrics/accuracy_top5', 0)),
                "training_time_sec": float(final_cls.get('time', 0))
            }

        # Get IoU analysis results
        iou_results_file = f"{exp_dir}/analysis/iou_variation/iou_variation_results.json"
        if Path(iou_results_file).exists():
            try:
                with open(iou_results_file, 'r') as f:
                    iou_data = json.load(f)
                    summary_data["iou_analysis"] = iou_data
            except Exception as e:
                print(f"   ⚠️ Could not load IoU analysis: {e}")

        # Save JSON summary
        with open(f"{exp_dir}/experiment_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2)

        # Create markdown summary
        total_time = (summary_data.get("detection", {}).get("training_time_sec", 0) +
                     summary_data.get("classification", {}).get("training_time_sec", 0)) / 60

        md_content = f"""# {model_key.upper()} Complete Pipeline Results

**Generated**: {summary_data['experiment_info']['timestamp']}

## 📊 Performance Summary

### Detection Performance (Training Data)
- **mAP50**: {summary_data.get('detection', {}).get('mAP50', 0):.3f} (validation set)
- **mAP50-95**: {summary_data.get('detection', {}).get('mAP50_95', 0):.3f} (validation set)
- **Precision**: {summary_data.get('detection', {}).get('precision', 0):.3f}
- **Recall**: {summary_data.get('detection', {}).get('recall', 0):.3f}
- **Training Time**: {summary_data.get('detection', {}).get('training_time_sec', 0)/60:.1f} minutes

"""

        # Add IoU Analysis section if available
        if 'iou_analysis' in summary_data:
            md_content += f"""### IoU Variation Analysis (TEST SET - Independent)

| IoU Threshold | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---------------|---------|--------------|-----------|--------|
"""
            for iou_key, metrics in summary_data['iou_analysis'].items():
                iou_val = metrics['iou_threshold']
                map50 = metrics['map50']
                map50_95 = metrics['map50_95']
                precision = metrics['precision']
                recall = metrics['recall']
                md_content += f"| {iou_val} | {map50:.3f} | {map50_95:.3f} | {precision:.3f} | {recall:.3f} |\n"

            # Find best performing IoU
            best_iou = max(summary_data['iou_analysis'].values(), key=lambda x: x['map50'])
            md_content += f"""
**Best TEST Performance**: mAP@0.5={best_iou['map50']:.3f} at IoU={best_iou['iou_threshold']}
⚠️ **IMPORTANT**: These are TEST SET results (independent evaluation)

"""

        md_content += f"""### Classification Performance
- **Top-1 Accuracy**: {summary_data.get('classification', {}).get('top1_accuracy', 0):.3f}
- **Top-5 Accuracy**: {summary_data.get('classification', {}).get('top5_accuracy', 0):.3f}
- **Training Time**: {summary_data.get('classification', {}).get('training_time_sec', 0)/60:.1f} minutes

### Total Training Time: {total_time:.1f} minutes

## 📁 Results Structure
```
{exp_dir}/
├── analysis/                           # Analysis results
│   ├── confusion_matrix/               # Classification confusion matrix
│   └── iou_variation/                  # IoU analysis (TEST SET)
│       ├── iou_variation_results.json  # Raw IoU data
│       ├── iou_comparison_table.csv    # Comparison table
│       └── iou_analysis_report.md      # IoU analysis report
├── experiment_summary.json             # This summary in JSON format
├── experiment_summary.md               # This report
├── run_analysis.py                     # Classification analysis script
└── run_iou_analysis.py                 # IoU analysis script
```

## 🔗 Original Results Locations
- **Detection**: results/current_experiments/training/detection/{detection_model}/{det_exp_name}/
- **Classification**: results/current_experiments/training/classification/yolov8_classification/{cls_exp_name}/
- **Crops**: data/crops_from_{model_key}_{det_exp_name}/

---
*Generated by Multiple Models Pipeline*
"""

        with open(f"{exp_dir}/experiment_summary.md", 'w') as f:
            f.write(md_content)

        print(f"   📋 Experiment summary created: {exp_dir}/experiment_summary.md")

    except Exception as e:
        print(f"   ⚠️ Could not create experiment summary: {e}")

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

        # STAGE 4: Create Organized Analysis
        print(f"\n🔬 STAGE 4: Creating organized analysis for {model_key}")

        # Create experiment directory structure
        exp_dir = f"experiments/{base_exp_name}_{model_key}_complete"
        analysis_dir = f"{exp_dir}/analysis"

        # Create directories
        for dir_path in [exp_dir, analysis_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Run unified analysis for this specific experiment
        classification_model = f"results/current_experiments/training/classification/yolov8_classification/{cls_exp_name}/weights/best.pt"
        test_data = f"data/crops_from_{model_key}_{det_exp_name}/yolo_classification/test"

        # Check if paths exist before running analysis
        if Path(classification_model).exists() and Path(test_data).exists():
            print(f"   📊 Running confusion matrix analysis...")

            # Create custom analysis script for this experiment
            analysis_script_content = f'''#!/usr/bin/env python3
import os
import sys
import shutil
sys.path.append('/home/akhiyarwaladi/hello_world')

# Import the analysis class
try:
    from scripts.analysis.classification_deep_analysis import DeepClassificationAnalyzer

    analyzer = DeepClassificationAnalyzer(
        model_path="{classification_model}",
        test_data_path="{test_data}",
        output_dir="{analysis_dir}"
    )

    print("🔬 Running comprehensive analysis...")
    analyzer.run_complete_analysis()
    print("✅ Analysis completed successfully!")

except Exception as e:
    print(f"⚠️ Analysis failed: {{e}}")
'''

            # Write and run analysis script
            script_path = f"{exp_dir}/run_analysis.py"
            with open(script_path, 'w') as f:
                f.write(analysis_script_content)

            # Run analysis
            analysis_cmd = ["python3", script_path]
            if run_command(analysis_cmd, f"STAGE 4: Analysis for {model_key}"):
                print(f"   ✅ Analysis completed for {model_key}")
            else:
                print(f"   ⚠️ Analysis had issues for {model_key}, but continuing...")
        else:
            print(f"   ⚠️ Skipping classification analysis - required files not found")

        # STAGE 4B: IoU Variation Analysis (on TEST SET)
        print(f"   📊 Running IoU variation analysis (TEST SET)...")
        detection_model_path = f"results/current_experiments/training/detection/{detection_model}/{det_exp_name}/weights/best.pt"

        if Path(detection_model_path).exists():
            iou_analysis_dir = f"{analysis_dir}/iou_variation"
            Path(iou_analysis_dir).mkdir(parents=True, exist_ok=True)

            # Create IoU analysis script with TEST SET correction
            iou_script_content = f'''#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
sys.path.append('/home/akhiyarwaladi/hello_world')

try:
    from ultralytics import YOLO
    import pandas as pd

    # Load detection model
    model = YOLO("{detection_model_path}")
    print(f"📦 Loaded model: {model_key.upper()}")

    # IoU thresholds to test (following reference paper)
    iou_thresholds = [0.3, 0.5, 0.7]
    results_summary = {{}}

    for iou_thresh in iou_thresholds:
        print(f"\\n🎯 Testing IoU threshold: {{iou_thresh}} (TEST SET)")

        # CORRECTED: Use TEST SET (split='test') not validation set
        metrics = model.val(
            data="data/integrated/yolo/data.yaml",
            split='test',  # Use test set, NOT validation set
            iou=iou_thresh,
            verbose=False,
            save=False
        )

        # Extract key metrics
        results_summary[f"iou_{{iou_thresh:.1f}}"] = {{
            "iou_threshold": iou_thresh,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "confidence_threshold": 0.25  # Default YOLO confidence
        }}

        print(f"   mAP@0.5: {{metrics.box.map50:.3f}}")
        print(f"   mAP@0.5:0.95: {{metrics.box.map:.3f}}")
        print(f"   Precision: {{metrics.box.mp:.3f}}")
        print(f"   Recall: {{metrics.box.mr:.3f}}")

    # Save results
    output_file = Path("{iou_analysis_dir}") / "iou_variation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Create comparison table
    comparison_data = []
    for iou_key, metrics in results_summary.items():
        comparison_data.append({{
            "IoU_Threshold": metrics["iou_threshold"],
            "mAP@0.5": f"{{metrics['map50']:.3f}}",
            "mAP@0.5:0.95": f"{{metrics['map50_95']:.3f}}",
            "Precision": f"{{metrics['precision']:.3f}}",
            "Recall": f"{{metrics['recall']:.3f}}"
        }})

    df = pd.DataFrame(comparison_data)
    csv_file = Path("{iou_analysis_dir}") / "iou_comparison_table.csv"
    df.to_csv(csv_file, index=False)

    # Create markdown report
    md_content = f"""# IoU Variation Analysis - {{model_key.upper()}}

## 📊 Performance at Different IoU Thresholds (TEST SET)

| IoU Threshold | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---------------|---------|--------------|-----------|--------|
"""

    for data in comparison_data:
        md_content += f"| {{data['IoU_Threshold']}} | {{data['mAP@0.5']}} | {{data['mAP@0.5:0.95']}} | {{data['Precision']}} | {{data['Recall']}} |\\n"

    md_content += f"""
## 📚 Reference Paper Comparison

**Paper Parameters:**
- IoU=0.7 for training
- Testing at IoU: 0.3, 0.5, 0.7
- YOLOv4: mAP@0.5=89% (source), 90% (cross-test)
- YOLOv5: mAP@0.5=96% (source), 59% (cross-test)

**Our Results (TEST SET - Independent Evaluation):**
- Model: {{model_key.upper()}}
- Best mAP@0.5: {{max(results_summary.values(), key=lambda x: x['map50'])['map50']:.3f}} at IoU={{max(results_summary.values(), key=lambda x: x['map50'])['iou_threshold']}}
- Detection confidence threshold: 0.25 (fixed)
- ⚠️ IMPORTANT: These are TEST SET results (not validation)

## 🔧 Analysis Notes

- **IoU Threshold**: Controls Non-Maximum Suppression (NMS) overlap tolerance
- **Confidence Threshold**: Fixed at 0.25 for detection filtering
- **Higher IoU**: More strict overlap requirement (fewer detections)
- **Lower IoU**: More lenient overlap (more detections, potential duplicates)

## 📁 Files Generated

- `iou_variation_results.json`: Raw metrics data
- `iou_comparison_table.csv`: Comparison table
- `iou_analysis_report.md`: This report

---
*Generated by IoU Variation Analysis (TEST SET)*
"""

    md_file = Path("{iou_analysis_dir}") / "iou_analysis_report.md"
    with open(md_file, 'w') as f:
        f.write(md_content)

    print(f"\\n✅ IoU variation analysis completed!")
    print(f"📁 Results saved to: {{Path('{iou_analysis_dir}')}}")
    print(f"📊 Best TEST mAP@0.5: {{max(results_summary.values(), key=lambda x: x['map50'])['map50']:.3f}} at IoU={{max(results_summary.values(), key=lambda x: x['map50'])['iou_threshold']}}")

except Exception as e:
    print(f"⚠️ IoU analysis failed: {{e}}")
    import traceback
    traceback.print_exc()
'''

            # Write and run IoU analysis script
            iou_script_path = f"{exp_dir}/run_iou_analysis.py"
            with open(iou_script_path, 'w') as f:
                f.write(iou_script_content)

            # Run IoU analysis
            iou_cmd = ["python3", iou_script_path]
            if run_command(iou_cmd, f"STAGE 4B: IoU Analysis for {model_key}"):
                print(f"   ✅ IoU analysis completed for {model_key}")
            else:
                print(f"   ⚠️ IoU analysis had issues for {model_key}, but continuing...")
        else:
            print(f"   ⚠️ Skipping IoU analysis - detection model not found")

        # Create experiment summary
        create_experiment_summary(exp_dir, model_key, det_exp_name, cls_exp_name, detection_model)

        successful_models.append(model_key)
        print(f"\n🎉 {model_key.upper()} COMPLETE PIPELINE FINISHED!")
        print(f"📁 Organized Results: {exp_dir}/")

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