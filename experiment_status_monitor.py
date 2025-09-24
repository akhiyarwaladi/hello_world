#!/usr/bin/env python3
"""
Real-time Experiment Status Monitor
Track progress of detection-classification variation experiments
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def check_detection_model_status(exp_path):
    """Check detection model training status"""
    detection_path = exp_path / "detection"
    status = []

    for detection_type in detection_path.glob("*_detection"):
        if detection_type.is_dir():
            model_folders = list(detection_type.glob("*"))
            for model_folder in model_folders:
                weights_path = model_folder / "weights" / "best.pt"
                results_path = model_folder / "results.csv"

                if weights_path.exists():
                    status.append({
                        "Detection_Model": detection_type.name.replace("_detection", ""),
                        "Experiment": model_folder.name,
                        "Status": "[DONE] COMPLETED" if results_path.exists() else "[TRAIN] TRAINING",
                        "Weights": "[OK]" if weights_path.exists() else "[NONE]",
                        "Last_Modified": datetime.fromtimestamp(weights_path.stat().st_mtime).strftime("%H:%M:%S")
                    })

    return status

def check_classification_models_status(exp_path):
    """Check classification models training status"""
    models_path = exp_path / "models"
    status = []

    if models_path.exists():
        for model_type in models_path.glob("*"):
            if model_type.is_dir():
                model_folders = list(model_type.glob("*"))
                for model_folder in model_folders:
                    results_file = model_folder / "results.txt"
                    best_pt = model_folder / "best.pt"

                    # Extract detection model from experiment name
                    exp_name = model_folder.name
                    detection_model = "yolo10"  # Default, but should extract from name
                    for det_type in ["yolo10", "yolo11", "yolo12", "rtdetr"]:
                        if det_type in exp_name:
                            detection_model = det_type
                            break

                    # Parse results if available
                    test_acc = "N/A"
                    training_time = "N/A"
                    current_epoch = "N/A"

                    # Check results.csv first (like detection models)
                    results_csv = model_folder / "results.csv"
                    if results_csv.exists():
                        try:
                            import pandas as pd
                            df = pd.read_csv(results_csv)
                            if not df.empty:
                                # Get latest epoch info
                                current_epoch = f"{len(df)}"
                                latest_val_acc = df['val_acc'].iloc[-1]
                                test_acc = f"{latest_val_acc:.2f}%"
                        except:
                            pass

                    # Fallback to results.txt for final summary
                    if results_file.exists() and test_acc == "N/A":
                        try:
                            with open(results_file, 'r') as f:
                                content = f.read()
                                for line in content.split('\n'):
                                    if "Test Acc:" in line:
                                        test_acc = line.split("Test Acc:")[1].strip()
                                    elif "Training Time:" in line:
                                        training_time = line.split("Training Time:")[1].strip()
                        except:
                            pass

                    status.append({
                        "Detection_Model": detection_model,
                        "Classification_Model": model_type.name,
                        "Experiment": model_folder.name,
                        "Status": "[DONE] COMPLETED" if results_file.exists() else ("[TRAIN] TRAINING" if best_pt.exists() else "[NONE] NOT_STARTED"),
                        "Test_Accuracy": test_acc,
                        "Training_Time": training_time,
                        "Last_Modified": datetime.fromtimestamp(best_pt.stat().st_mtime).strftime("%H:%M:%S") if best_pt.exists() else "N/A"
                    })

    return status

def create_results_matrix(classification_status):
    """Create detection-classification results matrix"""
    df = pd.DataFrame(classification_status)
    if df.empty:
        return "No classification results available yet."

    # Create pivot table
    matrix = df.pivot_table(
        index='Detection_Model',
        columns='Classification_Model',
        values='Test_Accuracy',
        aggfunc='first',
        fill_value='[NONE]'
    )

    return matrix

def monitor_experiment_status(experiment_name=None):
    """Monitor complete experiment status"""

    if experiment_name is None:
        # Find most recent experiment
        results_path = Path("results")
        exp_folders = [f for f in results_path.glob("exp_*") if f.is_dir()]
        if not exp_folders:
            print("[ERROR] No experiments found!")
            return

        exp_path = max(exp_folders, key=lambda x: x.stat().st_mtime)
        experiment_name = exp_path.name
    else:
        exp_path = Path("results") / experiment_name

    if not exp_path.exists():
        print(f"[ERROR] Experiment {experiment_name} not found!")
        return

    print(f"\n🔍 MONITORING EXPERIMENT: {experiment_name}")
    print(f"📁 Path: {exp_path}")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Check detection models
    detection_status = check_detection_model_status(exp_path)
    if detection_status:
        print("\n📊 DETECTION MODELS STATUS:")
        df_det = pd.DataFrame(detection_status)
        print(df_det.to_string(index=False))

    # Check classification models
    classification_status = check_classification_models_status(exp_path)
    if classification_status:
        print(f"\n🎯 CLASSIFICATION MODELS STATUS ({len(classification_status)} models):")
        df_cls = pd.DataFrame(classification_status)
        print(df_cls.to_string(index=False))

        # Count completed models
        completed = len([s for s in classification_status if "COMPLETED" in s['Status']])
        training = len([s for s in classification_status if "TRAINING" in s['Status']])
        total = len(classification_status)

        print(f"\n📈 PROGRESS SUMMARY:")
        print(f"   [DONE] Completed: {completed}/{total} ({completed/total*100:.1f}%)")
        print(f"   🔄 Training: {training}/{total}")
        print(f"   [NONE] Not Started: {total-completed-training}/{total}")

        # Results matrix (only completed models)
        completed_results = [s for s in classification_status if "COMPLETED" in s['Status']]
        if completed_results:
            print(f"\n🏆 RESULTS MATRIX (Completed Models):")
            matrix = create_results_matrix(completed_results)
            print(matrix)

    print("\n" + "=" * 80)

    # Save summary to JSON
    summary = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "detection_models": detection_status,
        "classification_models": classification_status
    }

    summary_file = exp_path / "experiment_status_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Status saved to: {summary_file}")

if __name__ == "__main__":
    import sys

    exp_name = sys.argv[1] if len(sys.argv) > 1 else None
    monitor_experiment_status(exp_name)