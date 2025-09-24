#!/usr/bin/env python3
"""
Detection-Classification Model Mapping Tool
Clearly shows which classification models belong to which detection models
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def analyze_detection_classification_mapping(exp_path):
    """Create comprehensive mapping between detection and classification models"""

    print(f"\n🔗 DETECTION ↔ CLASSIFICATION MODEL MAPPING")
    print(f"📁 Experiment: {exp_path.name}")
    print("=" * 80)

    # 1. DETECTION MODELS
    detection_path = exp_path / "detection"
    detection_models = {}

    if detection_path.exists():
        print(f"\n📊 DETECTION MODELS:")
        for detection_type in detection_path.glob("*_detection"):
            if detection_type.is_dir():
                model_folders = list(detection_type.glob("*"))
                for model_folder in model_folders:
                    weights_path = model_folder / "weights" / "best.pt"
                    results_csv = model_folder / "results.csv"

                    # Extract detection model type
                    det_type = detection_type.name.replace("_detection", "")
                    exp_name = model_folder.name

                    if weights_path.exists():
                        # Read mAP from results.csv if available
                        map50 = "N/A"
                        if results_csv.exists():
                            try:
                                import pandas as pd
                                df = pd.read_csv(results_csv)
                                if not df.empty and 'metrics/mAP50(B)' in df.columns:
                                    map50 = f"{df['metrics/mAP50(B)'].iloc[-1]:.3f}"
                            except:
                                pass

                        detection_models[det_type] = {
                            "experiment": exp_name,
                            "path": str(model_folder),
                            "weights": str(weights_path),
                            "mAP50": map50,
                            "status": "✅ COMPLETED" if weights_path.exists() else "🔄 TRAINING"
                        }

                        print(f"   {det_type:>10}: {exp_name} → mAP50: {map50}")

    # 2. CLASSIFICATION MODELS
    models_path = exp_path / "models"
    classification_models = {}

    if models_path.exists():
        print(f"\n🎯 CLASSIFICATION MODELS (grouped by detection model):")

        for model_type in models_path.glob("*"):
            if model_type.is_dir():
                model_folders = list(model_type.glob("*"))
                for model_folder in model_folders:
                    results_file = model_folder / "results.txt"
                    best_pt = model_folder / "best.pt"

                    # Extract detection model from experiment name
                    exp_name = model_folder.name
                    detection_model = "unknown"
                    for det_type in ["yolo10", "yolo11", "yolo12", "rtdetr"]:
                        if det_type in exp_name:
                            detection_model = det_type
                            break

                    # Parse classification results
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
                                # Get latest epoch info for progress tracking
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

                    # Store classification model info
                    if detection_model not in classification_models:
                        classification_models[detection_model] = []

                    classification_models[detection_model].append({
                        "classification_type": model_type.name,
                        "experiment": exp_name,
                        "path": str(model_folder),
                        "test_accuracy": test_acc,
                        "training_time": training_time,
                        "status": "✅ COMPLETED" if results_file.exists() else ("🔄 TRAINING" if best_pt.exists() else "❌ NOT_STARTED")
                    })

    # 3. GROUPED DISPLAY BY DETECTION MODEL
    for det_model in sorted(classification_models.keys()):
        det_info = detection_models.get(det_model, {"mAP50": "N/A", "status": "❌ NOT_FOUND"})

        print(f"\n🔸 DETECTION: {det_model.upper()} (mAP50: {det_info['mAP50']}) {det_info['status']}")
        print(f"   └─ Classification Models:")

        cls_models = classification_models[det_model]
        for i, cls_model in enumerate(sorted(cls_models, key=lambda x: x['classification_type'])):
            connector = "├─" if i < len(cls_models) - 1 else "└─"
            status_icon = "✅" if "COMPLETED" in cls_model['status'] else ("🔄" if "TRAINING" in cls_model['status'] else "❌")

            print(f"      {connector} {cls_model['classification_type']:>15}: {cls_model['test_accuracy']:>8} {status_icon}")

    # 4. RESULTS MATRIX TABLE
    print(f"\n📊 RESULTS MATRIX:")

    # Create matrix data
    matrix_data = []
    all_cls_types = set()

    for det_model, cls_models in classification_models.items():
        row_data = {"Detection_Model": det_model.upper()}
        for cls_model in cls_models:
            cls_type = cls_model['classification_type']
            all_cls_types.add(cls_type)
            accuracy = cls_model['test_accuracy']
            if accuracy != "N/A" and "%" in accuracy:
                row_data[cls_type] = accuracy
            else:
                row_data[cls_type] = "🔄" if "TRAINING" in cls_model['status'] else "❌"
        matrix_data.append(row_data)

    # Display matrix
    if matrix_data:
        df_matrix = pd.DataFrame(matrix_data)
        df_matrix = df_matrix.set_index("Detection_Model")

        # Fill missing values
        for col in sorted(all_cls_types):
            if col not in df_matrix.columns:
                df_matrix[col] = "❌"

        # Reorder columns
        df_matrix = df_matrix[sorted(all_cls_types)]
        print(df_matrix.to_string())

    # 5. SUMMARY STATISTICS
    print(f"\n📈 EXPERIMENT SUMMARY:")
    total_detection = len(detection_models)
    total_classification = sum(len(models) for models in classification_models.values())
    completed_classification = sum(1 for models in classification_models.values()
                                  for model in models if "COMPLETED" in model['status'])

    print(f"   🎯 Detection Models: {total_detection}")
    print(f"   🎯 Classification Models: {completed_classification}/{total_classification} completed")
    print(f"   📊 Total Combinations: {total_detection * 5} (expected)")
    print(f"   ⏰ Completion Rate: {completed_classification/total_classification*100:.1f}%")

    return {
        "detection_models": detection_models,
        "classification_models": classification_models,
        "matrix_data": matrix_data
    }

def main():
    # Find most recent experiment
    results_path = Path("results")
    exp_folders = [f for f in results_path.glob("exp_*") if f.is_dir()]
    if not exp_folders:
        print("❌ No experiments found!")
        return

    exp_path = max(exp_folders, key=lambda x: x.stat().st_mtime)

    # Analyze mapping
    mapping_data = analyze_detection_classification_mapping(exp_path)

    # Save detailed mapping
    output_file = exp_path / "detection_classification_mapping.json"
    with open(output_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)

    print(f"\n💾 Detailed mapping saved to: {output_file}")

if __name__ == "__main__":
    main()