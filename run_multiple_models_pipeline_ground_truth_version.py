#!/usr/bin/env python
"""
Multiple Models Pipeline: Train detection models -> Generate crops -> Train classification
Automatically handles folder routing based on experiment names
"""

import os
import sys
import argparse
import subprocess
import time
import json
import pandas as pd
import shutil
import zipfile
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from utils.results_manager import get_results_manager, get_experiment_path, get_crops_path, create_crops_path, get_analysis_path, create_analysis_path
from utils.pipeline_continue import (
    check_completed_stages,
    determine_next_stage,
    validate_experiment_dir,
    load_experiment_metadata,
    save_experiment_metadata,
    merge_parameters,
    print_experiment_status,
    list_available_experiments,
    find_detection_models,
    find_crop_data
)

# REMOVED: get_experiment_folder function - always use "training" for consistency
# Test mode and production mode now use same folder structure

def run_optimized_training(model_name, data_yaml, epochs, exp_name, centralized_path, dataset_type=""):
    """Run optimized training with configuration optimized for all datasets"""
    from ultralytics import YOLO

    try:
        print(f"[TARGET] OPTIMIZED TRAINING: {model_name}")
        print(f"   Dataset: {dataset_type}")
        print(f"   Full augmentation enabled")
        print(f"   Epochs: {epochs}")
        print(f"   Output: {centralized_path}")

        # Load model
        model = YOLO(model_name)

        # Configure parameters based on model and dataset
        # FIXED: RT-DETR needs different LR than YOLO due to transformer architecture
        if 'rtdetr' in model_name.lower():
            lr_value = 0.0001  # Lower LR for RT-DETR transformer
            print(f"   [FIXED] Using RT-DETR optimized LR: {lr_value}")
        else:
            lr_value = 0.0005  # Standard LR for YOLO models

        # Dataset-specific configurations
        if dataset_type == "mp_idb_species":
            # Single class detection - more conservative settings
            patience_val = 50
            batch_size = 32
            print(f"   [CONFIG] Species detection: batch={batch_size}, patience={patience_val}")
        elif dataset_type == "mp_idb_stages":
            # 4-class stage detection - balanced settings
            patience_val = 60
            batch_size = 24
            print(f"   [CONFIG] Stage detection: batch={batch_size}, patience={patience_val}")
        elif dataset_type == "iml_lifecycle":
            # 5-class lifecycle - need more patience for convergence
            patience_val = 70
            batch_size = 20
            print(f"   [CONFIG] Lifecycle detection: batch={batch_size}, patience={patience_val}")
        else:
            # Default settings
            patience_val = 50
            batch_size = 32
            print(f"   [CONFIG] Default: batch={batch_size}, patience={patience_val}")

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,    # Dataset-optimized batch size
            patience=patience_val,
            save=True,
            save_period=10,
            device='cuda' if torch.cuda.is_available() else 'cpu',  # Auto-detect GPU
            workers=4,           # GPU optimized workers
            exist_ok=True,
            optimizer='AdamW',
            lr0=lr_value,       # Model-specific learning rate
            weight_decay=0.0005, # L2 regularization
            warmup_epochs=5,     # Warmup for stability

            # CONSERVATIVE AUGMENTATION - Prevent overfitting on small dataset
            augment=True,
            hsv_h=0.010,        # Reduced color augmentation
            hsv_s=0.5,          # Reduced saturation change
            hsv_v=0.3,          # Reduced brightness change
            degrees=15,         # Reduced rotation (45->15)
            scale=0.3,          # Reduced scaling
            flipud=0.0,         # No vertical flip (can confuse parasite orientation)
            fliplr=0.5,         # Keep horizontal flip
            mosaic=0.5,         # Reduced mosaic (1.0->0.5)
            mixup=0.0,          # Disable mixup (too aggressive for small dataset)
            copy_paste=0.0,     # Disable copy_paste (too aggressive)

            # Output settings
            project=str(centralized_path.parent),
            name=exp_name,
            plots=True,
            val=True,
            verbose=True
        )

        print(f"[SUCCESS] Optimized training completed: {exp_name}")
        return True

    except Exception as e:
        print(f"[ERROR] Optimized training failed: {e}")
        return False

def run_command(cmd, description):
    """Run command with logging and proper error capture"""
    print(f"\n[START] {description}")
    # Convert all items to strings to handle Path objects
    cmd_str = [str(item) for item in cmd]
    print(f"Command: {' '.join(cmd_str)}")

    try:
        # Use UTF-8 encoding with replace errors and safe environment
        result = subprocess.run(
            cmd_str,
            capture_output=True,
            text=True,
            check=False,
            encoding='utf-8',
            errors='replace',
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )

        # Print stdout if available - handle Unicode safely
        if result.stdout:
            safe_stdout = result.stdout.strip().encode('ascii', 'replace').decode('ascii')
            print(f"[OUTPUT] {safe_stdout}")

        # Print stderr if available - handle Unicode safely
        if result.stderr:
            safe_stderr = result.stderr.strip().encode('ascii', 'replace').decode('ascii')
            print(f"[ERROR] {safe_stderr}")

        if result.returncode == 0:
            print(f"[SUCCESS] {description} completed successfully")
            return True
        else:
            print(f"[FAILED] {description} failed with return code: {result.returncode}")
            return False

    except Exception as e:
        # Safe error printing
        safe_error = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"[EXCEPTION] Failed to run command: {safe_error}")
        return False


def wait_for_file(file_path, max_wait_seconds=60, check_interval=2):
    """Wait for file to exist with size stability check"""
    print(f"[WAIT] Waiting for file: {file_path}")

    for attempt in range(max_wait_seconds // check_interval):
        if Path(file_path).exists():
            try:
                initial_size = Path(file_path).stat().st_size
                time.sleep(1)
                final_size = Path(file_path).stat().st_size

                if initial_size == final_size and final_size > 0:
                    print(f"[SUCCESS] File ready: {file_path}")
                    return True
            except Exception:
                pass

        time.sleep(check_interval)

    print(f"[ERROR] Timeout waiting for file: {file_path}")
    return False

# REMOVED: consolidate_and_zip_results function - using direct centralized save approach

def create_centralized_zip(base_exp_name, results_manager):
    """Create ZIP archive from centralized results folder"""
    print(f"\n[ARCHIVE] CREATING ZIP ARCHIVE FROM CENTRALIZED RESULTS")

    # The centralized folder is already created by results_manager
    centralized_dir = results_manager.pipeline_dir

    if not centralized_dir.exists():
        print(f"[ERROR] Centralized folder not found: {centralized_dir}")
        return None, None

    # Create master summary in centralized folder
    master_summary = {
        "experiment_name": base_exp_name,
        "timestamp": datetime.now().isoformat(),
        "pipeline_type": "centralized_results",
        "folder_structure": {
            "detection": len(list((centralized_dir / "detection").glob("*"))) if (centralized_dir / "detection").exists() else 0,
            "classification": len(list((centralized_dir / "classification").glob("*"))) if (centralized_dir / "classification").exists() else 0,
            "crop_data": len(list((centralized_dir / "crop_data").glob("*"))) if (centralized_dir / "crop_data").exists() else 0,
            "analysis": len(list((centralized_dir / "analysis").glob("*"))) if (centralized_dir / "analysis").exists() else 0
        }
    }

    with open(centralized_dir / "master_summary.json", "w") as f:
        json.dump(master_summary, f, indent=2)

    # Create Excel version of master summary (EASIER TO READ)
    create_master_summary_excel(centralized_dir, master_summary)

    # Create README
    readme_content = f"""# Centralized Pipeline Results: {base_exp_name}

## Summary
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Pipeline Type**: Centralized Results (Direct Save)
- **Total Components**: {sum(master_summary['folder_structure'].values())}

## Folder Structure
- `detection/` - Detection model results and weights
- `classification/` - Classification model results and weights
- `crop_data/` - Generated crop datasets
- `analysis/` - Analysis results and visualizations
- `master_summary.json` - Detailed summary

## Key Features
This archive contains results from a CENTRALIZED pipeline run where all components
were saved directly to this folder structure (not copied afterward).

All model weights, training logs, analysis results, and generated datasets are
organized in a clean hierarchy for easy access and distribution.
"""

    with open(centralized_dir / "README.md", "w") as f:
        f.write(readme_content)

    # Create ZIP archive in results directory (not root folder)
    zip_filename = centralized_dir.parent / f"{centralized_dir.name}.zip"
    if zip_filename.exists():
        zip_filename.unlink()

    print(f"[ARCHIVE] Creating ZIP archive: {zip_filename}")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in centralized_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(centralized_dir.parent)
                zipf.write(file_path, arcname)

    # Calculate size
    zip_size = zip_filename.stat().st_size / (1024 * 1024)  # MB
    print(f"[SUCCESS] ZIP created: {zip_filename} ({zip_size:.1f} MB)")

    return str(zip_filename), str(centralized_dir)

def create_experiment_summary(exp_dir, model_key, det_exp_name, cls_exp_name, detection_model, cls_model_name="yolo11"):
    """Create experiment summary - FIXED to use correct centralized paths"""
    try:
        # FIXED: Determine experiment base path from exp_dir
        experiment_path = Path(exp_dir).parent.parent if Path(exp_dir).name.endswith('_complete') else Path(exp_dir)

        summary_data = {
            "experiment_info": {
                "detection_model": model_key.upper(),
                "classification_model": cls_model_name.upper(),
                "detection_experiment": det_exp_name,
                "classification_experiment": cls_exp_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        # FIXED: Use direct experiment path for detection results
        det_results_path = experiment_path / "detection" / detection_model / det_exp_name / "results.csv"

        if det_results_path.exists():
            det_df = pd.read_csv(det_results_path)
            final_det = det_df.iloc[-1]
            summary_data["detection"] = {
                "epochs": len(det_df),
                "mAP50": float(final_det.get('metrics/mAP50(B)', 0)),
                "mAP50_95": float(final_det.get('metrics/mAP50-95(B)', 0)),
                "precision": float(final_det.get('metrics/precision(B)', 0)),
                "recall": float(final_det.get('metrics/recall(B)', 0))
            }
        else:
            print(f"[DEBUG] Detection results not found: {det_results_path}")

        # FIXED: Use direct experiment path for classification results
        if cls_model_name in ["yolo11"]:
            cls_results_path = experiment_path / "models" / "yolo11_classification" / cls_exp_name / "results.csv"
        else:
            # Extract base model name from cls_model_name (e.g., "densenet121_focal" -> "densenet121")
            base_model_name = cls_model_name.split('_')[0] if '_' in cls_model_name else cls_model_name
            cls_results_path = experiment_path / "models" / base_model_name / cls_exp_name / "results.csv"

        if cls_results_path.exists():
            cls_df = pd.read_csv(cls_results_path)
            final_cls = cls_df.iloc[-1]
            # FIXED: Handle both PyTorch and YOLO classification column names
            accuracy_col = final_cls.get('val_acc', final_cls.get('accuracy', final_cls.get('metrics/accuracy_top1', 0)))
            summary_data["classification"] = {
                "model_type": cls_model_name,
                "epochs": len(cls_df),
                "top1_accuracy": float(accuracy_col),
                "top5_accuracy": float(final_cls.get('metrics/accuracy_top5', 0))
            }
        else:
            print(f"[DEBUG] Classification results not found: {cls_results_path}")

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
        md_content = f"""# {model_key.upper()} -> {cls_model_name.upper()} Pipeline Results

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
- **Top-1 Accuracy**: {summary_data.get('classification', {}).get('top1_accuracy', 0):.3f}%
- **Top-5 Accuracy**: {summary_data.get('classification', {}).get('top5_accuracy', 0):.3f}

## Results Locations (CENTRALIZED)
- **Detection**: {experiment_path}/detection/{detection_model}/{det_exp_name}/
- **Classification**: {experiment_path}/models/{cls_model_name}/{cls_exp_name}/
- **Crops**: {experiment_path}/crop_data/
"""

        with open(f"{exp_dir}/experiment_summary.md", 'w') as f:
            f.write(md_content)

        # Create Excel version of experiment summary (EASIER TO READ)
        create_experiment_summary_excel(exp_dir, summary_data)

    except Exception as e:
        print(f"[WARNING] Could not create experiment summary: {e}")

def create_master_summary_excel(centralized_dir, master_summary):
    """Create Excel version of master summary - easier to read than JSON"""
    try:
        import pandas as pd

        # Create summary data for Excel
        summary_data = []

        # Basic info
        summary_data.append({
            'Category': 'Experiment Info',
            'Metric': 'Experiment Name',
            'Value': master_summary.get('experiment_name', 'Unknown'),
            'Description': 'Name of the experiment'
        })

        summary_data.append({
            'Category': 'Experiment Info',
            'Metric': 'Generated Timestamp',
            'Value': master_summary.get('timestamp', 'Unknown'),
            'Description': 'When the experiment was created'
        })

        summary_data.append({
            'Category': 'Experiment Info',
            'Metric': 'Pipeline Type',
            'Value': master_summary.get('pipeline_type', 'Unknown'),
            'Description': 'Type of pipeline execution'
        })

        # Folder structure
        folder_structure = master_summary.get('folder_structure', {})
        for folder_name, count in folder_structure.items():
            summary_data.append({
                'Category': 'Folder Structure',
                'Metric': f'{folder_name.title()} Files',
                'Value': count,
                'Description': f'Number of files/folders in {folder_name}/'
            })

        total_components = sum(folder_structure.values())
        summary_data.append({
            'Category': 'Folder Structure',
            'Metric': 'Total Components',
            'Value': total_components,
            'Description': 'Total number of files/folders generated'
        })

        # Save as Excel
        excel_path = centralized_dir / "master_summary.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Master_Summary', index=False)
            print(f"[EXCEL] ✅ Master summary saved: {excel_path}")
        except ImportError:
            # Fallback to xlsxwriter if openpyxl not available
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Master_Summary', index=False)
            print(f"[EXCEL] ✅ Master summary saved (xlsxwriter): {excel_path}")

    except Exception as e:
        print(f"[WARNING] Could not create Excel master summary: {e}")

def create_experiment_summary_excel(exp_dir, summary_data):
    """Create Excel version of experiment summary - easier to read than JSON"""
    try:
        import pandas as pd

        # Create structured data for Excel
        excel_data = []

        # Experiment info
        exp_info = summary_data.get('experiment_info', {})
        for key, value in exp_info.items():
            excel_data.append({
                'Category': 'Experiment Info',
                'Metric': key.replace('_', ' ').title(),
                'Value': value,
                'Stage': 'General'
            })

        # Detection metrics
        detection = summary_data.get('detection', {})
        for key, value in detection.items():
            excel_data.append({
                'Category': 'Detection Performance',
                'Metric': key.replace('_', ' ').title(),
                'Value': value,
                'Stage': 'Detection'
            })

        # Classification metrics
        classification = summary_data.get('classification', {})
        for key, value in classification.items():
            excel_data.append({
                'Category': 'Classification Performance',
                'Metric': key.replace('_', ' ').title(),
                'Value': value,
                'Stage': 'Classification'
            })

        # IoU Analysis (if available)
        iou_analysis = summary_data.get('iou_analysis', {})
        for iou_threshold, metrics in iou_analysis.items():
            for metric_key, metric_value in metrics.items():
                excel_data.append({
                    'Category': f'IoU Analysis ({iou_threshold})',
                    'Metric': metric_key.replace('_', ' ').title(),
                    'Value': metric_value,
                    'Stage': 'Analysis'
                })

        # Save as Excel
        excel_path = Path(exp_dir) / "experiment_summary.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                pd.DataFrame(excel_data).to_excel(writer, sheet_name='Experiment_Summary', index=False)
            print(f"[EXCEL] ✅ Experiment summary saved: {excel_path}")
        except ImportError:
            # Fallback to xlsxwriter if openpyxl not available
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                pd.DataFrame(excel_data).to_excel(writer, sheet_name='Experiment_Summary', index=False)
            print(f"[EXCEL] ✅ Experiment summary saved (xlsxwriter): {excel_path}")

    except Exception as e:
        print(f"[WARNING] Could not create Excel experiment summary: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Multiple Models Pipeline: Train multiple detection models -> Generate Crops -> Train Classification",
        epilog="""
Multi-Dataset Examples:
  All datasets:        (no --dataset parameter, runs all 3 datasets)
  Single dataset:      --dataset iml_lifecycle

Stage Control Examples:
  Detection only:      --stop-stage detection
  Crop generation:     --start-stage detection --stop-stage crop
  Classification only: --start-stage classification --stop-stage classification
  Analysis only:       --start-stage analysis
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--include", nargs="+",
                       choices=["yolo10", "yolo11", "yolo12", "rtdetr"],
                       help="Detection models to include (if not specified, includes yolo10, yolo11, yolo12)")
    parser.add_argument("--exclude-detection", nargs="+",
                       choices=["yolo10", "yolo11", "yolo12", "rtdetr"],
                       default=[],
                       help="Detection models to exclude")
    parser.add_argument("--epochs-det", type=int, default=50,
                       help="Epochs for detection training")
    parser.add_argument("--epochs-cls", type=int, default=30,
                       help="Epochs for classification training")
    parser.add_argument("--experiment-name", default="multi_pipeline",
                       help="Base name for experiments")
    parser.add_argument("--dataset", choices=["mp_idb_species", "mp_idb_stages", "iml_lifecycle", "all"], default="all",
                       help="Dataset selection: mp_idb_species (4 species), mp_idb_stages (4 stages), iml_lifecycle (4 stages), all (run all datasets)")
    parser.add_argument("--classification-models", nargs="+",
                       choices=["densenet121", "efficientnet_b1", "convnext_tiny", "mobilenet_v3_large", "efficientnet_b2", "resnet101", "all"],
                       default=["all"],
                       help="6 optimized classification models (2024): DenseNet121, EfficientNet-B1, ConvNeXt-Tiny, MobileNetV3-Large, EfficientNet-B2, ResNet101")
    parser.add_argument("--exclude-classification", nargs="+",
                       choices=["densenet121", "efficientnet_b1", "convnext_tiny", "mobilenet_v3_large", "efficientnet_b2", "resnet101"],
                       default=[],
                       help="Classification models to exclude")
    parser.add_argument("--no-zip", action="store_true",
                       help="Skip creating ZIP archive of results (default: always create ZIP)")

    # Continue/Resume functionality
    parser.add_argument("--continue-from", type=str, metavar="EXPERIMENT_NAME",
                       help="Continue from existing experiment (e.g., exp_multi_pipeline_20250921_144544)")
    parser.add_argument("--start-stage", choices=["detection", "crop", "classification", "analysis"],
                       help="Force start from specific stage (auto-detected if not specified)")
    parser.add_argument("--stop-stage", choices=["detection", "crop", "classification", "analysis"],
                       help="Stop after completing this stage (default: run all stages)")
    parser.add_argument("--list-experiments", action="store_true",
                       help="List available experiments and exit")

    args = parser.parse_args()

    # Handle multi-dataset execution
    if args.dataset == "all":
        print("[MULTI-DATASET] Running pipeline for ALL datasets!")
        datasets_to_run = ["mp_idb_species", "mp_idb_stages", "iml_lifecycle"]
        all_results = []

        for dataset in datasets_to_run:
            print(f"\n{'='*80}")
            print(f"[TARGET] STARTING DATASET: {dataset.upper()}")
            print(f"{'='*80}")

            # Create a copy of args with the specific dataset
            dataset_args = argparse.Namespace(**vars(args))
            dataset_args.dataset = dataset

            try:
                result = run_pipeline_for_dataset(dataset_args)
                all_results.append((dataset, result))
                print(f"[SUCCESS] COMPLETED DATASET: {dataset.upper()}")
            except Exception as e:
                print(f"[ERROR] FAILED DATASET: {dataset.upper()} - {e}")
                all_results.append((dataset, None))

        # Print final summary
        print(f"\n{'='*80}")
        print(f"[FINISH] MULTI-DATASET PIPELINE SUMMARY")
        print(f"{'='*80}")
        for dataset, result in all_results:
            status = "[SUCCESS]" if result else "[FAILED]"
            print(f"{dataset:15} {status}")

        return

    # Single dataset execution
    return run_pipeline_for_dataset(args)

def run_pipeline_for_dataset(args):
    """Run the complete pipeline for a single dataset"""

    # Handle special commands first
    if args.list_experiments:
        list_available_experiments()
        return

    # Handle continue/resume functionality
    continue_mode = False
    experiment_dir = None
    start_stage = None

    if args.continue_from:
        continue_mode = True
        experiment_name = args.continue_from

        # Handle both with and without "results/" prefix
        if experiment_name.startswith("results/"):
            experiment_dir = experiment_name
        else:
            experiment_dir = f"results/{experiment_name}"

        # Validate experiment exists
        if not validate_experiment_dir(experiment_dir):
            print("[ERROR] Cannot continue from invalid experiment")
            return

        print(f"[CONTINUE] CONTINUE MODE: {experiment_name}")
        print("=" * 60)

        # Show current experiment status
        print_experiment_status(experiment_dir)

        # Load existing metadata and merge parameters
        metadata = load_experiment_metadata(experiment_dir)
        if metadata.get('original_args'):
            original_args = metadata['original_args']
            print("[INFO] Merging parameters with original experiment...")
            merged_args_dict = merge_parameters(original_args, args)

            # Create new args object with merged parameters
            for key, value in merged_args_dict.items():
                if hasattr(args, key):
                    setattr(args, key, value)
        else:
            # No metadata available - try to infer dataset from experiment name
            print("[INFO] No metadata found - inferring dataset from experiment name...")
            if "mp_idb_stages" in experiment_name.lower() or "stages" in experiment_name.lower():
                args.dataset = "mp_idb_stages"
                print(f"[INFER] Detected dataset: mp_idb_stages")
            elif "mp_idb_species" in experiment_name.lower() or "species" in experiment_name.lower():
                args.dataset = "mp_idb_species"
                print(f"[INFER] Detected dataset: mp_idb_species")
            elif "iml_lifecycle" in experiment_name.lower() or "lifecycle" in experiment_name.lower():
                args.dataset = "iml_lifecycle"
                print(f"[INFER] Detected dataset: iml_lifecycle")
            else:
                print(f"[WARNING] Could not infer dataset from experiment name: {experiment_name}")
                print(f"[WARNING] Using default dataset: {args.dataset}")

        # Check completed stages and determine where to start
        completed_stages = check_completed_stages(experiment_dir)

        if args.start_stage:
            start_stage = args.start_stage
            print(f"[TARGET] Force starting from stage: {start_stage}")
        else:
            start_stage = determine_next_stage(completed_stages)
            print(f"[CONTINUE] Auto-determined next stage: {start_stage}")

        print()

    # Determine which detection models to run
    all_detection_models = ["yolo10", "yolo11", "yolo12", "rtdetr"]
    # Default models (exclude rtdetr since it's not yet fixed)
    default_detection_models = ["yolo10", "yolo11", "yolo12"]

    if args.include:
        models_to_run = args.include
    else:
        models_to_run = default_detection_models

    # Remove excluded detection models
    models_to_run = [model for model in models_to_run if model not in args.exclude_detection]

    if not models_to_run:
        print("[ERROR] No detection models to run after exclusions!")
        return

    # Define classification models - SYSTEMATIC COMPARISON: 7 Models × 2 Loss Functions = 14 Experiments
    # FIXED: Include EfficientNet-B0 (optimal model) instead of B1 (overfits)
    # ADDED: ResNet18 as lightweight baseline model for faster training
    base_models = ["resnet18", "densenet121", "efficientnet_b0", "convnext_tiny", "mobilenet_v3_large", "efficientnet_b2", "resnet101"]

    classification_configs = {}

    # Generate configurations for each model with both loss functions
    for model in base_models:
        # Configuration 1: Cross-Entropy (Baseline)
        classification_configs[f"{model}_ce"] = {
            "type": "pytorch",
            "script": "scripts/training/12_train_pytorch_classification.py",
            "model": model,
            "loss": "cross_entropy",
            "epochs": 25,        # Standardized epochs
            "batch": 32,         # Optimized for 224px images
            "lr": 0.0005,        # FIXED: Optimal LR (was 0.001)
            "display_name": f"{model.upper()} (Cross-Entropy)"
        }

        # Configuration 2: Focal Loss (Novel Contribution)
        classification_configs[f"{model}_focal"] = {
            "type": "pytorch",
            "script": "scripts/training/12_train_pytorch_classification.py",
            "model": model,
            "loss": "focal",
            "focal_alpha": 1.0,  # FIXED: Optimal alpha (was 2.0)
            "focal_gamma": 2.0,
            "epochs": 25,        # Standardized epochs
            "batch": 32,         # Optimized for 224px images
            "lr": 0.0005,        # Lower LR for focal loss stability
            "display_name": f"{model.upper()} (Focal Loss)"
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
        print("[ERROR] No classification models to run after exclusions!")
        return

    # Set confidence threshold
    confidence_threshold = "0.25"
    print(f"[TARGET] Using confidence threshold: {confidence_threshold}")

    # Handle experiment naming for continue vs new mode
    if continue_mode:
        # Use existing experiment directory
        base_exp_name = Path(experiment_dir).name.replace("exp_", "")
        results_manager = get_results_manager(pipeline_name=base_exp_name)
        print(f"[INFO] CONTINUING: {experiment_dir}/")
    else:
        # Add timestamp to experiment name for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_exp_name = f"{args.experiment_name}_{timestamp}"

        # Add dataset type to experiment name
        if args.dataset == "iml_lifecycle":
            base_exp_name += "_iml_lifecycle"
        elif args.dataset == "mp_idb_stages":
            base_exp_name += "_mp_idb_stages"
        elif args.dataset == "mp_idb_species":
            base_exp_name += "_mp_idb_species"

        # NEW: Initialize centralized results manager
        results_manager = get_results_manager(pipeline_name=base_exp_name)
        print(f"[INFO] RESULTS: results/exp_{base_exp_name}/")

    print("[TARGET] MULTIPLE MODELS PIPELINE - SYSTEMATIC COMPARISON")
    print(f"Detection models: {', '.join(models_to_run)}")
    print(f"Classification: {len(base_models)} architectures × 2 loss functions = {len(classification_configs)} experiments")
    print(f"Loss Functions: Cross-Entropy (baseline) vs Focal Loss (novel contribution)")
    print(f"Expected Results: {len(models_to_run)} × {len(classification_configs)} = {len(models_to_run) * len(classification_configs)} total combinations")
    print(f"Epochs: {args.epochs_det} det, 25 cls (standardized)")
    print(f"Confidence: {confidence_threshold}")

    # Auto-setup and auto-detect best dataset
    if args.dataset == "iml_lifecycle":
        # Setup IML lifecycle dataset paths
        lifecycle_ready_path = Path("data/processed/lifecycle/data.yaml")

        if not lifecycle_ready_path.exists():
            print("[SETUP] Setting up IML lifecycle dataset for pipeline...")
            import subprocess

            # First ensure raw dataset is downloaded
            download_result = subprocess.run([sys.executable, "scripts/data_setup/01_download_datasets.py", "--dataset", "malaria_lifecycle"],
                                           capture_output=True, text=True, encoding='utf-8', errors='replace')
            if download_result.returncode != 0:
                print(f"[ERROR] Failed to download IML lifecycle dataset: {download_result.stderr}")
                return

            # Setup using lifecycle setup script
            # Convert to lifecycle format (now defaults to single-class detection for consistency)
            setup_result = subprocess.run([sys.executable, "scripts/data_setup/08_setup_lifecycle_for_pipeline.py"],
                                        capture_output=True, text=True, encoding='utf-8', errors='replace')
            if setup_result.returncode != 0:
                print(f"[ERROR] Failed to setup IML lifecycle dataset: {setup_result.stderr}")
                return

            print("[SUCCESS] IML lifecycle dataset setup completed")

    elif args.dataset == "mp_idb_stages":
        # Setup MP-IDB Kaggle stage dataset paths
        stage_ready_path = Path("data/processed/stages/data.yaml")

        if not stage_ready_path.exists():
            print("[SETUP] Setting up MP-IDB stages dataset for pipeline...")
            import subprocess

            # First ensure Kaggle dataset is downloaded
            download_result = subprocess.run([sys.executable, "scripts/data_setup/01_download_datasets.py", "--dataset", "kaggle_mp_idb"],
                                           capture_output=True, text=True, encoding='utf-8', errors='replace')
            if download_result.returncode != 0:
                print(f"[ERROR] Failed to download MP-IDB Kaggle dataset: {download_result.stderr}")
                return

            # Convert to stage format (now defaults to single-class detection for consistency)
            setup_result = subprocess.run([sys.executable, "scripts/data_setup/09_setup_kaggle_stage_for_pipeline.py"],
                                        capture_output=True, text=True, encoding='utf-8', errors='replace')
            if setup_result.returncode != 0:
                print(f"[ERROR] Failed to setup MP-IDB stages dataset: {setup_result.stderr}")
                return

            print("[SUCCESS] MP-IDB stages dataset setup completed")

    else:  # mp_idb_species
        # MP-IDB species dataset setup
        kaggle_ready_path = Path("data/processed/species/data.yaml")

        if not kaggle_ready_path.exists():
            print("[SETUP] Setting up MP-IDB species dataset for pipeline...")
            import subprocess
            result = subprocess.run([sys.executable, "scripts/data_setup/07_setup_kaggle_species_for_pipeline.py"],
                                  capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode != 0:
                print(f"[ERROR] Failed to setup MP-IDB species dataset: {result.stderr}")
                return
        print(f"[INFO] Dataset: MP-IDB Species Pipeline Ready (data/processed/species/)")

    # Model mapping
    detection_models = {
        "yolo10": "yolov10_detection",
        "yolo11": "yolov11_detection",
        "yolo12": "yolov12_detection",
        "rtdetr": "rtdetr_detection"
    }

    successful_models = []
    failed_models = []

    # =============================================================================
    # RESTRUCTURED PIPELINE: 3 CLEAR STAGES INSTEAD OF PER-MODEL LOOPS
    # =============================================================================

    detection_results = {}  # Store detection model paths for crop generation

    # OPTION A: SHARED CLASSIFICATION ARCHITECTURE
    # ===========================================
    # STAGE 1: Train ALL Detection Models (independent)
    # STAGE 2: Generate Ground Truth Crops ONCE (shared)
    # STAGE 3: Train Classification Models ONCE (shared, independent of detection)
    # STAGE 4: Analysis (separate detection vs classification)

    print(f"\n{'='*80}")
    print(f"STAGE 1: DETECTION TRAINING - All Models (Independent)")
    print(f"{'='*80}")
    print(f"[ARCHITECTURE] Using Option A: Shared Classification")
    print(f"[EFFICIENCY] Crops and classification will be generated once, not per detection model")

    detection_models_trained = []
    detection_models_failed = []

    # STAGE 1: Train detection models only (no crops/classification per model)
    for model_key in models_to_run:
        print(f"\n[DETECTION] Training {model_key.upper()} detection model")

        detection_model = detection_models[model_key]
        det_exp_name = f"{base_exp_name}_{model_key}_det"

        # Initialize variables that may be used in later stages
        model_path = None
        centralized_detection_path = None
        centralized_crops_path = None
        crop_data_path = None
        classification_success = []
        classification_failed = []

        # STAGE 1: Train Detection Model - DIRECT SAVE to centralized folder
        if start_stage is None or start_stage in ['detection']:
            print(f"\n[INFO] STAGE 1: Training {detection_model}")

            # NEW: Get centralized path but let YOLO create the experiment folder to avoid conflicts
            centralized_detection_path = results_manager.get_experiment_path("training", detection_model, det_exp_name)
        elif start_stage in ['crop', 'classification', 'analysis']:
            print(f"\n[SKIP] STAGE 1: Skipping detection training (start_stage={start_stage})")
            # Try to find existing detection model
            if continue_mode:
                existing_models = find_detection_models(experiment_dir)
                # Map model_key to detection model naming
                if model_key == 'yolo10':
                    model_type = 'yolov10'
                elif model_key == 'yolo11':
                    model_type = 'yolov11'
                elif model_key == 'yolo12':
                    model_type = 'yolov12'
                elif model_key == 'rtdetr':
                    model_type = 'rtdetr'
                else:
                    model_type = model_key

                if model_type in existing_models:
                    model_path = existing_models[model_type][0]  # Use first available model
                    centralized_detection_path = model_path.parent.parent
                    # Extract actual experiment name from the found model path
                    actual_exp_name = centralized_detection_path.name
                    det_exp_name = actual_exp_name  # Update to use actual name
                    print(f"   [SUCCESS] Found existing model: {model_path}")
                    print(f"   [UPDATE] Using experiment name: {det_exp_name}")
                else:
                    print(f"   [ERROR] No existing {model_type} model found")
                    failed_models.append(f"{model_key} (no existing detection model)")
                    continue
            else:
                print(f"   [ERROR] Cannot skip detection in non-continue mode")
                failed_models.append(f"{model_key} (detection required)")
                continue

        # Only run detection training if we're not skipping it
        if start_stage is None or start_stage == 'detection':
            # Direct YOLO training command with auto-download for YOLOv10, YOLOv11, YOLOv12
            if detection_model == "yolov10_detection":
                yolo_model = "yolov10m.pt"  # YOLOv10 medium
            elif detection_model == "yolov11_detection":
                yolo_model = "yolo11m.pt"
            elif detection_model == "yolov12_detection":
                yolo_model = "yolo12m.pt"  # YOLOv12 medium
            elif detection_model == "rtdetr_detection":
                yolo_model = "rtdetr-l.pt"

            # Auto-detect dataset based on selection
            if args.dataset == "iml_lifecycle":
                lifecycle_path = Path("data/processed/lifecycle")
                if lifecycle_path.exists() and (lifecycle_path / "data.yaml").exists():
                    data_yaml = "data/processed/lifecycle/data.yaml"
                    print(f"[IML_LIFECYCLE] Using IML lifecycle dataset for detection training (1 class: parasite)")
                else:
                    print(f"[ERROR] IML lifecycle dataset not found! Setup first.")
                    failed_models.append(f"{model_key} (no IML lifecycle dataset)")
                    continue
            elif args.dataset == "mp_idb_stages":
                stage_path = Path("data/processed/stages")
                if stage_path.exists() and (stage_path / "data.yaml").exists():
                    data_yaml = "data/processed/stages/data.yaml"
                    print(f"[MP_IDB_STAGES] Using MP-IDB stages dataset for detection training (1 class: parasite)")
                else:
                    print(f"[ERROR] MP-IDB stages dataset not found! Setup first.")
                    failed_models.append(f"{model_key} (no MP-IDB stages dataset)")
                    continue
            else:  # mp_idb_species
                # MP-IDB species dataset logic (single-class detection)
                species_path = Path("data/processed/species")
                if species_path.exists() and (species_path / "data.yaml").exists():
                    data_yaml = "data/processed/species/data.yaml"
                    print(f"[MP_IDB_SPECIES] Using MP-IDB species dataset for detection training (1 class: parasite)")
                else:
                    print(f"[ERROR] MP-IDB species dataset not found! Setup first.")
                    failed_models.append(f"{model_key} (no MP-IDB species dataset)")
                    continue

            # Use consistent optimized training for ALL datasets
            print(f"[TARGET] Using OPTIMIZED training for {args.dataset}")
            if not run_optimized_training(yolo_model, data_yaml, args.epochs_det,
                                        det_exp_name, centralized_detection_path, args.dataset):
                failed_models.append(f"{model_key} (detection)")
                continue

            # Wait for weights directly in centralized location
            # Handle YOLO's automatic folder name increments (e.g., exp, exp2, exp3...)
            def find_actual_model_path(base_path, exp_name):
                """Find the actual folder created by YOLO (handles auto-increment)"""
                parent_dir = base_path.parent

                # Look for exact match first
                exact_path = parent_dir / exp_name / "weights" / "best.pt"
                if exact_path.exists():
                    return exact_path

                # Look for numbered variants (exp2, exp3, etc.)
                for i in range(2, 20):  # Check up to exp19
                    numbered_path = parent_dir / f"{exp_name}{i}" / "weights" / "best.pt"
                    if numbered_path.exists():
                        return numbered_path

                return None

            model_path = find_actual_model_path(centralized_detection_path, det_exp_name)

            if model_path is None:
                # Fall back to waiting for the expected path
                model_path = centralized_detection_path / "weights" / "best.pt"
                if not wait_for_file(str(model_path), max_wait_seconds=120, check_interval=3):
                    failed_models.append(f"{model_key} (weights missing)")
                    continue
            else:
                # Update experiment name and path to reflect what YOLO actually created
                actual_exp_name = model_path.parent.parent.name
                if actual_exp_name != det_exp_name:
                    print(f"   [UPDATE] YOLO created experiment: {actual_exp_name} (requested: {det_exp_name})")
                    det_exp_name = actual_exp_name
                centralized_detection_path = model_path.parent.parent
                print(f"[SUCCESS] Found model at: {model_path}")

            print(f"[SUCCESS] Detection model saved directly to: {centralized_detection_path}")

        # Store detection model info for later stages
        if centralized_detection_path:
            detection_models_trained.append({
                'model_key': model_key,
                'detection_model': detection_model,
                'det_exp_name': det_exp_name,
                'path': centralized_detection_path
            })
            print(f"[SUCCESS] {model_key.upper()} detection training completed")
        else:
            detection_models_failed.append(model_key)
            print(f"[FAILED] {model_key.upper()} detection training failed")

        # CHECK: Stop after detection stage if requested
        if hasattr(args, 'stop_stage') and args.stop_stage == 'detection':
            print(f"\n[STOP] Stopping after detection stage as requested (--stop-stage detection)")
            continue

    # END OF DETECTION TRAINING LOOP
    print(f"\n[SUMMARY] Detection Training Results:")
    print(f"   Successful: {len(detection_models_trained)} models")
    print(f"   Failed: {len(detection_models_failed)} models")

    if not detection_models_trained and start_stage in [None, 'detection']:
        print(f"[ERROR] No detection models trained successfully. Cannot continue.")
        return

    # =============================================================================
    # STAGE 2: GENERATE GROUND TRUTH CROPS (ONCE, SHARED)
    # =============================================================================

    shared_crops_path = None
    print(f"\n{'='*80}")
    print(f"STAGE 2: GROUND TRUTH CROPS GENERATION (SHARED)")
    print(f"{'='*80}")

    if start_stage is None or start_stage in ['detection', 'crop']:
            print(f"\n[PROCESS] STAGE 2: Generating ground truth crops for {model_key}")
            print(f"   [IMPROVED] Using ground truth annotations for cleaner classification training")
            print(f"   [INFO] Ground truth crops eliminate detection noise from classification training")

            # Auto-detect raw dataset and type for ground truth crop generation
            if args.dataset == "iml_lifecycle":
                raw_dataset_path = "data/raw/malaria_lifecycle"
                dataset_type = "iml_lifecycle"
                print(f"[IML_LIFECYCLE] Using IML lifecycle raw annotations for ground truth crops")
            elif args.dataset == "mp_idb_stages":
                raw_dataset_path = "data/raw/kaggle_dataset/MP-IDB-YOLO"
                dataset_type = "mp_idb_stages"
                print(f"[MP_IDB_STAGES] Using MP-IDB raw annotations for ground truth crops")
            elif args.dataset == "mp_idb_species":
                raw_dataset_path = "data/raw/kaggle_dataset/MP-IDB-YOLO"
                dataset_type = "mp_idb_species"
                print(f"[MP_IDB_SPECIES] Using MP-IDB raw annotations for ground truth crops")
            else:
                print(f"[ERROR] Unknown dataset type: {args.dataset}")
                failed_models.append(f"{model_key} (unknown dataset)")
                continue

            # Check if raw dataset exists
            if not Path(raw_dataset_path).exists():
                print(f"[ERROR] Raw dataset not found: {raw_dataset_path}")
                failed_models.append(f"{model_key} (raw dataset missing)")
                continue

            # NEW: Use centralized crops path (create directory when needed)
            centralized_crops_path = results_manager.create_crops_path(model_key, det_exp_name)
            output_path = str(centralized_crops_path)

            # Generate ground truth crops using our improved script
            cmd2 = [
                "python", "scripts/training/generate_ground_truth_crops.py",
                "--dataset", raw_dataset_path,
                "--output", output_path,
                "--type", dataset_type,
                "--crop_size", "224"  # FIXED: Use 224px to match pre-processed ground truth
            ]

            if not run_command(cmd2, f"Generating ground truth crops for {model_key}"):
                failed_models.append(f"{model_key} (ground truth crops)")
                continue

            # Verify crop data in CENTRALIZED location (ground truth crops use different structure)
            crop_data_path = centralized_crops_path / "crops"
            if not crop_data_path.exists():
                # Try direct structure (ground truth crops may use different structure)
                crop_data_path = centralized_crops_path
                if not any(crop_data_path.glob("*/images")):  # Check for train/val/test structure
                    failed_models.append(f"{model_key} (ground truth crop data missing)")
                    continue

            # Count ground truth crops
            total_crops = 0
            for split in ['train', 'val', 'test']:
                split_path = crop_data_path / split
                if split_path.exists():
                    for class_dir in split_path.iterdir():
                        if class_dir.is_dir():
                            total_crops += len(list(class_dir.glob("*.jpg")))

            print(f"   [INFO] Generated {total_crops} ground truth crops")
            print(f"   [IMPROVED] Ground truth crops provide cleaner classification training data")
            if total_crops == 0:
                failed_models.append(f"{model_key} (no ground truth crops generated)")
                continue

        # CHECK: Stop after crop generation stage if requested
        if hasattr(args, 'stop_stage') and args.stop_stage == 'crop':
            print(f"\n[STOP] Stopping after crop generation stage as requested (--stop-stage crop)")
            successful_models.append(model_key)
            continue
        elif start_stage in ['classification', 'analysis']:
            print(f"\n[SKIP] STAGE 2: Skipping crop generation (start_stage={start_stage})")
            # Try to find existing crop data
            if continue_mode:
                crop_dirs = find_crop_data(experiment_dir)
                # Find crop data for this specific model
                model_crop_dir = None
                for crop_dir in crop_dirs:
                    if model_key in crop_dir.name:
                        model_crop_dir = crop_dir
                        break

                if model_crop_dir:
                    centralized_crops_path = model_crop_dir
                    crop_data_path = model_crop_dir / "crops"
                    if not crop_data_path.exists():
                        crop_data_path = model_crop_dir  # Direct structure
                    print(f"   [SUCCESS] Found existing crop data: {crop_data_path}")
                else:
                    print(f"   [ERROR] No existing crop data found for {model_key}")
                    failed_models.append(f"{model_key} (no existing crop data)")
                    continue
            else:
                print(f"   [ERROR] Cannot skip crop generation in non-continue mode")
                failed_models.append(f"{model_key} (crops required)")
                continue

        # STAGE 3: Train Classification Models
        if start_stage is None or start_stage in ['detection', 'crop', 'classification']:
            print(f"\n[TRAIN] STAGE 3: Training classification for {model_key}")
            classification_success = []
            classification_failed = []

            for cls_model_name in selected_classification:
                if cls_model_name not in classification_configs:
                    continue

                cls_config = classification_configs[cls_model_name]
                cls_exp_name = f"{base_exp_name}_{model_key}_{cls_model_name}_cls"

                print(f"   [START] Training {cls_config.get('display_name', cls_model_name.upper())}")

                # NEW: Create centralized path for classification (folder needed for saving)
                centralized_cls_path = results_manager.create_experiment_path("training", cls_config['model'], cls_exp_name)

                if cls_config["type"] == "yolo":
                    # YOLO classification - direct training command
                    yolo_cls_model = "yolov11n-cls.pt"  # Default to YOLOv11 classification

                    cmd3 = [
                        "yolo", "classify", "train",
                        f"model={yolo_cls_model}",
                        f"data={crop_data_path}",
                        f"epochs={args.epochs_cls}",
                        f"name={cls_exp_name}",
                        f"project={centralized_cls_path.parent}",
                        f"device={'cuda' if torch.cuda.is_available() else 'cpu'}"
                    ]
                else:
                    # PyTorch classification - systematic comparison with loss functions
                    cmd3 = [
                        "python", cls_config["script"],
                        "--data", str(crop_data_path),
                        "--model", cls_config["model"],
                        "--epochs", str(cls_config["epochs"]),  # Use config epochs
                        "--batch", str(cls_config["batch"]),
                        "--lr", str(cls_config["lr"]),
                        "--loss", cls_config["loss"],           # Cross-entropy or focal
                        "--device", "cuda" if torch.cuda.is_available() else "cpu",
                        "--name", cls_exp_name,
                        "--save-dir", str(centralized_cls_path)
                    ]

                    # Add focal loss parameters if needed
                    if cls_config["loss"] == "focal":
                        cmd3.extend([
                            "--focal_alpha", str(cls_config["focal_alpha"]),
                            "--focal_gamma", str(cls_config["focal_gamma"])
                        ])

                if run_command(cmd3, f"Training {cls_model_name.upper()}"):
                    print(f"[SUCCESS] Classification model saved directly to: {centralized_cls_path}")
                    classification_success.append(cls_model_name)
                else:
                    classification_failed.append(cls_model_name)

            if not classification_success:
                failed_models.append(f"{model_key} (all classification)")
                continue

        # CHECK: Stop after classification stage if requested
        if hasattr(args, 'stop_stage') and args.stop_stage == 'classification':
            print(f"\n[STOP] Stopping after classification stage as requested (--stop-stage classification)")
            successful_models.append(model_key)
            continue
        elif start_stage == 'analysis':
            print(f"\n[SKIP] STAGE 3: Skipping classification training (start_stage={start_stage})")
            # Try to find existing classification models
            if continue_mode:
                # Look for existing classification models in the experiment
                exp_path = Path(experiment_dir)
                classification_success = []

                # Look through all model type directories
                for model_type_dir in (exp_path / "models").glob("*"):
                    if model_type_dir.is_dir():
                        # Look for experiment directories within each model type
                        for exp_dir in model_type_dir.glob("*"):
                            if exp_dir.is_dir() and model_key in exp_dir.name:
                                # Extract the classification model name from the path
                                cls_model_name = model_type_dir.name
                                classification_success.append(cls_model_name)

                if classification_success:
                    print(f"   [SUCCESS] Found existing classification models: {classification_success}")
                else:
                    print(f"   [ERROR] No existing classification models found for {model_key}")
                    failed_models.append(f"{model_key} (no existing classification models)")
                    continue
            else:
                print(f"   [ERROR] Cannot skip classification in non-continue mode")
                failed_models.append(f"{model_key} (classification required)")
                continue

        # STAGE 4: Create Organized Analysis for each successful classification model
        if start_stage is None or start_stage in ['detection', 'crop', 'classification', 'analysis']:
            print(f"\n[ANALYZE] STAGE 4: Creating organized analysis for {model_key}")
        else:
            print(f"\n[SKIP] STAGE 4: Skipping analysis (start_stage={start_stage})")

        if start_stage is None or start_stage in ['detection', 'crop', 'classification', 'analysis']:
            for cls_model_name in classification_success:
                cls_exp_name = f"{base_exp_name}_{model_key}_{cls_model_name}_cls"

                # NEW: Use centralized analysis path
                centralized_analysis_path = results_manager.create_analysis_path(f"{model_key}_{cls_model_name}_complete")
                analysis_dir = str(centralized_analysis_path)

                # Find classification model in CENTRALIZED location (do NOT create folders)
                if cls_model_name in ["yolo11"]:
                    cls_config_name = "yolov11_classification"  # Use YOLOv11 classification
                    # Look in centralized classification results
                    classification_model = results_manager.find_experiment_path("training", cls_config_name, cls_exp_name) / "weights" / "best.pt"
                else:
                    # PyTorch models in centralized location - uses .pt extension
                    classification_model = results_manager.find_experiment_path("training", f"pytorch_classification_{cls_model_name}", cls_exp_name) / "best.pt"

                # Use centralized test data path
                test_data = centralized_crops_path / "yolo_classification" / "test"

                # Check if paths exist before running analysis
                if Path(classification_model).exists() and Path(test_data).exists():
                    print(f"   [INFO] Running analysis for {cls_model_name.upper()}")

                    # Use standalone classification analysis script
                    analysis_cmd = [
                        "python", "scripts/analysis/classification_deep_analysis.py",
                        "--model", classification_model,
                        "--test-data", test_data,
                        "--output", analysis_dir
                    ]

                    run_command(analysis_cmd, f"Analysis for {cls_model_name.upper()}")

            # STAGE 4B: IoU Variation Analysis (NO RE-TESTING) - once per detection model
            # Use pre-computed training results for IoU analysis (much faster!)
            print(f"   [INFO] Running IoU variation analysis from training results (no re-testing)")

            # Use CENTRALIZED detection results.csv path
            detection_results_csv = centralized_detection_path / "results.csv"

            if detection_results_csv.exists() and classification_success:
                first_cls = classification_success[0]
                # Use centralized analysis path for IoU
                iou_analysis_path = results_manager.create_analysis_path(f"{model_key}_{first_cls}_iou_variation")
                iou_analysis_dir = str(iou_analysis_path)

                # Use improved IoU analysis script (no model loading or re-testing)
                iou_cmd = [
                    "python", "scripts/analysis/compare_models_performance.py",
                    "--iou-from-results",
                    "--results-csv", str(detection_results_csv),
                    "--output", iou_analysis_dir,
                    "--experiment-name", f"{model_key}_{det_exp_name}"
                ]

                print(f"   [FAST] Using pre-computed results from: {detection_results_csv}")
                print(f"   [ADVANTAGE] No model loading or re-prediction required")
                run_command(iou_cmd, f"IoU Analysis (Fast) for {model_key}")
            else:
                print(f"   [WARNING] Skipping IoU analysis - detection results.csv not found or no classification success")
                if not detection_results_csv.exists():
                    print(f"   [DEBUG] Expected results.csv at: {detection_results_csv}")

            # STAGE 4C: IEEE Access Compliant Analysis (Journal Ready)
            if args.continue_from and classification_success:
                print(f"   [INFO] Running IEEE Access compliant analysis")
                ieee_cmd = [
                    "python", "scripts/analysis/unified_journal_analysis.py",
                    "--centralized-experiment", args.continue_from
                ]
                try:
                    run_command(ieee_cmd, f"IEEE Analysis for {args.continue_from}")
                    print(f"   [SUCCESS] IEEE compliant analysis completed")
                except Exception as e:
                    print(f"   [WARNING] IEEE analysis failed: {e}")

            # STAGE 4D: Dataset Statistics Analysis (Train/Val/Test + Augmentation Effects)
            print(f"   [INFO] Running dataset statistics analysis")
            try:
                # Use centralized analysis path for dataset stats
                dataset_stats_path = results_manager.create_analysis_path(f"{model_key}_dataset_statistics")
                dataset_stats_dir = str(dataset_stats_path)

                # Run dataset statistics analyzer
                dataset_stats_cmd = [
                    "python", "scripts/analysis/dataset_statistics_analyzer.py",
                    "--output", dataset_stats_dir
                ]

                run_command(dataset_stats_cmd, f"Dataset Statistics Analysis")
                print(f"   [SUCCESS] Dataset statistics analysis completed")
                print(f"   [INFO] Dataset splits and augmentation effects analyzed")
            except Exception as e:
                print(f"   [WARNING] Dataset statistics analysis failed: {e}")

            # Create experiment summaries in centralized location
            for cls_model_name in classification_success:
                cls_exp_name = f"{base_exp_name}_{model_key}_{cls_model_name}_cls"
                # Use centralized summary path
                summary_path = results_manager.create_analysis_path(f"{model_key}_{cls_model_name}_complete")
                create_experiment_summary(str(summary_path), model_key, det_exp_name, cls_exp_name, detection_model, cls_model_name)

        successful_models.append(f"{model_key} ({', '.join(classification_success)})")
        print(f"\n[SUCCESS] {model_key.upper()} PIPELINE COMPLETED")
        if classification_failed:
            print(f"[ERROR] Failed: {', '.join(classification_failed)}")

    # Final summary
    print(f"\n[DONE] PIPELINE FINISHED")
    if successful_models:
        print(f"\n[SUCCESS] SUCCESSFUL ({len(successful_models)}):")
        for model in successful_models:
            print(f"   {model}")
    if failed_models:
        print(f"\n[ERROR] FAILED ({len(failed_models)}):")
        for model in failed_models:
            print(f"   {model}")
    print(f"\nSuccess Rate: {len(successful_models)}/{len(models_to_run)}")

    # STAGE 5: Create ZIP from centralized results (automatic by default)
    if not args.no_zip and successful_models:
        try:
            zip_filename, centralized_dir = create_centralized_zip(base_exp_name, results_manager)
            if zip_filename:
                print(f"\n[TARGET] FINAL DELIVERABLE:")
                print(f"[ARCHIVE] Download: {zip_filename}")
                print(f"[INFO] Or browse: {centralized_dir}/")
            else:
                print(f"[ERROR] Failed to create ZIP archive")
        except Exception as e:
            print(f"[ERROR] Failed to create ZIP: {e}")
    elif not args.no_zip:
        print(f"\n[WARNING] No successful experiments to zip")
    else:
        print(f"\n[INFO] Results saved in centralized structure:")
        print(f"[SUCCESS] All results: {results_manager.pipeline_dir}/")

if __name__ == "__main__":
    main()
