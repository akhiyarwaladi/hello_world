#!/usr/bin/env python3
"""
Copy Full Test Visualizations to Centralized Folder

Collects ALL test visualizations from individual experiment folders
and organizes them into a centralized location for easy access.

Structure:
  visualization_outputs/test_visualizations/
    ├── full/              # ALL test images with bbox (this script)
    │   ├── detection/
    │   │   ├── yolo10/
    │   │   ├── yolo11/
    │   │   └── yolo12/
    │   └── classification/
    │       ├── densenet121/
    │       ├── efficientnet_b0/
    │       ├── efficientnet_b1/
    │       ├── efficientnet_b2/
    │       ├── resnet50/
    │       └── resnet101/
    └── selected/          # Top error cases (generate_all_centralized.py)
        ├── detection/
        └── classification/
"""

import sys
import shutil
from pathlib import Path
from typing import Dict, List

def discover_experiments(experiments_dir: Path) -> List[Dict]:
    """Auto-discover experiment structure."""
    experiments = []

    if not experiments_dir.exists():
        print(f"[ERROR] Experiments directory not found: {experiments_dir}")
        return experiments

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith('experiment_'):
            continue

        dataset_name = exp_dir.name.replace('experiment_', '')
        viz_dir = exp_dir / "visualizations"

        detection_models = [d.name.replace('pred_detection_', '')
                          for d in viz_dir.iterdir()
                          if d.is_dir() and d.name.startswith('pred_detection_')] if viz_dir.exists() else []

        classification_models = [d.name.replace('pred_classification_', '').replace('_focal', '')
                               for d in viz_dir.iterdir()
                               if d.is_dir() and d.name.startswith('pred_classification_')] if viz_dir.exists() else []

        experiments.append({
            'dataset': dataset_name,
            'path': exp_dir,
            'viz_dir': viz_dir,
            'detection_models': detection_models,
            'classification_models': classification_models
        })

        print(f"[DISCOVERED] {dataset_name}: {len(detection_models)} det, {len(classification_models)} cls")

    return experiments


def copy_full_test_visualizations(
    experiment_dir: Path,
    output_dir: Path
):
    """Copy all test visualizations to centralized folder."""

    print("\n" + "="*80)
    print("COPYING FULL TEST VISUALIZATIONS")
    print("="*80)
    print(f"Experiment: {experiment_dir}")
    print(f"Output: {output_dir}")
    print("="*80)

    # Create output structure
    full_det_dir = output_dir / "full" / "detection"
    full_cls_dir = output_dir / "full" / "classification"
    full_det_dir.mkdir(parents=True, exist_ok=True)
    full_cls_dir.mkdir(parents=True, exist_ok=True)

    # Discover experiments
    experiments = discover_experiments(experiment_dir)

    if not experiments:
        print("[WARNING] No experiments found!")
        return

    total_det_copied = 0
    total_cls_copied = 0

    # Best models configuration (based on performance analysis)
    # Detection: YOLO11 is best overall across all datasets
    best_detection_model = "yolo11"

    # Classification: Best model per dataset (from verified results)
    best_classification_models = {
        'iml_lifecycle': 'efficientnet_b1',      # 91.51% accuracy
        'mp_idb_species': 'efficientnet_b1',     # Best for species
        'mp_idb_stages': 'resnet50',             # 89.0% accuracy
        'md_2019_stages': 'efficientnet_b0'      # 91.0% accuracy
    }

    # Copy detection visualizations (BEST MODEL ONLY)
    print("\n[1/2] COPYING DETECTION VISUALIZATIONS (BEST MODEL ONLY)")
    print("-"*80)
    print(f"   Best detection model: {best_detection_model.upper()}")
    print()

    for exp in experiments:
        dataset = exp['dataset']
        viz_dir = exp['viz_dir']

        # Only copy best detection model
        if best_detection_model not in exp['detection_models']:
            print(f"  ⚠️  {dataset}: {best_detection_model} not found, skipping")
            continue

        src_dir = viz_dir / f"pred_detection_{best_detection_model}"

        if not src_dir.exists():
            print(f"  ⚠️  {dataset}: source folder not found")
            continue

        # Create output folder with standardized naming: yolo11_iml_lifecycle/
        dest_dir = full_det_dir / f"{best_detection_model}_{dataset}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy all PNG files
        copied = 0
        for png_file in src_dir.glob("*.png"):
            dest_path = dest_dir / png_file.name
            shutil.copy2(png_file, dest_path)
            copied += 1

        if copied > 0:
            print(f"  ✓ {dataset}: {copied} files ({best_detection_model})")
            total_det_copied += copied

    # Copy classification visualizations (BEST MODEL ONLY PER DATASET)
    print("\n[2/2] COPYING CLASSIFICATION VISUALIZATIONS (BEST MODEL PER DATASET)")
    print("-"*80)
    print(f"   Best models:")
    for ds, model in best_classification_models.items():
        print(f"     - {ds}: {model}")
    print()

    for exp in experiments:
        dataset = exp['dataset']
        viz_dir = exp['viz_dir']

        # Get best model for this dataset
        best_cls_model = best_classification_models.get(dataset)

        if not best_cls_model:
            print(f"  ⚠️  {dataset}: no best model configured, skipping")
            continue

        if best_cls_model not in exp['classification_models']:
            print(f"  ⚠️  {dataset}: {best_cls_model} not found, skipping")
            continue

        src_dir = viz_dir / f"pred_classification_{best_cls_model}_focal"

        # Try without _focal suffix if not found
        if not src_dir.exists():
            src_dir = viz_dir / f"pred_classification_{best_cls_model}"

        if not src_dir.exists():
            print(f"  ⚠️  {dataset}: source folder not found")
            continue

        # Create output folder with standardized naming: efficientnet_b1_iml_lifecycle/
        dest_dir = full_cls_dir / f"{best_cls_model}_{dataset}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy all PNG files
        copied = 0
        for png_file in src_dir.glob("*.png"):
            dest_path = dest_dir / png_file.name
            shutil.copy2(png_file, dest_path)
            copied += 1

        if copied > 0:
            print(f"  ✓ {dataset}: {copied} files ({best_cls_model})")
            total_cls_copied += copied

    print("\n" + "="*80)
    print("✅ COPY COMPLETE")
    print("="*80)
    print(f"  Detection: {total_det_copied} files")
    print(f"  Classification: {total_cls_copied} files")
    print(f"  Total: {total_det_copied + total_cls_copied} visualizations")
    print(f"\n  Location: {output_dir / 'full'}")
    print("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Copy all test visualizations to centralized folder'
    )
    parser.add_argument('--experiment', type=str,
                       help='Experiment directory (default: auto-detect latest)')
    parser.add_argument('--output', type=str,
                       default='visualization_outputs/test_visualizations',
                       help='Output directory (default: visualization_outputs/test_visualizations)')

    args = parser.parse_args()

    # Auto-detect latest experiment if not specified
    if args.experiment:
        experiment_dir = Path(args.experiment)
    else:
        results_dir = Path("results")
        experiments = sorted([d for d in results_dir.glob("optA_*") if d.is_dir()],
                           key=lambda x: x.stat().st_mtime, reverse=True)

        if not experiments:
            print("[ERROR] No experiments found in results/")
            return 1

        experiment_dir = experiments[0] / "experiments"
        print(f"[AUTO-DETECT] Latest: {experiments[0].name}\n")

    output_dir = Path(args.output)

    copy_full_test_visualizations(experiment_dir, output_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())
