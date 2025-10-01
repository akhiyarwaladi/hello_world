#!/usr/bin/env python3
"""
Batch Generate Detection + Classification Figures for All Datasets

Automatically generates figures for all available datasets in an Option A experiment folder.
Processes all datasets with all detection and classification model combinations.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def find_experiments(parent_folder):
    """Find all experiment folders in parent directory"""
    experiments_dir = Path(parent_folder) / "experiments"

    if not experiments_dir.exists():
        print(f"[ERROR] Experiments directory not found: {experiments_dir}")
        return []

    experiments = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith("experiment_"):
            experiments.append(exp_dir)

    return sorted(experiments)


def get_dataset_name(experiment_dir):
    """Extract dataset name from experiment folder name"""
    # experiment_mp_idb_species -> mp_idb_species
    return experiment_dir.name.replace("experiment_", "")


def get_processed_dataset_path(dataset_name):
    """Map dataset name to processed data path"""
    mapping = {
        "iml_lifecycle": "lifecycle",
        "mp_idb_species": "species",
        "mp_idb_stages": "stages"
    }

    return mapping.get(dataset_name, dataset_name)


def find_detection_models(experiment_dir):
    """Find all detection model folders in experiment"""
    det_models = []
    for folder in experiment_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("det_"):
            weights_file = folder / "weights" / "best.pt"
            if weights_file.exists():
                det_models.append(folder)
    return sorted(det_models)


def find_classification_models(experiment_dir):
    """Find all classification model folders in experiment"""
    cls_models = []
    for folder in experiment_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("cls_"):
            weights_file = folder / "best.pt"
            if weights_file.exists():
                cls_models.append(folder)
    return sorted(cls_models)


def get_crops_dir(experiment_dir):
    """Find ground truth crops directory"""
    for folder in experiment_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("crops_"):
            return folder
    return None


def generate_figures_for_combination(
    detection_model,
    classification_model,
    test_images_dir,
    test_labels_dir,
    crops_dir,
    output_dir,
    max_images=None,
    conf_threshold=0.25
):
    """Generate figures for one detection+classification combination"""

    script_path = Path(__file__).parent / "generate_detection_classification_figures.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--detection-model", str(detection_model),
        "--classification-model", str(classification_model),
        "--test-images", str(test_images_dir),
        "--test-labels", str(test_labels_dir),
        "--gt-crops", str(crops_dir),
        "--output", str(output_dir),
        "--det-conf-threshold", str(conf_threshold)
    ]

    if max_images is not None:
        cmd.extend(["--max-images", str(max_images)])

    print(f"\n{'='*80}")
    print(f"Running: {' '.join([str(x) for x in cmd])}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"[ERROR] Failed to generate figures:")
        print(result.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Batch generate detection+classification figures for all datasets'
    )
    parser.add_argument('--parent-folder', type=str, required=True,
                       help='Parent Option A experiment folder (e.g., results/optA_20251001_183508)')
    parser.add_argument('--output-base', type=str, default='paper_figures_all',
                       help='Base output directory (default: paper_figures_all)')
    parser.add_argument('--detection-models', type=str, nargs='+', default=['all'],
                       help='Detection models to use (yolo10, yolo11, yolo12, or all)')
    parser.add_argument('--classification-models', type=str, nargs='+', default=['all'],
                       help='Classification models to use (specific names or all)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum images per dataset (default: all)')
    parser.add_argument('--det-conf-threshold', type=float, default=0.25,
                       help='Detection confidence threshold (default: 0.25)')
    parser.add_argument('--datasets', type=str, nargs='+', default=['all'],
                       help='Specific datasets to process (default: all)')

    args = parser.parse_args()

    print("="*80)
    print("BATCH DETECTION + CLASSIFICATION FIGURE GENERATION")
    print("="*80)
    print(f"Parent folder: {args.parent_folder}")
    print(f"Output base: {args.output_base}")
    print(f"Max images per dataset: {args.max_images or 'ALL'}")
    print(f"Detection models: {args.detection_models}")
    print(f"Classification models: {args.classification_models}")
    print("="*80)

    # Find all experiments
    experiments = find_experiments(args.parent_folder)

    if not experiments:
        print("[ERROR] No experiment folders found!")
        return

    print(f"\n[FOUND] {len(experiments)} experiment(s):")
    for exp in experiments:
        print(f"  - {exp.name}")

    # Process each experiment
    total_generated = 0
    total_failed = 0

    for exp_dir in experiments:
        dataset_name = get_dataset_name(exp_dir)

        # Check if this dataset should be processed
        if 'all' not in args.datasets and dataset_name not in args.datasets:
            print(f"\n[SKIP] {dataset_name} (not in requested datasets)")
            continue

        print(f"\n{'='*80}")
        print(f"PROCESSING: {dataset_name}")
        print(f"{'='*80}")

        # Get paths
        processed_dataset = get_processed_dataset_path(dataset_name)
        test_images = Path("data/processed") / processed_dataset / "test" / "images"
        test_labels = Path("data/processed") / processed_dataset / "test" / "labels"
        crops_dir = get_crops_dir(exp_dir)

        # Validate paths
        if not test_images.exists():
            print(f"[ERROR] Test images not found: {test_images}")
            total_failed += 1
            continue

        if not test_labels.exists():
            print(f"[ERROR] Test labels not found: {test_labels}")
            total_failed += 1
            continue

        if not crops_dir:
            print(f"[ERROR] Crops directory not found in {exp_dir}")
            total_failed += 1
            continue

        # Find models
        det_models = find_detection_models(exp_dir)
        cls_models = find_classification_models(exp_dir)

        print(f"[FOUND] {len(det_models)} detection models, {len(cls_models)} classification models")

        # Filter detection models
        if 'all' not in args.detection_models:
            det_models = [m for m in det_models if any(name in m.name for name in args.detection_models)]

        # Filter classification models
        if 'all' not in args.classification_models:
            cls_models = [m for m in cls_models if any(name in m.name for name in args.classification_models)]

        print(f"[SELECTED] {len(det_models)} detection models, {len(cls_models)} classification models")

        if not det_models or not cls_models:
            print("[ERROR] No models selected after filtering!")
            total_failed += 1
            continue

        # Generate figures for each combination
        for det_model in det_models:
            for cls_model in cls_models:
                det_name = det_model.name.replace("det_", "")
                cls_name = cls_model.name.replace("cls_", "").replace("_classification", "")

                output_dir = Path(args.output_base) / dataset_name / f"{det_name}_{cls_name}"

                print(f"\n[GENERATE] {dataset_name} - {det_name} + {cls_name}")

                success = generate_figures_for_combination(
                    detection_model=det_model / "weights" / "best.pt",
                    classification_model=cls_model / "best.pt",
                    test_images_dir=test_images,
                    test_labels_dir=test_labels,
                    crops_dir=crops_dir,
                    output_dir=output_dir,
                    max_images=args.max_images,
                    conf_threshold=args.det_conf_threshold
                )

                if success:
                    total_generated += 1
                else:
                    total_failed += 1

    # Summary
    print(f"\n{'='*80}")
    print("BATCH GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total combinations generated: {total_generated}")
    print(f"Total failures: {total_failed}")
    print(f"Output directory: {args.output_base}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
