#!/usr/bin/env python3
"""
Generate ALL detection + classification visualizations for pipeline experiments.
This script is called from main_pipeline.py at STAGE 4H.

Generates 4 folders of visualizations:
1. gt_detection/      - Ground truth boxes with 'parasite' labels (blue)
2. pred_detection/    - Predicted detection boxes with confidence (green)
3. gt_classification/ - Ground truth boxes with class labels (blue)
4. pred_classification/ - GT boxes with predicted labels (green=correct, red=wrong)

For EACH combination of detection model × classification model.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def generate_visualizations_for_experiment(
    detection_model_path,
    classification_model_path,
    test_images_dir,
    test_labels_dir,
    gt_crops_dir,
    output_base_dir,
    model_combination_name,
    max_images=None
):
    """Generate visualizations for a single model combination"""

    # Ensure paths exist
    if not Path(detection_model_path).exists():
        print(f"   [ERROR] Detection model not found: {detection_model_path}")
        return False

    if not Path(classification_model_path).exists():
        print(f"   [ERROR] Classification model not found: {classification_model_path}")
        return False

    if not Path(test_images_dir).exists():
        print(f"   [ERROR] Test images not found: {test_images_dir}")
        return False

    # Create output directory
    output_dir = Path(output_base_dir) / f"paper_figures_{model_combination_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n   [GENERATE] {model_combination_name.upper()}")
    print(f"      Detection: {Path(detection_model_path).parent.parent.name}")
    print(f"      Classification: {Path(classification_model_path).parent.name}")
    print(f"      Output: {output_dir}")

    # Build command
    cmd = [
        sys.executable,
        "scripts/visualization/generate_detection_classification_figures.py",
        "--detection-model", str(detection_model_path),
        "--classification-model", str(classification_model_path),
        "--test-images", str(test_images_dir),
        "--test-labels", str(test_labels_dir),
        "--gt-crops", str(gt_crops_dir),
        "--output", str(output_dir),
        "--det-conf-threshold", "0.25"
    ]

    if max_images:
        cmd.extend(["--max-images", str(max_images)])

    # Run command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode == 0:
            print(f"      [SUCCESS] Generated visualizations: {output_dir}")
            return True
        else:
            print(f"      [ERROR] Failed to generate visualizations")
            if result.stderr:
                print(f"      {result.stderr[:200]}")
            return False

    except Exception as e:
        print(f"      [ERROR] Exception: {e}")
        return False


def generate_all_pipeline_visualizations(
    experiment_dir,
    dataset_name,
    detection_models_trained,
    classification_models_trained,
    max_images=None
):
    """Generate visualizations for all model combinations in pipeline experiment"""

    print(f"\n{'='*80}")
    print(f"[VISUALIZE] GENERATING ALL DETECTION + CLASSIFICATION FIGURES")
    print(f"{'='*80}")
    print(f"   Experiment: {experiment_dir}")
    print(f"   Dataset: {dataset_name}")
    print(f"   Detection models: {len(detection_models_trained)}")
    print(f"   Classification models: {len(classification_models_trained)}")
    print(f"   Total combinations: {len(detection_models_trained) * len(classification_models_trained)}")

    if max_images:
        print(f"   Max images per combination: {max_images}")
    else:
        print(f"   Processing: ALL test images")

    # Determine test images/labels paths based on dataset
    dataset_paths = {
        "iml_lifecycle": ("data/processed/lifecycle/test/images", "data/processed/lifecycle/test/labels"),
        "mp_idb_species": ("data/processed/species/test/images", "data/processed/species/test/labels"),
        "mp_idb_stages": ("data/processed/stages/test/images", "data/processed/stages/test/labels"),
        "md_2019_stages": ("data/processed/md_2019_stages/test/images", "data/processed/md_2019_stages/test/labels")
    }

    if dataset_name not in dataset_paths:
        print(f"   [ERROR] Unknown dataset: {dataset_name}")
        return False

    test_images_dir, test_labels_dir = dataset_paths[dataset_name]

    # Find ground truth crops directory
    gt_crops_dir = Path(experiment_dir) / "crops_gt_crops"
    if not gt_crops_dir.exists():
        print(f"   [ERROR] Ground truth crops not found: {gt_crops_dir}")
        return False

    successful = 0
    failed = 0

    # Generate for each combination
    for det_model_info in detection_models_trained:
        det_model_key = det_model_info['model_key']
        det_model_path = det_model_info['path'] / "weights" / "best.pt"

        for cls_model_name in classification_models_trained:
            # Find classification model path
            cls_model_dir = Path(experiment_dir) / f"cls_{cls_model_name}"
            if not cls_model_dir.exists():
                # Try without cls_ prefix
                cls_model_dir = Path(experiment_dir) / cls_model_name

            if not cls_model_dir.exists():
                print(f"   [WARNING] Classification model not found: {cls_model_name}")
                failed += 1
                continue

            cls_model_path = cls_model_dir / "best.pt"

            # Create combination name
            model_combination_name = f"{det_model_key}_{cls_model_name}"

            # Generate visualizations
            success = generate_visualizations_for_experiment(
                detection_model_path=det_model_path,
                classification_model_path=cls_model_path,
                test_images_dir=test_images_dir,
                test_labels_dir=test_labels_dir,
                gt_crops_dir=gt_crops_dir,
                output_base_dir=experiment_dir,
                model_combination_name=model_combination_name,
                max_images=max_images
            )

            if success:
                successful += 1
            else:
                failed += 1

    # Summary
    print(f"\n{'='*80}")
    print(f"[SUMMARY] VISUALIZATION GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Successful: {successful} combinations")
    print(f"   Failed: {failed} combinations")
    print(f"   Total outputs: {successful * 4} folders (4 per combination)")
    print(f"   Location: {experiment_dir}/paper_figures_*/")

    return successful > 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate all visualizations for pipeline experiment"
    )
    parser.add_argument("--experiment-dir", type=str, required=True,
                       help="Experiment directory (e.g., results/optA_XX/experiments/experiment_YY)")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["iml_lifecycle", "mp_idb_species", "mp_idb_stages", "md_2019_stages"],
                       help="Dataset name")
    parser.add_argument("--detection-models", nargs="+", required=True,
                       help="Detection model keys (e.g., yolo10 yolo11)")
    parser.add_argument("--classification-models", nargs="+", required=True,
                       help="Classification model names (e.g., densenet121_focal efficientnet_b1_focal)")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum images to process per combination (default: all)")

    args = parser.parse_args()

    # Build detection model info list
    detection_models_trained = []
    for model_key in args.detection_models:
        model_path = Path(args.experiment_dir) / f"det_{model_key}"
        if model_path.exists():
            detection_models_trained.append({
                'model_key': model_key,
                'path': model_path
            })
        else:
            print(f"[WARNING] Detection model not found: det_{model_key}")

    # Run visualization generation
    success = generate_all_pipeline_visualizations(
        experiment_dir=args.experiment_dir,
        dataset_name=args.dataset,
        detection_models_trained=detection_models_trained,
        classification_models_trained=args.classification_models,
        max_images=args.max_images
    )

    if success:
        print("\n[SUCCESS] All visualizations generated!")
        return 0
    else:
        print("\n[ERROR] Failed to generate visualizations")
        return 1


if __name__ == "__main__":
    exit(main())
