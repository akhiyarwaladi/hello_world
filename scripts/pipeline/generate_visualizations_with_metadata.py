#!/usr/bin/env python3
"""
Generate visualizations WITH CSV metadata for paper selection
Enhanced version that exports CSV to help identify best images
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def generate_detection_visualizations_with_metadata(
    detection_models_trained,
    test_images_dir,
    test_labels_dir,
    output_base_dir,
    max_images=None
):
    """Generate detection visualizations with CSV metadata"""

    print(f"\n{'='*80}")
    print(f"[DETECTION VIZ + CSV] Generating detection visualizations with metadata")
    print(f"{'='*80}")

    successful = 0
    failed = 0

    for det_model_info in detection_models_trained:
        det_model_key = det_model_info['model_key']
        det_model_path = det_model_info['path'] / "weights" / "best.pt"

        if not det_model_path.exists():
            print(f"   [ERROR] Detection model not found: {det_model_path}")
            failed += 1
            continue

        print(f"\n   [GENERATE] Detection + CSV: {det_model_key.upper()}")

        # Output folder for this detection model
        output_dir = Path(output_base_dir) / f"pred_detection_{det_model_key}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command to generate detection visualization WITH METADATA
        cmd = [
            sys.executable,
            "scripts/visualization/generate_detection_only_with_metadata.py",
            "--detection-model", str(det_model_path),
            "--test-images", str(test_images_dir),
            "--test-labels", str(test_labels_dir),
            "--output", str(output_dir),
            "--det-conf-threshold", "0.25",
            "--iou-threshold", "0.5"
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
                print(f"      [SUCCESS] Generated: {output_dir}")
                print(f"      [CSV] Metadata: {output_dir}/detection_metadata.csv")
                successful += 1
            else:
                print(f"      [ERROR] Failed with return code {result.returncode}")
                print(f"      {result.stderr[:500]}")  # First 500 chars
                failed += 1

        except Exception as e:
            print(f"      [ERROR] Exception: {e}")
            failed += 1

    print(f"\n   [SUMMARY] Detection: {successful} successful, {failed} failed")
    return successful > 0


def generate_classification_visualizations_with_metadata(
    classification_models_trained,
    test_images_dir,
    test_labels_dir,
    crops_dir,
    experiment_dir,
    output_base_dir,
    num_classes,
    max_images=None
):
    """Generate classification visualizations with CSV metadata"""

    print(f"\n{'='*80}")
    print(f"[CLASSIFICATION VIZ + CSV] Generating classification visualizations with metadata")
    print(f"{'='*80}")

    successful = 0
    failed = 0

    for cls_model_name in classification_models_trained:
        print(f"\n   [GENERATE] Classification + CSV: {cls_model_name.upper()}")

        # Find classification model path
        # Try with _classification suffix (single-dataset mode)
        cls_path_with_suffix = Path(experiment_dir) / f"cls_{cls_model_name}_classification" / "best.pt"
        # Try without suffix (multi-dataset mode)
        cls_path_no_suffix = Path(experiment_dir) / f"cls_{cls_model_name}" / "best.pt"

        if cls_path_with_suffix.exists():
            cls_model_path = cls_path_with_suffix
        elif cls_path_no_suffix.exists():
            cls_model_path = cls_path_no_suffix
        else:
            print(f"      [ERROR] Classification model not found: {cls_model_name}")
            print(f"         Tried: {cls_path_with_suffix}")
            print(f"         Tried: {cls_path_no_suffix}")
            failed += 1
            continue

        # Output folder for this classification model
        output_dir = Path(output_base_dir) / f"pred_classification_{cls_model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            "scripts/visualization/generate_classification_only_with_metadata.py",
            "--classification-model", str(cls_model_path),
            "--test-images", str(test_images_dir),
            "--test-labels", str(test_labels_dir),
            "--crops-dir", str(crops_dir),
            "--output", str(output_dir),
            "--num-classes", str(num_classes)
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
                print(f"      [SUCCESS] Generated: {output_dir}")
                print(f"      [CSV] Image metadata: {output_dir}/classification_metadata_images.csv")
                print(f"      [CSV] Box metadata: {output_dir}/classification_metadata_boxes.csv")
                successful += 1
            else:
                print(f"      [ERROR] Failed with return code {result.returncode}")
                print(f"      {result.stderr[:500]}")
                failed += 1

        except Exception as e:
            print(f"      [ERROR] Exception: {e}")
            failed += 1

    print(f"\n   [SUMMARY] Classification: {successful} successful, {failed} failed")
    return successful > 0


def generate_all_visualizations_with_metadata(
    experiment_dir,
    dataset_name,
    detection_models_trained,
    classification_models_trained,
    num_classes,
    max_images=None
):
    """
    Generate all visualizations with CSV metadata

    Args:
        experiment_dir: Experiment folder (e.g., results/optA_.../experiments/experiment_md_2019_stages)
        dataset_name: Dataset name (md_2019_stages, iml_lifecycle, etc.)
        detection_models_trained: List of detection model info dicts
        classification_models_trained: List of classification model names
        num_classes: Number of classes for this dataset
        max_images: Limit number of images (None = all)
    """

    print(f"\n{'='*80}")
    print(f"[VISUALIZATION WITH METADATA] Starting generation for {dataset_name}")
    print(f"{'='*80}")
    print(f"   Detection models: {len(detection_models_trained)}")
    print(f"   Classification models: {len(classification_models_trained)}")
    print(f"   Max images: {max_images if max_images else 'ALL'}")

    experiment_dir = Path(experiment_dir)

    # Find test data paths
    data_dir = Path("data") / dataset_name
    test_images_dir = data_dir / "images" / "test"
    test_labels_dir = data_dir / "labels" / "test"

    # Find crops directory
    crops_gt_path = experiment_dir / "crops_gt_crops"
    if not crops_gt_path.exists():
        print(f"   [ERROR] Crops directory not found: {crops_gt_path}")
        return False

    # Output directory for visualizations
    viz_dir = experiment_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Generate detection visualizations with CSV
    det_success = generate_detection_visualizations_with_metadata(
        detection_models_trained,
        test_images_dir,
        test_labels_dir,
        viz_dir,
        max_images
    )

    # Generate classification visualizations with CSV
    cls_success = generate_classification_visualizations_with_metadata(
        classification_models_trained,
        test_images_dir,
        test_labels_dir,
        crops_gt_path,
        experiment_dir,
        viz_dir,
        num_classes,
        max_images
    )

    # Summary
    print(f"\n{'='*80}")
    print(f"[COMPLETE] Visualization generation finished")
    print(f"{'='*80}")
    print(f"   Detection: {'✓ Success' if det_success else '✗ Failed'}")
    print(f"   Classification: {'✓ Success' if cls_success else '✗ Failed'}")
    print(f"   Output directory: {viz_dir}")
    print(f"\n[USAGE] To find best images for paper:")
    print(f"   1. Open CSV files in {viz_dir}")
    print(f"   2. Sort by 'paper_score' (descending)")
    print(f"   3. Look at 'recommendation' column")
    print(f"   4. Images with score ≥9 are best for paper")

    return det_success or cls_success


def main():
    """
    Standalone mode for re-generating visualizations with metadata
    """
    parser = argparse.ArgumentParser(description='Generate visualizations with CSV metadata')
    parser.add_argument('--experiment-dir', type=str, required=True,
                       help='Experiment directory (e.g., results/optA_20251016_200330/experiments/experiment_md_2019_stages)')
    parser.add_argument('--dataset-name', type=str, required=True,
                       help='Dataset name (md_2019_stages, iml_lifecycle, mp_idb_species, mp_idb_stages)')
    parser.add_argument('--num-classes', type=int, required=True,
                       help='Number of classes')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Limit number of images (default: all)')

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)

    # Auto-detect detection models
    detection_models_trained = []
    for yolo_variant in ['yolo10', 'yolo11', 'yolo12']:
        det_path = experiment_dir / f"det_{yolo_variant}"
        if det_path.exists():
            detection_models_trained.append({
                'model_key': yolo_variant,
                'path': det_path
            })

    # Auto-detect classification models
    classification_models_trained = []
    for cls_model in ['densenet121', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'resnet50', 'resnet101']:
        for loss_type in ['focal', 'ce', 'cb']:
            cls_folder_name = f"cls_{cls_model}_{loss_type}"
            cls_path = experiment_dir / cls_folder_name
            if cls_path.exists():
                classification_models_trained.append(f"{cls_model}_{loss_type}")

    print(f"[AUTO-DETECT] Found {len(detection_models_trained)} detection models")
    print(f"[AUTO-DETECT] Found {len(classification_models_trained)} classification models")

    if not detection_models_trained and not classification_models_trained:
        print(f"[ERROR] No models found in {experiment_dir}")
        return 1

    # Generate visualizations
    success = generate_all_visualizations_with_metadata(
        str(experiment_dir),
        args.dataset_name,
        detection_models_trained,
        classification_models_trained,
        args.num_classes,
        args.max_images
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
