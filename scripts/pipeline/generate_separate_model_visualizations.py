#!/usr/bin/env python3
"""
Generate separate visualizations per model (NOT combinations)

EFFICIENT STRUCTURE:
- 3 detection folders: pred_detection_yolo10/, pred_detection_yolo11/, pred_detection_yolo12/
- 6 classification folders: pred_classification_densenet121/, pred_classification_efficientnet_b1/, etc.
- 1 gt_detection/ folder (shared)
- 1 gt_classification/ folder (shared)

Total: 10 folders (not 3×6=18)
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def generate_detection_visualizations(
    detection_models_trained,
    test_images_dir,
    test_labels_dir,
    output_base_dir,
    max_images=None
):
    """Generate detection visualizations for each detection model"""

    print(f"\n{'='*80}")
    print(f"[DETECTION VIZ] Generating detection visualizations")
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

        print(f"\n   [GENERATE] Detection: {det_model_key.upper()}")

        # Output folder for this detection model
        output_dir = Path(output_base_dir) / f"pred_detection_{det_model_key}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command to generate detection visualization WITH METADATA
        cmd = [
            sys.executable,
            "scripts/visualization/generate_detection_only_with_metadata.py",  # NEW: with CSV metadata
            "--detection-model", str(det_model_path),
            "--test-images", str(test_images_dir),
            "--test-labels", str(test_labels_dir),
            "--output", str(output_dir),
            "--det-conf-threshold", "0.25",
            "--iou-threshold", "0.5"
            # Default: backup old folder to _backup_timestamp (no --no-backup or --keep-old)
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
                successful += 1
            else:
                print(f"      [ERROR] Failed to generate visualizations")
                if result.stderr:
                    print(f"      {result.stderr[:200]}")
                failed += 1

        except Exception as e:
            print(f"      [ERROR] Exception: {e}")
            failed += 1

    return successful, failed


def generate_classification_visualizations(
    classification_models_trained,
    gt_crops_dir,
    test_images_dir,
    test_labels_dir,
    output_base_dir,
    num_classes,
    max_images=None
):
    """Generate classification visualizations for each classification model"""

    print(f"\n{'='*80}")
    print(f"[CLASSIFICATION VIZ] Generating classification visualizations")
    print(f"{'='*80}")

    successful = 0
    failed = 0

    for cls_model_name in classification_models_trained:
        # Find classification model path
        cls_model_dir = Path(output_base_dir).parent / f"cls_{cls_model_name}"
        if not cls_model_dir.exists():
            # Try without cls_ prefix
            cls_model_dir = Path(output_base_dir).parent / cls_model_name

        if not cls_model_dir.exists():
            print(f"   [WARNING] Classification model not found: {cls_model_name}")
            failed += 1
            continue

        cls_model_path = cls_model_dir / "best.pt"
        if not cls_model_path.exists():
            print(f"   [ERROR] Model weights not found: {cls_model_path}")
            failed += 1
            continue

        print(f"\n   [GENERATE] Classification: {cls_model_name.upper()}")

        # Output folder for this classification model
        output_dir = Path(output_base_dir) / f"pred_classification_{cls_model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command to generate classification visualization WITH METADATA
        cmd = [
            sys.executable,
            "scripts/visualization/generate_classification_only_with_metadata.py",  # NEW: with CSV metadata
            "--classification-model", str(cls_model_path),
            "--test-images", str(test_images_dir),
            "--test-labels", str(test_labels_dir),
            "--crops-dir", str(gt_crops_dir),  # Changed from --gt-crops to --crops-dir
            "--output", str(output_dir),
            "--num-classes", str(num_classes)  # NEW: required parameter
            # Default: backup old folder to _backup_timestamp (no --no-backup or --keep-old)
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
                successful += 1
            else:
                print(f"      [ERROR] Failed to generate visualizations")
                if result.stderr:
                    print(f"      {result.stderr[:200]}")
                failed += 1

        except Exception as e:
            print(f"      [ERROR] Exception: {e}")
            failed += 1

    return successful, failed


def generate_ground_truth_visualizations(
    test_images_dir,
    test_labels_dir,
    gt_crops_dir,
    output_base_dir,
    max_images=None
):
    """Generate ground truth visualizations (detection + classification)"""

    print(f"\n{'='*80}")
    print(f"[GT VIZ] Generating ground truth visualizations")
    print(f"{'='*80}")

    # GT Detection
    gt_det_dir = Path(output_base_dir) / "gt_detection"
    gt_det_dir.mkdir(parents=True, exist_ok=True)

    cmd_det = [
        sys.executable,
        "scripts/visualization/generate_gt_detection.py",
        "--test-images", str(test_images_dir),
        "--test-labels", str(test_labels_dir),
        "--output", str(gt_det_dir)
    ]

    if max_images:
        cmd_det.extend(["--max-images", str(max_images)])

    print(f"   [GENERATE] GT Detection")
    try:
        result = subprocess.run(cmd_det, capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
        if result.returncode == 0:
            print(f"      [SUCCESS] Generated: {gt_det_dir}")
        else:
            print(f"      [ERROR] Failed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
    except Exception as e:
        print(f"      [ERROR] Exception: {e}")

    # GT Classification
    gt_cls_dir = Path(output_base_dir) / "gt_classification"
    gt_cls_dir.mkdir(parents=True, exist_ok=True)

    cmd_cls = [
        sys.executable,
        "scripts/visualization/generate_gt_classification.py",
        "--test-images", str(test_images_dir),
        "--test-labels", str(test_labels_dir),
        "--gt-crops", str(gt_crops_dir),
        "--output", str(gt_cls_dir)
    ]

    if max_images:
        cmd_cls.extend(["--max-images", str(max_images)])

    print(f"   [GENERATE] GT Classification")
    try:
        result = subprocess.run(cmd_cls, capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
        if result.returncode == 0:
            print(f"      [SUCCESS] Generated: {gt_cls_dir}")
        else:
            print(f"      [ERROR] Failed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
    except Exception as e:
        print(f"      [ERROR] Exception: {e}")


def generate_all_separate_visualizations(
    experiment_dir,
    dataset_name,
    detection_models_trained,
    classification_models_trained,
    max_images=None,
    centralized_output=None
):
    """Generate all visualizations separately per model (not combinations)

    Args:
        centralized_output: If provided, also generate to centralized folder
                           (e.g., 'visualization_outputs/test_visualizations/full')
    """

    print(f"\n{'='*80}")
    print(f"[VISUALIZE] GENERATING SEPARATE MODEL VISUALIZATIONS")
    print(f"{'='*80}")
    print(f"   Experiment: {experiment_dir}")
    print(f"   Dataset: {dataset_name}")
    print(f"   Detection models: {len(detection_models_trained)}")
    print(f"   Classification models: {len(classification_models_trained)}")
    print(f"   Total folders: {len(detection_models_trained) + len(classification_models_trained) + 2}")
    print(f"   Structure: {len(detection_models_trained)} detection + {len(classification_models_trained)} classification + 2 GT")

    if max_images:
        print(f"   Max images per model: {max_images}")
    else:
        print(f"   Processing: ALL test images")

    if centralized_output:
        print(f"   Centralized output: {centralized_output} (generating to BOTH locations)")

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

    # Determine number of classes per dataset
    num_classes_map = {
        "iml_lifecycle": 4,        # ring, trophozoite, schizont, gametocyte
        "mp_idb_species": 4,       # P_falciparum, P_vivax, P_malariae, P_ovale
        "mp_idb_stages": 4,        # ring, trophozoite, schizont, gametocyte
        "md_2019_stages": 3        # ring, schizont, trophozoite
    }
    num_classes = num_classes_map.get(dataset_name, 4)  # default to 4 if unknown

    # Find ground truth crops directory
    gt_crops_dir = Path(experiment_dir) / "crops_gt_crops"
    if not gt_crops_dir.exists():
        print(f"   [ERROR] Ground truth crops not found: {gt_crops_dir}")
        return False

    # Output base directory
    output_base = Path(experiment_dir) / "visualizations"
    output_base.mkdir(parents=True, exist_ok=True)

    # 1. Generate ground truth visualizations (once)
    generate_ground_truth_visualizations(
        test_images_dir=test_images_dir,
        test_labels_dir=test_labels_dir,
        gt_crops_dir=gt_crops_dir,
        output_base_dir=output_base,
        max_images=max_images
    )

    # 2. Generate detection visualizations (one per detection model)
    det_success, det_failed = generate_detection_visualizations(
        detection_models_trained=detection_models_trained,
        test_images_dir=test_images_dir,
        test_labels_dir=test_labels_dir,
        output_base_dir=output_base,
        max_images=max_images
    )

    # 3. Generate classification visualizations (one per classification model)
    cls_success, cls_failed = generate_classification_visualizations(
        classification_models_trained=classification_models_trained,
        gt_crops_dir=gt_crops_dir,
        test_images_dir=test_images_dir,
        test_labels_dir=test_labels_dir,
        output_base_dir=output_base,
        num_classes=num_classes,  # NEW: pass num_classes
        max_images=max_images
    )

    # 4. Copy to centralized location if requested
    if centralized_output:
        print(f"\n{'='*80}")
        print(f"[CENTRALIZED] COPYING TO CENTRALIZED LOCATION")
        print(f"{'='*80}")

        centralized_base = Path(centralized_output)
        centralized_det = centralized_base / "detection"
        centralized_cls = centralized_base / "classification"
        centralized_det.mkdir(parents=True, exist_ok=True)
        centralized_cls.mkdir(parents=True, exist_ok=True)

        import shutil

        # Copy detection visualizations
        print(f"\n   [COPY] Detection visualizations...")
        det_copied = 0
        for det_model_info in detection_models_trained:
            det_model_key = det_model_info['model_key']
            src_dir = output_base / f"pred_detection_{det_model_key}"
            dest_dir = centralized_det / f"{det_model_key}_{dataset_name}"

            if src_dir.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)
                for png_file in src_dir.glob("*.png"):
                    shutil.copy2(png_file, dest_dir / png_file.name)
                    det_copied += 1
                print(f"      ✓ {det_model_key}_{dataset_name}")

        # Copy classification visualizations
        print(f"\n   [COPY] Classification visualizations...")
        cls_copied = 0
        for cls_model_name in classification_models_trained.keys():
            src_dir = output_base / f"pred_classification_{cls_model_name}"
            dest_dir = centralized_cls / f"{cls_model_name}_{dataset_name}"

            if src_dir.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)
                for png_file in src_dir.glob("*.png"):
                    shutil.copy2(png_file, dest_dir / png_file.name)
                    cls_copied += 1
                print(f"      ✓ {cls_model_name}_{dataset_name}")

        print(f"\n   ✅ Centralized copy complete: {det_copied + cls_copied} files")
        print(f"      Detection: {det_copied} files")
        print(f"      Classification: {cls_copied} files")
        print(f"      Location: {centralized_base}")

    # Summary
    print(f"\n{'='*80}")
    print(f"[SUMMARY] VISUALIZATION GENERATION COMPLETE (WITH CSV METADATA)")
    print(f"{'='*80}")
    print(f"   Detection: {det_success} success, {det_failed} failed")
    print(f"   Classification: {cls_success} success, {cls_failed} failed")
    print(f"   Ground truth: 2 folders (detection + classification)")
    print(f"   Total folders: {det_success + cls_success + 2}")
    print(f"   Location: {output_base}/")
    print(f"\n   [NEW] CSV Metadata Files:")
    print(f"      - detection_metadata.csv (in each pred_detection_* folder)")
    print(f"      - classification_metadata_images.csv (in each pred_classification_* folder)")
    print(f"      - classification_metadata_boxes.csv (in each pred_classification_* folder)")
    print(f"\n   [USAGE] To find best images for paper:")
    print(f"      1. Open CSV files with Excel or pandas")
    print(f"      2. Sort by 'paper_score' (descending)")
    print(f"      3. Images with score ≥9 are best for paper")
    print(f"\n   Structure:")
    print(f"      gt_detection/")
    print(f"      gt_classification/")
    for det_model in detection_models_trained:
        print(f"      pred_detection_{det_model['model_key']}/")
    for cls_model in classification_models_trained:
        print(f"      pred_classification_{cls_model}/")

    return (det_success + cls_success) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate separate visualizations per model (not combinations)"
    )
    parser.add_argument("--experiment-dir", type=str, required=True,
                       help="Experiment directory")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["iml_lifecycle", "mp_idb_species", "mp_idb_stages", "md_2019_stages"],
                       help="Dataset name")
    parser.add_argument("--detection-models", nargs="+", required=True,
                       help="Detection model keys (e.g., yolo10 yolo11)")
    parser.add_argument("--classification-models", nargs="+", required=True,
                       help="Classification model names")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum images to process per model (default: all)")

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
    success = generate_all_separate_visualizations(
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
