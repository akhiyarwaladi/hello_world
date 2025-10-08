#!/usr/bin/env python3
"""
Quick Wrapper: Generate Detection+Classification Figures for Experiment
Similar to run_improved_gradcam_on_experiments.py but for bounding box visualizations
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def find_detection_models(exp_folder):
    """Find all detection model checkpoints"""
    exp_path = Path(exp_folder)
    models = []

    for det_folder in exp_path.glob("det_*"):
        best_model = det_folder / "weights" / "best.pt"
        if best_model.exists():
            models.append(best_model)

    return sorted(models)


def find_classification_models(exp_folder):
    """Find all classification model checkpoints"""
    exp_path = Path(exp_folder)
    models = []

    for cls_folder in list(exp_path.glob("cls_*")) + list(exp_path.glob("cls_*_classification")):
        for model_name in ["best.pt", "best_model.pt"]:
            best_model = cls_folder / model_name
            if best_model.exists():
                models.append(best_model)
                break

    return sorted(models)


def find_crops_dir(exp_folder):
    """Find ground truth crops directory"""
    exp_path = Path(exp_folder)

    for folder in exp_path.glob("crops_*"):
        if folder.is_dir():
            return folder

    return None


def get_dataset_name(exp_folder):
    """Extract dataset name from experiment folder"""
    exp_path = Path(exp_folder)
    folder_name = exp_path.name

    if folder_name.startswith('experiment_'):
        return folder_name.replace('experiment_', '')

    return "unknown"


def get_processed_data_path(dataset_name):
    """Map dataset name to processed data folder"""
    mapping = {
        "iml_lifecycle": "lifecycle",
        "mp_idb_species": "species",
        "mp_idb_stages": "stages"
    }

    base_dataset = mapping.get(dataset_name, dataset_name)
    data_path = project_root / "data" / "processed" / base_dataset

    return data_path


def run_visualization(detection_model, classification_model, test_images_dir,
                     test_labels_dir, crops_dir, output_dir, max_images=20):
    """Run detection+classification figure generation"""

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
        "--max-images", str(max_images)
    ]

    print(f"\n{'='*80}")
    print(f"Running: {' '.join([str(c) for c in cmd])}")
    print(f"{'='*80}\n")

    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Detection+Classification figures for experiment'
    )
    parser.add_argument('--exp-folder', type=str, required=True,
                       help='Path to experiment folder (e.g., results/.../experiment_iml_lifecycle/)')
    parser.add_argument('--detection-models', type=str, nargs='+',
                       help='Filter specific detection models (e.g., yolo11)')
    parser.add_argument('--classification-models', type=str, nargs='+',
                       help='Filter specific classification models (e.g., efficientnet_b1_focal)')
    parser.add_argument('--max-images', type=int, default=20,
                       help='Max images to process')
    parser.add_argument('--output-base', type=str, default=None,
                       help='Base output directory (default: exp_folder/detection_classification_figures/)')

    args = parser.parse_args()

    print("="*80)
    print("DETECTION + CLASSIFICATION FIGURE GENERATION")
    print("="*80)
    print(f"Experiment: {args.exp_folder}")

    # Find models
    print("\n[1/4] Finding detection models...")
    det_models = find_detection_models(args.exp_folder)

    if not det_models:
        print(f"[ERROR] No detection models found in {args.exp_folder}")
        return

    print(f"   Found {len(det_models)} detection models:")
    for m in det_models:
        print(f"      - {m.parent.parent.name}")

    # Filter detection models
    if args.detection_models:
        print(f"\n   Filtering detection models: {args.detection_models}")
        filtered = []
        for m in det_models:
            model_name = m.parent.parent.name
            for filter_name in args.detection_models:
                if filter_name.lower() in model_name.lower():
                    filtered.append(m)
                    break
        det_models = filtered
        print(f"   After filtering: {len(det_models)} models")

    print("\n[2/4] Finding classification models...")
    cls_models = find_classification_models(args.exp_folder)

    if not cls_models:
        print(f"[ERROR] No classification models found in {args.exp_folder}")
        return

    print(f"   Found {len(cls_models)} classification models:")
    for m in cls_models:
        print(f"      - {m.parent.name}")

    # Filter classification models
    if args.classification_models:
        print(f"\n   Filtering classification models: {args.classification_models}")
        filtered = []
        for m in cls_models:
            model_name = m.parent.name
            for filter_name in args.classification_models:
                if filter_name.lower() in model_name.lower():
                    filtered.append(m)
                    break
        cls_models = filtered
        print(f"   After filtering: {len(cls_models)} models")

    # Find processed data
    print("\n[3/4] Finding processed data...")
    dataset_name = get_dataset_name(args.exp_folder)
    data_path = get_processed_data_path(dataset_name)

    test_images_dir = data_path / "test" / "images"
    test_labels_dir = data_path / "test" / "labels"

    if not test_images_dir.exists():
        print(f"[ERROR] Test images not found: {test_images_dir}")
        return

    if not test_labels_dir.exists():
        print(f"[ERROR] Test labels not found: {test_labels_dir}")
        return

    print(f"   Dataset: {dataset_name}")
    print(f"   Test images: {test_images_dir}")
    print(f"   Test labels: {test_labels_dir}")

    # Find crops
    crops_dir = find_crops_dir(args.exp_folder)
    if not crops_dir:
        print(f"[WARNING] Crops directory not found in {args.exp_folder}")
        print("   Classification prediction will not be available")

    # Setup output
    if args.output_base:
        output_base = Path(args.output_base)
    else:
        output_base = Path(args.exp_folder) / "detection_classification_figures"

    output_base.mkdir(parents=True, exist_ok=True)

    # Process combinations
    n_combinations = len(det_models) * len(cls_models)
    print(f"\n[4/4] Processing {n_combinations} combinations...")
    print("="*80)

    processed = 0
    for i, det_model in enumerate(det_models, 1):
        det_name = det_model.parent.parent.name

        for j, cls_model in enumerate(cls_models, 1):
            cls_name = cls_model.parent.name

            combo_name = f"{det_name}_{cls_name}"
            output_dir = output_base / combo_name

            print(f"\n{'#'*80}")
            print(f"Combination {processed+1}/{n_combinations}: {combo_name}")
            print(f"{'#'*80}")

            try:
                run_visualization(
                    detection_model=det_model,
                    classification_model=cls_model,
                    test_images_dir=test_images_dir,
                    test_labels_dir=test_labels_dir,
                    crops_dir=crops_dir,
                    output_dir=output_dir,
                    max_images=args.max_images
                )
                processed += 1
            except Exception as e:
                print(f"\n[ERROR] Failed to process {combo_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "="*80)
    print("ALL COMBINATIONS PROCESSED!")
    print("="*80)
    print(f"[OK] Processed {processed}/{n_combinations} combinations")
    print(f"[OK] Output directory: {output_base}")
    print("\nView results:")
    for det_model in det_models:
        det_name = det_model.parent.parent.name
        for cls_model in cls_models:
            cls_name = cls_model.parent.name
            combo_name = f"{det_name}_{cls_name}"
            output_dir = output_base / combo_name
            if output_dir.exists():
                n_files = len(list(output_dir.glob("pred_detection/*.jpg")))
                print(f"  - {combo_name}: {n_files} images")
    print("="*80)


if __name__ == "__main__":
    main()
