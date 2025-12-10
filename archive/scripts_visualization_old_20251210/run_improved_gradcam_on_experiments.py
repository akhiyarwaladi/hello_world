#!/usr/bin/env python3
"""
Helper script to run improved Grad-CAM on experiment results
Automatically detects models and test images from experiment folders
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def find_classification_models(exp_folder):
    """Find all classification model checkpoints in experiment folder"""
    exp_path = Path(exp_folder)

    # Pattern: cls_<model>_<loss>/best.pt or best_model.pt
    models = []

    for cls_folder in list(exp_path.glob("cls_*")) + list(exp_path.glob("cls_*_classification")):
        # Try both naming conventions
        for model_name in ["best.pt", "best_model.pt"]:
            best_model = cls_folder / model_name
            if best_model.exists():
                models.append(best_model)
                break  # Found model, move to next folder

    return sorted(models)


def find_test_images(exp_folder):
    """Find test images from crops_gt_crops folder"""
    exp_path = Path(exp_folder)

    # Look for crops folder (standard structure)
    crops_folder = exp_path / "crops_gt_crops"

    if crops_folder.exists():
        # Try crops/test/ first (new structure)
        test_folder = crops_folder / "crops" / "test"
        if test_folder.exists():
            return test_folder

        # Fallback to direct test/ (old structure)
        test_folder = crops_folder / "test"
        if test_folder.exists():
            return test_folder

    return None


def get_dataset_name(exp_folder):
    """Extract dataset name from experiment folder"""
    exp_path = Path(exp_folder)

    # Pattern: experiment_<dataset_name>
    folder_name = exp_path.name
    if folder_name.startswith('experiment_'):
        return folder_name.replace('experiment_', '')

    return "unknown"


def get_class_names(dataset_name):
    """Get class names for dataset"""

    # Life cycle stages datasets
    lifecycle_classes = ['gametocyte', 'ring', 'schizont', 'trophozoite']

    # Species dataset
    species_classes = ['P_falciparum', 'P_malariae', 'P_ovale', 'P_vivax']

    if 'species' in dataset_name.lower():
        return species_classes
    else:
        return lifecycle_classes


def run_improved_gradcam(model_path, images_path, class_names, output_dir, max_images=10):
    """Run improved Grad-CAM script"""

    script_path = Path(__file__).parent / "generate_improved_gradcam.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--model", str(model_path),
        "--images", str(images_path),
        "--class-names"] + class_names + [
        "--output", str(output_dir),
        "--max-images", str(max_images)
    ]

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description='Run improved Grad-CAM on experiment results'
    )
    parser.add_argument('--exp-folder', type=str, required=True,
                       help='Path to experiment folder (e.g., results/.../experiment_iml_lifecycle/)')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Specific model names to process (e.g., densenet121_focal efficientnet_b1_focal)')
    parser.add_argument('--max-images', type=int, default=10,
                       help='Max images per model to process')
    parser.add_argument('--output-base', type=str, default=None,
                       help='Base output directory (default: exp_folder/gradcam_visualizations/)')

    args = parser.parse_args()

    print("="*80)
    print("IMPROVED GRAD-CAM ON EXPERIMENTS")
    print("="*80)
    print(f"Experiment: {args.exp_folder}")

    # Find models
    print("\n[1/4] Finding classification models...")
    all_models = find_classification_models(args.exp_folder)

    if not all_models:
        print(f"[ERROR] No classification models found in {args.exp_folder}")
        return

    print(f"   Found {len(all_models)} models:")
    for m in all_models:
        print(f"      • {m.parent.name}")

    # Filter models if specified
    if args.models:
        print(f"\n   Filtering for: {args.models}")
        filtered_models = []
        for m in all_models:
            model_name = m.parent.name
            for filter_name in args.models:
                if filter_name.lower() in model_name.lower():
                    filtered_models.append(m)
                    break
        all_models = filtered_models
        print(f"   After filtering: {len(all_models)} models")

    # Find test images
    print("\n[2/4] Finding test images...")
    test_images = find_test_images(args.exp_folder)

    if not test_images or not test_images.exists():
        print(f"[ERROR] No test images found in {args.exp_folder}")
        return

    n_images = len(list(test_images.glob("*.jpg")) + list(test_images.glob("*.png")))
    print(f"   Test images folder: {test_images}")
    print(f"   Total images: {n_images}")

    # Get dataset info
    print("\n[3/4] Getting dataset info...")
    dataset_name = get_dataset_name(args.exp_folder)
    class_names = get_class_names(dataset_name)

    print(f"   Dataset: {dataset_name}")
    print(f"   Classes: {class_names}")

    # Setup output
    if args.output_base:
        output_base = Path(args.output_base)
    else:
        output_base = Path(args.exp_folder) / "gradcam_visualizations"

    output_base.mkdir(parents=True, exist_ok=True)

    print(f"\n[4/4] Processing {len(all_models)} models...")
    print("="*80)

    # Process each model
    for i, model_path in enumerate(all_models, 1):
        model_name = model_path.parent.name

        print(f"\n{'#'*80}")
        print(f"Model {i}/{len(all_models)}: {model_name}")
        print(f"{'#'*80}")

        # Create output dir for this model
        output_dir = output_base / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            run_improved_gradcam(
                model_path=model_path,
                images_path=test_images,
                class_names=class_names,
                output_dir=output_dir,
                max_images=args.max_images
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("ALL MODELS PROCESSED!")
    print("="*80)
    print(f"[OK] Processed {len(all_models)} models")
    print(f"[OK] Output directory: {output_base}")
    print("\nView results:")
    for model_path in all_models:
        model_name = model_path.parent.name
        output_dir = output_base / model_name
        if output_dir.exists():
            n_files = len(list(output_dir.glob("*.png")))
            print(f"  - {model_name}: {n_files} visualizations")
    print("="*80)


if __name__ == "__main__":
    main()
