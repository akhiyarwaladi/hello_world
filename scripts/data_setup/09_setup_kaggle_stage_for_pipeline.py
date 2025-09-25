#!/usr/bin/env python3
"""
Setup Kaggle Stage Classification Dataset for pipeline use.
Converts 16-class Kaggle MP-IDB dataset (4 species x 4 stages) to 4 stage classes only.
"""

import os
import sys
import shutil
import yaml
import subprocess
from pathlib import Path


def check_kaggle_dataset():
    """Check if Kaggle dataset exists"""
    kaggle_dir = Path("data/kaggle_dataset/MP-IDB-YOLO")

    if not kaggle_dir.exists():
        print(f"[ERROR] Kaggle dataset not found at {kaggle_dir}")
        print("[TIP] Download first: python scripts/data_setup/01_download_datasets.py --dataset kaggle_mp_idb")
        return False

    data_yaml = kaggle_dir / "data.yaml"
    images_dir = kaggle_dir / "images"
    labels_dir = kaggle_dir / "labels"

    if not all([data_yaml.exists(), images_dir.exists(), labels_dir.exists()]):
        print(f"[ERROR] Incomplete Kaggle dataset structure at {kaggle_dir}")
        return False

    print(f"[INFO] Kaggle dataset found at {kaggle_dir}")
    return str(kaggle_dir)


def convert_kaggle_to_stage_format(kaggle_dir, output_dir):
    """Convert Kaggle 16-class dataset to 4 stage classes using converter"""
    print(f"[CONVERT] Converting Kaggle 16-class to 4 stage classes...")
    print(f"   Source: {kaggle_dir}")
    print(f"   Target: {output_dir}")

    # Use the converter script
    result = subprocess.run([
        sys.executable, "scripts/data_setup/04_convert_to_yolo.py",
        "--dataset", "stage",
        "--output-dir", output_dir,
        "--task", "detect"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Failed to convert: {result.stderr}")
        return False

    print(f"[SUCCESS] Conversion completed successfully")
    return True


def create_stage_data_yaml(output_dir):
    """Create data.yaml configuration file for stage classification"""
    output_path = Path(output_dir)

    # Stage class names
    class_names = [
        'ring',
        'schizont',
        'trophozoite',
        'gametocyte'
    ]

    yaml_content = {
        'path': str(output_path.absolute()).replace('\\', '/'),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 4,
        'names': class_names
    }

    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    return yaml_path


def update_pipeline_for_stage_dataset():
    """Update pipeline script to recognize stage dataset"""
    pipeline_script = "run_multiple_models_pipeline.py"

    if not Path(pipeline_script).exists():
        print(f"[WARNING] Pipeline script not found: {pipeline_script}")
        return

    # Check if kaggle_stage is already supported
    with open(pipeline_script, 'r') as f:
        content = f.read()

    if 'kaggle_stage' in content:
        print(f"[INFO] Pipeline already supports kaggle_stage dataset")
        return

    print(f"[INFO] Stage dataset ready - use --dataset-type stage with pipeline")


def setup_kaggle_stage_for_pipeline():
    """Main setup function for Kaggle stage dataset"""
    print("="*60)
    print(" SETUP KAGGLE STAGE CLASSIFICATION DATASET ")
    print("="*60)
    print("Converting Kaggle MP-IDB 16-class to 4 stage classes for pipeline use...")

    # Step 1: Check if Kaggle dataset exists
    kaggle_dir = check_kaggle_dataset()
    if not kaggle_dir:
        return False

    # Step 2: Convert to stage format
    output_dir = "data/kaggle_stage_pipeline_ready"
    success = convert_kaggle_to_stage_format(kaggle_dir, output_dir)
    if not success:
        print("[ERROR] Failed to convert Kaggle dataset to stage format")
        return False

    # Step 3: Create proper data.yaml
    yaml_path = create_stage_data_yaml(output_dir)

    # Step 4: Update pipeline support
    update_pipeline_for_stage_dataset()

    # Count converted data
    output_path = Path(output_dir)
    train_images = len(list((output_path / "train" / "images").glob("*.jpg")))
    val_images = len(list((output_path / "val" / "images").glob("*.jpg")))
    test_images = len(list((output_path / "test" / "images").glob("*.jpg")))
    total_images = train_images + val_images + test_images

    # Print summary
    print(f"\n[SUCCESS] Kaggle stage dataset setup completed!")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Configuration: {yaml_path}")
    print(f"\n[DATASET SUMMARY]")
    print(f"   Total images: {total_images}")
    print(f"   Train: {train_images} images")
    print(f"   Val: {val_images} images")
    print(f"   Test: {test_images} images")
    print(f"\n[STAGE CLASSES]")
    print(f"   0: ring")
    print(f"   1: schizont")
    print(f"   2: trophozoite")
    print(f"   3: gametocyte")
    print(f"\n[READY] Kaggle stage dataset ready for pipeline!")
    print(f"[USAGE] Use --dataset-type stage flag in pipeline commands")
    print(f"[EXAMPLE] python run_multiple_models_pipeline.py --dataset-type stage --include yolo11 --epochs-det 10 --epochs-cls 10")

    return str(yaml_path)


if __name__ == "__main__":
    setup_kaggle_stage_for_pipeline()