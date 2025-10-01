#!/usr/bin/env python3
"""
DEPRECATED: This script is now a wrapper for the unified Kaggle processor.
Setup Kaggle Stage Classification Dataset for pipeline use.
Converts 16-class Kaggle MP-IDB dataset (4 species x 4 stages) to 4 stage classes only.
"""

import os
import sys
import shutil
import yaml
import subprocess
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add path for import
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))


def polygon_to_bbox(polygon_coords):
    """Convert polygon coordinates to bounding box format"""
    if len(polygon_coords) < 4:
        return None

    # Convert to float coordinates
    coords = [float(x) for x in polygon_coords]

    # Separate x and y coordinates
    x_coords = coords[::2]  # Even indices (0, 2, 4, ...)
    y_coords = coords[1::2]  # Odd indices (1, 3, 5, ...)

    if len(x_coords) < 2 or len(y_coords) < 2:
        return None

    # Calculate bounding box
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Convert to YOLO format (center_x, center_y, width, height)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return [center_x, center_y, width, height]


def convert_kaggle_class_to_stage(kaggle_class_id):
    """Convert Kaggle 16-class ID to 4-stage class ID"""
    # Kaggle MP-IDB has 16 classes: 4 species × 4 stages
    # Classes 0-3: P_falciparum (Ring, Schizont, Trophozoite, Gametocyte)
    # Classes 4-7: P_vivax (Ring, Schizont, Trophozoite, Gametocyte)
    # Classes 8-11: P_malariae (Ring, Schizont, Trophozoite, Gametocyte)
    # Classes 12-15: P_ovale (Ring, Schizont, Trophozoite, Gametocyte)

    stage_id = kaggle_class_id % 4
    return stage_id  # 0=Ring, 1=Schizont, 2=Trophozoite, 3=Gametocyte


def check_kaggle_dataset():
    """Check if Kaggle dataset exists"""
    kaggle_dir = Path("data/raw/kaggle_dataset/MP-IDB-YOLO")

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


def convert_kaggle_to_stage_format(kaggle_dir, output_dir, single_class=False,
                                   train_ratio=0.70, val_ratio=0.20, test_ratio=0.10):
    """Simply copy dataset and convert class IDs for single-class detection

    Args:
        kaggle_dir: Source directory path
        output_dir: Output directory path
        single_class: If True, single-class detection. If False, multi-class
        train_ratio: Training set ratio (default: 0.70 = 70%)
        val_ratio: Validation set ratio (default: 0.20 = 20%)
        test_ratio: Test set ratio (default: 0.10 = 10%)
    """
    # Verify ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio:.4f}")

    print(f"[SPLIT] Using split ratios: Train={train_ratio:.2%}, Val={val_ratio:.2%}, Test={test_ratio:.2%}")

    if single_class:
        print(f"[CONVERT] Converting to single parasite class (copying all files)...")
    else:
        print(f"[CONVERT] Converting to 4 stage classes (copying all files)...")
    print(f"   Source: {kaggle_dir}")
    print(f"   Target: {output_dir}")

    kaggle_path = Path(kaggle_dir)
    output_path = Path(output_dir)

    # Get all images with their labels
    images_dir = kaggle_path / "images"
    labels_dir = kaggle_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[ERROR] Missing images or labels directory in {kaggle_dir}")
        return False

    # Collect image files that have corresponding label files
    valid_images = []
    for image_file in images_dir.glob("*.jpg"):
        label_file = labels_dir / f"{image_file.stem}.txt"
        if label_file.exists():
            valid_images.append((image_file, label_file))

    print(f"[INFO] Found {len(valid_images)} images with labels")

    # Create stratification labels based on stages in each image
    print(f"[STRATIFY] Analyzing stage distribution for stratified splitting...")
    stratify_labels = []
    stage_mapping = {
        # Ring (classes 0, 4, 8, 12) -> stage 0
        0: 0, 4: 0, 8: 0, 12: 0,
        # Schizont (classes 1, 5, 9, 13) -> stage 1
        1: 1, 5: 1, 9: 1, 13: 1,
        # Trophozoite (classes 2, 6, 10, 14) -> stage 2
        2: 2, 6: 2, 10: 2, 14: 2,
        # Gametocyte (classes 3, 7, 11, 15) -> stage 3
        3: 3, 7: 3, 11: 3, 15: 3
    }

    from collections import Counter

    for image_file, label_file in valid_images:
        stages_in_image = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    stage_id = stage_mapping.get(class_id, 0)
                    stages_in_image.append(stage_id)

        # Use most frequent stage in image for stratification
        if stages_in_image:
            dominant_stage = max(set(stages_in_image), key=stages_in_image.count)
            stratify_labels.append(dominant_stage)
        else:
            stratify_labels.append(0)  # Default to ring

    # Print stage distribution
    stage_dist = Counter(stratify_labels)
    stage_names = ['ring', 'schizont', 'trophozoite', 'gametocyte']
    print(f"[STRATIFY] Stage distribution for stratification:")
    for stage_id, count in sorted(stage_dist.items()):
        stage_name = stage_names[stage_id] if stage_id < len(stage_names) else f"stage_{stage_id}"
        print(f"   {stage_name}: {count} images")

    # Stratified split with custom ratios
    try:
        # First split: separate train set
        temp_size = val_ratio + test_ratio  # Remaining after train
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            valid_images, stratify_labels,
            test_size=temp_size, random_state=42, stratify=stratify_labels
        )
        # Second split: separate val and test from remaining
        val_adjusted = val_ratio / (val_ratio + test_ratio)
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels,
            test_size=(1 - val_adjusted), random_state=42, stratify=temp_labels
        )
        print(f"[STRATIFY] Successfully applied stratified splitting")
    except ValueError as e:
        print(f"[WARNING] Stratified split failed ({e}), using random split")
        temp_size = val_ratio + test_ratio
        train_images, temp_images = train_test_split(valid_images, test_size=temp_size, random_state=42)
        val_adjusted = val_ratio / (val_ratio + test_ratio)
        val_images, test_images = train_test_split(temp_images, test_size=(1 - val_adjusted), random_state=42)

    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    # Validate split distribution
    def validate_stage_distribution(images, split_name):
        split_stage_dist = Counter()
        for i, (image_file, label_file) in enumerate(images):
            # Find the stratify label for this image
            image_idx = valid_images.index((image_file, label_file))
            stage_id = stratify_labels[image_idx]
            split_stage_dist[stage_id] += 1

        print(f"   {split_name} stage distribution:")
        for stage_id, count in sorted(split_stage_dist.items()):
            stage_name = stage_names[stage_id] if stage_id < len(stage_names) else f"stage_{stage_id}"
            print(f"      {stage_name}: {count} images")

        return split_stage_dist

    print(f"\n[VALIDATION] Final split distribution:")
    train_dist = validate_stage_distribution(train_images, "Train")
    val_dist = validate_stage_distribution(val_images, "Val")
    test_dist = validate_stage_distribution(test_images, "Test")

    # Create output structure and copy files
    if output_path.exists():
        shutil.rmtree(output_path)

    total_converted = 0
    for split_name, split_files in splits.items():
        # Create directories
        split_images_dir = output_path / split_name / "images"
        split_labels_dir = output_path / split_name / "labels"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"[SPLIT] Processing {split_name}: {len(split_files)} images")

        for image_file, label_file in split_files:
            # Copy image
            shutil.copy2(image_file, split_images_dir / image_file.name)

            # Convert label file from polygon to bbox format
            converted_lines = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:  # At least class_id + 2 coordinates
                        try:
                            kaggle_class_id = int(parts[0])
                            polygon_coords = parts[1:]

                            # Convert polygon to bounding box
                            bbox = polygon_to_bbox(polygon_coords)
                            if bbox is None:
                                continue

                            # Convert class ID
                            if single_class:
                                # Convert all to single class (parasite detection)
                                stage_class_id = 0
                            else:
                                # Convert to stage classification (0-3)
                                stage_class_id = convert_kaggle_class_to_stage(kaggle_class_id)

                            # Format as YOLO bbox: class_id x_center y_center width height
                            bbox_line = f"{stage_class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                            converted_lines.append(bbox_line)
                            total_converted += 1

                        except (ValueError, IndexError) as e:
                            print(f"[WARNING] Skipped invalid line in {label_file.name}: {line.strip()}")
                            continue

            # Write converted labels
            dst_label = split_labels_dir / label_file.name
            with open(dst_label, 'w') as f:
                for line in converted_lines:
                    f.write(line + '\n')

    if single_class:
        print(f"[SUCCESS] Converted {total_converted} polygon annotations to single class bbox format")
    else:
        print(f"[SUCCESS] Converted {total_converted} polygon annotations to 4-stage bbox format")

    total_images = len(train_images) + len(val_images) + len(test_images)
    print(f"[SUCCESS] Dataset ready with {total_images} images")
    print(f"[INFO] Split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    return True




def create_stage_data_yaml(output_dir, single_class=False):
    """Create data.yaml configuration file for stage classification"""
    output_path = Path(output_dir)

    if single_class:
        # Single parasite class for detection
        class_names = ['parasite']
        print(f"[CONFIG] Creating single-class configuration (parasite detection)")
    else:
        # Stage class names for multi-class detection
        class_names = [
            'ring',
            'schizont',
            'trophozoite',
            'gametocyte'
        ]
        print(f"[CONFIG] Creating multi-class configuration (stage detection)")

    yaml_content = {
        'path': str(output_path.absolute()).replace('\\', '/'),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"[SUCCESS] Created data.yaml with {len(class_names)} class(es)")
    print(f"   Classes: {', '.join(class_names)}")

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


def setup_kaggle_stage_for_pipeline(single_class=True, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10):
    """Main setup function for Kaggle stage dataset

    Args:
        single_class: If True, single-class detection. If False, multi-class
        train_ratio: Training set ratio (default: 0.70 = 70%)
        val_ratio: Validation set ratio (default: 0.20 = 20%)
        test_ratio: Test set ratio (default: 0.10 = 10%)
    """
    print("="*60)
    if single_class:
        print(" SETUP KAGGLE SINGLE-CLASS DETECTION DATASET ")
        print("Converting Kaggle MP-IDB 16-class to single parasite class for detection...")
    else:
        print(" SETUP KAGGLE STAGE CLASSIFICATION DATASET ")
        print("Converting Kaggle MP-IDB 16-class to 4 stage classes for pipeline use...")
    print("="*60)

    # Step 0: Clean existing processed data for fresh setup
    output_dir = Path("data/processed/stages")
    if output_dir.exists():
        print(f"[CLEAN] Removing existing processed data: {output_dir}")
        shutil.rmtree(output_dir)
        print(f"[CLEAN] Cleaned successfully - ensuring fresh setup")

    # Step 1: Check if Kaggle dataset exists
    kaggle_dir = check_kaggle_dataset()
    if not kaggle_dir:
        return False

    # Step 2: Convert to stage format
    if single_class:
        output_dir = "data/processed/stages"
    else:
        output_dir = "data/processed/stages"
    success = convert_kaggle_to_stage_format(kaggle_dir, output_dir, single_class,
                                            train_ratio, val_ratio, test_ratio)
    if not success:
        print("[ERROR] Failed to convert Kaggle dataset to stage format")
        return False

    # Step 3: Create proper data.yaml
    yaml_path = create_stage_data_yaml(output_dir, single_class)

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
    if single_class:
        print(f"\n[DETECTION CLASS]")
        print(f"   0: parasite")
        print(f"\n[READY] Kaggle single-class detection dataset ready for pipeline!")
        print(f"[USAGE] Use --dataset mp_idb_stages flag in pipeline commands")
        print(f"[EXAMPLE] python run_multiple_models_pipeline.py --dataset mp_idb_stages --include yolo11 --epochs-det 10 --epochs-cls 10")
    else:
        print(f"\n[STAGE CLASSES]")
        print(f"   0: ring")
        print(f"   1: schizont")
        print(f"   2: trophozoite")
        print(f"   3: gametocyte")
        print(f"\n[READY] Kaggle stage dataset ready for pipeline!")
        print(f"[USAGE] Use --dataset mp_idb_stages flag in pipeline commands")
        print(f"[EXAMPLE] python run_multiple_models_pipeline.py --dataset mp_idb_stages --include yolo11 --epochs-det 10 --epochs-cls 10")

    return str(yaml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Kaggle MP-IDB dataset for pipeline use")
    parser.add_argument("--multi-class", action="store_true",
                       help="Generate 4-class stages detection dataset instead of single-class (parasite)")
    parser.add_argument("--train-ratio", type=float, default=0.66,
                       help="Training set ratio (default: 0.66 = 66%%)")
    parser.add_argument("--val-ratio", type=float, default=0.17,
                       help="Validation set ratio (default: 0.17 = 17%%)")
    parser.add_argument("--test-ratio", type=float, default=0.17,
                       help="Test set ratio (default: 0.17 = 17%%)")

    args = parser.parse_args()
    # Default is now single-class, unless --multi-class is specified
    setup_kaggle_stage_for_pipeline(single_class=not args.multi_class,
                                    train_ratio=args.train_ratio,
                                    val_ratio=args.val_ratio,
                                    test_ratio=args.test_ratio)