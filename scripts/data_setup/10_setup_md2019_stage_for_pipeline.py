#!/usr/bin/env python3
"""
Setup MD_2019 Stage Classification Dataset for pipeline use.
Converts 10-class MD_2019 dataset with binary masks to 4 lifecycle stages.

Author: Malaria Detection Team
Date: 2025-10-13
"""

import os
import sys
import shutil
import yaml
import argparse
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

# Add path for import
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))


def extract_bbox_from_binary_mask(mask_path):
    """Extract bounding boxes from binary ground truth mask

    Args:
        mask_path: Path to binary mask image

    Returns:
        List of bounding boxes [(x_min, y_min, x_max, y_max), ...]
    """
    # Read binary mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((x, y, x + w, y + h))

    return bboxes


def map_md2019_stage_to_3class(stage_name):
    """Map MD_2019 10-class stages to 3 lifecycle stages (EXCLUDE Gametocyte)

    Args:
        stage_name: MD_2019 stage name

    Returns:
        Stage ID (0-2) or None if excluded
    """
    # Mapping (3 classes only - Gametocyte excluded due to insufficient samples)
    stage_mapping = {
        # Ring: R + LR-ET → class 0
        'R': 0,
        'LR-ET': 0,
        # Schizont: Esch + Lsch + Seg → class 1
        'Esch': 1,
        'Lsch': 1,
        'Seg': 1,
        # Trophozoite: MT + LT → class 2
        'MT': 2,
        'LT': 2,
        # Excluded: Gametocyte (only 2 samples), DEBRIS, WBC
        'Gam': None,
        'DEBRIS': None,
        'WBC': None
    }

    return stage_mapping.get(stage_name)


def convert_md2019_to_yolo_format(md2019_dir):
    """Convert MD_2019 dataset to YOLO format in-place

    Creates images/ and labels/ subdirectories within md2019_dir

    Args:
        md2019_dir: Path to MD_2019 dataset directory

    Returns:
        True if successful, False otherwise
    """
    md2019_path = Path(md2019_dir)

    # Check required files
    images_source_dir = md2019_path / "Giemsa stained images"
    masks_dir = md2019_path / "Ground truth images"
    annotations_file = md2019_path / "LifeStages.xlsx"

    if not all([images_source_dir.exists(), masks_dir.exists(), annotations_file.exists()]):
        print(f"[ERROR] MD_2019 dataset incomplete at {md2019_dir}")
        return False

    # Create YOLO format directories
    yolo_images_dir = md2019_path / "images"
    yolo_labels_dir = md2019_path / "labels"

    # Skip if already converted
    if yolo_images_dir.exists() and yolo_labels_dir.exists():
        print(f"[INFO] MD_2019 already converted to YOLO format")
        return True

    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONVERT] Converting MD_2019 to YOLO format...")

    # Read annotations from Excel
    df = pd.read_excel(annotations_file)

    # Process each image
    total_converted = 0
    total_excluded = 0
    stage_counts = {'ring': 0, 'schizont': 0, 'trophozoite': 0}

    # Group annotations by image
    for image_name in df['imageName'].unique():
        image_annotations = df[df['imageName'] == image_name]

        # Source paths
        src_image_path = images_source_dir / image_name
        mask_path = masks_dir / image_name.replace('.jpg', '_GT.png')

        if not src_image_path.exists():
            print(f"[WARNING] Image not found: {src_image_path}")
            continue

        # Load image to get dimensions
        img = cv2.imread(str(src_image_path))
        if img is None:
            continue

        img_height, img_width = img.shape[:2]

        # Copy image to YOLO images directory
        dst_image_path = yolo_images_dir / image_name
        shutil.copy2(src_image_path, dst_image_path)

        # Extract bboxes from mask if available
        bboxes_from_mask = []
        if mask_path.exists():
            bboxes_from_mask = extract_bbox_from_binary_mask(mask_path)

        # Create YOLO label file
        label_lines = []

        for idx, row in image_annotations.iterrows():
            stage = row['stage']
            stage_id = map_md2019_stage_to_3class(stage)

            if stage_id is None:
                # Exclude Gametocyte (only 2 samples), DEBRIS, and WBC
                total_excluded += 1
                continue

            # Get center coordinates from Excel
            center_x_px = float(row['center_x'])
            center_y_px = float(row['center_y'])

            # Find matching bbox from mask or estimate bbox size
            if bboxes_from_mask:
                # Find closest bbox to this center point
                min_dist = float('inf')
                best_bbox = None

                for bbox in bboxes_from_mask:
                    x_min, y_min, x_max, y_max = bbox
                    bbox_center_x = (x_min + x_max) / 2
                    bbox_center_y = (y_min + y_max) / 2

                    dist = np.sqrt((bbox_center_x - center_x_px)**2 + (bbox_center_y - center_y_px)**2)

                    if dist < min_dist:
                        min_dist = dist
                        best_bbox = bbox

                if best_bbox and min_dist < 50:  # Within 50 pixels
                    x_min, y_min, x_max, y_max = best_bbox
                else:
                    # Fallback: estimate bbox size based on stage
                    # From analysis: Ring (39.8×36.4), Trophozoite (71.0×70.3), Schizont (91.1×95.5)
                    if stage_id == 0:  # Ring
                        w, h = 40, 36
                    elif stage_id == 1:  # Schizont
                        w, h = 91, 96
                    else:  # Trophozoite
                        w, h = 71, 70

                    x_min = max(0, center_x_px - w/2)
                    y_min = max(0, center_y_px - h/2)
                    x_max = min(img_width, center_x_px + w/2)
                    y_max = min(img_height, center_y_px + h/2)
            else:
                # Fallback: estimate bbox size
                if stage_id == 0:  # Ring
                    w, h = 40, 36
                elif stage_id == 1:  # Schizont
                    w, h = 91, 96
                else:  # Trophozoite
                    w, h = 71, 70

                x_min = max(0, center_x_px - w/2)
                y_min = max(0, center_y_px - h/2)
                x_max = min(img_width, center_x_px + w/2)
                y_max = min(img_height, center_y_px + h/2)

            # Convert to YOLO format (normalized)
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # Ensure normalized coordinates are in [0,1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            # YOLO format: class_id x_center y_center width height
            label_line = f"{stage_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            label_lines.append(label_line)

            # Update statistics
            stage_names = ['ring', 'schizont', 'trophozoite']
            if stage_id < len(stage_names):
                stage_counts[stage_names[stage_id]] += 1
            total_converted += 1

        # Write label file
        if label_lines:
            # Create label filename: replace image extension with .txt
            label_filename = Path(image_name).with_suffix('.txt').name
            label_path = yolo_labels_dir / label_filename
            with open(label_path, 'w') as f:
                for line in label_lines:
                    f.write(line + '\n')

    print(f"[SUCCESS] MD_2019 converted to YOLO format")
    print(f"[INFO] Converted {total_converted} annotations")
    print(f"[INFO] Excluded {total_excluded} annotations (Gametocyte: 2, DEBRIS: 691, WBC: 51)")
    print(f"[INFO] Stage distribution (3 classes):")
    for stage, count in stage_counts.items():
        print(f"   {stage}: {count}")

    return True


def check_md2019_dataset():
    """Check if MD_2019 dataset exists and convert if needed

    Returns:
        Path to MD_2019 dataset or False if not found
    """
    md2019_dir = Path("data/raw/md_2019")

    if not md2019_dir.exists():
        print(f"[ERROR] MD_2019 dataset not found at {md2019_dir}")
        print("[TIP] Download MD_2019 dataset manually to data/raw/md_2019/")
        return False

    # Check if already converted to YOLO format
    yolo_images_dir = md2019_dir / "images"
    yolo_labels_dir = md2019_dir / "labels"

    if not (yolo_images_dir.exists() and yolo_labels_dir.exists()):
        # Need to convert MD_2019 to YOLO format first
        print(f"[INFO] MD_2019 dataset found (needs conversion)")
        success = convert_md2019_to_yolo_format(md2019_dir)
        if not success:
            print(f"[ERROR] Failed to convert MD_2019 to YOLO format")
            return False
    else:
        print(f"[INFO] MD_2019 dataset found (already in YOLO format)")

    return str(md2019_dir)


def convert_md2019_to_stage_format(md2019_dir, output_dir, single_class=False,
                                    train_ratio=0.70, val_ratio=0.20, test_ratio=0.10):
    """Convert MD_2019 dataset to 4-stage format

    Args:
        md2019_dir: Source MD_2019 directory path
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
        print(f"[CONVERT] Converting MD_2019 to single parasite class...")
    else:
        print(f"[CONVERT] Converting MD_2019 to 4 stage classes...")
    print(f"   Source: {md2019_dir}")
    print(f"   Target: {output_dir}")

    md2019_path = Path(md2019_dir)
    output_path = Path(output_dir)

    # Get all images with their labels
    images_dir = md2019_path / "images"
    labels_dir = md2019_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[ERROR] Missing images or labels directory in {md2019_dir}")
        return False

    # Collect image files that have corresponding label files (support both .jpg and .png)
    valid_images = []
    for pattern in ["*.jpg", "*.png", "*.JPG", "*.PNG"]:
        for image_file in images_dir.glob(pattern):
            label_file = labels_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                valid_images.append((image_file, label_file))

    print(f"[INFO] Found {len(valid_images)} images with labels")

    # Create stratification labels based on stages in each image
    print(f"[STRATIFY] Analyzing stage distribution for stratified splitting...")
    stratify_labels = []

    for image_file, label_file in valid_images:
        stages_in_image = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    stages_in_image.append(class_id)

        # Use most frequent stage in image for stratification
        if stages_in_image:
            dominant_stage = max(set(stages_in_image), key=stages_in_image.count)
            stratify_labels.append(dominant_stage)
        else:
            stratify_labels.append(0)  # Default to ring

    # Print stage distribution
    stage_dist = Counter(stratify_labels)
    stage_names = ['ring', 'schizont', 'trophozoite']
    print(f"[STRATIFY] Stage distribution for stratification (3 classes):")
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

            # Copy and convert label file if needed
            converted_lines = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    try:
                        # MD_2019 labels are already in YOLO bbox format with 4 classes
                        # Format: class_id x_center y_center width height
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if single_class:
                                # Convert all to single class
                                stage_class_id = 0
                            else:
                                # Already in 4-class format
                                stage_class_id = class_id

                            # Keep bbox coordinates as-is
                            bbox_line = f"{stage_class_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}"
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
        print(f"[SUCCESS] Converted {total_converted} annotations to single class bbox format")
    else:
        print(f"[SUCCESS] Converted {total_converted} annotations to 4-stage bbox format")

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
        # Stage class names for multi-class detection (3 classes - no Gametocyte)
        class_names = [
            'ring',
            'schizont',
            'trophozoite'
        ]
        print(f"[CONFIG] Creating multi-class configuration (3 stage classes)")

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


def setup_md2019_stage_for_pipeline(single_class=True, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10):
    """Main setup function for MD_2019 stage dataset

    Args:
        single_class: If True, single-class detection. If False, multi-class
        train_ratio: Training set ratio (default: 0.70 = 70%)
        val_ratio: Validation set ratio (default: 0.20 = 20%)
        test_ratio: Test set ratio (default: 0.10 = 10%)
    """
    print("="*60)
    if single_class:
        print(" SETUP MD_2019 SINGLE-CLASS DETECTION DATASET ")
        print("Converting MD_2019 10-class to single parasite class for detection...")
    else:
        print(" SETUP MD_2019 STAGE CLASSIFICATION DATASET (3 CLASSES) ")
        print("Converting MD_2019 10-class to 3 stage classes (NO Gametocyte)...")
    print("="*60)

    # Step 0: Clean existing processed data for fresh setup
    output_dir = Path("data/processed/md_2019_stages")
    if output_dir.exists():
        print(f"[CLEAN] Removing existing processed data: {output_dir}")
        shutil.rmtree(output_dir)
        print(f"[CLEAN] Cleaned successfully - ensuring fresh setup")

    # Step 1: Check if MD_2019 dataset exists
    md2019_dir = check_md2019_dataset()
    if not md2019_dir:
        return False

    # Step 2: Convert to stage format
    output_dir = "data/processed/md_2019_stages"
    success = convert_md2019_to_stage_format(md2019_dir, output_dir, single_class,
                                             train_ratio, val_ratio, test_ratio)
    if not success:
        print("[ERROR] Failed to convert MD_2019 dataset to stage format")
        return False

    # Step 3: Create proper data.yaml
    yaml_path = create_stage_data_yaml(output_dir, single_class)

    # Count converted data (support both .jpg and .png)
    output_path = Path(output_dir)
    train_images = len(list((output_path / "train" / "images").glob("*.jpg"))) + \
                   len(list((output_path / "train" / "images").glob("*.png")))
    val_images = len(list((output_path / "val" / "images").glob("*.jpg"))) + \
                 len(list((output_path / "val" / "images").glob("*.png")))
    test_images = len(list((output_path / "test" / "images").glob("*.jpg"))) + \
                  len(list((output_path / "test" / "images").glob("*.png")))
    total_images = train_images + val_images + test_images

    # Print summary
    print(f"\n[SUCCESS] MD_2019 stage dataset setup completed!")
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
        print(f"\n[READY] MD_2019 single-class detection dataset ready for pipeline!")
        print(f"[USAGE] Use --dataset md_2019_stages flag in pipeline commands")
        print(f"[EXAMPLE] python main_pipeline.py --dataset md_2019_stages --include yolo11 --epochs-det 10 --epochs-cls 10")
    else:
        print(f"\n[STAGE CLASSES (3 ONLY)]")
        print(f"   0: ring")
        print(f"   1: schizont")
        print(f"   2: trophozoite")
        print(f"   NOTE: Gametocyte EXCLUDED (only 2 samples in original data)")
        print(f"\n[READY] MD_2019 stage dataset ready for pipeline!")
        print(f"[USAGE] Use --dataset md_2019_stages flag in pipeline commands")
        print(f"[EXAMPLE] python main_pipeline.py --dataset md_2019_stages --include yolo11 --epochs-det 10 --epochs-cls 10")

    return str(yaml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup MD_2019 dataset for pipeline use (3 classes only - no Gametocyte)")
    parser.add_argument("--multi-class", action="store_true",
                       help="Generate 3-class stages detection dataset instead of single-class (parasite)")
    parser.add_argument("--train-ratio", type=float, default=0.66,
                       help="Training set ratio (default: 0.66 = 66%%)")
    parser.add_argument("--val-ratio", type=float, default=0.17,
                       help="Validation set ratio (default: 0.17 = 17%%)")
    parser.add_argument("--test-ratio", type=float, default=0.17,
                       help="Test set ratio (default: 0.17 = 17%%)")

    args = parser.parse_args()
    # Default is now single-class, unless --multi-class is specified
    setup_md2019_stage_for_pipeline(single_class=not args.multi_class,
                                     train_ratio=args.train_ratio,
                                     val_ratio=args.val_ratio,
                                     test_ratio=args.test_ratio)
