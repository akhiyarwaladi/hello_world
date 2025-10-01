#!/usr/bin/env python3
"""
Setup Malaria Lifecycle Classification Dataset for pipeline use.
Downloads, converts JSON to YOLO format, and creates train/val/test splits.
"""

import os
import sys
import json
import shutil
import yaml
import subprocess
import cv2
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def download_lifecycle_dataset():
    """Download lifecycle dataset if not exists"""
    raw_dir = Path("data/raw/malaria_lifecycle")

    if raw_dir.exists() and (raw_dir / "annotations.json").exists():
        print(f"[INFO] Lifecycle dataset already exists at {raw_dir}")
        return str(raw_dir)

    print("[DOWNLOAD] Downloading malaria lifecycle dataset...")

    # Use the download script
    result = subprocess.run([
        sys.executable, "scripts/data_setup/01_download_datasets.py",
        "--dataset", "malaria_lifecycle"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Failed to download: {result.stderr}")
        return None

    return str(raw_dir)

def convert_lifecycle_json_to_yolo(raw_dir, output_dir):
    """Convert lifecycle JSON annotations to YOLO format"""
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    annotations_file = raw_path / "annotations.json"
    images_dir = raw_path / "IML_Malaria"

    if not annotations_file.exists():
        print(f"[ERROR] Annotations file not found: {annotations_file}")
        return False

    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        return False

    print(f"[CONVERT] Converting JSON annotations to YOLO format...")
    print(f"   Source: {raw_path}")
    print(f"   Target: {output_path}")

    # Load JSON annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    # Lifecycle class mapping
    lifecycle_classes = {
        "red blood cell": 0,
        "ring": 1,
        "gametocyte": 2,
        "trophozoite": 3,
        "schizont": 4
    }

    # Process annotations
    converted_data = []
    skipped_images = 0

    for img_data in annotations:
        image_name = img_data['image_name']
        image_path = images_dir / image_name

        if not image_path.exists():
            skipped_images += 1
            continue

        # Load image to get dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            skipped_images += 1
            continue

        img_height, img_width = img.shape[:2]

        # Process each object in the image
        image_annotations = []
        for obj in img_data.get('objects', []):
            obj_type = obj.get('type', '').lower()
            bbox = obj.get('bbox', {})

            if obj_type not in lifecycle_classes:
                continue

            # Convert bbox to YOLO format (normalized x_center, y_center, width, height)
            x = float(bbox.get('x', 0))
            y = float(bbox.get('y', 0))
            w = float(bbox.get('w', 0))
            h = float(bbox.get('h', 0))

            # Convert to center coordinates and normalize
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            width_norm = w / img_width
            height_norm = h / img_height

            # Ensure normalized coordinates are in [0,1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width_norm = max(0, min(1, width_norm))
            height_norm = max(0, min(1, height_norm))

            image_annotations.append({
                'class_id': lifecycle_classes[obj_type],
                'x_center': x_center,
                'y_center': y_center,
                'width': width_norm,
                'height': height_norm
            })

        if image_annotations:  # Only add images with annotations
            converted_data.append({
                'image_name': image_name,
                'image_path': image_path,
                'annotations': image_annotations
            })

    print(f"[INFO] Processed {len(converted_data)} images ({skipped_images} skipped)")
    print(f"[INFO] Total annotations: {sum(len(item['annotations']) for item in converted_data)}")

    return converted_data

def create_yolo_splits(converted_data, output_dir, single_class=False,
                      train_ratio=0.70, val_ratio=0.20, test_ratio=0.10):
    """Create train/val/test splits in YOLO format with stratified splitting

    Args:
        converted_data: List of converted image data
        output_dir: Output directory for splits
        single_class: If True, use single class (parasite). If False, multi-class
        train_ratio: Ratio for training set (default: 0.70 = 70%)
        val_ratio: Ratio for validation set (default: 0.20 = 20%)
        test_ratio: Ratio for test set (default: 0.10 = 10%)
    """
    # Verify ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio:.4f}")

    print(f"[SPLIT] Using split ratios: Train={train_ratio:.2%}, Val={val_ratio:.2%}, Test={test_ratio:.2%}")

    output_path = Path(output_dir)

    # Remove existing output if exists
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create output structure
    output_path.mkdir(exist_ok=True)
    for split in ['train', 'val', 'test']:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # Prepare stratification labels based on dominant class in each image
    stratify_labels = []
    for item in converted_data:
        if item['annotations']:
            # Use the most frequent class in the image for stratification
            class_counts = {}
            for ann in item['annotations']:
                class_id = ann['class_id']
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

            # Get dominant class
            dominant_class = max(class_counts.keys(), key=lambda k: class_counts[k])
            stratify_labels.append(dominant_class)
        else:
            stratify_labels.append(0)  # Default class if no annotations

    print(f"[STRATIFY] Class distribution for stratification:")
    from collections import Counter
    class_dist = Counter(stratify_labels)
    for class_id, count in sorted(class_dist.items()):
        print(f"   Class {class_id}: {count} images")

    # Stratified split with custom ratios
    try:
        # First split: separate test set
        temp_size = val_ratio + test_ratio  # Remaining after train
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            converted_data, stratify_labels,
            test_size=temp_size, random_state=42, stratify=stratify_labels
        )
        # Second split: separate val and test from remaining
        val_adjusted = val_ratio / (val_ratio + test_ratio)
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels,
            test_size=(1 - val_adjusted), random_state=42, stratify=temp_labels
        )
        print(f"[STRATIFY] Successfully applied stratified splitting")
    except ValueError as e:
        print(f"[WARNING] Stratified split failed ({e}), using random split with manual balancing")

        # Fallback: manual balancing for very small classes
        class_groups = {}
        for i, item in enumerate(converted_data):
            label = stratify_labels[i]
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(item)

        train_data, val_data, test_data = [], [], []

        for class_id, items in class_groups.items():
            n_total = len(items)
            # Ensure at least 1 sample in each split for each class
            n_test = max(1, int(n_total * test_ratio))
            n_val = max(1, int(n_total * val_ratio))
            n_train = n_total - n_test - n_val

            if n_train < 1:
                n_train = 1
                n_val = max(0, n_total - n_train - n_test)
                n_test = n_total - n_train - n_val

            # Shuffle and split
            import random
            random.shuffle(items)

            test_data.extend(items[:n_test])
            val_data.extend(items[n_test:n_test+n_val])
            train_data.extend(items[n_test+n_val:])

            print(f"   Class {class_id}: Train={n_train}, Val={n_val}, Test={n_test}")

        print(f"[BALANCED] Manual balancing completed")

    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    # Validate split distribution
    print(f"\n[VALIDATION] Final split distribution:")
    for split_name, split_data in splits.items():
        split_class_counts = Counter()
        for item in split_data:
            for ann in item['annotations']:
                split_class_counts[ann['class_id']] += 1

        print(f"   {split_name} ({len(split_data)} images):")
        for class_id in sorted(split_class_counts.keys()):
            print(f"      Class {class_id}: {split_class_counts[class_id]} annotations")

        # Check for missing classes
        all_classes = set(range(5))  # Assuming 5 classes (0-4)
        missing_classes = all_classes - set(split_class_counts.keys())
        if missing_classes:
            print(f"      WARNING: Missing classes in {split_name}: {missing_classes}")

    stats = {'total_images': 0, 'total_annotations': 0, 'class_counts': [0] * 5}

    for split_name, split_data in splits.items():
        print(f"[SPLIT] Processing {split_name}: {len(split_data)} images")

        split_images_dir = output_path / split_name / "images"
        split_labels_dir = output_path / split_name / "labels"

        for item in split_data:
            # Filter annotations based on single_class mode
            filtered_annotations = []
            for ann in item['annotations']:
                if single_class:
                    # For detection: EXCLUDE red blood cells (class_id 0), only include parasites
                    if ann['class_id'] != 0:  # Skip red blood cells
                        filtered_annotations.append({
                            **ann,
                            'class_id': 0  # All parasites become class 0 for detection
                        })
                else:
                    # For classification: include all classes as-is
                    filtered_annotations.append(ann)

            # Only process images that have valid annotations after filtering
            if filtered_annotations:
                # Copy image
                src_image_path = item['image_path']
                dst_image_path = split_images_dir / item['image_name']
                shutil.copy2(src_image_path, dst_image_path)

                # Create label file
                label_path = split_labels_dir / f"{Path(item['image_name']).stem}.txt"
                with open(label_path, 'w') as f:
                    for ann in filtered_annotations:
                        # YOLO detection format: class_id x_center y_center width height
                        line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                        f.write(line)

                        # Update statistics (use original class_id for stats even in single_class mode)
                        if single_class:
                            # For single class, count all parasites as parasite class
                            stats['class_counts'][1] += 1  # Count as parasite (non-red blood cell)
                        else:
                            stats['class_counts'][ann['class_id']] += 1
                        stats['total_annotations'] += 1

                stats['total_images'] += 1

    return splits, stats

def create_data_yaml(output_dir, single_class=False):
    """Create data.yaml configuration file"""
    output_path = Path(output_dir)

    if single_class:
        # Single parasite class for detection
        class_names = ['parasite']
        print(f"[CONFIG] Creating single-class configuration (parasite detection)")
    else:
        # Class names for lifecycle stages (including red blood cells)
        class_names = [
            'red_blood_cell',
            'ring',
            'gametocyte',
            'trophozoite',
            'schizont'
        ]
        print(f"[CONFIG] Creating multi-class configuration (lifecycle detection)")

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

def setup_lifecycle_for_pipeline(single_class=True, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10):
    """Main setup function for lifecycle dataset

    Args:
        single_class: If True, single-class detection. If False, multi-class
        train_ratio: Training set ratio (default: 0.70 = 70%)
        val_ratio: Validation set ratio (default: 0.20 = 20%)
        test_ratio: Test set ratio (default: 0.10 = 10%)
    """
    if single_class:
        print("[SETUP] Setting up Malaria Lifecycle Single-Class Detection Dataset for pipeline...")
    else:
        print("[SETUP] Setting up Malaria Lifecycle Classification Dataset for pipeline...")

    # Step 0: Clean existing processed data for fresh setup
    output_dir = Path("data/processed/lifecycle")
    if output_dir.exists():
        print(f"[CLEAN] Removing existing processed data: {output_dir}")
        shutil.rmtree(output_dir)
        print(f"[CLEAN] Cleaned successfully - ensuring fresh setup")

    # Step 1: Download dataset
    raw_dir = download_lifecycle_dataset()
    if not raw_dir:
        return False

    # Step 2: Convert JSON to YOLO format
    converted_data = convert_lifecycle_json_to_yolo(raw_dir, "temp_lifecycle")
    if not converted_data:
        print("[ERROR] Failed to convert lifecycle dataset")
        return False

    # Step 3: Create train/val/test splits
    if single_class:
        output_dir = "data/processed/lifecycle"
    else:
        output_dir = "data/processed/lifecycle"
    splits, stats = create_yolo_splits(converted_data, output_dir, single_class,
                                      train_ratio, val_ratio, test_ratio)

    # Step 4: Create data.yaml
    yaml_path = create_data_yaml(output_dir, single_class)

    # Print summary
    print(f"\\n[SUCCESS] Lifecycle dataset setup completed!")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Configuration: {yaml_path}")
    print(f"\\n[DATASET SUMMARY]")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Total annotations: {stats['total_annotations']}")
    print(f"   Train: {len(splits['train'])} images")
    print(f"   Val: {len(splits['val'])} images")
    print(f"   Test: {len(splits['test'])} images")

    if single_class:
        print(f"\\n[DETECTION CLASS]")
        print(f"   0: parasite (excluding red blood cells from detection training)")
        print(f"\\n[PARASITE ANNOTATIONS] {stats['class_counts'][1]} parasite objects (red blood cells excluded)")
        print(f"\\n[READY] Lifecycle single-class detection dataset ready for pipeline!")
        print(f"[USAGE] Use --dataset iml_lifecycle flag in pipeline commands")
        print(f"[EXAMPLE] python run_multiple_models_pipeline.py --dataset iml_lifecycle --include yolo11 --epochs-det 10 --epochs-cls 10")
    else:
        print(f"\\n[CLASS DISTRIBUTION]")
        class_names = ['red_blood_cell', 'ring', 'gametocyte', 'trophozoite', 'schizont']
        for i, count in enumerate(stats['class_counts']):
            percentage = (count / stats['total_annotations']) * 100 if stats['total_annotations'] > 0 else 0
            print(f"   {class_names[i]}: {count} ({percentage:.1f}%)")
        print(f"\\n[READY] Lifecycle dataset ready for pipeline!")
        print(f"[USAGE] Use --dataset iml_lifecycle flag in pipeline commands")

    return str(yaml_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Malaria Lifecycle dataset for pipeline use")
    parser.add_argument("--multi-class", action="store_true",
                       help="Generate 5-class lifecycle detection dataset instead of single-class (parasite)")
    parser.add_argument("--train-ratio", type=float, default=0.66,
                       help="Training set ratio (default: 0.66 = 66%%)")
    parser.add_argument("--val-ratio", type=float, default=0.17,
                       help="Validation set ratio (default: 0.17 = 17%%)")
    parser.add_argument("--test-ratio", type=float, default=0.17,
                       help="Test set ratio (default: 0.17 = 17%%)")

    args = parser.parse_args()
    # Default is now single-class, unless --multi-class is specified
    setup_lifecycle_for_pipeline(single_class=not args.multi_class,
                                train_ratio=args.train_ratio,
                                val_ratio=args.val_ratio,
                                test_ratio=args.test_ratio)