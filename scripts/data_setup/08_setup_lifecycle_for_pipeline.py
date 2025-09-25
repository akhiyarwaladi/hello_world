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

def create_yolo_splits(converted_data, output_dir):
    """Create train/val/test splits in YOLO format"""
    output_path = Path(output_dir)

    # Remove existing output if exists
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create output structure
    output_path.mkdir(exist_ok=True)
    for split in ['train', 'val', 'test']:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # Split data: 70% train, 20% val, 10% test
    train_data, temp_data = train_test_split(converted_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.33, random_state=42)

    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    stats = {'total_images': 0, 'total_annotations': 0, 'class_counts': [0] * 5}

    for split_name, split_data in splits.items():
        print(f"[SPLIT] Processing {split_name}: {len(split_data)} images")

        split_images_dir = output_path / split_name / "images"
        split_labels_dir = output_path / split_name / "labels"

        for item in split_data:
            # Copy image
            src_image_path = item['image_path']
            dst_image_path = split_images_dir / item['image_name']
            shutil.copy2(src_image_path, dst_image_path)

            # Create label file
            label_path = split_labels_dir / f"{Path(item['image_name']).stem}.txt"
            with open(label_path, 'w') as f:
                for ann in item['annotations']:
                    # YOLO detection format: class_id x_center y_center width height
                    line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                    f.write(line)

                    # Update statistics
                    stats['class_counts'][ann['class_id']] += 1
                    stats['total_annotations'] += 1

            stats['total_images'] += 1

    return splits, stats

def create_data_yaml(output_dir):
    """Create data.yaml configuration file"""
    output_path = Path(output_dir)

    # Class names for lifecycle stages
    class_names = [
        'red_blood_cell',
        'ring',
        'gametocyte',
        'trophozoite',
        'schizont'
    ]

    yaml_content = {
        'path': output_dir.replace('\\', '/'),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 5,
        'names': class_names
    }

    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    return yaml_path

def setup_lifecycle_for_pipeline():
    """Main setup function for lifecycle dataset"""
    print("[SETUP] Setting up Malaria Lifecycle Classification Dataset for pipeline...")

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
    output_dir = "data/lifecycle_pipeline_ready"
    splits, stats = create_yolo_splits(converted_data, output_dir)

    # Step 4: Create data.yaml
    yaml_path = create_data_yaml(output_dir)

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
    print(f"\\n[CLASS DISTRIBUTION]")
    class_names = ['red_blood_cell', 'ring', 'gametocyte', 'trophozoite', 'schizont']
    for i, count in enumerate(stats['class_counts']):
        percentage = (count / stats['total_annotations']) * 100 if stats['total_annotations'] > 0 else 0
        print(f"   {class_names[i]}: {count} ({percentage:.1f}%)")

    print(f"\\n[READY] Lifecycle dataset ready for pipeline!")
    print(f"[USAGE] Use --dataset-type lifecycle flag in pipeline commands")

    return str(yaml_path)

if __name__ == "__main__":
    setup_lifecycle_for_pipeline()