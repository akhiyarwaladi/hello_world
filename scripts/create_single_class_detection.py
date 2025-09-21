#!/usr/bin/env python3
"""
Create single-class detection dataset for malaria
Only detect "parasit" vs "background" - no species classification
Author: Assistant
Date: 2025-09-21
"""

import os
import shutil
import yaml
from pathlib import Path

def create_single_class_detection():
    """Convert multi-class detection to single-class (parasit only)"""

    # Source and destination paths
    src_dir = Path("data/integrated/yolo")
    dst_dir = Path("data/single_class_detection")

    # Create destination structure
    for split in ['train', 'val', 'test']:
        (dst_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    print("Converting multi-class to single-class detection...")

    # Process each split
    total_images = 0
    total_objects = 0

    for split in ['train', 'val', 'test']:
        src_images = src_dir / split / "images"
        src_labels = src_dir / split / "labels"
        dst_images = dst_dir / split / "images"
        dst_labels = dst_dir / split / "labels"

        if not src_images.exists():
            print(f"Warning: {src_images} not found")
            continue

        print(f"Processing {split} split...")

        split_images = 0
        split_objects = 0

        # Copy all images
        for img_file in src_images.glob("*.jpg"):
            shutil.copy2(img_file, dst_images / img_file.name)
            split_images += 1

        # Convert labels (all classes → class 0 "parasit")
        for label_file in src_labels.glob("*.txt"):
            src_label_path = src_labels / label_file.name
            dst_label_path = dst_labels / label_file.name

            with open(src_label_path, 'r') as f:
                lines = f.readlines()

            # Convert all classes to class 0 (parasit)
            converted_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Change class to 0, keep coordinates
                    new_line = f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n"
                    converted_lines.append(new_line)
                    split_objects += 1

            # Write converted labels
            with open(dst_label_path, 'w') as f:
                f.writelines(converted_lines)

        print(f"  {split}: {split_images} images, {split_objects} objects")
        total_images += split_images
        total_objects += split_objects

    # Create single-class data.yaml
    data_config = {
        'path': str(dst_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,  # SINGLE CLASS!
        'names': {
            0: 'parasit'  # Only one class
        }
    }

    yaml_path = dst_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"\n✅ Single-class detection dataset created!")
    print(f"Total: {total_images} images, {total_objects} parasite objects")
    print(f"Classes: 1 (parasit only)")
    print(f"Dataset path: {dst_dir}")
    print(f"Config file: {yaml_path}")

    return dst_dir

if __name__ == "__main__":
    create_single_class_detection()