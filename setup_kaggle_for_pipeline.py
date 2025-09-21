#!/usr/bin/env python3
"""
Setup Kaggle MP-IDB original dataset for pipeline use.
Creates train/val/test splits and converts to single-class detection.
"""

import os
import shutil
import yaml
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def setup_kaggle_for_pipeline():
    """Setup Kaggle dataset for pipeline use"""

    print("🚀 Setting up Kaggle MP-IDB original dataset for pipeline...")

    # Paths
    kaggle_source = Path("data/kaggle_dataset/MP-IDB-YOLO")
    output_dir = Path("data/kaggle_pipeline_ready")

    if not kaggle_source.exists():
        print(f"❌ Kaggle dataset not found at {kaggle_source}")
        return False

    # Remove existing output if exists
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Create output structure
    output_dir.mkdir(exist_ok=True)
    train_img_dir = output_dir / "train" / "images"
    train_label_dir = output_dir / "train" / "labels"
    val_img_dir = output_dir / "val" / "images"
    val_label_dir = output_dir / "val" / "labels"
    test_img_dir = output_dir / "test" / "images"
    test_label_dir = output_dir / "test" / "labels"

    for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    all_images_dir = kaggle_source / "images"
    all_labels_dir = kaggle_source / "labels"

    image_files = [f for f in os.listdir(all_images_dir) if f.lower().endswith(('.jpg', '.png'))]
    print(f"📊 Found {len(image_files)} images in Kaggle dataset")

    # Split data: 70% train, 20% val, 10% test
    train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.33, random_state=42)  # 0.33 * 0.3 = 0.1

    print(f"📂 Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    def copy_files(files, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
        """Copy image and label files"""
        for img_file in files:
            # Copy image
            shutil.copy(src_img_dir / img_file, dst_img_dir / img_file)

            # Copy corresponding label
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            if (src_label_dir / label_file).exists():
                shutil.copy(src_label_dir / label_file, dst_label_dir / label_file)

    # Copy files to splits
    copy_files(train_files, all_images_dir, all_labels_dir, train_img_dir, train_label_dir)
    copy_files(val_files, all_images_dir, all_labels_dir, val_img_dir, val_label_dir)
    copy_files(test_files, all_images_dir, all_labels_dir, test_img_dir, test_label_dir)

    def convert_to_single_class_detection(labels_dir):
        """Convert multi-class to single class detection (0 = parasite)"""
        converted_count = 0
        for label_file in os.listdir(labels_dir):
            label_path = labels_dir / label_file

            with open(label_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]

            new_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) < 5:  # Skip invalid lines
                    continue

                # Keep polygon format but change class to 0
                parts[0] = '0'  # Single class detection (0 = parasite)
                new_lines.append(' '.join(parts))

            # Write converted labels
            with open(label_path, 'w') as f:
                for line in new_lines:
                    f.write(line + '\n')

            converted_count += len(new_lines)

        return converted_count

    # Convert all labels to single-class detection format
    print("🔄 Converting to single-class detection...")
    train_objects = convert_to_single_class_detection(train_label_dir)
    val_objects = convert_to_single_class_detection(val_label_dir)
    test_objects = convert_to_single_class_detection(test_label_dir)

    total_objects = train_objects + val_objects + test_objects
    print(f"✅ Converted {total_objects} objects ({train_objects} train, {val_objects} val, {test_objects} test)")

    # Create data.yaml for pipeline
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['parasite']
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"✅ Created data.yaml at {yaml_path}")
    print(f"📊 Dataset summary:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Total objects: {total_objects}")
    print(f"   Train: {len(train_files)} images, {train_objects} objects")
    print(f"   Val: {len(val_files)} images, {val_objects} objects")
    print(f"   Test: {len(test_files)} images, {test_objects} objects")
    print(f"   Classes: 1 (parasite detection)")
    print(f"🎯 Dataset ready for pipeline at: {output_dir}")

    return str(yaml_path)

if __name__ == "__main__":
    setup_kaggle_for_pipeline()