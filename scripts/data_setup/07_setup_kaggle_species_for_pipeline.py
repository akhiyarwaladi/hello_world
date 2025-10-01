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

def convert_kaggle_class_to_stage(kaggle_class_id):
    """Convert Kaggle 16-class ID to 4-stage class ID"""
    # Kaggle MP-IDB has 16 classes: 4 species × 4 stages
    # Classes 0-3: P_falciparum (Ring, Schizont, Trophozoite, Gametocyte)
    # Classes 4-7: P_vivax (Ring, Schizont, Trophozoite, Gametocyte)
    # Classes 8-11: P_malariae (Ring, Schizont, Trophozoite, Gametocyte)
    # Classes 12-15: P_ovale (Ring, Schizont, Trophozoite, Gametocyte)

    stage_id = kaggle_class_id % 4
    return stage_id  # 0=Ring, 1=Schizont, 2=Trophozoite, 3=Gametocyte

def polygon_to_bbox(polygon_coords):
    """Convert polygon coordinates to bounding box format"""
    if len(polygon_coords) < 6:  # Need at least 3 points (6 coordinates)
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

def setup_kaggle_for_pipeline(output_type="species", single_class=True,
                             train_ratio=0.70, val_ratio=0.20, test_ratio=0.10):
    """
    Setup Kaggle dataset for pipeline use

    Args:
        output_type: "species" or "stages"
        single_class: True for single parasite class, False for multi-class
        train_ratio: Training set ratio (default: 0.70 = 70%)
        val_ratio: Validation set ratio (default: 0.20 = 20%)
        test_ratio: Test set ratio (default: 0.10 = 10%)
    """
    # Verify ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio:.4f}")

    print(f"[SPLIT] Using split ratios: Train={train_ratio:.2%}, Val={val_ratio:.2%}, Test={test_ratio:.2%}")

    if output_type == "species":
        print("[SETUP] Setting up Kaggle MP-IDB for SPECIES classification...")
        output_dir = Path("data/processed/species")
    elif output_type == "stages":
        print("[SETUP] Setting up Kaggle MP-IDB for STAGES classification...")
        output_dir = Path("data/processed/stages")
    else:
        raise ValueError(f"Invalid output_type: {output_type}. Must be 'species' or 'stages'")

    # Paths
    kaggle_source = Path("data/raw/kaggle_dataset/MP-IDB-YOLO")

    if not kaggle_source.exists():
        print(f"[ERROR] Kaggle dataset not found at {kaggle_source}")
        return False

    # Step 0: Clean existing processed data for fresh setup
    if output_dir.exists():
        print(f"[CLEAN] Removing existing processed data: {output_dir}")
        shutil.rmtree(output_dir)
        print(f"[CLEAN] Cleaned successfully - ensuring fresh setup")

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
    print(f"[INFO] Found {len(image_files)} images in Kaggle dataset")

    # Create stratification labels based on species in each image
    print(f"[STRATIFY] Analyzing species distribution for stratified splitting...")
    stratify_labels = []
    species_mapping = {
        # P_falciparum (classes 0-3) -> species 0
        0: 0, 1: 0, 2: 0, 3: 0,
        # P_vivax (classes 4-7) -> species 1
        4: 1, 5: 1, 6: 1, 7: 1,
        # P_malariae (classes 8-11) -> species 2
        8: 2, 9: 2, 10: 2, 11: 2,
        # P_ovale (classes 12-15) -> species 3
        12: 3, 13: 3, 14: 3, 15: 3
    }

    from collections import Counter

    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = all_labels_dir / label_file

        if label_path.exists():
            species_in_image = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        species_id = species_mapping.get(class_id, 0)
                        species_in_image.append(species_id)

            # Use most frequent species in image for stratification
            if species_in_image:
                dominant_species = max(set(species_in_image), key=species_in_image.count)
                stratify_labels.append(dominant_species)
            else:
                stratify_labels.append(0)  # Default to P_falciparum
        else:
            stratify_labels.append(0)  # Default if no label file

    # Print species distribution
    species_dist = Counter(stratify_labels)
    species_names = ['P_falciparum', 'P_vivax', 'P_malariae', 'P_ovale']
    print(f"[STRATIFY] Species distribution for stratification:")
    for species_id, count in sorted(species_dist.items()):
        species_name = species_names[species_id] if species_id < len(species_names) else f"species_{species_id}"
        print(f"   {species_name}: {count} images")

    # Stratified split with custom ratios
    try:
        # First split: separate train set
        temp_size = val_ratio + test_ratio  # Remaining after train
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            image_files, stratify_labels,
            test_size=temp_size, random_state=42, stratify=stratify_labels
        )
        # Second split: separate val and test from remaining
        val_adjusted = val_ratio / (val_ratio + test_ratio)
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels,
            test_size=(1 - val_adjusted), random_state=42, stratify=temp_labels
        )
        print(f"[STRATIFY] Successfully applied stratified splitting")
    except ValueError as e:
        print(f"[WARNING] Stratified split failed ({e}), using random split")
        temp_size = val_ratio + test_ratio
        train_files, temp_files = train_test_split(image_files, test_size=temp_size, random_state=42)
        val_adjusted = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(temp_files, test_size=(1 - val_adjusted), random_state=42)

    print(f"[SPLIT] {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # Validate split distribution
    def validate_species_distribution(files, split_name):
        split_species_dist = Counter()
        for img_file in files:
            img_idx = image_files.index(img_file)
            species_id = stratify_labels[img_idx]
            split_species_dist[species_id] += 1

        print(f"   {split_name} species distribution:")
        for species_id, count in sorted(split_species_dist.items()):
            species_name = species_names[species_id] if species_id < len(species_names) else f"species_{species_id}"
            print(f"      {species_name}: {count} images")

        return split_species_dist

    print(f"\n[VALIDATION] Final split distribution:")
    train_dist = validate_species_distribution(train_files, "Train")
    val_dist = validate_species_distribution(val_files, "Val")
    test_dist = validate_species_distribution(test_files, "Test")

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
        """Convert segmentation polygons to bounding boxes and single class"""
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

                # Extract polygon coordinates (skip class_id)
                coords = [float(x) for x in parts[1:]]
                if len(coords) < 6:  # Need at least 3 points (6 coordinates)
                    continue

                # Convert polygon to bounding box
                x_coords = coords[::2]  # Every other coordinate starting from 0
                y_coords = coords[1::2]  # Every other coordinate starting from 1

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # YOLO format: class x_center y_center width height
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                # Single class detection (0 = parasite)
                new_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Write converted labels
            with open(label_path, 'w') as f:
                for line in new_lines:
                    f.write(line + '\n')

            converted_count += len(new_lines)

        return converted_count

    # Convert all labels to single-class detection format
    print("[PROCESSING] Converting segmentation polygons to bounding boxes...")
    train_objects = convert_to_single_class_detection(train_label_dir)
    val_objects = convert_to_single_class_detection(val_label_dir)
    test_objects = convert_to_single_class_detection(test_label_dir)

    total_objects = train_objects + val_objects + test_objects
    print(f"[SUCCESS] Converted {total_objects} polygons to bounding boxes ({train_objects} train, {val_objects} val, {test_objects} test)")

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

    print(f"[SUCCESS] Created data.yaml at {yaml_path}")
    print(f"[INFO] Dataset summary:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Total objects: {total_objects}")
    print(f"   Train: {len(train_files)} images, {train_objects} objects")
    print(f"   Val: {len(val_files)} images, {val_objects} objects")
    print(f"   Test: {len(test_files)} images, {test_objects} objects")
    print(f"   Classes: 1 (parasite detection)")
    print(f"[READY] Dataset ready for pipeline at: {output_dir}")

    return str(yaml_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setup Kaggle MP-IDB dataset for species classification")
    parser.add_argument("--train-ratio", type=float, default=0.66,
                       help="Training set ratio (default: 0.66 = 66%%)")
    parser.add_argument("--val-ratio", type=float, default=0.17,
                       help="Validation set ratio (default: 0.17 = 17%%)")
    parser.add_argument("--test-ratio", type=float, default=0.17,
                       help="Test set ratio (default: 0.17 = 17%%)")

    args = parser.parse_args()
    setup_kaggle_for_pipeline(output_type="species", single_class=True,
                             train_ratio=args.train_ratio,
                             val_ratio=args.val_ratio,
                             test_ratio=args.test_ratio)