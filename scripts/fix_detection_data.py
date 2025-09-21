#!/usr/bin/env python3
"""
Fix detection training data to use full images instead of crops
Author: Assistant
Date: 2025-09-21
"""

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
from collections import defaultdict
import yaml

class DetectionDataFixer:
    """Fix detection training data format"""

    def __init__(self):
        self.raw_data_dir = Path("data/raw/mp_idb")
        self.output_dir = Path("data/detection_format")
        self.integrated_dir = Path("data/integrated")

        # Create output structure
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    def load_mp_idb_annotations(self):
        """Load original MP-IDB annotations"""
        annotations = {}

        for species in ['Falciparum', 'Vivax', 'Malariae', 'Ovale']:
            csv_file = self.raw_data_dir / species / f"mp-idb-{species.lower()}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                annotations[species] = df
                print(f"Loaded {len(df)} annotations for {species}")

        return annotations

    def convert_bbox_to_yolo(self, xmin, xmax, ymin, ymax, img_width, img_height):
        """Convert absolute bbox to YOLO format (normalized)"""
        # Calculate center and dimensions
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = abs(xmax - xmin) / img_width
        height = abs(ymax - ymin) / img_height

        # Ensure values are within [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        return x_center, y_center, width, height

    def create_detection_dataset(self):
        """Create proper detection dataset from raw data"""
        print("Creating detection dataset from raw MP-IDB data...")

        # Load annotations
        annotations = self.load_mp_idb_annotations()

        # Species to class mapping
        species_to_class = {
            'Falciparum': 0,  # P_falciparum
            'Vivax': 1,       # P_vivax
            'Malariae': 2,    # P_malariae
            'Ovale': 3        # P_ovale
        }

        image_counter = 0
        all_images = {}

        # Process each species
        for species, df in annotations.items():
            class_id = species_to_class[species]
            img_dir = self.raw_data_dir / species / "img"

            # Group annotations by image
            grouped = df.groupby('filename')

            for filename, group in grouped:
                img_path = img_dir / filename
                if not img_path.exists():
                    print(f"Warning: Image not found: {img_path}")
                    continue

                # Load image to get dimensions
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                img_height, img_width = img.shape[:2]
                print(f"Processing {filename}: {img_width}x{img_height}")

                # Collect all bounding boxes for this image
                yolo_labels = []
                for _, row in group.iterrows():
                    # Extract bbox coordinates
                    xmin, xmax = int(row['xmin']), int(row['xmax'])
                    ymin, ymax = int(row['ymin']), int(row['ymax'])

                    # Convert to YOLO format
                    x_center, y_center, width, height = self.convert_bbox_to_yolo(
                        xmin, xmax, ymin, ymax, img_width, img_height
                    )

                    # Add to labels
                    yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # Store image info
                new_filename = f"{image_counter:06d}.jpg"
                all_images[new_filename] = {
                    'source_path': img_path,
                    'labels': yolo_labels,
                    'species': species,
                    'original_filename': filename
                }
                image_counter += 1

        print(f"Total images to process: {len(all_images)}")
        return all_images

    def split_and_save_dataset(self, all_images):
        """Split dataset and save to train/val/test"""
        from sklearn.model_selection import train_test_split

        # Convert to list for splitting
        image_list = list(all_images.items())

        # Split: 70% train, 20% val, 10% test
        train_images, temp_images = train_test_split(image_list, test_size=0.3, random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=0.33, random_state=42)

        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

        # Save each split
        for split_name, split_images in splits.items():
            print(f"\\nProcessing {split_name} split: {len(split_images)} images")

            for new_filename, info in split_images:
                # Copy image
                src_path = info['source_path']
                dst_image_path = self.output_dir / split_name / "images" / new_filename
                shutil.copy2(src_path, dst_image_path)

                # Create label file
                label_filename = new_filename.replace('.jpg', '.txt')
                dst_label_path = self.output_dir / split_name / "labels" / label_filename

                with open(dst_label_path, 'w') as f:
                    for label in info['labels']:
                        f.write(label + '\\n')

        # Create data.yaml
        self.create_data_yaml()

        # Print statistics
        self.print_statistics(splits)

    def create_data_yaml(self):
        """Create YOLO data.yaml configuration"""
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 4,  # Number of classes
            'names': {
                0: 'P_falciparum',
                1: 'P_vivax',
                2: 'P_malariae',
                3: 'P_ovale'
            }
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        print(f"Created data.yaml at {yaml_path}")

    def print_statistics(self, splits):
        """Print dataset statistics"""
        print("\\n" + "="*50)
        print("DETECTION DATASET STATISTICS")
        print("="*50)

        for split_name, split_images in splits.items():
            print(f"\\n{split_name.upper()} SET:")
            print(f"  Total images: {len(split_images)}")

            # Count by species
            species_count = defaultdict(int)
            bbox_count = defaultdict(int)

            for _, info in split_images:
                species_count[info['species']] += 1
                bbox_count[info['species']] += len(info['labels'])

            print("  Images by species:")
            for species, count in species_count.items():
                print(f"    {species}: {count} images")

            print("  Bounding boxes by species:")
            for species, count in bbox_count.items():
                print(f"    {species}: {count} objects")

    def run(self):
        """Run the complete fixing process"""
        print("Starting detection data fixing process...")

        # Create detection dataset
        all_images = self.create_detection_dataset()

        # Split and save
        self.split_and_save_dataset(all_images)

        print(f"\\nDetection dataset created at: {self.output_dir}")
        print("Ready for YOLO detection training!")

if __name__ == "__main__":
    fixer = DetectionDataFixer()
    fixer.run()