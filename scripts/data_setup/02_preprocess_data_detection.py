#!/usr/bin/env python3
"""
FIXED preprocessing script for detection training
Creates proper detection dataset instead of individual crops
Author: Assistant
Date: 2025-09-21
"""

import os
import cv2
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import shutil
from sklearn.model_selection import train_test_split


class MalariaDetectionPreprocessor:
    """Preprocessor for malaria detection dataset (full images + YOLO annotations)"""

    def __init__(self, raw_data_dir: str = "data/raw", detection_data_dir: str = "data/detection_ready"):
        """Initialize preprocessor for detection format"""
        self.raw_data_dir = Path(raw_data_dir)
        self.detection_data_dir = Path(detection_data_dir)
        self.detection_data_dir.mkdir(parents=True, exist_ok=True)

        # Create YOLO structure
        for split in ['train', 'val', 'test']:
            (self.detection_data_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.detection_data_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Class mapping for detection
        self.species_to_class = {
            'P_falciparum': 0,
            'P_vivax': 1,
            'P_malariae': 2,
            'P_ovale': 3
        }

        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'total_objects': 0,
            'by_species': {},
            'image_sizes': [],
            'object_sizes': []
        }

    def load_mp_idb_annotations(self, species_dir: Path) -> pd.DataFrame:
        """Load CSV annotations for a species"""
        species_name = species_dir.name
        csv_file = species_dir / f"mp-idb-{species_name.lower()}.csv"

        if not csv_file.exists():
            print(f"Warning: {csv_file} not found")
            return pd.DataFrame()

        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} annotations for {species_name}")
        return df

    def convert_bbox_to_yolo(self, xmin, xmax, ymin, ymax, img_width, img_height):
        """Convert absolute bbox to YOLO format (normalized)"""
        # Ensure correct order (handle swapped coordinates)
        x1, x2 = min(xmin, xmax), max(xmin, xmax)
        y1, y2 = min(ymin, ymax), max(ymin, ymax)

        # Calculate center and dimensions
        x_center = (x1 + x2) / 2.0 / img_width
        y_center = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        # Ensure values are within [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        return x_center, y_center, width, height

    def process_mp_idb_for_detection(self) -> Dict[str, List]:
        """Process MP-IDB dataset for detection training (FIXED VERSION)"""
        print("\\nProcessing MP-IDB Dataset for Detection...")

        dataset_dir = self.raw_data_dir / "mp_idb"
        if not dataset_dir.exists():
            print(f"MP-IDB dataset not found at {dataset_dir}")
            return {}

        all_images = {}

        # Process each species
        for species_folder in ['Falciparum', 'Vivax', 'Malariae', 'Ovale']:
            species_dir = dataset_dir / species_folder
            if not species_dir.exists():
                continue

            species_name = f"P_{species_folder.lower()}"
            class_id = self.species_to_class[species_name]

            print(f"\\nProcessing {species_name} (class {class_id})...")

            # Load annotations
            annotations_df = self.load_mp_idb_annotations(species_dir)
            if annotations_df.empty:
                continue

            # Get image directory
            img_dir = species_dir / "img"
            if not img_dir.exists():
                print(f"  Image directory not found: {img_dir}")
                continue

            # Group annotations by image
            grouped = annotations_df.groupby('filename')
            processed_count = 0

            for filename, group in grouped:
                img_path = img_dir / filename
                if not img_path.exists():
                    print(f"  Warning: Image not found: {img_path}")
                    continue

                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                img_height, img_width = img.shape[:2]

                # Process all annotations for this image
                yolo_labels = []
                valid_objects = 0

                for _, row in group.iterrows():
                    try:
                        # Extract bbox coordinates
                        xmin, xmax = int(row['xmin']), int(row['xmax'])
                        ymin, ymax = int(row['ymin']), int(row['ymax'])

                        # Convert to YOLO format
                        x_center, y_center, width, height = self.convert_bbox_to_yolo(
                            xmin, xmax, ymin, ymax, img_width, img_height
                        )

                        # Skip invalid boxes
                        if width <= 0 or height <= 0:
                            continue

                        # Add to labels
                        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                        valid_objects += 1

                        # Track statistics
                        self.stats['object_sizes'].append(width * height)

                    except (ValueError, KeyError) as e:
                        print(f"    Error processing annotation: {e}")
                        continue

                if yolo_labels:  # Only add if we have valid labels
                    # Generate new filename
                    new_filename = f"{self.stats['total_images']:06d}.jpg"

                    all_images[new_filename] = {
                        'source_path': img_path,
                        'labels': yolo_labels,
                        'species': species_name,
                        'class_id': class_id,
                        'original_filename': filename,
                        'image_size': (img_width, img_height),
                        'object_count': valid_objects
                    }

                    self.stats['total_images'] += 1
                    self.stats['total_objects'] += valid_objects
                    self.stats['image_sizes'].append((img_width, img_height))
                    processed_count += 1

            # Update species statistics
            self.stats['by_species'][species_name] = {
                'images': processed_count,
                'objects': sum(info['object_count'] for info in all_images.values()
                              if info['species'] == species_name)
            }

            print(f"  Processed {processed_count} images for {species_name}")

        print(f"\\nTotal: {len(all_images)} images with {self.stats['total_objects']} objects")
        return all_images

    def split_and_save_dataset(self, all_images: Dict):
        """Split dataset and save to train/val/test with YOLO format"""
        print("\\nSplitting and saving dataset...")

        # Convert to list for splitting
        image_list = list(all_images.items())

        # Split: 70% train, 20% val, 10% test
        train_images, temp_images = train_test_split(image_list, test_size=0.3,
                                                   random_state=42, shuffle=True)
        val_images, test_images = train_test_split(temp_images, test_size=0.33,
                                                 random_state=42, shuffle=True)

        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

        # Save each split
        for split_name, split_images in splits.items():
            print(f"  Saving {split_name} split: {len(split_images)} images")

            for new_filename, info in tqdm(split_images, desc=f"Processing {split_name}"):
                # Copy image (keep original resolution!)
                src_path = info['source_path']
                dst_image_path = self.detection_data_dir / split_name / "images" / new_filename
                shutil.copy2(src_path, dst_image_path)

                # Create YOLO label file
                label_filename = new_filename.replace('.jpg', '.txt')
                dst_label_path = self.detection_data_dir / split_name / "labels" / label_filename

                with open(dst_label_path, 'w') as f:
                    for label in info['labels']:
                        f.write(label + '\n')

        # Create data.yaml
        self.create_data_yaml()

        # Print statistics
        self.print_dataset_statistics(splits)

    def create_data_yaml(self):
        """Create YOLO data.yaml configuration"""
        data_config = {
            'path': str(self.detection_data_dir.absolute()),
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

        yaml_path = self.detection_data_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        print(f"\\nCreated data.yaml at {yaml_path}")

    def print_dataset_statistics(self, splits: Dict):
        """Print comprehensive dataset statistics"""
        print("\\n" + "="*60)
        print("MALARIA DETECTION DATASET STATISTICS (FIXED)")
        print("="*60)

        # Overall statistics
        print(f"Total Images: {self.stats['total_images']}")
        print(f"Total Objects: {self.stats['total_objects']}")
        print(f"Avg Objects per Image: {self.stats['total_objects']/self.stats['total_images']:.1f}")

        # Image size statistics
        if self.stats['image_sizes']:
            widths = [s[0] for s in self.stats['image_sizes']]
            heights = [s[1] for s in self.stats['image_sizes']]
            print(f"Image Sizes: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")

        # Object size statistics
        if self.stats['object_sizes']:
            obj_sizes = self.stats['object_sizes']
            print(f"Object Sizes (normalized): {min(obj_sizes):.6f} to {max(obj_sizes):.6f}")
            print(f"Avg Object Size: {np.mean(obj_sizes):.6f}")

        # By species
        print("\\nBy Species:")
        for species, stats in self.stats['by_species'].items():
            print(f"  {species}: {stats['images']} images, {stats['objects']} objects")

        # By split
        print("\\nBy Split:")
        for split_name, split_images in splits.items():
            images_count = len(split_images)
            objects_count = sum(info['object_count'] for _, info in split_images)
            print(f"  {split_name}: {images_count} images, {objects_count} objects")

            # Species distribution in this split
            species_in_split = {}
            for _, info in split_images:
                species = info['species']
                species_in_split[species] = species_in_split.get(species, 0) + 1

            for species, count in species_in_split.items():
                print(f"    {species}: {count} images")

    def run(self):
        """Run the complete preprocessing for detection"""
        print("="*60)
        print("FIXED MALARIA DETECTION PREPROCESSING")
        print("Creating full images + YOLO annotations (NOT crops)")
        print("="*60)

        # Process MP-IDB dataset
        all_images = self.process_mp_idb_for_detection()

        if not all_images:
            print("No images processed. Check your data directory.")
            return

        # Split and save
        self.split_and_save_dataset(all_images)

        print(f"\\n✅ FIXED detection dataset created at: {self.detection_data_dir}")
        print("This dataset contains:")
        print("  - Full resolution microscopy images")
        print("  - YOLO format bounding box annotations")
        print("  - Multiple parasites per image")
        print("  - Ready for proper detection training!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process malaria dataset for detection training")
    parser.add_argument("--raw-data", default="data/raw", help="Raw data directory")
    parser.add_argument("--output", default="data/detection_ready", help="Output directory")

    args = parser.parse_args()

    preprocessor = MalariaDetectionPreprocessor(args.raw_data, args.output)
    preprocessor.run()