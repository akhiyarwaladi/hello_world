#!/usr/bin/env python3
"""
Generate crops from ground truth annotations for clean classification training.
Supports IML lifecycle, MP-IDB species, and MP-IDB stages datasets.
"""

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import shutil

class GroundTruthCropGenerator:
    """Generate crops from ground truth annotations"""

    def __init__(self, dataset_path, output_path, crop_size=224, train_ratio=0.66, val_ratio=0.17, test_ratio=0.17):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.crop_size = crop_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

        # Remove existing output directory to prevent duplicates
        if self.output_path.exists():
            print(f"[CLEAN] Removing existing output directory: {self.output_path}")
            import shutil
            shutil.rmtree(self.output_path)
            print(f"[CLEAN] Directory cleaned successfully")

        # Create output directory structure
        self.crops_dir = self.output_path / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'total_images': 0,
            'total_crops': 0,
            'split_distribution': defaultdict(lambda: defaultdict(int)),
            'class_distribution': defaultdict(int)
        }

    def detect_dataset_type(self):
        """Detect which dataset we're processing"""
        # Check for IML lifecycle raw data structure
        if (self.dataset_path / "annotations.json").exists() and (self.dataset_path / "IML_Malaria").exists():
            return 'iml_lifecycle'

        # Check for processed data with data.yaml
        if (self.dataset_path / "data.yaml").exists():
            # Read data.yaml to understand structure
            import yaml
            with open(self.dataset_path / "data.yaml", 'r') as f:
                config = yaml.safe_load(f)

            if config.get('nc', 0) == 1 and 'parasite' in config.get('names', []):
                return 'iml_lifecycle'
            elif 'ring' in str(config.get('names', [])):
                return 'mp_idb_stages'
            elif 'falciparum' in str(config.get('names', [])):
                return 'mp_idb_species'

        # Fallback detection based on path
        dataset_name = str(self.dataset_path).lower()
        if 'lifecycle' in dataset_name or 'malaria_lifecycle' in dataset_name:
            return 'iml_lifecycle'
        elif 'species' in dataset_name:
            return 'mp_idb_species'
        elif 'stages' in dataset_name:
            return 'mp_idb_stages'

        return 'unknown'

    def get_class_mapping(self, dataset_type):
        """Get class mapping for each dataset type"""
        if dataset_type == 'iml_lifecycle':
            # Map single detection class to 4 lifecycle classes based on actual annotations
            return {
                'detection_to_classification': {0: 'needs_mapping'},  # Will be determined from source data
                'class_names': ['ring', 'gametocyte', 'trophozoite', 'schizont'],
                'class_ids': {'ring': 0, 'gametocyte': 1, 'trophozoite': 2, 'schizont': 3}
            }
        elif dataset_type == 'mp_idb_species':
            # Map from 16 classes to 4 species based on actual MP-IDB structure
            # Classes: falciparum(0-3), vivax(4-7), ovale(8-11), malariae(12-15)
            species_mapping = {}
            for i in range(16):
                species_id = i // 4  # 0-3 -> 0, 4-7 -> 1, 8-11 -> 2, 12-15 -> 3
                species_mapping[i] = species_id

            return {
                'detection_to_classification': species_mapping,
                'class_names': ['P_falciparum', 'P_vivax', 'P_ovale', 'P_malariae'],
                'class_ids': {'P_falciparum': 0, 'P_vivax': 1, 'P_ovale': 2, 'P_malariae': 3}
            }
        elif dataset_type == 'mp_idb_stages':
            # Map from 16 classes to 4 stages
            stage_mapping = {}
            for i in range(16):
                stage_id = i % 4  # 0,4,8,12 -> 0; 1,5,9,13 -> 1; etc.
                stage_mapping[i] = stage_id

            return {
                'detection_to_classification': stage_mapping,
                'class_names': ['ring', 'schizont', 'trophozoite', 'gametocyte'],
                'class_ids': {'ring': 0, 'schizont': 1, 'trophozoite': 2, 'gametocyte': 3}
            }

        return None

    def load_original_annotations_iml(self):
        """Load original annotations for IML lifecycle to get true class labels"""
        # Look for original annotation source
        raw_lifecycle_dir = Path("data/raw/malaria_lifecycle")

        if not raw_lifecycle_dir.exists():
            print(f"[WARNING] Raw lifecycle data not found at {raw_lifecycle_dir}")
            return {}

        # Find annotation file
        annotation_file = raw_lifecycle_dir / "annotations.json"
        if not annotation_file.exists():
            print(f"[WARNING] No annotation file found at {annotation_file}")
            return {}

        print(f"[INFO] Loading original annotations from {annotation_file}")

        import json
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        # Map image filename to annotations with true class labels
        image_to_annotations = defaultdict(list)

        # Statistics for filtering
        total_objects = 0
        skipped_rbc = 0
        skipped_difficult = 0
        skipped_unknown = 0

        # Handle custom IML lifecycle format
        # Format: [{"image_name": "...", "objects": [{"type": "...", "bbox": {"x": "...", "y": "...", "h": "...", "w": "..."}}]}]

        if isinstance(annotations, list):
            for image_entry in annotations:
                if 'image_name' in image_entry and 'objects' in image_entry:
                    filename = image_entry['image_name']

                    for obj in image_entry['objects']:
                        object_type = obj.get('type', '').lower()
                        bbox_info = obj.get('bbox', {})
                        total_objects += 1

                        # Skip red blood cells - we only want parasites
                        if 'red blood' in object_type:
                            skipped_rbc += 1
                            continue

                        # Skip difficult annotations - ambiguous cases
                        if object_type == 'difficult':
                            skipped_difficult += 1
                            continue

                        # Map object types to our class IDs
                        class_mapping = {
                            'ring': 0, 'gametocyte': 1, 'trophozoite': 2, 'schizont': 3
                        }

                        class_id = class_mapping.get(object_type, -1)
                        if class_id == -1:
                            skipped_unknown += 1
                            print(f"[WARNING] Unknown type '{object_type}' in {filename}")
                            continue  # Skip unknown types

                        # Convert bbox format from {"x": "...", "y": "...", "h": "...", "w": "..."} to [x, y, width, height]
                        try:
                            x = int(bbox_info['x'])
                            y = int(bbox_info['y'])
                            w = int(bbox_info['w'])
                            h = int(bbox_info['h'])

                            image_to_annotations[filename].append({
                                'bbox': [x, y, w, h],  # [x, y, width, height]
                                'class_id': class_id,
                                'class_name': object_type
                            })
                        except (KeyError, ValueError) as e:
                            print(f"[WARNING] Invalid bbox format for {filename}: {bbox_info}")
                            continue

        print(f"[INFO] Loaded annotations for {len(image_to_annotations)} images")
        print(f"[FILTER] Processed {total_objects} total objects:")
        print(f"  - Skipped {skipped_rbc} red blood cells")
        print(f"  - Skipped {skipped_difficult} difficult annotations")
        print(f"  - Skipped {skipped_unknown} unknown types")

        # Print class distribution from original annotations
        class_counts = defaultdict(int)
        for filename, anns in image_to_annotations.items():
            for ann in anns:
                class_counts[ann['class_id']] += 1

        total_lifecycle = sum(class_counts.values())
        print(f"[INFO] Using {total_lifecycle} lifecycle stage annotations:")
        class_names = ['ring', 'gametocyte', 'trophozoite', 'schizont']
        for class_id, count in sorted(class_counts.items()):
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            percentage = (count / total_lifecycle * 100) if total_lifecycle > 0 else 0
            print(f"  {class_name}: {count} annotations ({percentage:.1f}%)")

        return image_to_annotations

    def create_stratified_splits(self, dataset_type, original_annotations=None):
        """Create stratified train/val/test splits for single directory datasets"""
        from sklearn.model_selection import train_test_split
        from collections import Counter

        image_files = []
        stratify_labels = []

        if dataset_type == 'iml_lifecycle' and original_annotations:
            # For IML lifecycle, stratify by dominant class in each image
            # Sort image names for deterministic file ordering
            sorted_image_names = sorted(original_annotations.keys())
            for image_name in sorted_image_names:
                anns = original_annotations[image_name]
                if anns:
                    image_files.append(image_name)
                    # Use most frequent class in image for stratification
                    class_ids = [ann['class_id'] for ann in anns]
                    dominant_class = max(set(class_ids), key=class_ids.count)
                    stratify_labels.append(dominant_class)
        else:
            # For MP-IDB datasets, scan labels
            images_dir = self.dataset_path / "images"
            labels_dir = self.dataset_path / "labels"

            # Sort image files for deterministic file ordering
            for image_file in sorted(images_dir.glob("*.jpg")):
                label_file = labels_dir / f"{image_file.stem}.txt"
                if label_file.exists():
                    image_files.append(image_file.name)

                    # Read label file and get dominant class
                    class_ids = []
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                class_id = int(parts[0])
                                if dataset_type == 'mp_idb_species':
                                    # Map to species: 0-3->0, 4-7->1, 8-11->2, 12-15->3
                                    class_id = class_id // 4
                                elif dataset_type == 'mp_idb_stages':
                                    # Map to stages: 0,4,8,12->0, 1,5,9,13->1, etc.
                                    class_id = class_id % 4
                                class_ids.append(class_id)

                    if class_ids:
                        dominant_class = max(set(class_ids), key=class_ids.count)
                        stratify_labels.append(dominant_class)
                    else:
                        image_files.pop()  # Remove if no valid labels

        if not image_files:
            print(f"[WARNING] No valid images found for stratification")
            return {}

        # Print distribution before split
        class_dist = Counter(stratify_labels)
        print(f"[STRATIFY] Overall class distribution: {dict(class_dist)}")

        # Stratified split using configured ratios (default: 66% train, 17% val, 17% test)
        temp_size = self.val_ratio + self.test_ratio
        test_relative_size = self.test_ratio / temp_size if temp_size > 0 else 0.5

        print(f"[SPLIT RATIOS] Train={self.train_ratio:.0%}, Val={self.val_ratio:.0%}, Test={self.test_ratio:.0%}")

        try:
            train_files, temp_files, train_labels, temp_labels = train_test_split(
                image_files, stratify_labels,
                test_size=temp_size, random_state=42, stratify=stratify_labels
            )
            val_files, test_files, val_labels, test_labels = train_test_split(
                temp_files, temp_labels,
                test_size=test_relative_size, random_state=42, stratify=temp_labels
            )
            print(f"[STRATIFY] Split: {len(train_files)} train ({len(train_files)/len(image_files):.1%}), "
                  f"{len(val_files)} val ({len(val_files)/len(image_files):.1%}), "
                  f"{len(test_files)} test ({len(test_files)/len(image_files):.1%})")
        except ValueError as e:
            print(f"[WARNING] Stratified split failed ({e}), using random split")
            train_files, temp_files = train_test_split(image_files, test_size=temp_size, random_state=42)
            val_files, test_files = train_test_split(temp_files, test_size=test_relative_size, random_state=42)

        # Create assignment dictionary
        assignment = {}
        for file in train_files:
            assignment[file] = 'train'
        for file in val_files:
            assignment[file] = 'val'
        for file in test_files:
            assignment[file] = 'test'

        return assignment

    def yolo_to_pixel_coords(self, yolo_coords, img_width, img_height):
        """Convert YOLO normalized coordinates to pixel coordinates"""
        x_center, y_center, width, height = yolo_coords

        # Convert to pixel coordinates
        pixel_width = width * img_width
        pixel_height = height * img_height
        pixel_x_center = x_center * img_width
        pixel_y_center = y_center * img_height

        # Convert to x1, y1, x2, y2
        x1 = int(pixel_x_center - pixel_width / 2)
        y1 = int(pixel_y_center - pixel_height / 2)
        x2 = int(pixel_x_center + pixel_width / 2)
        y2 = int(pixel_y_center + pixel_height / 2)

        return x1, y1, x2, y2

    def polygon_to_bbox(self, polygon_coords, img_width, img_height):
        """Convert polygon coordinates to bounding box"""
        if len(polygon_coords) < 6:  # Need at least 3 points (6 coordinates)
            return None

        # Convert normalized coordinates to pixel coordinates
        pixel_coords = []
        for i, coord in enumerate(polygon_coords):
            if i % 2 == 0:  # x coordinate
                pixel_coords.append(float(coord) * img_width)
            else:  # y coordinate
                pixel_coords.append(float(coord) * img_height)

        # Separate x and y coordinates
        x_coords = pixel_coords[::2]  # Even indices
        y_coords = pixel_coords[1::2]  # Odd indices

        if len(x_coords) < 2 or len(y_coords) < 2:
            return None

        # Calculate bounding box
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Return as x1, y1, x2, y2
        return int(x_min), int(y_min), int(x_max), int(y_max)

    def generate_crop(self, image, bbox, target_size=None):
        """Generate crop from image using bounding box"""
        if target_size is None:
            target_size = self.crop_size

        x1, y1, x2, y2 = bbox

        # Ensure coordinates are within image bounds
        img_height, img_width = image.shape[:2]
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(x1 + 1, min(x2, img_width))
        y2 = max(y1 + 1, min(y2, img_height))

        # Extract crop
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Resize to target size using high-quality interpolation
        # Use INTER_CUBIC for upscaling (better quality) or INTER_AREA for downscaling
        current_size = max(crop.shape[:2])
        if current_size < target_size:
            # Upscaling - use cubic interpolation for better quality
            crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        else:
            # Downscaling - use area interpolation to preserve details
            crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)

        return crop_resized

    def process_dataset(self, dataset_type):
        """Process dataset to generate ground truth crops"""
        print(f"[PROCESSING] Generating ground truth crops for {dataset_type}")

        class_mapping = self.get_class_mapping(dataset_type)
        if not class_mapping:
            print(f"[ERROR] Unknown dataset type: {dataset_type}")
            return False

        class_names = class_mapping['class_names']

        # Create class directories for each split
        for split in ['train', 'val', 'test']:
            for class_name in class_names:
                (self.crops_dir / split / class_name).mkdir(parents=True, exist_ok=True)

        # Load original annotations for IML lifecycle
        original_annotations = {}
        if dataset_type == 'iml_lifecycle':
            original_annotations = self.load_original_annotations_iml()

        metadata = []

        # Check if dataset has train/val/test structure or is a single directory
        has_splits = any((self.dataset_path / split).exists() for split in ['train', 'val', 'test'])

        # For single directory datasets, prepare stratified splits
        image_split_assignment = {}
        if not has_splits:
            print(f"[STRATIFY] Creating stratified train/val/test splits...")
            image_split_assignment = self.create_stratified_splits(dataset_type, original_annotations)

        if has_splits:
            # Process existing splits
            splits_to_process = ['train', 'val', 'test']
        else:
            # Single directory dataset - process all images and create splits later
            print(f"[INFO] No train/val/test splits found, processing all images from root directory")
            splits_to_process = ['all']

        # Process each split
        for split in splits_to_process:
            if split == 'all':
                # Handle different raw data structures
                if dataset_type == 'iml_lifecycle':
                    # IML lifecycle raw data structure
                    images_dir = self.dataset_path / "IML_Malaria"
                    labels_dir = None  # Use original_annotations instead
                else:
                    # Standard raw data structure (MP-IDB)
                    images_dir = self.dataset_path / "images"
                    labels_dir = self.dataset_path / "labels"
                # We'll determine splits dynamically per image based on stratified sampling
            else:
                # Process existing split directory
                split_dir = self.dataset_path / split
                if not split_dir.exists():
                    continue
                images_dir = split_dir / "images"
                labels_dir = split_dir / "labels"
                actual_split = split

            # Check if images directory exists
            if not images_dir.exists():
                continue

            # For IML lifecycle raw data, labels are in original_annotations
            if dataset_type == 'iml_lifecycle' and split == 'all':
                if not original_annotations:
                    print(f"[WARNING] No original annotations loaded for IML lifecycle")
                    continue
            elif labels_dir and not labels_dir.exists():
                continue

            print(f"[SPLIT] Processing {split}...")

            # Process each image
            image_pattern = "*.JPG" if dataset_type == 'iml_lifecycle' else "*.jpg"
            for image_path in images_dir.glob(image_pattern):
                # For IML lifecycle, we use original annotations directly
                if dataset_type == 'iml_lifecycle' and image_path.name in original_annotations:
                    # Load image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        continue

                    img_height, img_width = image.shape[:2]
                    self.stats['total_images'] += 1

                    # Determine split assignment for this image
                    if split == 'all' and image_split_assignment:
                        actual_split = image_split_assignment.get(image_path.name, 'train')
                    elif split != 'all':
                        actual_split = split
                    else:
                        actual_split = 'train'  # fallback

                    # Use original annotations with true class labels
                    original_anns = original_annotations[image_path.name]

                    for i, ann in enumerate(original_anns):
                        # Convert from [x, y, width, height] to [x1, y1, x2, y2]
                        x, y, w, h = ann['bbox']
                        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

                        # Generate crop
                        crop = self.generate_crop(image, (x1, y1, x2, y2))
                        if crop is None:
                            continue

                        class_id = ann['class_id']
                        class_name = class_names[class_id]

                        # Save crop
                        crop_filename = f"{image_path.stem}_crop_{i:03d}.jpg"
                        crop_path = self.crops_dir / actual_split / class_name / crop_filename
                        cv2.imwrite(str(crop_path), crop)

                        # Add to metadata
                        metadata.append({
                            'original_image': str(image_path.relative_to(self.dataset_path)),
                            'crop_filename': crop_filename,
                            'split': actual_split,
                            'class_id': class_id,
                            'class_name': class_name,
                            'bbox_x1': x1, 'bbox_y1': y1, 'bbox_x2': x2, 'bbox_y2': y2,
                            'source': 'ground_truth'
                        })

                        self.stats['total_crops'] += 1
                        self.stats['split_distribution'][actual_split][class_id] += 1
                        self.stats['class_distribution'][class_id] += 1

                elif dataset_type != 'iml_lifecycle':
                    # For MP-IDB datasets, use YOLO labels
                    label_path = labels_dir / f"{image_path.stem}.txt"

                    if not label_path.exists():
                        continue

                    # Load image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        continue

                    img_height, img_width = image.shape[:2]
                    self.stats['total_images'] += 1

                    # Determine split assignment for this image
                    if split == 'all' and image_split_assignment:
                        actual_split = image_split_assignment.get(image_path.name, 'train')
                    elif split != 'all':
                        actual_split = split
                    else:
                        actual_split = 'train'  # fallback

                    # Load YOLO labels
                    with open(label_path, 'r') as f:
                        lines = f.readlines()

                    # Use YOLO labels for MP-IDB datasets
                    for i, line in enumerate(lines):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        yolo_class_id = int(parts[0])
                        polygon_coords = parts[1:]  # All coordinates after class_id

                        # Convert polygon to bounding box
                        bbox_coords = self.polygon_to_bbox(polygon_coords, img_width, img_height)
                        if bbox_coords is None:
                            continue
                        x1, y1, x2, y2 = bbox_coords

                        # Map to classification class
                        if dataset_type in ['mp_idb_species', 'mp_idb_stages']:
                            class_id = class_mapping['detection_to_classification'].get(yolo_class_id, 0)
                        else:
                            class_id = 0  # Default for single class

                        class_name = class_names[class_id]

                        # Generate crop
                        crop = self.generate_crop(image, (x1, y1, x2, y2))
                        if crop is None:
                            continue

                        # Save crop
                        crop_filename = f"{image_path.stem}_crop_{i:03d}.jpg"
                        crop_path = self.crops_dir / actual_split / class_name / crop_filename
                        cv2.imwrite(str(crop_path), crop)

                        # Add to metadata
                        metadata.append({
                            'original_image': str(image_path.relative_to(self.dataset_path)),
                            'crop_filename': crop_filename,
                            'split': actual_split,
                            'class_id': class_id,
                            'class_name': class_name,
                            'bbox_x1': x1, 'bbox_y1': y1, 'bbox_x2': x2, 'bbox_y2': y2,
                            'source': 'ground_truth'
                        })

                        self.stats['total_crops'] += 1
                        self.stats['split_distribution'][actual_split][class_id] += 1
                        self.stats['class_distribution'][class_id] += 1

        # Save metadata
        if metadata:
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_csv(self.output_path / 'ground_truth_crop_metadata.csv', index=False)

        return True

    def print_statistics(self):
        """Print generation statistics"""
        print(f"\n[STATISTICS] Ground Truth Crop Generation:")
        print(f"  Total images processed: {self.stats['total_images']}")
        print(f"  Total crops generated: {self.stats['total_crops']}")

        print(f"\n[SPLIT DISTRIBUTION]:")
        for split, class_counts in self.stats['split_distribution'].items():
            total_split = sum(class_counts.values())
            print(f"  {split}: {total_split} crops")
            for class_id, count in sorted(class_counts.items()):
                print(f"    Class {class_id}: {count} crops")

        print(f"\n[CLASS DISTRIBUTION]:")
        total_crops = sum(self.stats['class_distribution'].values())
        for class_id, count in sorted(self.stats['class_distribution'].items()):
            percentage = (count / total_crops * 100) if total_crops > 0 else 0
            print(f"  Class {class_id}: {count} crops ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Generate ground truth crops for classification training')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory (with train/val/test structure)')
    parser.add_argument('--output', required=True, help='Output directory for generated crops')
    parser.add_argument('--crop_size', type=int, default=224, help='Size of generated crops (default: 224 for paper compatibility)')
    parser.add_argument('--type', choices=['iml_lifecycle', 'mp_idb_species', 'mp_idb_stages'],
                       help='Force dataset type (overrides auto-detection)')
    parser.add_argument('--train-ratio', type=float, default=0.66,
                       help='Training set ratio (default: 0.66 = 66%%)')
    parser.add_argument('--val-ratio', type=float, default=0.17,
                       help='Validation set ratio (default: 0.17 = 17%%)')
    parser.add_argument('--test-ratio', type=float, default=0.17,
                       help='Test set ratio (default: 0.17 = 17%%)')

    args = parser.parse_args()

    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"[ERROR] Train/val/test ratios must sum to 1.0, got {total_ratio:.4f}")
        print(f"  --train-ratio: {args.train_ratio}")
        print(f"  --val-ratio: {args.val_ratio}")
        print(f"  --test-ratio: {args.test_ratio}")
        return

    # Initialize generator with split ratios
    generator = GroundTruthCropGenerator(
        args.dataset, args.output, args.crop_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    # Detect or use specified dataset type
    if args.type:
        dataset_type = args.type
        print(f"[SPECIFIED] Dataset type: {dataset_type}")
    else:
        dataset_type = generator.detect_dataset_type()
        print(f"[DETECTED] Dataset type: {dataset_type}")

    if dataset_type == 'unknown':
        print("[ERROR] Could not detect dataset type")
        return

    # Process dataset
    success = generator.process_dataset(dataset_type)

    if success:
        generator.print_statistics()
        print(f"\n[SUCCESS] Ground truth crops generated at: {args.output}")
    else:
        print("[ERROR] Failed to generate crops")

if __name__ == "__main__":
    main()