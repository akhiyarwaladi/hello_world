#!/usr/bin/env python3
"""
Crop individual parasites from detection dataset for classification training
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse

class ParasiteCropper:
    """Crop individual parasites from detection bounding boxes"""

    def __init__(self,
                 detection_path: str = "data/detection_fixed",
                 output_path: str = "data/classification_crops",
                 crop_size: int = 128,
                 padding: int = 16):

        self.detection_path = Path(detection_path)
        self.output_path = Path(output_path)
        self.crop_size = crop_size
        self.padding = padding

        # Create output structure
        self.setup_output_directories()

        print(f"Parasite Cropper initialized")
        print(f"Detection path: {self.detection_path}")
        print(f"Output path: {self.output_path}")
        print(f"Crop size: {self.crop_size}x{self.crop_size}")
        print(f"Padding: {self.padding}px")

    def setup_output_directories(self):
        """Create output directory structure for classification"""

        dirs_to_create = [
            self.output_path,
            self.output_path / "train",
            self.output_path / "val",
            self.output_path / "test",
            self.output_path / "train" / "parasite",
            self.output_path / "val" / "parasite",
            self.output_path / "test" / "parasite"
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

        print("✓ Created output directories")

    def load_yolo_annotations(self, label_file: Path) -> list:
        """Load YOLO format annotations"""

        annotations = []
        if not label_file.exists():
            return annotations

        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        annotations.append({
                            'class_id': class_id,
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': width,
                            'height': height
                        })
        except Exception as e:
            print(f"❌ Error reading {label_file}: {e}")

        return annotations

    def denormalize_bbox(self, annotation: dict, img_width: int, img_height: int) -> tuple:
        """Convert normalized YOLO bbox to pixel coordinates"""

        center_x = annotation['center_x'] * img_width
        center_y = annotation['center_y'] * img_height
        width = annotation['width'] * img_width
        height = annotation['height'] * img_height

        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)

        return x1, y1, x2, y2

    def crop_parasite(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Crop parasite with padding and resize to standard size"""

        img_height, img_width = image.shape[:2]

        # Add padding
        x1_pad = max(0, x1 - self.padding)
        y1_pad = max(0, y1 - self.padding)
        x2_pad = min(img_width, x2 + self.padding)
        y2_pad = min(img_height, y2 + self.padding)

        # Crop the region
        crop = image[y1_pad:y2_pad, x1_pad:x2_pad]

        if crop.size == 0:
            return None

        # Resize to standard size
        crop_resized = cv2.resize(crop, (self.crop_size, self.crop_size))

        return crop_resized

    def process_detection_dataset(self):
        """Process all detection images and crop parasites"""

        images_dir = self.detection_path / "images"
        labels_dir = self.detection_path / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            print(f"❌ Detection dataset not found in {self.detection_path}")
            return False

        # Get all image files
        image_files = list(images_dir.glob("*.jpg"))
        print(f"Found {len(image_files)} images to process")

        cropped_count = 0
        total_parasites = 0
        crop_metadata = []

        for img_file in tqdm(image_files, desc="Cropping parasites"):
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"⚠️ Cannot load image: {img_file}")
                continue

            img_height, img_width = image.shape[:2]

            # Load corresponding label file
            label_file = labels_dir / f"{img_file.stem}.txt"
            annotations = self.load_yolo_annotations(label_file)

            if not annotations:
                print(f"⚠️ No annotations for {img_file.name}")
                continue

            total_parasites += len(annotations)

            # Process each parasite in the image
            for i, annotation in enumerate(annotations):
                # Convert to pixel coordinates
                x1, y1, x2, y2 = self.denormalize_bbox(annotation, img_width, img_height)

                # Crop parasite
                crop = self.crop_parasite(image, x1, y1, x2, y2)

                if crop is None:
                    continue

                # Generate crop filename
                crop_filename = f"{img_file.stem}_parasite_{i:02d}.jpg"

                # Save to train folder (we'll split later)
                crop_path = self.output_path / "train" / "parasite" / crop_filename

                success = cv2.imwrite(str(crop_path), crop)

                if success:
                    cropped_count += 1

                    # Store metadata
                    crop_metadata.append({
                        'crop_filename': crop_filename,
                        'source_image': img_file.name,
                        'bbox_pixel': [x1, y1, x2, y2],
                        'bbox_normalized': [annotation['center_x'], annotation['center_y'],
                                          annotation['width'], annotation['height']],
                        'crop_size': [self.crop_size, self.crop_size]
                    })

        print(f"\\n✓ Cropped {cropped_count} parasites from {total_parasites} total")

        # Save metadata
        metadata_file = self.output_path / "crop_metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(crop_metadata, f, indent=2)
            print(f"✓ Saved crop metadata to {metadata_file}")
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")

        return cropped_count > 0

    def split_crops_for_training(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Split cropped parasites into train/val/test sets"""

        train_dir = self.output_path / "train" / "parasite"
        val_dir = self.output_path / "val" / "parasite"
        test_dir = self.output_path / "test" / "parasite"

        # Get all cropped files
        crop_files = list(train_dir.glob("*.jpg"))

        if not crop_files:
            print("❌ No cropped files found to split")
            return False

        # Shuffle for random split
        np.random.seed(42)
        np.random.shuffle(crop_files)

        # Calculate split indices
        total_files = len(crop_files)
        train_end = int(total_files * train_ratio)
        val_end = int(total_files * (train_ratio + val_ratio))

        # Move files to appropriate directories
        val_files = crop_files[train_end:val_end]
        test_files = crop_files[val_end:]

        print(f"\\nSplitting {total_files} crops:")
        print(f"   Train: {train_end} files (staying in place)")
        print(f"   Val: {len(val_files)} files")
        print(f"   Test: {len(test_files)} files")

        # Move validation files
        for file_path in val_files:
            new_path = val_dir / file_path.name
            file_path.rename(new_path)

        # Move test files
        for file_path in test_files:
            new_path = test_dir / file_path.name
            file_path.rename(new_path)

        print("✓ Dataset split completed")
        return True

    def create_dataset_summary(self):
        """Create summary of the cropped dataset"""

        train_count = len(list((self.output_path / "train" / "parasite").glob("*.jpg")))
        val_count = len(list((self.output_path / "val" / "parasite").glob("*.jpg")))
        test_count = len(list((self.output_path / "test" / "parasite").glob("*.jpg")))
        total_count = train_count + val_count + test_count

        summary = {
            'dataset_type': 'parasite_classification_crops',
            'crop_size': f'{self.crop_size}x{self.crop_size}',
            'padding': self.padding,
            'total_crops': total_count,
            'train_crops': train_count,
            'val_crops': val_count,
            'test_crops': test_count,
            'classes': ['parasite'],
            'source': 'MP-IDB detection dataset'
        }

        # Save summary
        summary_file = self.output_path / "dataset_summary.json"
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving summary: {e}")
            return False

        # Print summary
        print(f"\\n{'='*50}")
        print("PARASITE CLASSIFICATION CROPS SUMMARY")
        print(f"{'='*50}")
        print(f"Total crops: {total_count}")
        print(f"Train: {train_count} ({train_count/total_count*100:.1f}%)")
        print(f"Val: {val_count} ({val_count/total_count*100:.1f}%)")
        print(f"Test: {test_count} ({test_count/total_count*100:.1f}%)")
        print(f"Crop size: {self.crop_size}x{self.crop_size}")
        print(f"Output directory: {self.output_path}")
        print(f"{'='*50}\\n")

        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Crop parasites from detection dataset")
    parser.add_argument('--detection-path', default='data/detection_fixed',
                       help='Detection dataset path')
    parser.add_argument('--output-path', default='data/classification_crops',
                       help='Output path for cropped parasites')
    parser.add_argument('--crop-size', type=int, default=128,
                       help='Crop size (default: 128)')
    parser.add_argument('--padding', type=int, default=16,
                       help='Padding around bounding box (default: 16)')

    args = parser.parse_args()

    # Create cropper
    cropper = ParasiteCropper(
        detection_path=args.detection_path,
        output_path=args.output_path,
        crop_size=args.crop_size,
        padding=args.padding
    )

    print("Starting parasite cropping from detection dataset...")

    # Process detection dataset
    if not cropper.process_detection_dataset():
        print("❌ Failed to crop parasites")
        return

    # Split into train/val/test
    if not cropper.split_crops_for_training():
        print("❌ Failed to split dataset")
        return

    # Create summary
    if not cropper.create_dataset_summary():
        print("❌ Failed to create summary")
        return

    print("🎉 Parasite cropping completed successfully!")
    print("✅ Cropped parasite dataset ready for classification training!")
    print(f"\\nNext steps:")
    print(f"1. Train classification: python scripts/train_classification.py --data {args.output_path}")
    print(f"2. Compare with detection results")

if __name__ == "__main__":
    main()