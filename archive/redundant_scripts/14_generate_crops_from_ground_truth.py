#!/usr/bin/env python3
"""
Generate Classification Crops from Ground Truth Annotations
This script uses ground truth bounding boxes to crop parasites for classification training
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.results_manager import ResultsManager

def parse_yolo_annotation(annotation_file, image_width, image_height):
    """Parse YOLO format annotation file"""
    boxes = []
    if not annotation_file.exists():
        return boxes

    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert YOLO format to pixel coordinates
                x_center_px = x_center * image_width
                y_center_px = y_center * image_height
                width_px = width * image_width
                height_px = height * image_height

                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)

                boxes.append({
                    'class_id': class_id,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': 1.0  # Ground truth has perfect confidence
                })

    return boxes

def crop_from_ground_truth(image_path, annotation_path, crop_size=128):
    """Generate crops from ground truth annotations"""
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    image_height, image_width = image.shape[:2]
    boxes = parse_yolo_annotation(annotation_path, image_width, image_height)

    crops = []
    for box in boxes:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

        # Calculate center and expand to square crop
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Create square crop around center
        half_size = crop_size // 2
        crop_x1 = max(0, center_x - half_size)
        crop_y1 = max(0, center_y - half_size)
        crop_x2 = min(image_width, center_x + half_size)
        crop_y2 = min(image_height, center_y + half_size)

        # Extract crop
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize to exact crop size if needed
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            crop = cv2.resize(crop, (crop_size, crop_size))

        crops.append({
            'crop': crop,
            'confidence': box['confidence'],
            'bbox': [x1, y1, x2, y2],
            'crop_coords': [crop_x1, crop_y1, crop_x2, crop_y2],
            'class_id': box['class_id']
        })

    return crops

def process_ground_truth_dataset(input_dir, output_dir, crop_size=128):
    """Process entire dataset and generate crops from ground truth"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    crops_dir = output_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for train/val/test
    for split in ['train', 'val', 'test']:
        split_path = input_path / split
        if split_path.exists():
            (crops_dir / split).mkdir(exist_ok=True)

    # Process images and collect metadata
    metadata = []
    processed_count = 0
    crop_count = 0

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

    for split in ['train', 'val', 'test']:
        images_path = input_path / split / "images"
        labels_path = input_path / split / "labels"

        if not images_path.exists() or not labels_path.exists():
            continue

        print(f"Processing {split} split...")

        # Get all images in this split
        split_images = []
        for ext in image_extensions:
            split_images.extend(images_path.glob(ext))

        for image_path in tqdm(split_images, desc=f"Processing {split}"):
            try:
                # Find corresponding annotation file
                annotation_file = labels_path / f"{image_path.stem}.txt"

                crops = crop_from_ground_truth(image_path, annotation_file, crop_size)
                processed_count += 1

                for i, crop_data in enumerate(crops):
                    # Generate crop filename
                    crop_filename = f"{image_path.stem}_gt_crop_{i:03d}.jpg"
                    crop_output_path = crops_dir / split / crop_filename

                    # Save crop
                    cv2.imwrite(str(crop_output_path), crop_data['crop'])

                    # Add metadata
                    metadata.append({
                        'original_image': str(image_path.relative_to(input_path)),
                        'crop_filename': crop_filename,
                        'split': split,
                        'confidence': crop_data['confidence'],
                        'bbox_x1': crop_data['bbox'][0],
                        'bbox_y1': crop_data['bbox'][1],
                        'bbox_x2': crop_data['bbox'][2],
                        'bbox_y2': crop_data['bbox'][3],
                        'crop_x1': crop_data['crop_coords'][0],
                        'crop_y1': crop_data['crop_coords'][1],
                        'crop_x2': crop_data['crop_coords'][2],
                        'crop_y2': crop_data['crop_coords'][3],
                        'dataset_source': 'ground_truth',
                        'class_id': crop_data['class_id']
                    })

                    crop_count += 1

            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                continue

    # Save metadata
    if metadata:
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(output_path / 'ground_truth_crop_metadata.csv', index=False)

        print(f"\n✅ Ground truth processing completed:")
        print(f"   📸 Images processed: {processed_count}")
        print(f"   ✂️  Crops generated: {crop_count}")
        print(f"   📊 Average crops per image: {crop_count/processed_count:.2f}")

        # Show split distribution
        if 'split' in metadata_df.columns:
            split_counts = metadata_df['split'].value_counts()
            print(f"   📂 Split distribution:")
            for split, count in split_counts.items():
                print(f"      {split}: {count} crops")

        return metadata_df
    else:
        print("❌ No crops were generated from ground truth!")
        return None

def create_yolo_classification_structure_gt(crops_dir, metadata_df, output_dir):
    """Create YOLO classification directory structure from ground truth crops"""
    yolo_dir = Path(output_dir) / "yolo_classification"

    # For now, create single class structure (all crops are "parasite")
    # Later this can be extended to use species labels if available

    for split in ['train', 'val', 'test']:
        split_crops = metadata_df[metadata_df['split'] == split]
        if len(split_crops) > 0:
            # Create parasite class directory
            class_dir = yolo_dir / split / "parasite"
            class_dir.mkdir(parents=True, exist_ok=True)

            # Copy crops to class directory
            for _, row in split_crops.iterrows():
                src_path = Path(crops_dir) / split / row['crop_filename']
                dst_path = class_dir / row['crop_filename']

                if src_path.exists():
                    import shutil
                    shutil.copy2(src_path, dst_path)

    print(f"✅ YOLO classification structure (ground truth) created at: {yolo_dir}")
    return yolo_dir

def main():
    parser = argparse.ArgumentParser(description="Generate crops from ground truth annotations")
    parser.add_argument("--input", required=True,
                       help="Input dataset directory (with train/val/test/images and labels structure)")
    parser.add_argument("--output", required=True,
                       help="Output directory for generated crops")
    parser.add_argument("--crop_size", type=int, default=128,
                       help="Size of generated crops")
    parser.add_argument("--create_yolo_structure", action="store_true",
                       help="Create YOLO classification directory structure")

    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING CROPS FROM GROUND TRUTH ANNOTATIONS")
    print("=" * 60)

    # Validate inputs
    if not Path(args.input).exists():
        print(f"❌ Input directory not found: {args.input}")
        return

    print(f"📁 Input dataset: {args.input}")
    print(f"📂 Output directory: {args.output}")
    print(f"📏 Crop size: {args.crop_size}x{args.crop_size}")

    try:
        # Process dataset
        metadata = process_ground_truth_dataset(
            input_dir=args.input,
            output_dir=args.output,
            crop_size=args.crop_size
        )

        if metadata is not None and args.create_yolo_structure:
            # Create YOLO classification structure
            crops_dir = Path(args.output) / "crops"
            yolo_dir = create_yolo_classification_structure_gt(
                crops_dir, metadata, args.output
            )

        print(f"\n🎉 Ground truth crop generation completed successfully!")
        print(f"📊 Results saved to: {args.output}")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return

if __name__ == "__main__":
    main()