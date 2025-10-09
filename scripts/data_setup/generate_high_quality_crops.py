#!/usr/bin/env python3
"""
Generate High-Quality Crops for Visualization

Creates higher resolution crops with padding for better visualization quality.
Instead of tight bounding boxes, adds margin and upscales to desired resolution.

Usage:
    python generate_high_quality_crops.py \
        --dataset iml_lifecycle \
        --output-size 512 \
        --padding 0.2 \
        --output crops_high_quality

Features:
    - Adds padding around parasites (default 20%)
    - Upscales to target resolution (default 512x512)
    - Maintains aspect ratio with smart cropping
    - High-quality LANCZOS interpolation
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_yolo_annotations(label_file):
    """Load YOLO format annotations"""
    annotations = []
    if not Path(label_file).exists():
        return annotations

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # class_id, x_center, y_center, width, height
                annotations.append([float(x) for x in parts[:5]])

    return annotations


def yolo_to_bbox(annotation, img_width, img_height, padding=0.2):
    """
    Convert YOLO format to bounding box with padding

    Args:
        annotation: [class_id, x_center, y_center, width, height] (normalized)
        img_width: Image width
        img_height: Image height
        padding: Padding ratio (e.g., 0.2 = 20% padding around box)

    Returns:
        [x1, y1, x2, y2] with padding
    """
    class_id, x_center, y_center, w, h = annotation

    # Add padding
    w_padded = w * (1 + padding)
    h_padded = h * (1 + padding)

    # Convert to absolute coordinates
    x1 = int((x_center - w_padded / 2) * img_width)
    y1 = int((y_center - h_padded / 2) * img_height)
    x2 = int((x_center + w_padded / 2) * img_width)
    y2 = int((y_center + h_padded / 2) * img_height)

    # Clip to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)

    return [x1, y1, x2, y2]


def crop_and_resize(image, bbox, target_size=512, interpolation=Image.LANCZOS):
    """
    Crop region and resize to target size with high quality

    Args:
        image: PIL Image
        bbox: [x1, y1, x2, y2]
        target_size: Target resolution (square)
        interpolation: PIL interpolation method

    Returns:
        PIL Image at target_size x target_size
    """
    x1, y1, x2, y2 = bbox

    # Crop
    crop = image.crop((x1, y1, x2, y2))

    # Get crop dimensions
    crop_w, crop_h = crop.size

    # Calculate aspect ratio
    aspect = crop_w / crop_h

    if aspect > 1:
        # Width > Height: scale width to target, then crop height
        new_w = target_size
        new_h = int(target_size / aspect)
    else:
        # Height > Width: scale height to target, then crop width
        new_h = target_size
        new_w = int(target_size * aspect)

    # Resize with high quality
    crop_resized = crop.resize((new_w, new_h), interpolation)

    # Center crop to target_size x target_size
    final_crop = Image.new('RGB', (target_size, target_size), (255, 255, 255))

    # Calculate paste position (center)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    final_crop.paste(crop_resized, (paste_x, paste_y))

    return final_crop


def get_class_name_from_filename(filename, dataset_name):
    """
    Extract class name from filename suffix

    For detection datasets, class info is in filename, not class_id.
    Example: 1305121398-0012-S.jpg -> schizont
    """
    stem = Path(filename).stem

    # Extract suffix after last hyphen (e.g., "R", "T", "S", "G", "R_T", etc.)
    parts = stem.split('-')
    if len(parts) < 3:
        return 'unknown'

    suffix = parts[-1]  # Last part after hyphen

    # Parse suffix - may have multiple classes like "R_T" or "S_G"
    # Take the first class letter
    class_letters = suffix.split('_')[0] if '_' in suffix else suffix
    first_letter = class_letters[0] if class_letters else 'U'

    # Map letter to class name based on dataset
    if dataset_name in ['mp_idb_stages', 'iml_lifecycle']:
        letter_to_class = {
            'R': 'ring',
            'T': 'trophozoite',
            'S': 'schizont',
            'G': 'gametocyte'
        }
        return letter_to_class.get(first_letter, 'unknown')

    elif dataset_name == 'mp_idb_species':
        # Species dataset uses different naming convention
        # Need to check actual file naming pattern
        return 'P_falciparum'  # Default for now

    return 'unknown'


def process_dataset(dataset_name, data_dir, output_dir, output_size=512, padding=0.2):
    """
    Process entire dataset to generate high-quality crops

    Args:
        dataset_name: Dataset name (iml_lifecycle, mp_idb_species, mp_idb_stages)
        data_dir: Path to processed data (images + labels)
        output_dir: Output directory for high-quality crops
        output_size: Target crop size (default 512x512)
        padding: Padding ratio around bounding box (default 0.2 = 20%)
    """

    # Map dataset names
    dataset_map = {
        'iml_lifecycle': 'lifecycle',
        'mp_idb_species': 'species',
        'mp_idb_stages': 'stages'
    }

    data_subdir = dataset_map.get(dataset_name, dataset_name)

    # Paths
    images_dir = Path(data_dir) / data_subdir / 'test' / 'images'
    labels_dir = Path(data_dir) / data_subdir / 'test' / 'labels'

    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        return

    if not labels_dir.exists():
        print(f"[ERROR] Labels directory not found: {labels_dir}")
        return

    # Get image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG"))

    if not image_files:
        print(f"[ERROR] No images found in {images_dir}")
        return

    print(f"\n{'='*80}")
    print(f"HIGH-QUALITY CROP GENERATION")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Images: {len(image_files)}")
    print(f"Output size: {output_size}x{output_size}")
    print(f"Padding: {padding*100:.0f}%")
    print(f"Interpolation: LANCZOS (high quality)")
    print()

    # Statistics
    total_crops = 0
    stats_by_class = {}

    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Load image
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size

        # Load annotations
        label_file = labels_dir / f"{img_path.stem}.txt"
        annotations = load_yolo_annotations(label_file)

        if not annotations:
            continue

        # Process each annotation
        for crop_id, annotation in enumerate(annotations):
            # Get bounding box with padding
            bbox = yolo_to_bbox(annotation, img_width, img_height, padding)

            # Crop and resize
            high_quality_crop = crop_and_resize(image, bbox, output_size)

            # Get class name from filename (not annotation class_id)
            class_name = get_class_name_from_filename(img_path.name, dataset_name)

            # Create output directory for this class
            class_output_dir = Path(output_dir) / dataset_name / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)

            # Save crop
            output_filename = f"{img_path.stem}_crop_{crop_id:03d}_hq.jpg"
            output_path = class_output_dir / output_filename
            high_quality_crop.save(output_path, quality=95)

            # Update statistics
            total_crops += 1
            if class_name not in stats_by_class:
                stats_by_class[class_name] = 0
            stats_by_class[class_name] += 1

    # Print statistics
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETED!")
    print(f"{'='*80}")
    print(f"Total crops generated: {total_crops}")
    print(f"\nBreakdown by class:")
    for class_name, count in sorted(stats_by_class.items()):
        print(f"  {class_name:20s}: {count:4d} crops")
    print(f"\nOutput directory: {output_dir}/{dataset_name}/")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate high-quality crops for visualization'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['iml_lifecycle', 'mp_idb_species', 'mp_idb_stages', 'all'],
        help='Dataset to process'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Path to processed data directory (default: data/processed)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='crops_high_quality',
        help='Output directory (default: crops_high_quality)'
    )
    parser.add_argument(
        '--output-size',
        type=int,
        default=512,
        help='Target crop size in pixels (default: 512)'
    )
    parser.add_argument(
        '--padding',
        type=float,
        default=0.2,
        help='Padding ratio around bounding box (default: 0.2 = 20%%)'
    )

    args = parser.parse_args()

    # Process datasets
    if args.dataset == 'all':
        datasets = ['iml_lifecycle', 'mp_idb_species', 'mp_idb_stages']
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        process_dataset(
            dataset,
            args.data_dir,
            args.output,
            args.output_size,
            args.padding
        )


if __name__ == "__main__":
    main()
