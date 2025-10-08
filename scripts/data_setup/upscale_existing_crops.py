#!/usr/bin/env python3
"""
Upscale Existing Crops to Higher Resolution

Takes existing ground truth crops and upscales them to higher resolution
for better visualization quality.

Usage:
    python upscale_existing_crops.py \
        --input results/optA_*/experiments/experiment_iml_lifecycle/crops_gt_crops/crops/test \
        --output crops_upscaled_512 \
        --size 512

Features:
    - High-quality LANCZOS interpolation
    - Preserves class folder structure
    - Adds padding (optional)
    - Square crops with white background
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def upscale_crop(input_path, output_path, target_size=512, padding=0.1):
    """
    Upscale crop to target size with optional padding

    Args:
        input_path: Path to input crop
        output_path: Path to save upscaled crop
        target_size: Target resolution (square)
        padding: Padding ratio (0.1 = 10% padding)
    """
    # Load image
    img = Image.open(input_path).convert('RGB')
    w, h = img.size

    # Add padding if requested
    if padding > 0:
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        # Create padded canvas (white background)
        padded = Image.new('RGB', (w + 2*pad_w, h + 2*pad_h), (255, 255, 255))
        padded.paste(img, (pad_w, pad_h))
        img = padded
        w, h = img.size

    # Calculate aspect ratio
    aspect = w / h

    if aspect > 1:
        # Width > Height
        new_w = target_size
        new_h = int(target_size / aspect)
    else:
        # Height >= Width
        new_h = target_size
        new_w = int(target_size * aspect)

    # Resize with high quality LANCZOS
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Center on white canvas
    canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))

    # Save with high quality
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def process_crops_directory(input_dir, output_dir, target_size=512, padding=0.1):
    """
    Process entire directory of crops

    Args:
        input_dir: Input directory (with class subfolders)
        output_dir: Output directory
        target_size: Target crop size
        padding: Padding ratio
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    # Find all class folders
    class_folders = [f for f in input_dir.iterdir() if f.is_dir()]

    if not class_folders:
        print(f"[ERROR] No class folders found in {input_dir}")
        return

    print(f"\n{'='*80}")
    print(f"CROP UPSCALING")
    print(f"{'='*80}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Padding: {padding*100:.0f}%")
    print(f"Interpolation: LANCZOS (high quality)")
    print(f"Classes found: {len(class_folders)}")
    print()

    total_crops = 0
    stats_by_class = {}

    # Process each class
    for class_folder in class_folders:
        class_name = class_folder.name
        print(f"Processing class: {class_name}")

        # Get all image files
        crop_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.JPG"))

        if not crop_files:
            print(f"  [WARNING] No crops found in {class_folder}")
            continue

        # Output directory for this class
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        # Process each crop
        for crop_file in tqdm(crop_files, desc=f"  {class_name}", leave=False):
            output_filename = crop_file.stem + "_upscaled.jpg"
            output_path = class_output_dir / output_filename

            try:
                upscale_crop(crop_file, output_path, target_size, padding)
                total_crops += 1

                if class_name not in stats_by_class:
                    stats_by_class[class_name] = 0
                stats_by_class[class_name] += 1

            except Exception as e:
                print(f"  [ERROR] Failed to process {crop_file.name}: {e}")

    # Print statistics
    print(f"\n{'='*80}")
    print(f"UPSCALING COMPLETED!")
    print(f"{'='*80}")
    print(f"Total crops upscaled: {total_crops}")
    print(f"\nBreakdown by class:")
    for class_name, count in sorted(stats_by_class.items()):
        print(f"  {class_name:20s}: {count:4d} crops")
    print(f"\nOutput directory: {output_dir}/")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Upscale existing crops to higher resolution'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory (containing class subfolders with crops)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=512,
        help='Target crop size in pixels (default: 512)'
    )
    parser.add_argument(
        '--padding',
        type=float,
        default=0.1,
        help='Padding ratio (default: 0.1 = 10%%)'
    )

    args = parser.parse_args()

    process_crops_directory(
        args.input,
        args.output,
        args.size,
        args.padding
    )


if __name__ == "__main__":
    main()
