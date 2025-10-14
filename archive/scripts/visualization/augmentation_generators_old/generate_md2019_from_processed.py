#!/usr/bin/env python3
"""
Generate Publication-Quality Augmentation from MD_2019 Processed Data
Note: MD_2019 parasites are very small (40-96px native)
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageEnhance
import warnings
warnings.filterwarnings('ignore')


def load_md2019_annotations(processed_dir):
    """Load MD_2019 annotations from YOLO format"""
    label_dir = Path(processed_dir) / "train" / "labels"
    image_dir = Path(processed_dir) / "train" / "images"

    # Stage mapping
    stage_map = {0: 'ring', 1: 'trophozoite', 2: 'schizont'}

    # Store all parasites by class
    class_to_parasites = {name: [] for name in stage_map.values()}

    # Parse all labels
    for label_file in label_dir.glob("*.txt"):
        # Find corresponding image
        image_path = None
        for ext in ['.png', '.jpg']:
            candidate = image_dir / (label_file.stem + ext)
            if candidate.exists():
                image_path = candidate
                break

        if not image_path:
            continue

        # Load image dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        img_h, img_w = image.shape[:2]

        # Parse labels
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                # Convert to pixel coordinates
                w_px = int(width * img_w)
                h_px = int(height * img_h)
                native_size = max(w_px, h_px)
                native_with_margin = native_size * 1.5

                if class_id in stage_map:
                    class_name = stage_map[class_id]
                    class_to_parasites[class_name].append({
                        'image_path': str(image_path),
                        'class_id': class_id,
                        'native_size': native_size,
                        'size_with_margin': native_with_margin,
                        'bbox_yolo': [x_center, y_center, width, height],
                        'image_dims': [img_h, img_w]
                    })

    return class_to_parasites


def crop_with_margin(image, bbox_yolo, image_dims, margin_ratio=0.25):
    """Crop image from YOLO bbox with margin"""
    img_h, img_w = image_dims
    x_center, y_center, width, height = bbox_yolo

    # Convert to pixel coordinates
    w_px = int(width * img_w)
    h_px = int(height * img_h)
    x_px = int(x_center * img_w)
    y_px = int(y_center * img_h)

    # Calculate bbox corners
    x1 = int(x_px - w_px / 2)
    y1 = int(y_px - h_px / 2)
    x2 = int(x_px + w_px / 2)
    y2 = int(y_px + h_px / 2)

    # Add margin
    margin_x = int(w_px * margin_ratio)
    margin_y = int(h_px * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image.shape[1], x2 + margin_x)
    y2 = min(image.shape[0], y2 + margin_y)

    # Crop at native resolution
    crop = image[y1:y2, x1:x2]
    return crop


def smart_resize(crop):
    """Smart adaptive resize"""
    h, w = crop.shape[:2]
    native_size = max(h, w)

    # Determine optimal target size
    if native_size >= 400:
        target_size = 512
        quality_rating = "⭐ EXCELLENT"
    elif native_size >= 300:
        target_size = 512
        quality_rating = "✅ VERY GOOD"
    elif native_size >= 250:
        target_size = 448
        quality_rating = "✅ GOOD"
    elif native_size >= 200:
        target_size = 384
        quality_rating = "⚠️ OK"
    elif native_size >= 150:
        target_size = 300
        quality_rating = "❌ NOT IDEAL"
    else:
        target_size = min(256, int(native_size * 2.0))
        quality_rating = "❌ POOR"

    # Convert and resize
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    crop_resized = crop_pil.resize((target_size, target_size), Image.LANCZOS)

    upscale_ratio = target_size / native_size
    print(f"       Upscale: {native_size}px → {target_size}px (ratio: {upscale_ratio:.2f}x) {quality_rating}")

    return np.array(crop_resized)


def apply_rotation(img, degrees):
    return img.rotate(degrees, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

def apply_brightness(img, factor):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def apply_contrast(img, factor):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def apply_saturation(img, factor):
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)

def apply_sharpness(img, factor):
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)

def flip_horizontal(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def flip_vertical(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def get_augmentation_grid(crop_array):
    """Generate augmentations"""
    crop_pil = Image.fromarray(crop_array)

    augmentations = {
        'Original\ndetected\ninfected cell': crop_pil.copy(),
        "90° clockwise\ndirection": apply_rotation(crop_pil, -90),
        "180° clockwise\ndirection": apply_rotation(crop_pil, 180),
        "90° anti-clockwise\ndirection": apply_rotation(crop_pil, 90),
        "270° clockwise\ndirection": apply_rotation(crop_pil, -270),
        "Brightness 0.8": apply_brightness(crop_pil, 0.8),
        "Brightness 1.2": apply_brightness(crop_pil, 1.2),
        "Contrast 0.5": apply_contrast(crop_pil, 0.5),
        "Brightness 1.5": apply_brightness(crop_pil, 1.5),
        "Flip horizontal": flip_horizontal(crop_pil),
        "Flip vertical": flip_vertical(crop_pil),
        "Saturation 0.5": apply_saturation(crop_pil, 0.5),
        "Saturation 1.5": apply_saturation(crop_pil, 1.5),
        "Sharpness 1.5": apply_sharpness(crop_pil, 1.5),
    }

    return augmentations


def create_publication_figure(crops_data, output_path, dpi=300):
    """Create publication-quality augmentation figure"""
    n_samples = len(crops_data)
    n_cols = 7
    n_rows = n_samples * 2

    max_crop_size = max([crop_data.get('final_size', 512) for crop_data in crops_data])
    print(f"\n[FIGURE] Creating {n_rows}×{n_cols} grid")
    print(f"         Max crop size: {max_crop_size}px")

    # Create figure
    fig = plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5))

    # Add title
    fig.suptitle(
        'Figure 1. Example of data augmentation on the detected infected cell\nconducted on the training dataset',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    # Create grid
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.1)

    # Process each sample
    for sample_idx, crop_data in enumerate(crops_data):
        crop_array = crop_data['crop']
        class_name = crop_data['class']

        # Get augmentations
        augmentations = get_augmentation_grid(crop_array)

        # Calculate row offset
        row_offset = sample_idx * 2

        # Layout positions
        positions = [
            (row_offset, 0, 'Original\ndetected\ninfected cell'),
            (row_offset, 1, "90° clockwise\ndirection"),
            (row_offset, 2, "180° clockwise\ndirection"),
            (row_offset, 3, "90° anti-clockwise\ndirection"),
            (row_offset, 4, "270° clockwise\ndirection"),
            (row_offset, 5, "Brightness 0.8"),
            (row_offset, 6, "Brightness 1.2"),
            (row_offset + 1, 0, "Contrast 0.5"),
            (row_offset + 1, 1, "Brightness 1.5"),
            (row_offset + 1, 2, "Flip horizontal"),
            (row_offset + 1, 3, "Flip vertical"),
            (row_offset + 1, 4, "Saturation 0.5"),
            (row_offset + 1, 5, "Saturation 1.5"),
            (row_offset + 1, 6, "Sharpness 1.5"),
        ]

        for row, col, aug_name in positions:
            ax = fig.add_subplot(gs[row, col])

            if aug_name in augmentations:
                img_array = np.array(augmentations[aug_name])
                ax.imshow(img_array)
                ax.set_title(aug_name, fontsize=9, fontweight='bold', pad=5)

                if col == 0 and row == row_offset:
                    ax.text(
                        -0.15, 0.5,
                        f'{class_name.upper()}',
                        transform=ax.transAxes,
                        fontsize=10,
                        fontweight='bold',
                        rotation=90,
                        verticalalignment='center',
                        horizontalalignment='right'
                    )

            ax.axis('off')

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        format='png'
    )
    plt.close()

    print(f"[SUCCESS] Saved: {output_path}")
    print(f"          DPI: {dpi}")
    print(f"          Samples: {n_samples}")


def process_md2019_stages(processed_dir, output_path):
    """Process MD_2019 Stages"""
    print("\n[PROCESSING] MD_2019 Stages Dataset")

    # Load annotations
    class_to_parasites = load_md2019_annotations(processed_dir)

    crops_data = []

    for class_name in ['ring', 'trophozoite', 'schizont']:
        parasites = class_to_parasites[class_name]

        if not parasites:
            print(f"\n[ERROR] {class_name}: No samples found")
            continue

        # Sort by size (largest first)
        parasites.sort(key=lambda x: x['size_with_margin'], reverse=True)

        print(f"\n[SELECT] {class_name}: Found {len(parasites)} samples")
        print(f"         Top 5 largest: {[int(p['size_with_margin']) for p in parasites[:5]]}px")

        # Select LARGEST
        sample = parasites[0]

        # Load image
        image = cv2.imread(sample['image_path'])

        print(f"\n[CROP] {class_name.upper()}")
        print(f"       Original image: {image.shape[:2]}")
        print(f"       Native parasite: {sample['native_size']}px")

        # Crop with margin
        crop = crop_with_margin(image, sample['bbox_yolo'], sample['image_dims'], margin_ratio=0.25)
        print(f"       Native crop: {crop.shape[:2]} (with 25% margin)")

        # Smart resize
        crop_resized = smart_resize(crop)

        crops_data.append({
            'crop': crop_resized,
            'class': class_name,
            'final_size': crop_resized.shape[0]
        })

    if len(crops_data) == 3:
        create_publication_figure(crops_data, output_path)
        return True
    else:
        print(f"\n[WARNING] Could not find all 3 classes (found {len(crops_data)})")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', type=str, default='data/processed/md_2019_stages')
    parser.add_argument('--output', type=str,
                       default='luaran/auto_generated/figures/augmentation/augmentation_md_2019_stages_from_processed.png')
    parser.add_argument('--dpi', type=int, default=300)

    args = parser.parse_args()

    print("=" * 80)
    print("MD_2019 STAGES AUGMENTATION")
    print("=" * 80)
    print(f"Processed dir: {args.processed_dir}")
    print(f"Output: {args.output}")
    print(f"DPI: {args.dpi}")
    print("\n⚠️  WARNING: MD_2019 parasites are very small (40-96px native)")
    print("   Expected quality: ❌ POOR (for comparison purposes only)")

    success = process_md2019_stages(args.processed_dir, args.output)

    if success:
        output_path = Path(args.output)
        file_size = output_path.stat().st_size / 1024 / 1024
        print(f"\n[COMPLETED] MD_2019 Stages")
        print(f"            File: {output_path}")
        print(f"            Size: {file_size:.2f} MB")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
