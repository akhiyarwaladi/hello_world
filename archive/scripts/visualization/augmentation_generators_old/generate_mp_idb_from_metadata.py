#!/usr/bin/env python3
"""
Generate Publication-Quality Augmentation from MP-IDB Metadata
Uses ground truth crop metadata to find largest parasites per species
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
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def load_metadata(metadata_path):
    """Load and process crop metadata"""
    df = pd.read_csv(metadata_path)

    # Calculate bbox size
    df['width'] = df['bbox_x2'] - df['bbox_x1']
    df['height'] = df['bbox_y2'] - df['bbox_y1']
    df['size'] = df[['width', 'height']].max(axis=1)
    df['size_with_margin'] = df['size'] * 1.5  # 25% margin

    return df


def crop_with_margin(image, bbox, margin_ratio=0.25):
    """Crop image with margin at NATIVE resolution"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Add margin
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image.shape[1], x2 + margin_x)
    y2 = min(image.shape[0], y2 + margin_y)

    # Crop at native resolution
    crop = image[y1:y2, x1:x2]
    return crop


def smart_resize(crop):
    """Smart adaptive resize: minimize blur by limiting upscale ratio"""
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
        target_size = min(256, int(native_size * 1.8))
        quality_rating = "❌ POOR"

    # Convert and resize
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    crop_resized = crop_pil.resize((target_size, target_size), Image.LANCZOS)

    upscale_ratio = target_size / native_size
    print(f"       Upscale: {native_size}px → {target_size}px (ratio: {upscale_ratio:.2f}x) {quality_rating}")

    return np.array(crop_resized)


def apply_rotation(img, degrees):
    """Apply rotation with white background"""
    return img.rotate(degrees, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))


def apply_brightness(img, factor):
    """Apply brightness adjustment"""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_contrast(img, factor):
    """Apply contrast adjustment"""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def apply_saturation(img, factor):
    """Apply saturation adjustment"""
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)


def apply_sharpness(img, factor):
    """Apply sharpness adjustment"""
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)


def flip_horizontal(img):
    """Flip image horizontally"""
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def flip_vertical(img):
    """Flip image vertically"""
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def get_augmentation_grid(crop_array):
    """Generate augmentations from numpy array"""
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

        # Layout positions (2 rows × 7 cols per sample)
        positions = [
            # Row 1
            (row_offset, 0, 'Original\ndetected\ninfected cell'),
            (row_offset, 1, "90° clockwise\ndirection"),
            (row_offset, 2, "180° clockwise\ndirection"),
            (row_offset, 3, "90° anti-clockwise\ndirection"),
            (row_offset, 4, "270° clockwise\ndirection"),
            (row_offset, 5, "Brightness 0.8"),
            (row_offset, 6, "Brightness 1.2"),
            # Row 2
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

                # Add class label on first column
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


def process_mp_idb_species(metadata_path, image_dir, output_path):
    """Process MP-IDB Species dataset from metadata"""
    print("\n[PROCESSING] MP-IDB Species Dataset (from metadata)")

    # Load metadata
    df = load_metadata(metadata_path)
    print(f"[INFO] Loaded metadata for {len(df)} crops")

    # Species to process
    species_list = ['P_falciparum', 'P_vivax', 'P_ovale', 'P_malariae']

    crops_data = []

    for species in species_list:
        # Filter by species and sort by size
        species_df = df[df['class_name'] == species].sort_values('size_with_margin', ascending=False)

        print(f"\n[SELECT] {species}: Found {len(species_df)} samples")

        if len(species_df) == 0:
            print(f"         [ERROR] No samples found for {species}")
            continue

        # Show top 10
        print(f"         Top 10 largest (with 25% margin):")
        for i, (idx, row) in enumerate(species_df.head(10).iterrows()):
            print(f"         #{i+1}: Native size ~{row['size_with_margin']:.0f}px")

        # Select LARGEST sample
        sample = species_df.iloc[0]

        # Load original image (remove 'images/' prefix if present)
        img_file = sample['original_image']
        if img_file.startswith('images/'):
            img_file = img_file[7:]  # Remove 'images/' prefix

        # Try all splits (train, val, test)
        image_path = None
        base_dir = Path(image_dir).parent  # Go up to species/ folder

        for split in ['train', 'val', 'test']:
            candidate_path = base_dir / split / 'images' / img_file
            if candidate_path.exists():
                image_path = candidate_path
                break

        if image_path is None:
            print(f"         [ERROR] Image not found in any split: {img_file}")
            continue

        image = cv2.imread(str(image_path))

        print(f"\n[CROP] {species.upper()}")
        print(f"       Original image: {image.shape[:2]}")
        print(f"       Image: {sample['original_image']}")
        print(f"       Bbox: [{sample['bbox_x1']}, {sample['bbox_y1']}, {sample['bbox_x2']}, {sample['bbox_y2']}]")

        # Crop with margin at NATIVE resolution
        bbox = [sample['bbox_x1'], sample['bbox_y1'], sample['bbox_x2'], sample['bbox_y2']]
        crop = crop_with_margin(image, bbox, margin_ratio=0.25)
        print(f"       Native crop: {crop.shape[:2]} (with 25% margin)")

        # Smart adaptive resize
        crop_resized = smart_resize(crop)

        crops_data.append({
            'crop': crop_resized,
            'class': species.replace('_', ' ').title(),
            'final_size': crop_resized.shape[0]
        })

    # Create figure
    if len(crops_data) == 4:
        create_publication_figure(crops_data, output_path)
        return True
    else:
        print(f"\n[WARNING] Could not find all 4 species (found {len(crops_data)})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality augmentation from MP-IDB metadata'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='results/optA_20251013_220815/experiments/experiment_mp_idb_species/crops_gt_crops/ground_truth_crop_metadata.csv',
        help='Path to ground truth crop metadata CSV'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='data/processed/species/train',
        help='Directory containing original images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='luaran/auto_generated/figures/augmentation/augmentation_mp_idb_species_from_metadata.png',
        help='Output file path'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output DPI (default: 300)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MP-IDB SPECIES AUGMENTATION FROM METADATA")
    print("=" * 80)
    print(f"Metadata: {args.metadata}")
    print(f"Image dir: {args.image_dir}")
    print(f"Output: {args.output}")
    print(f"DPI: {args.dpi}")

    success = process_mp_idb_species(args.metadata, args.image_dir, args.output)

    if success:
        output_path = Path(args.output)
        file_size = output_path.stat().st_size / 1024 / 1024
        print(f"\n[COMPLETED] MP-IDB Species")
        print(f"            File: {output_path}")
        print(f"            Size: {file_size:.2f} MB")

    print("\n" + "=" * 80)
    print("GENERATION COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
