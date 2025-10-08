#!/usr/bin/env python3
"""
Generate High-Quality Augmentation Visualization for Paper/Report

Creates publication-quality figures showing various data augmentation techniques
applied to malaria parasite images. Output images are high-resolution (512x512 or larger)
suitable for papers and reports.

Example augmentations:
- Rotations: 90°, 180°, 270°
- Brightness adjustments: 0.8, 1.2, 1.5
- Contrast adjustments: 0.5, 1.5
- Saturation adjustments: 0.5, 1.5
- Sharpness adjustments: 1.5
- Flips: horizontal, vertical
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageEnhance
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def apply_rotation(img, degrees):
    """Apply rotation with white background (no black corners)"""
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


def get_augmentation_grid(image_path, output_size=512, samples_per_class=3):
    """
    Generate comprehensive augmentation examples

    Args:
        image_path: Path to input image
        output_size: Output image size (default: 512x512 for high quality)
        samples_per_class: Number of sample images to show (for multiple originals)

    Returns:
        Dictionary of augmentation name -> augmented image
    """
    # Load and resize image to high quality
    img = Image.open(image_path).convert('RGB')
    img = img.resize((output_size, output_size), Image.BICUBIC)

    augmentations = {}

    # Original (row 1, col 1)
    augmentations['Original\ndetected\ninfected cell'] = img.copy()

    # Rotations (row 1: 90°, 180°, 270°)
    augmentations["90° clockwise\ndirection"] = apply_rotation(img, -90)
    augmentations["180° clockwise\ndirection"] = apply_rotation(img, 180)
    augmentations["90° anti-clockwise\ndirection"] = apply_rotation(img, 90)
    augmentations["270° clockwise\ndirection"] = apply_rotation(img, -270)

    # Brightness adjustments (row 1: 0.8, 1.2)
    augmentations["Brightness 0.8"] = apply_brightness(img, 0.8)
    augmentations["Brightness 1.2"] = apply_brightness(img, 1.2)

    # Row 2: Contrast, Brightness 1.5, Flips, Saturation
    augmentations["Contrast 0.5"] = apply_contrast(img, 0.5)
    augmentations["Brightness 1.5"] = apply_brightness(img, 1.5)
    augmentations["Flip horizontal"] = flip_horizontal(img)
    augmentations["Flip vertical"] = flip_vertical(img)
    augmentations["Saturation 0.5"] = apply_saturation(img, 0.5)
    augmentations["Saturation 1.5"] = apply_saturation(img, 1.5)
    augmentations["Sharpness 1.5"] = apply_sharpness(img, 1.5)

    return augmentations


def create_publication_figure(image_paths, output_path, output_size=512, dpi=300):
    """
    Create publication-quality augmentation figure

    Args:
        image_paths: List of paths to parasite images (1-4 images recommended)
        output_path: Output file path
        output_size: Size of each image in grid (default: 512)
        dpi: Output DPI (default: 300 for publication quality)
    """

    if not isinstance(image_paths, list):
        image_paths = [image_paths]

    n_samples = len(image_paths)

    # Get augmentations for first image (to know structure)
    first_aug = get_augmentation_grid(image_paths[0], output_size)
    aug_names = list(first_aug.keys())
    n_augmentations = len(aug_names)

    # Calculate grid layout
    # We want: n_cols = 7 (original + 6 augmentations per row)
    # n_rows = n_samples * 2 (2 rows per sample)
    n_cols = 7
    n_rows = n_samples * 2

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

    # Process each sample image
    for sample_idx, img_path in enumerate(image_paths):
        augmentations = get_augmentation_grid(img_path, output_size)

        # Get class name
        class_name = Path(img_path).parent.name

        # Calculate row offset for this sample
        row_offset = sample_idx * 2

        # Place augmentations in grid (2 rows × 7 cols per sample)
        positions = [
            # Row 1: Original, 90°, 180°, 90° anti, 270°, Brightness 0.8, Brightness 1.2
            (row_offset, 0, 'Original\ndetected\ninfected cell'),
            (row_offset, 1, "90° clockwise\ndirection"),
            (row_offset, 2, "180° clockwise\ndirection"),
            (row_offset, 3, "90° anti-clockwise\ndirection"),
            (row_offset, 4, "270° clockwise\ndirection"),
            (row_offset, 5, "Brightness 0.8"),
            (row_offset, 6, "Brightness 1.2"),

            # Row 2: Contrast, Brightness 1.5, Flip H, Flip V, Saturation 0.5, Saturation 1.5, Sharpness
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

            # Display image
            if aug_name in augmentations:
                img_array = np.array(augmentations[aug_name])
                ax.imshow(img_array)

                # Add title
                ax.set_title(aug_name, fontsize=9, fontweight='bold', pad=5)

                # Add class label only on first column of each sample
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

    print(f"[SUCCESS] Saved publication figure: {output_path}")
    print(f"          Resolution: {output_size}x{output_size} per image")
    print(f"          DPI: {dpi}")
    print(f"          Total augmentations: {n_augmentations}")
    print(f"          Samples: {n_samples}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate high-quality augmentation visualization for paper/report'
    )
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to parasite crop images (1-4 images recommended)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='augmentation_figure_high_quality.png',
        help='Output file path (default: augmentation_figure_high_quality.png)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=512,
        help='Size of each image in grid (default: 512 for high quality)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output DPI (default: 300 for publication)'
    )

    args = parser.parse_args()

    print("="*80)
    print("HIGH-QUALITY AUGMENTATION VISUALIZATION")
    print("="*80)
    print(f"Input images: {len(args.images)} sample(s)")
    for img_path in args.images:
        print(f"  - {img_path}")
    print(f"Output: {args.output}")
    print(f"Image size: {args.size}x{args.size}")
    print(f"DPI: {args.dpi}")
    print()

    # Validate input images
    for img_path in args.images:
        if not Path(img_path).exists():
            print(f"[ERROR] Image not found: {img_path}")
            return

    # Create figure
    create_publication_figure(
        args.images,
        args.output,
        output_size=args.size,
        dpi=args.dpi
    )

    print()
    print("="*80)
    print("VISUALIZATION COMPLETED!")
    print("="*80)
    print(f"\nOutput file: {args.output}")
    print(f"File size: {Path(args.output).stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
