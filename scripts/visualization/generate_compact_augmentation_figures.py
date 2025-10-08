#!/usr/bin/env python3
"""
Generate Compact Augmentation Visualizations for Journal Publication

Creates 3 publication-ready compact versions:
1. Single-row version (1 class × 7 augmentations) - Most compact
2. Two-class version (2 classes × 7 augmentations) - Balanced
3. Three-class version (3 classes × 7 augmentations) - Representative

Optimized for journal space constraints while maintaining clarity.
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


def get_compact_augmentations(image_path, output_size=512):
    """
    Generate compact set of most representative augmentations

    Returns dict with 7 key augmentations:
    - Original
    - Rotation (90° or 180°)
    - Brightness (darker/lighter)
    - Contrast
    - Saturation
    - Flip (horizontal)
    - Sharpness
    """
    # Load and resize image to high quality
    img = Image.open(image_path).convert('RGB')
    img = img.resize((output_size, output_size), Image.BICUBIC)

    # Select 7 most representative augmentations for journal
    augmentations = {
        'Original': img.copy(),
        '90° rotation': apply_rotation(img, -90),
        'Brightness 0.7': apply_brightness(img, 0.7),
        'Brightness 1.3': apply_brightness(img, 1.3),
        'Contrast 0.6': apply_contrast(img, 0.6),
        'Saturation 1.4': apply_saturation(img, 1.4),
        'Flip horizontal': flip_horizontal(img),
    }

    return augmentations


def create_compact_figure_v1(image_path, output_path, output_size=512, dpi=300):
    """
    Version 1: Single-row compact (1 class × 7 augmentations)
    Most space-efficient for journals with tight space constraints
    """
    augmentations = get_compact_augmentations(image_path, output_size)
    aug_names = list(augmentations.keys())

    n_cols = len(aug_names)
    n_rows = 1

    # Create figure - horizontal strip
    fig = plt.figure(figsize=(n_cols * 1.8, n_rows * 2))

    # Add title
    class_name = Path(image_path).parent.name.upper()
    fig.suptitle(
        f'Data augmentation examples ({class_name} stage)',
        fontsize=12,
        fontweight='bold',
        y=0.98
    )

    # Create grid
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.1, wspace=0.15)

    for col_idx, aug_name in enumerate(aug_names):
        ax = fig.add_subplot(gs[0, col_idx])

        # Display image
        img_array = np.array(augmentations[aug_name])
        ax.imshow(img_array)

        # Add title
        ax.set_title(aug_name, fontsize=8, fontweight='bold', pad=3)
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

    print(f"[SUCCESS] Saved V1 (single-row): {output_path}")
    print(f"          Layout: {n_rows} row × {n_cols} augmentations")


def create_compact_figure_v2(image_paths, output_path, output_size=512, dpi=300):
    """
    Version 2: Two-class compact (2 classes × 7 augmentations)
    Balanced view showing diversity across classes
    """
    if not isinstance(image_paths, list):
        image_paths = [image_paths]

    # Limit to 2 samples for compact version
    image_paths = image_paths[:2]
    n_samples = len(image_paths)

    # Get augmentations
    first_aug = get_compact_augmentations(image_paths[0], output_size)
    n_cols = len(first_aug)
    n_rows = n_samples

    # Create figure
    fig = plt.figure(figsize=(n_cols * 1.8, n_rows * 2))

    # Add title
    fig.suptitle(
        'Data augmentation examples on different lifecycle stages',
        fontsize=12,
        fontweight='bold',
        y=0.995
    )

    # Create grid
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.25, wspace=0.15)

    # Process each sample
    for sample_idx, img_path in enumerate(image_paths):
        augmentations = get_compact_augmentations(img_path, output_size)
        aug_names = list(augmentations.keys())
        class_name = Path(img_path).parent.name.upper()

        for col_idx, aug_name in enumerate(aug_names):
            ax = fig.add_subplot(gs[sample_idx, col_idx])

            # Display image
            img_array = np.array(augmentations[aug_name])
            ax.imshow(img_array)

            # Add title
            ax.set_title(aug_name, fontsize=8, fontweight='bold', pad=3)

            # Add class label on first column
            if col_idx == 0:
                ax.text(
                    -0.15, 0.5,
                    class_name,
                    transform=ax.transAxes,
                    fontsize=9,
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

    print(f"[SUCCESS] Saved V2 (two-class): {output_path}")
    print(f"          Layout: {n_rows} rows × {n_cols} augmentations")


def create_compact_figure_v3(image_paths, output_path, output_size=512, dpi=300):
    """
    Version 3: Three-class compact (3 classes × 7 augmentations)
    More representative while still compact for journals
    """
    if not isinstance(image_paths, list):
        image_paths = [image_paths]

    # Limit to 3 samples for compact version
    image_paths = image_paths[:3]
    n_samples = len(image_paths)

    # Get augmentations
    first_aug = get_compact_augmentations(image_paths[0], output_size)
    n_cols = len(first_aug)
    n_rows = n_samples

    # Create figure
    fig = plt.figure(figsize=(n_cols * 1.8, n_rows * 2))

    # Add title
    fig.suptitle(
        'Representative data augmentation examples across lifecycle stages',
        fontsize=12,
        fontweight='bold',
        y=0.997
    )

    # Create grid
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.25, wspace=0.15)

    # Process each sample
    for sample_idx, img_path in enumerate(image_paths):
        augmentations = get_compact_augmentations(img_path, output_size)
        aug_names = list(augmentations.keys())
        class_name = Path(img_path).parent.name.upper()

        for col_idx, aug_name in enumerate(aug_names):
            ax = fig.add_subplot(gs[sample_idx, col_idx])

            # Display image
            img_array = np.array(augmentations[aug_name])
            ax.imshow(img_array)

            # Add title
            ax.set_title(aug_name, fontsize=8, fontweight='bold', pad=3)

            # Add class label on first column
            if col_idx == 0:
                ax.text(
                    -0.15, 0.5,
                    class_name,
                    transform=ax.transAxes,
                    fontsize=9,
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

    print(f"[SUCCESS] Saved V3 (three-class): {output_path}")
    print(f"          Layout: {n_rows} rows × {n_cols} augmentations")


def main():
    parser = argparse.ArgumentParser(
        description='Generate compact augmentation visualizations for journal publication'
    )
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to parasite crop images (1-3 images for compact versions)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='luaran/figures',
        help='Output directory (default: luaran/figures)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='augmentation_compact',
        help='Output filename prefix (default: augmentation_compact)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=512,
        help='Size of each image in grid (default: 512)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output DPI (default: 300 for publication)'
    )
    parser.add_argument(
        '--versions',
        type=str,
        nargs='+',
        choices=['v1', 'v2', 'v3', 'all'],
        default=['all'],
        help='Which versions to generate (v1=single-row, v2=two-class, v3=three-class, all=all versions)'
    )

    args = parser.parse_args()

    print("="*80)
    print("COMPACT AUGMENTATION VISUALIZATIONS FOR JOURNAL")
    print("="*80)
    print(f"Input images: {len(args.images)} sample(s)")
    for img_path in args.images:
        print(f"  - {img_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.size}x{args.size}")
    print(f"DPI: {args.dpi}")
    print(f"Versions: {args.versions}")
    print()

    # Validate input images
    for img_path in args.images:
        if not Path(img_path).exists():
            print(f"[ERROR] Image not found: {img_path}")
            return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which versions to generate
    versions_to_generate = args.versions
    if 'all' in versions_to_generate:
        versions_to_generate = ['v1', 'v2', 'v3']

    # Generate requested versions
    if 'v1' in versions_to_generate:
        # V1: Single-row (use first image only)
        output_path_v1 = output_dir / f"{args.prefix}_v1_single_row.png"
        create_compact_figure_v1(
            args.images[0],
            output_path_v1,
            output_size=args.size,
            dpi=args.dpi
        )
        print()

    if 'v2' in versions_to_generate and len(args.images) >= 2:
        # V2: Two-class
        output_path_v2 = output_dir / f"{args.prefix}_v2_two_class.png"
        create_compact_figure_v2(
            args.images[:2],
            output_path_v2,
            output_size=args.size,
            dpi=args.dpi
        )
        print()

    if 'v3' in versions_to_generate and len(args.images) >= 3:
        # V3: Three-class
        output_path_v3 = output_dir / f"{args.prefix}_v3_three_class.png"
        create_compact_figure_v3(
            args.images[:3],
            output_path_v3,
            output_size=args.size,
            dpi=args.dpi
        )
        print()

    print()
    print("="*80)
    print("COMPACT VISUALIZATIONS COMPLETED!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for version_file in output_dir.glob(f"{args.prefix}_*.png"):
        size_mb = version_file.stat().st_size / 1024 / 1024
        print(f"  - {version_file.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
