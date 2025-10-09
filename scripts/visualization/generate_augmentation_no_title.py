#!/usr/bin/env python3
"""
Generate augmentation visualizations WITHOUT TITLES for journal publication
Supports 1-4 class versions for comprehensive dataset visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageEnhance
import argparse
from pathlib import Path


def apply_rotation(image, angle):
    """Apply rotation to image"""
    return image.rotate(angle, expand=False, fillcolor=(255, 255, 255))


def apply_brightness(image, factor):
    """Apply brightness adjustment"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_contrast(image, factor):
    """Apply contrast adjustment"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def apply_saturation(image, factor):
    """Apply saturation adjustment"""
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def apply_sharpness(image, factor):
    """Apply sharpness adjustment"""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


def flip_horizontal(image):
    """Flip image horizontally"""
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def get_compact_augmentations(image_path, output_size=512):
    """
    Generate 7 key augmentations for compact journal visualization
    NO resizing if already at target size
    """
    img = Image.open(image_path).convert('RGB')

    # Only resize if not already target size
    if img.size != (output_size, output_size):
        img = img.resize((output_size, output_size), Image.BICUBIC)

    # Select 7 diverse augmentations for medical imaging
    augmentations = {
        'Original': img.copy(),
        '90° rotation': apply_rotation(img, -90),
        'Brightness 0.7': apply_brightness(img, 0.7),
        'Contrast 1.4': apply_contrast(img, 1.4),
        'Saturation 1.4': apply_saturation(img, 1.4),
        'Sharpness 2.0': apply_sharpness(img, 2.0),
        'Flip horizontal': flip_horizontal(img),
    }

    return augmentations


def create_figure_multi_class(image_paths, output_path, output_size=512, dpi=300):
    """
    Create multi-class augmentation figure (1-4 classes × 7 augmentations)
    NO TITLES for clean journal presentation
    """
    if not isinstance(image_paths, list):
        image_paths = [image_paths]

    n_samples = len(image_paths)

    # Get augmentations for first image to determine columns
    first_aug = get_compact_augmentations(image_paths[0], output_size)
    aug_names = list(first_aug.keys())
    n_cols = len(aug_names)
    n_rows = n_samples

    # Create figure - NO TITLE
    fig = plt.figure(figsize=(n_cols * 1.8, n_rows * 2))

    # Create grid with minimal spacing
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.25, wspace=0.15)

    # Process each sample
    for sample_idx, img_path in enumerate(image_paths):
        augmentations = get_compact_augmentations(img_path, output_size)
        class_name = Path(img_path).parent.name.upper()

        for col_idx, aug_name in enumerate(aug_names):
            ax = fig.add_subplot(gs[sample_idx, col_idx])

            # Display image
            img_array = np.array(augmentations[aug_name])
            ax.imshow(img_array)

            # Add column title ONLY on first row
            if sample_idx == 0:
                ax.set_title(aug_name, fontsize=8, fontweight='bold', pad=3)

            # Add class label on first column (left side)
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

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] {output_path.name}")
    print(f"          Layout: {n_rows} classes × {n_cols} augmentations ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate augmentation visualizations WITHOUT titles for journal'
    )
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to parasite crop images (1-4 images for multi-class)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path'
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

    args = parser.parse_args()

    print("="*80)
    print("AUGMENTATION VISUALIZATION (NO TITLE) FOR JOURNAL")
    print("="*80)
    print(f"Input images: {len(args.images)} class(es)")
    for img_path in args.images:
        print(f"  - {img_path}")
    print(f"Output file: {args.output}")
    print(f"Image size: {args.size}x{args.size}")
    print(f"DPI: {args.dpi}")
    print()

    # Validate input images
    for img_path in args.images:
        if not Path(img_path).exists():
            print(f"[ERROR] Image not found: {img_path}")
            return

    # Limit to 4 classes max
    if len(args.images) > 4:
        print(f"[WARNING] Maximum 4 classes supported, using first 4 images")
        args.images = args.images[:4]

    # Generate figure
    create_figure_multi_class(
        args.images,
        args.output,
        output_size=args.size,
        dpi=args.dpi
    )

    print()
    print("="*80)
    print("VISUALIZATION COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
