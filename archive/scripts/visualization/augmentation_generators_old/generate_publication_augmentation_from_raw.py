#!/usr/bin/env python3
"""
Generate Publication-Quality Augmentation Visualization from Raw Images

Strategy:
1. Load from ORIGINAL high-resolution images
2. Crop with margin (25% padding) at NATIVE resolution
3. Minimal resize (only if needed for consistency)
4. No aggressive processing (preserve quality)
5. Generate high-quality augmentation visualization
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
import json
import warnings
warnings.filterwarnings('ignore')


def load_iml_lifecycle_annotations(raw_data_dir):
    """Load IML lifecycle annotations from JSON"""
    annotation_file = Path(raw_data_dir) / "annotations.json"

    if not annotation_file.exists():
        print(f"[ERROR] Annotation file not found: {annotation_file}")
        return {}

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Parse annotations
    image_to_parasites = {}

    for image_entry in annotations:
        if 'image_name' not in image_entry or 'objects' not in image_entry:
            continue

        filename = image_entry['image_name']
        parasites = []

        for obj in image_entry['objects']:
            obj_type = obj.get('type', '').lower()

            # Skip non-parasites
            if 'red blood' in obj_type or obj_type == 'difficult':
                continue

            # Get bbox
            bbox_info = obj.get('bbox', {})
            try:
                x = int(bbox_info['x'])
                y = int(bbox_info['y'])
                w = int(bbox_info['w'])
                h = int(bbox_info['h'])

                parasites.append({
                    'class': obj_type,
                    'bbox': [x, y, w, h]  # x, y, width, height
                })
            except (KeyError, ValueError):
                continue

        if parasites:
            image_to_parasites[filename] = parasites

    return image_to_parasites


def crop_with_margin(image, bbox, margin_ratio=0.25):
    """
    Crop image with margin at NATIVE resolution

    Args:
        image: Original high-res image
        bbox: [x, y, width, height]
        margin_ratio: Extra padding (0.25 = 25%)

    Returns:
        Cropped image at native resolution
    """
    x, y, w, h = bbox

    # Add margin
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image.shape[1], x + w + margin_x)
    y2 = min(image.shape[0], y + h + margin_y)

    # Crop at native resolution
    crop = image[y1:y2, x1:x2]

    return crop


def smart_resize(crop, max_upscale_ratio=1.8, target_for_publication=512):
    """
    Smart adaptive resize: minimize blur by limiting upscale ratio

    Strategy (UPDATED):
    - If native size ≥ 400px: Keep at 512px (1.3x upscale max) ⭐ BEST
    - If native size ≥ 300px: Keep at 512px (~1.7x upscale) ✅ GOOD
    - If native size ≥ 250px: Target 448px (~1.8x upscale) ✅ GOOD
    - If native size ≥ 200px: Target 384px (~1.9x upscale) ⚠️ OK
    - If native size < 200px: Target 300px (limit to 2x) ❌ NOT IDEAL

    Args:
        crop: Cropped image
        max_upscale_ratio: Maximum upscale ratio (1.8x default)
        target_for_publication: Target for large crops (512)

    Returns:
        Resized crop with optimal quality
    """
    h, w = crop.shape[:2]
    native_size = max(h, w)

    # Determine optimal target size (prioritize quality)
    if native_size >= 400:
        # EXCELLENT - minimal upscale
        target_size = 512
        quality_rating = "⭐ EXCELLENT"
    elif native_size >= 300:
        # VERY GOOD - acceptable upscale
        target_size = 512
        quality_rating = "✅ VERY GOOD"
    elif native_size >= 250:
        # GOOD - moderate upscale
        target_size = 448
        quality_rating = "✅ GOOD"
    elif native_size >= 200:
        # OK - some blur expected
        target_size = 384
        quality_rating = "⚠️ OK"
    elif native_size >= 150:
        # NOT IDEAL - visible blur
        target_size = 300
        quality_rating = "❌ NOT IDEAL"
    else:
        # POOR - significant blur
        target_size = min(256, int(native_size * max_upscale_ratio))
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
    """
    Generate augmentations from numpy array

    Args:
        crop_array: Numpy array (H, W, 3) RGB format

    Returns:
        Dictionary of augmentation name -> augmented image (PIL)
    """
    # Convert to PIL
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


def create_publication_figure(crops_data, output_path, target_size=512, dpi=300):
    """
    Create publication-quality augmentation figure

    Args:
        crops_data: List of dicts with 'crop' (numpy array), 'class' (string), 'final_size' (int)
        output_path: Output file path
        target_size: Target size for crops (512)
        dpi: Output DPI (300)
    """
    n_samples = len(crops_data)
    n_cols = 7
    n_rows = n_samples * 2

    # Get max size for consistent figure sizing
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
    print(f"          Resolution: {target_size}×{target_size} per image")
    print(f"          DPI: {dpi}")
    print(f"          Samples: {n_samples}")


def load_mp_idb_yolo_annotations(processed_dir):
    """Load MP-IDB annotations from YOLO format"""
    label_dir = Path(processed_dir) / "train" / "labels"
    image_dir = Path(processed_dir) / "train" / "images"

    if not label_dir.exists():
        print(f"[ERROR] Label directory not found: {label_dir}")
        return {}

    image_to_parasites = {}

    # Iterate through all label files
    for label_file in label_dir.glob("*.txt"):
        # Try both .jpg and .png extensions
        image_path = None
        for ext in ['.jpg', '.png']:
            candidate_path = image_dir / (label_file.stem + ext)
            if candidate_path.exists():
                image_path = candidate_path
                break

        if image_path is None:
            continue

        # Load image to get dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        img_h, img_w = image.shape[:2]

        # Parse YOLO annotations
        parasites = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                # Convert from YOLO normalized to pixel coordinates
                x = int((x_center - width / 2) * img_w)
                y = int((y_center - height / 2) * img_h)
                w = int(width * img_w)
                h = int(height * img_h)

                parasites.append({
                    'class_id': class_id,
                    'bbox': [x, y, w, h]
                })

        if parasites:
            image_to_parasites[str(image_path)] = parasites

    return image_to_parasites


def load_md2019_yolo_annotations(processed_dir):
    """Load MD_2019 annotations from YOLO format"""
    return load_mp_idb_yolo_annotations(processed_dir)


def process_iml_lifecycle(raw_data_dir, output_path, target_size=512, min_native_size=100):
    """Process IML Lifecycle dataset"""
    print("\n[PROCESSING] IML Lifecycle Dataset")

    # Load annotations
    annotations = load_iml_lifecycle_annotations(raw_data_dir)
    print(f"[INFO] Loaded annotations for {len(annotations)} images")

    # Find images with 4 different classes (prioritize LARGER parasites)
    class_to_images = {'ring': [], 'trophozoite': [], 'schizont': [], 'gametocyte': []}

    for image_name, parasites in annotations.items():
        for parasite in parasites:
            class_name = parasite['class']
            bbox = parasite['bbox']

            # Calculate native size (with margin)
            w, h = bbox[2], bbox[3]
            native_size_with_margin = max(w, h) * 1.5  # 25% margin on each side

            if class_name in class_to_images:
                image_path = Path(raw_data_dir) / "IML_Malaria" / image_name
                if image_path.exists():
                    class_to_images[class_name].append({
                        'image_path': image_path,
                        'bbox': parasite['bbox'],
                        'class': class_name,
                        'native_size': native_size_with_margin
                    })

    # Sort each class by native size (largest first) and select best samples
    for class_name in class_to_images:
        class_to_images[class_name].sort(key=lambda x: x['native_size'], reverse=True)
        print(f"[SELECT] {class_name}: Found {len(class_to_images[class_name])} samples")
        if class_to_images[class_name]:
            # Show top 10 to find really large ones
            top_10 = class_to_images[class_name][:10]
            for i, sample in enumerate(top_10):
                print(f"         #{i+1}: Native size ~{sample['native_size']:.0f}px")

            # Filter for large samples (>250px preferred)
            large_samples = [s for s in class_to_images[class_name] if s['native_size'] >= 250]
            if large_samples:
                print(f"         [FOUND] {len(large_samples)} samples with native size ≥250px")
            else:
                print(f"         [WARNING] No samples ≥250px, using largest available")

    # Select LARGEST sample per class (best quality)
    crops_data = []
    for class_name in ['ring', 'trophozoite', 'schizont', 'gametocyte']:
        if class_to_images[class_name]:
            sample = class_to_images[class_name][0]  # Already sorted by size

            # Load original image (FULL RESOLUTION)
            image = cv2.imread(str(sample['image_path']))
            print(f"\n[CROP] {class_name.upper()}")
            print(f"       Original image: {image.shape[:2]}")
            print(f"       Bbox: {sample['bbox']}")

            # Crop with margin at NATIVE resolution
            crop = crop_with_margin(image, sample['bbox'], margin_ratio=0.25)
            print(f"       Native crop: {crop.shape[:2]} (with 25% margin)")

            # Smart adaptive resize
            crop_resized = smart_resize(crop)

            crops_data.append({
                'crop': crop_resized,
                'class': class_name,
                'final_size': crop_resized.shape[0]
            })

    # Create figure
    if len(crops_data) == 4:
        create_publication_figure(crops_data, output_path, target_size=target_size)
        return True
    else:
        print(f"[WARNING] Could not find all 4 classes")
        return False


def process_mp_idb_species(processed_dir, output_path, target_size=512):
    """Process MP-IDB Species dataset"""
    print("\n[PROCESSING] MP-IDB Species Dataset")

    # Load annotations
    annotations = load_mp_idb_yolo_annotations(processed_dir)
    print(f"[INFO] Loaded annotations for {len(annotations)} images")

    # Species classes (based on processed data structure)
    species_map = {
        0: 'P_falciparum',
        1: 'P_vivax',
        2: 'P_ovale',
        3: 'P_malariae'
    }

    class_to_images = {name: [] for name in species_map.values()}

    for image_path, parasites in annotations.items():
        for parasite in parasites:
            class_id = parasite['class_id']
            bbox = parasite['bbox']

            # Calculate native size (with margin)
            w, h = bbox[2], bbox[3]
            native_size_with_margin = max(w, h) * 1.5

            if class_id in species_map:
                class_name = species_map[class_id]
                class_to_images[class_name].append({
                    'image_path': image_path,
                    'bbox': bbox,
                    'class': class_name,
                    'native_size': native_size_with_margin
                })

    # Sort and select largest
    for class_name in class_to_images:
        class_to_images[class_name].sort(key=lambda x: x['native_size'], reverse=True)
        print(f"[SELECT] {class_name}: Found {len(class_to_images[class_name])} samples")
        if class_to_images[class_name]:
            top_10 = class_to_images[class_name][:10]
            for i, sample in enumerate(top_10):
                print(f"         #{i+1}: Native size ~{sample['native_size']:.0f}px")

            large_samples = [s for s in class_to_images[class_name] if s['native_size'] >= 250]
            if large_samples:
                print(f"         [FOUND] {len(large_samples)} samples with native size ≥250px")
            else:
                print(f"         [WARNING] No samples ≥250px, using largest available")

    # Generate crops
    crops_data = []
    for class_name in ['P_falciparum', 'P_vivax', 'P_ovale', 'P_malariae']:
        if class_to_images[class_name]:
            sample = class_to_images[class_name][0]

            image = cv2.imread(str(sample['image_path']))
            print(f"\n[CROP] {class_name.upper()}")
            print(f"       Original image: {image.shape[:2]}")
            print(f"       Bbox: {sample['bbox']}")

            crop = crop_with_margin(image, sample['bbox'], margin_ratio=0.25)
            print(f"       Native crop: {crop.shape[:2]} (with 25% margin)")

            crop_resized = smart_resize(crop)

            crops_data.append({
                'crop': crop_resized,
                'class': class_name,
                'final_size': crop_resized.shape[0]
            })

    if len(crops_data) == 4:
        create_publication_figure(crops_data, output_path, target_size=target_size)
        return True
    else:
        print(f"[WARNING] Could not find all 4 classes")
        return False


def process_mp_idb_stages(processed_dir, output_path, target_size=512):
    """Process MP-IDB Stages dataset"""
    print("\n[PROCESSING] MP-IDB Stages Dataset")

    annotations = load_mp_idb_yolo_annotations(processed_dir)
    print(f"[INFO] Loaded annotations for {len(annotations)} images")

    # Stage classes
    stage_map = {
        0: 'ring',
        1: 'trophozoite',
        2: 'schizont',
        3: 'gametocyte'
    }

    class_to_images = {name: [] for name in stage_map.values()}

    for image_path, parasites in annotations.items():
        for parasite in parasites:
            class_id = parasite['class_id']
            bbox = parasite['bbox']

            w, h = bbox[2], bbox[3]
            native_size_with_margin = max(w, h) * 1.5

            if class_id in stage_map:
                class_name = stage_map[class_id]
                class_to_images[class_name].append({
                    'image_path': image_path,
                    'bbox': bbox,
                    'class': class_name,
                    'native_size': native_size_with_margin
                })

    # Sort and select
    for class_name in class_to_images:
        class_to_images[class_name].sort(key=lambda x: x['native_size'], reverse=True)
        print(f"[SELECT] {class_name}: Found {len(class_to_images[class_name])} samples")
        if class_to_images[class_name]:
            top_10 = class_to_images[class_name][:10]
            for i, sample in enumerate(top_10):
                print(f"         #{i+1}: Native size ~{sample['native_size']:.0f}px")

            large_samples = [s for s in class_to_images[class_name] if s['native_size'] >= 250]
            if large_samples:
                print(f"         [FOUND] {len(large_samples)} samples with native size ≥250px")
            else:
                print(f"         [WARNING] No samples ≥250px, using largest available")

    # Generate crops
    crops_data = []
    for class_name in ['ring', 'trophozoite', 'schizont', 'gametocyte']:
        if class_to_images[class_name]:
            sample = class_to_images[class_name][0]

            image = cv2.imread(str(sample['image_path']))
            print(f"\n[CROP] {class_name.upper()}")
            print(f"       Original image: {image.shape[:2]}")
            print(f"       Bbox: {sample['bbox']}")

            crop = crop_with_margin(image, sample['bbox'], margin_ratio=0.25)
            print(f"       Native crop: {crop.shape[:2]} (with 25% margin)")

            crop_resized = smart_resize(crop)

            crops_data.append({
                'crop': crop_resized,
                'class': class_name,
                'final_size': crop_resized.shape[0]
            })

    if len(crops_data) == 4:
        create_publication_figure(crops_data, output_path, target_size=target_size)
        return True
    else:
        print(f"[WARNING] Could not find all 4 classes")
        return False


def process_md2019_stages(processed_dir, output_path, target_size=512):
    """Process MD_2019 Stages dataset"""
    print("\n[PROCESSING] MD_2019 Stages Dataset")

    annotations = load_md2019_yolo_annotations(processed_dir)
    print(f"[INFO] Loaded annotations for {len(annotations)} images")

    # MD_2019 has 3 classes (ring, trophozoite, schizont)
    stage_map = {
        0: 'ring',
        1: 'trophozoite',
        2: 'schizont'
    }

    class_to_images = {name: [] for name in stage_map.values()}

    for image_path, parasites in annotations.items():
        for parasite in parasites:
            class_id = parasite['class_id']
            bbox = parasite['bbox']

            w, h = bbox[2], bbox[3]
            native_size_with_margin = max(w, h) * 1.5

            if class_id in stage_map:
                class_name = stage_map[class_id]
                class_to_images[class_name].append({
                    'image_path': image_path,
                    'bbox': bbox,
                    'class': class_name,
                    'native_size': native_size_with_margin
                })

    # Sort and select
    for class_name in class_to_images:
        class_to_images[class_name].sort(key=lambda x: x['native_size'], reverse=True)
        print(f"[SELECT] {class_name}: Found {len(class_to_images[class_name])} samples")
        if class_to_images[class_name]:
            top_10 = class_to_images[class_name][:10]
            for i, sample in enumerate(top_10):
                print(f"         #{i+1}: Native size ~{sample['native_size']:.0f}px")

            large_samples = [s for s in class_to_images[class_name] if s['native_size'] >= 250]
            if large_samples:
                print(f"         [FOUND] {len(large_samples)} samples with native size ≥250px")
            else:
                print(f"         [WARNING] No samples ≥250px, using largest available")

    # Generate crops (only 3 classes for MD_2019)
    crops_data = []
    for class_name in ['ring', 'trophozoite', 'schizont']:
        if class_to_images[class_name]:
            sample = class_to_images[class_name][0]

            image = cv2.imread(str(sample['image_path']))
            print(f"\n[CROP] {class_name.upper()}")
            print(f"       Original image: {image.shape[:2]}")
            print(f"       Bbox: {sample['bbox']}")

            crop = crop_with_margin(image, sample['bbox'], margin_ratio=0.25)
            print(f"       Native crop: {crop.shape[:2]} (with 25% margin)")

            crop_resized = smart_resize(crop)

            crops_data.append({
                'crop': crop_resized,
                'class': class_name,
                'final_size': crop_resized.shape[0]
            })

    if len(crops_data) == 3:
        create_publication_figure(crops_data, output_path, target_size=target_size)
        return True
    else:
        print(f"[WARNING] Could not find all 3 classes")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality augmentation from raw images'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['iml_lifecycle', 'mp_idb_species', 'mp_idb_stages', 'md_2019_stages', 'all'],
        help='Dataset to process'
    )
    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default='data/raw',
        help='Raw data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='luaran/auto_generated/figures/augmentation',
        help='Output directory'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=512,
        help='Target size for crops (default: 512)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output DPI (default: 300)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PUBLICATION-QUALITY AUGMENTATION FROM RAW IMAGES")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Raw data dir: {args.raw_data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Target size: {args.target_size}×{args.target_size}")
    print(f"DPI: {args.dpi}")

    # Process datasets
    datasets_to_process = []
    if args.dataset == 'all':
        datasets_to_process = ['iml_lifecycle', 'mp_idb_species', 'mp_idb_stages', 'md_2019_stages']
    else:
        datasets_to_process = [args.dataset]

    results = []

    for dataset in datasets_to_process:
        if dataset == 'iml_lifecycle':
            raw_dir = Path(args.raw_data_dir) / "malaria_lifecycle"
            output_file = Path(args.output_dir) / f"augmentation_{dataset}_from_raw.png"

            success = process_iml_lifecycle(raw_dir, output_file, args.target_size)

            if success:
                file_size = output_file.stat().st_size / 1024 / 1024
                results.append({
                    'dataset': dataset,
                    'file': output_file,
                    'size_mb': file_size
                })
                print(f"\n[COMPLETED] {dataset}")
                print(f"            File: {output_file}")
                print(f"            Size: {file_size:.2f} MB")

        elif dataset == 'mp_idb_species':
            processed_dir = Path(args.raw_data_dir).parent / "processed" / "species"
            output_file = Path(args.output_dir) / f"augmentation_{dataset}_from_raw.png"

            success = process_mp_idb_species(processed_dir, output_file, args.target_size)

            if success:
                file_size = output_file.stat().st_size / 1024 / 1024
                results.append({
                    'dataset': dataset,
                    'file': output_file,
                    'size_mb': file_size
                })
                print(f"\n[COMPLETED] {dataset}")
                print(f"            File: {output_file}")
                print(f"            Size: {file_size:.2f} MB")

        elif dataset == 'mp_idb_stages':
            processed_dir = Path(args.raw_data_dir).parent / "processed" / "stages"
            output_file = Path(args.output_dir) / f"augmentation_{dataset}_from_raw.png"

            success = process_mp_idb_stages(processed_dir, output_file, args.target_size)

            if success:
                file_size = output_file.stat().st_size / 1024 / 1024
                results.append({
                    'dataset': dataset,
                    'file': output_file,
                    'size_mb': file_size
                })
                print(f"\n[COMPLETED] {dataset}")
                print(f"            File: {output_file}")
                print(f"            Size: {file_size:.2f} MB")

        elif dataset == 'md_2019_stages':
            processed_dir = Path(args.raw_data_dir).parent / "processed" / "md_2019_stages"
            output_file = Path(args.output_dir) / f"augmentation_{dataset}_from_raw.png"

            success = process_md2019_stages(processed_dir, output_file, args.target_size)

            if success:
                file_size = output_file.stat().st_size / 1024 / 1024
                results.append({
                    'dataset': dataset,
                    'file': output_file,
                    'size_mb': file_size
                })
                print(f"\n[COMPLETED] {dataset}")
                print(f"            File: {output_file}")
                print(f"            Size: {file_size:.2f} MB")

    print("\n" + "=" * 80)
    print("AUGMENTATION GENERATION COMPLETED!")
    print("=" * 80)
    print(f"\nGenerated {len(results)} figure(s):")
    for result in results:
        print(f"  - {result['dataset']}: {result['size_mb']:.2f} MB")
    print()


if __name__ == "__main__":
    main()
