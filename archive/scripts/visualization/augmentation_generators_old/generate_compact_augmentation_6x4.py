#!/usr/bin/env python3
"""
Generate Compact Augmentation Figure: 4 classes × 6 augmentations
Layout: 4 rows (one per class) × 6 columns (6 augmentations)
Total: 24 images per figure
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


def crop_with_margin(image, bbox, margin_ratio=0.25):
    """Crop image with margin at NATIVE resolution"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image.shape[1], x2 + margin_x)
    y2 = min(image.shape[0], y2 + margin_y)

    crop = image[y1:y2, x1:x2]
    return crop


def smart_resize(crop, target_size=384):
    """Smart adaptive resize"""
    h, w = crop.shape[:2]
    native_size = max(h, w)

    if native_size >= 300:
        final_size = 512
    elif native_size >= 250:
        final_size = 448
    elif native_size >= 200:
        final_size = 384
    else:
        final_size = 320

    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    crop_resized = crop_pil.resize((final_size, final_size), Image.LANCZOS)

    return np.array(crop_resized)


def apply_rotation(img, degrees):
    return img.rotate(degrees, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

def apply_brightness(img, factor):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def apply_contrast(img, factor):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def flip_horizontal(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def flip_vertical(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def get_compact_augmentations(crop_array):
    """
    Generate 6 augmentations for compact layout

    Layout (4 rows × 6 cols):
    Row 1: [Original, Rotate 90°, Rotate 180°, Rotate 270°, Flip H, Flip V]
    Row 2: [Original, Rotate 90°, Rotate 180°, Rotate 270°, Flip H, Flip V]
    Row 3: [Original, Rotate 90°, Rotate 180°, Rotate 270°, Flip H, Flip V]
    Row 4: [Original, Rotate 90°, Rotate 180°, Rotate 270°, Flip H, Flip V]
    """
    crop_pil = Image.fromarray(crop_array)

    augmentations = [
        ('Original', crop_pil.copy()),
        ('Rotate 90°', apply_rotation(crop_pil, -90)),
        ('Rotate 180°', apply_rotation(crop_pil, 180)),
        ('Rotate 270°', apply_rotation(crop_pil, -270)),
        ('Flip Horizontal', flip_horizontal(crop_pil)),
        ('Flip Vertical', flip_vertical(crop_pil)),
    ]

    return augmentations


def create_compact_figure(crops_data, output_path, title, dpi=300):
    """
    Create compact augmentation figure
    Layout: 4 classes × 6 augmentations = 24 images
    """
    n_classes = len(crops_data)
    n_cols = 6  # 6 augmentations
    n_rows = n_classes  # 1 row per class

    print(f"\n[FIGURE] Creating {n_rows}×{n_cols} compact grid ({n_rows * n_cols} images)")

    # Create figure (wider for 6 columns)
    fig = plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5))

    # Add title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # Create grid
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.25, wspace=0.15)

    # Process each class
    for row_idx, crop_data in enumerate(crops_data):
        crop_array = crop_data['crop']
        class_name = crop_data['class']

        # Get 6 augmentations
        augmentations = get_compact_augmentations(crop_array)

        # Add augmentations to grid
        for col_idx, (aug_name, aug_img) in enumerate(augmentations):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            # Display image
            img_array = np.array(aug_img)
            ax.imshow(img_array)

            # Add augmentation name as title (only on first row)
            if row_idx == 0:
                ax.set_title(aug_name, fontsize=10, fontweight='bold', pad=5)

            # Add class label on first column
            if col_idx == 0:
                ax.text(
                    -0.15, 0.5,
                    class_name.upper(),
                    transform=ax.transAxes,
                    fontsize=11,
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
    print(f"          Layout: {n_rows} rows × {n_cols} columns")
    print(f"          DPI: {dpi}")


def process_mp_idb_species(metadata_path, image_base_dir, output_path):
    """Process MP-IDB Species"""
    print("\n[PROCESSING] MP-IDB Species (Compact 4×6)")

    # Load metadata
    df = pd.read_csv(metadata_path)
    df['width'] = df['bbox_x2'] - df['bbox_x1']
    df['height'] = df['bbox_y2'] - df['bbox_y1']
    df['size'] = df[['width', 'height']].max(axis=1)
    df['size_with_margin'] = df['size'] * 1.5

    species_list = ['P_falciparum', 'P_vivax', 'P_ovale', 'P_malariae']
    crops_data = []

    for species in species_list:
        species_df = df[df['class_name'] == species].sort_values('size_with_margin', ascending=False)

        if len(species_df) == 0:
            continue

        sample = species_df.iloc[0]

        # Load image
        img_file = sample['original_image']
        if img_file.startswith('images/'):
            img_file = img_file[7:]

        image_path = None
        for split in ['train', 'val', 'test']:
            candidate = Path(image_base_dir) / split / 'images' / img_file
            if candidate.exists():
                image_path = candidate
                break

        if not image_path:
            continue

        image = cv2.imread(str(image_path))

        print(f"[CROP] {species}: {sample['size_with_margin']:.0f}px")

        # Crop with margin
        bbox = [sample['bbox_x1'], sample['bbox_y1'], sample['bbox_x2'], sample['bbox_y2']]
        crop = crop_with_margin(image, bbox, margin_ratio=0.25)

        # Resize
        crop_resized = smart_resize(crop)

        crops_data.append({
            'crop': crop_resized,
            'class': species.replace('_', ' ').title()
        })

    if len(crops_data) == 4:
        title = 'Data Augmentation Examples - MP-IDB Species Dataset'
        create_compact_figure(crops_data, output_path, title)
        return True
    return False


def process_iml_lifecycle(raw_data_dir, output_path):
    """Process IML Lifecycle"""
    import json

    print("\n[PROCESSING] IML Lifecycle (Compact 4×6)")

    # Load annotations
    annotation_file = Path(raw_data_dir) / "annotations.json"
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Parse annotations
    class_to_images = {'ring': [], 'trophozoite': [], 'schizont': [], 'gametocyte': []}

    for image_entry in annotations:
        if 'image_name' not in image_entry or 'objects' not in image_entry:
            continue

        filename = image_entry['image_name']

        for obj in image_entry['objects']:
            obj_type = obj.get('type', '').lower()

            if 'red blood' in obj_type or obj_type == 'difficult':
                continue

            bbox_info = obj.get('bbox', {})
            try:
                x = int(bbox_info['x'])
                y = int(bbox_info['y'])
                w = int(bbox_info['w'])
                h = int(bbox_info['h'])

                if obj_type in class_to_images:
                    image_path = Path(raw_data_dir) / "IML_Malaria" / filename
                    if image_path.exists():
                        native_size = max(w, h) * 1.5
                        class_to_images[obj_type].append({
                            'image_path': image_path,
                            'bbox': [x, y, x+w, y+h],
                            'native_size': native_size
                        })
            except (KeyError, ValueError):
                continue

    crops_data = []

    for class_name in ['ring', 'trophozoite', 'schizont', 'gametocyte']:
        if not class_to_images[class_name]:
            continue

        # Sort by size and select largest
        class_to_images[class_name].sort(key=lambda x: x['native_size'], reverse=True)
        sample = class_to_images[class_name][0]

        image = cv2.imread(str(sample['image_path']))

        print(f"[CROP] {class_name}: {sample['native_size']:.0f}px")

        # Crop and resize
        crop = crop_with_margin(image, sample['bbox'], margin_ratio=0.25)
        crop_resized = smart_resize(crop)

        crops_data.append({
            'crop': crop_resized,
            'class': class_name.title()
        })

    if len(crops_data) == 4:
        title = 'Data Augmentation Examples - IML Lifecycle Dataset'
        create_compact_figure(crops_data, output_path, title)
        return True
    return False


def process_md2019_stages(processed_dir, output_path):
    """Process MD_2019 Stages"""
    print("\n[PROCESSING] MD_2019 Stages (Compact 3×6)")

    label_dir = Path(processed_dir) / "train" / "labels"
    image_dir = Path(processed_dir) / "train" / "images"

    stage_map = {0: 'ring', 1: 'trophozoite', 2: 'schizont'}
    class_to_parasites = {name: [] for name in stage_map.values()}

    # Parse labels
    for label_file in label_dir.glob("*.txt"):
        image_path = None
        for ext in ['.png', '.jpg']:
            candidate = image_dir / (label_file.stem + ext)
            if candidate.exists():
                image_path = candidate
                break

        if not image_path:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        img_h, img_w = image.shape[:2]

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                w_px = int(width * img_w)
                h_px = int(height * img_h)
                native_size = max(w_px, h_px) * 1.5

                if class_id in stage_map:
                    class_name = stage_map[class_id]
                    class_to_parasites[class_name].append({
                        'image_path': str(image_path),
                        'bbox_yolo': [x_center, y_center, width, height],
                        'image_dims': [img_h, img_w],
                        'native_size': native_size
                    })

    crops_data = []

    for class_name in ['ring', 'trophozoite', 'schizont']:
        parasites = class_to_parasites[class_name]
        if not parasites:
            continue

        parasites.sort(key=lambda x: x['native_size'], reverse=True)
        sample = parasites[0]

        image = cv2.imread(sample['image_path'])

        print(f"[CROP] {class_name}: {sample['native_size']:.0f}px")

        # Crop from YOLO bbox
        img_h, img_w = sample['image_dims']
        x_center, y_center, width, height = sample['bbox_yolo']

        w_px = int(width * img_w)
        h_px = int(height * img_h)
        x_px = int(x_center * img_w)
        y_px = int(y_center * img_h)

        x1 = max(0, int(x_px - w_px * 0.625))  # 25% margin
        y1 = max(0, int(y_px - h_px * 0.625))
        x2 = min(img_w, int(x_px + w_px * 0.625))
        y2 = min(img_h, int(y_px + h_px * 0.625))

        crop = image[y1:y2, x1:x2]
        crop_resized = smart_resize(crop)

        crops_data.append({
            'crop': crop_resized,
            'class': class_name.title()
        })

    if len(crops_data) == 3:
        title = 'Data Augmentation Examples - MD_2019 Stages Dataset'
        create_compact_figure(crops_data, output_path, title)
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['mp_idb_species', 'iml_lifecycle', 'md_2019_stages', 'all'])
    parser.add_argument('--output-dir', type=str,
                       default='luaran/auto_generated/figures/augmentation')
    parser.add_argument('--dpi', type=int, default=300)

    args = parser.parse_args()

    print("=" * 80)
    print("COMPACT AUGMENTATION FIGURES (4×6 OR 3×6 LAYOUT)")
    print("=" * 80)

    datasets_to_process = []
    if args.dataset == 'all':
        datasets_to_process = ['mp_idb_species', 'iml_lifecycle', 'md_2019_stages']
    else:
        datasets_to_process = [args.dataset]

    results = []

    for dataset in datasets_to_process:
        if dataset == 'mp_idb_species':
            metadata = 'results/optA_20251013_220815/experiments/experiment_mp_idb_species/crops_gt_crops/ground_truth_crop_metadata.csv'
            image_dir = 'data/processed/species'
            output = Path(args.output_dir) / 'augmentation_mp_idb_species_compact_4x6.png'

            success = process_mp_idb_species(metadata, image_dir, output)

            if success:
                size_mb = output.stat().st_size / 1024 / 1024
                results.append({'dataset': dataset, 'file': output, 'size_mb': size_mb})
                print(f"\n[COMPLETED] {dataset}: {size_mb:.2f} MB")

        elif dataset == 'iml_lifecycle':
            raw_dir = Path('data/raw/malaria_lifecycle')
            output = Path(args.output_dir) / 'augmentation_iml_lifecycle_compact_4x6.png'

            success = process_iml_lifecycle(raw_dir, output)

            if success:
                size_mb = output.stat().st_size / 1024 / 1024
                results.append({'dataset': dataset, 'file': output, 'size_mb': size_mb})
                print(f"\n[COMPLETED] {dataset}: {size_mb:.2f} MB")

        elif dataset == 'md_2019_stages':
            processed = 'data/processed/md_2019_stages'
            output = Path(args.output_dir) / 'augmentation_md_2019_stages_compact_3x6.png'

            success = process_md2019_stages(processed, output)

            if success:
                size_mb = output.stat().st_size / 1024 / 1024
                results.append({'dataset': dataset, 'file': output, 'size_mb': size_mb})
                print(f"\n[COMPLETED] {dataset}: {size_mb:.2f} MB")

    print("\n" + "=" * 80)
    print(f"COMPLETED! Generated {len(results)} compact figure(s)")
    for result in results:
        print(f"  - {result['dataset']}: {result['size_mb']:.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
