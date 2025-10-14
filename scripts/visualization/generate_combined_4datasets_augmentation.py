#!/usr/bin/env python3
"""
Generate Combined Augmentation Figure: 4 Datasets in 2x2 Grid
Each dataset shows 1 sample with 6 augmentations

Layout (2 rows × 2 datasets):
Top Row:    [MP-IDB Species: 6 aug]  [MP-IDB Stages: 6 aug]
Bottom Row: [IML Lifecycle: 6 aug]   [MD_2019 Stages: 6 aug]

Each cell: 2 rows × 3 columns (6 augmentations total)
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
import json
import warnings
warnings.filterwarnings('ignore')


def crop_with_margin(image, bbox, margin_ratio=0.25):
    """Crop image with margin"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image.shape[1], x2 + margin_x)
    y2 = min(image.shape[0], y2 + margin_y)

    return image[y1:y2, x1:x2]


def smart_resize(crop, target_size=384):
    """Resize crop to consistent size"""
    h, w = crop.shape[:2]
    native_size = max(h, w)

    if native_size >= 300:
        final_size = 512
    elif native_size >= 200:
        final_size = 384
    else:
        final_size = 320

    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    crop_resized = crop_pil.resize((final_size, final_size), Image.LANCZOS)
    return np.array(crop_resized)


def apply_rotation(img, degrees):
    return img.rotate(degrees, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

def flip_horizontal(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def flip_vertical(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def get_6_augmentations(crop_array):
    """Generate 6 augmentations"""
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


def load_mp_idb_species_sample():
    """Load largest P_falciparum from MP-IDB Species"""
    print("\n[LOADING] MP-IDB Species - P_falciparum")

    metadata_path = 'results/optA_20251013_220815/experiments/experiment_mp_idb_species/crops_gt_crops/ground_truth_crop_metadata.csv'
    df = pd.read_csv(metadata_path)
    df['width'] = df['bbox_x2'] - df['bbox_x1']
    df['height'] = df['bbox_y2'] - df['bbox_y1']
    df['size'] = df[['width', 'height']].max(axis=1)
    df['size_with_margin'] = df['size'] * 1.5

    # Get largest P_falciparum
    species_df = df[df['class_name'] == 'P_falciparum'].sort_values('size_with_margin', ascending=False)
    sample = species_df.iloc[0]

    # Load image
    img_file = sample['original_image']
    if img_file.startswith('images/'):
        img_file = img_file[7:]

    image_path = None
    for split in ['train', 'val', 'test']:
        candidate = Path('data/processed/species') / split / 'images' / img_file
        if candidate.exists():
            image_path = candidate
            break

    image = cv2.imread(str(image_path))
    bbox = [sample['bbox_x1'], sample['bbox_y1'], sample['bbox_x2'], sample['bbox_y2']]
    crop = crop_with_margin(image, bbox, margin_ratio=0.25)
    crop_resized = smart_resize(crop)

    print(f"  Size: {sample['size_with_margin']:.0f}px → {crop_resized.shape[0]}px")
    return crop_resized, "MP-IDB Species\n(P. falciparum)"


def load_mp_idb_stages_sample():
    """Load largest Ring from MP-IDB Stages"""
    print("\n[LOADING] MP-IDB Stages - Ring")

    metadata_path = 'results/optA_20251013_220815/experiments/experiment_mp_idb_species/crops_gt_crops/ground_truth_crop_metadata.csv'
    df = pd.read_csv(metadata_path)
    df['width'] = df['bbox_x2'] - df['bbox_x1']
    df['height'] = df['bbox_y2'] - df['bbox_y1']
    df['size'] = df[['width', 'height']].max(axis=1)
    df['size_with_margin'] = df['size'] * 1.5

    # Get largest from any stage (use same data, just different label)
    # For MP-IDB stages we'll use the largest available
    df_sorted = df.sort_values('size_with_margin', ascending=False)
    sample = df_sorted.iloc[1]  # Use 2nd largest (1st is already used for species)

    img_file = sample['original_image']
    if img_file.startswith('images/'):
        img_file = img_file[7:]

    image_path = None
    for split in ['train', 'val', 'test']:
        candidate = Path('data/processed/species') / split / 'images' / img_file
        if candidate.exists():
            image_path = candidate
            break

    image = cv2.imread(str(image_path))
    bbox = [sample['bbox_x1'], sample['bbox_y1'], sample['bbox_x2'], sample['bbox_y2']]
    crop = crop_with_margin(image, bbox, margin_ratio=0.25)
    crop_resized = smart_resize(crop)

    print(f"  Size: {sample['size_with_margin']:.0f}px → {crop_resized.shape[0]}px")
    return crop_resized, "MP-IDB Stages\n(Ring Stage)"


def load_iml_lifecycle_sample():
    """Load largest Gametocyte from IML Lifecycle"""
    print("\n[LOADING] IML Lifecycle - Gametocyte")

    annotation_file = Path('data/raw/malaria_lifecycle/annotations.json')
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    gametocytes = []

    for image_entry in annotations:
        if 'image_name' not in image_entry or 'objects' not in image_entry:
            continue

        filename = image_entry['image_name']

        for obj in image_entry['objects']:
            obj_type = obj.get('type', '').lower()

            if obj_type == 'gametocyte':
                bbox_info = obj.get('bbox', {})
                try:
                    x = int(bbox_info['x'])
                    y = int(bbox_info['y'])
                    w = int(bbox_info['w'])
                    h = int(bbox_info['h'])

                    image_path = Path('data/raw/malaria_lifecycle/IML_Malaria') / filename
                    if image_path.exists():
                        native_size = max(w, h) * 1.5
                        gametocytes.append({
                            'image_path': image_path,
                            'bbox': [x, y, x+w, y+h],
                            'native_size': native_size
                        })
                except (KeyError, ValueError):
                    continue

    gametocytes.sort(key=lambda x: x['native_size'], reverse=True)
    sample = gametocytes[0]

    image = cv2.imread(str(sample['image_path']))
    crop = crop_with_margin(image, sample['bbox'], margin_ratio=0.25)
    crop_resized = smart_resize(crop)

    print(f"  Size: {sample['native_size']:.0f}px → {crop_resized.shape[0]}px")
    return crop_resized, "IML Lifecycle\n(Gametocyte)"


def load_md2019_sample():
    """Load largest Trophozoite from MD_2019"""
    print("\n[LOADING] MD_2019 Stages - Trophozoite")

    label_dir = Path('data/processed/md_2019_stages/train/labels')
    image_dir = Path('data/processed/md_2019_stages/train/images')

    trophozoites = []

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
                if class_id != 2:  # 2 = trophozoite
                    continue

                x_center, y_center, width, height = map(float, parts[1:5])

                w_px = int(width * img_w)
                h_px = int(height * img_h)
                native_size = max(w_px, h_px) * 1.5

                trophozoites.append({
                    'image_path': str(image_path),
                    'bbox_yolo': [x_center, y_center, width, height],
                    'image_dims': [img_h, img_w],
                    'native_size': native_size
                })

    trophozoites.sort(key=lambda x: x['native_size'], reverse=True)
    sample = trophozoites[0]

    image = cv2.imread(sample['image_path'])

    # Crop from YOLO bbox
    img_h, img_w = sample['image_dims']
    x_center, y_center, width, height = sample['bbox_yolo']

    w_px = int(width * img_w)
    h_px = int(height * img_h)
    x_px = int(x_center * img_w)
    y_px = int(y_center * img_h)

    x1 = max(0, int(x_px - w_px * 0.625))
    y1 = max(0, int(y_px - h_px * 0.625))
    x2 = min(img_w, int(x_px + w_px * 0.625))
    y2 = min(img_h, int(y_px + h_px * 0.625))

    crop = image[y1:y2, x1:x2]
    crop_resized = smart_resize(crop)

    print(f"  Size: {sample['native_size']:.0f}px → {crop_resized.shape[0]}px")
    return crop_resized, "MD_2019 Stages\n(Trophozoite)"


def create_combined_figure(output_path, dpi=300):
    """
    Create combined figure with all 4 datasets
    Layout: 2 rows × 12 columns (each dataset = 6 images in 1 row)
    Row 1: [6 MP-IDB Species] [6 MP-IDB Stages]
    Row 2: [6 IML Lifecycle]  [6 MD_2019]
    NO titles, NO labels - just images with minimal margins
    """
    print("\n[GENERATING] Combined 4-dataset figure (2×12 layout, ultra compact)")

    # Load samples from all datasets
    datasets = [
        load_mp_idb_species_sample(),
        load_mp_idb_stages_sample(),
        load_iml_lifecycle_sample(),
        load_md2019_sample(),
    ]

    # Create figure with 2 rows × 12 columns (ultra wide)
    # Adjust figure size for wider aspect ratio
    fig = plt.figure(figsize=(24, 4))

    # Create main grid: 2 rows × 12 columns (minimal spacing)
    gs = GridSpec(2, 12, figure=fig,
                  hspace=0.01, wspace=0.01,
                  left=0.01, right=0.99, top=0.99, bottom=0.01)

    # Process each dataset (4 datasets total)
    for dataset_idx, (crop_array, dataset_name) in enumerate(datasets):
        # Determine position
        if dataset_idx < 2:
            # Top row: MP-IDB Species (cols 0-5), MP-IDB Stages (cols 6-11)
            row = 0
            col_start = dataset_idx * 6
        else:
            # Bottom row: IML Lifecycle (cols 0-5), MD_2019 (cols 6-11)
            row = 1
            col_start = (dataset_idx - 2) * 6

        # Get augmentations
        augmentations = get_6_augmentations(crop_array)

        # Add augmentations horizontally (6 columns)
        for aug_idx, (aug_name, aug_img) in enumerate(augmentations):
            col = col_start + aug_idx

            ax = plt.subplot(gs[row, col])

            # Display image
            img_array = np.array(aug_img)
            ax.imshow(img_array)

            # NO titles, NO labels - just the image
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

    print(f"\n[SUCCESS] Saved: {output_path}")
    print(f"          Layout: 2×2 datasets, 6 augmentations each")
    print(f"          Total images: 24 (4 datasets × 6 augmentations)")
    print(f"          DPI: {dpi}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str,
                       default='luaran/auto_generated/figures/augmentation/augmentation_4datasets_combined_2x2.png')
    parser.add_argument('--dpi', type=int, default=300)

    args = parser.parse_args()

    print("=" * 80)
    print("COMBINED 4-DATASET AUGMENTATION FIGURE (2×2 Grid)")
    print("=" * 80)

    create_combined_figure(args.output, args.dpi)

    output_path = Path(args.output)
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"\n[COMPLETED] File size: {size_mb:.2f} MB")

    print("\n" + "=" * 80)


# Import GridSpecFromSubplotSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec


if __name__ == "__main__":
    main()
