#!/usr/bin/env python3
"""
Generate ONLY ground truth classification visualizations (blue boxes with GT class labels)
"""

import sys
import argparse
from pathlib import Path
import cv2
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.visualization_utils import draw_boxes, yolo_to_absolute, load_gt_annotations, COLORS


def load_gt_class_mapping(crops_dir, image_name):
    """Load GT class labels from metadata CSV"""
    mapping = {}

    # Try to load from metadata CSV
    metadata_file = Path(crops_dir) / "ground_truth_crop_metadata.csv"

    if metadata_file.exists():
        try:
            df = pd.read_csv(metadata_file)

            # Filter for this image
            image_filter = (df['original_image'].str.contains(image_name, case=False, na=False))
            image_rows = df[image_filter]

            # Extract crop_id and map to class_name
            for _, row in image_rows.iterrows():
                crop_filename = row['crop_filename']
                class_name = row['class_name']

                # Extract crop ID: PA171697_crop_000.jpg -> 0
                crop_id_str = crop_filename.split('_crop_')[-1].replace('.jpg', '').replace('.png', '')
                crop_id = int(crop_id_str)
                mapping[crop_id] = class_name

            return mapping
        except Exception as e:
            print(f"   [WARNING] Failed to load metadata: {e}")

    # Fallback: Try to load from crops directory structure
    crops_test_dir = Path(crops_dir) / "crops" / "test"
    if not crops_test_dir.exists():
        return mapping

    for class_dir in crops_test_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        pattern = f"{image_name}_crop_*.png"

        for crop_file in class_dir.glob(pattern):
            crop_id_str = crop_file.stem.split('_crop_')[-1]
            crop_id = int(crop_id_str)
            mapping[crop_id] = class_name

    return mapping


def generate_gt_classification_visualization(image_path, label_file, gt_crops_dir, output_dir):
    """Generate GT classification visualization"""
    image = cv2.imread(str(image_path))
    img_height, img_width = image.shape[:2]
    image_name = Path(image_path).stem

    # Load GT boxes and class mapping (utils returns list of dicts with 'box' key)
    gt_annotations = load_gt_annotations(label_file)
    gt_boxes = [yolo_to_absolute(ann['box'], img_width, img_height) for ann in gt_annotations]
    gt_class_mapping = load_gt_class_mapping(gt_crops_dir, image_name)

    # Create labels
    labels = []
    for idx in range(len(gt_boxes)):
        if idx in gt_class_mapping:
            labels.append(gt_class_mapping[idx])
        else:
            labels.append('parasite')

    # Draw blue boxes with GT class labels
    colors = [COLORS['blue']] * len(gt_boxes)
    img_with_boxes = draw_boxes(image, gt_boxes, labels, colors)

    # Save
    output_file = Path(output_dir) / f"{image_name}.png"
    cv2.imwrite(str(output_file), img_with_boxes)

    return len(gt_boxes)


def main():
    parser = argparse.ArgumentParser(description='Generate GT classification visualizations')
    parser.add_argument('--test-images', type=str, required=True)
    parser.add_argument('--test-labels', type=str, required=True)
    parser.add_argument('--gt-crops', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max-images', type=int, default=None)

    args = parser.parse_args()

    # Get test images
    test_images_dir = Path(args.test_images)
    test_labels_dir = Path(args.test_labels)
    image_files = sorted(list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.JPG")) +
                        list(test_images_dir.glob("*.png")) + list(test_images_dir.glob("*.PNG")))

    if args.max_images:
        image_files = image_files[:args.max_images]

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    processed = 0
    for img_path in image_files:
        label_file = test_labels_dir / f"{img_path.stem}.txt"
        if not label_file.exists():
            continue

        try:
            n_boxes = generate_gt_classification_visualization(img_path, label_file, args.gt_crops, output_dir)
            processed += 1
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print(f"[SUCCESS] Processed {processed} images")
    return 0


if __name__ == "__main__":
    exit(main())
