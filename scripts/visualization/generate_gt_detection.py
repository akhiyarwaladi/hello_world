#!/usr/bin/env python3
"""
Generate ONLY ground truth detection visualizations (blue boxes with 'parasite' label)
"""

import sys
import argparse
from pathlib import Path
import cv2

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.visualization_utils import draw_boxes, yolo_to_absolute, load_gt_annotations, COLORS


def generate_gt_detection_visualization(image_path, label_file, output_dir):
    """Generate GT detection visualization"""
    image = cv2.imread(str(image_path))
    img_height, img_width = image.shape[:2]
    image_name = Path(image_path).stem

    # Load GT boxes (utils returns list of dicts with 'class_id' and 'box' keys)
    gt_annotations = load_gt_annotations(label_file)
    gt_boxes = [yolo_to_absolute(ann['box'], img_width, img_height) for ann in gt_annotations]

    # Draw blue boxes with 'parasite' labels
    colors = [COLORS['blue']] * len(gt_boxes)
    labels = ['parasite'] * len(gt_boxes)
    img_with_boxes = draw_boxes(image, gt_boxes, labels, colors)

    # Save
    output_file = Path(output_dir) / f"{image_name}.png"
    cv2.imwrite(str(output_file), img_with_boxes)

    return len(gt_boxes)


def main():
    parser = argparse.ArgumentParser(description='Generate GT detection visualizations')
    parser.add_argument('--test-images', type=str, required=True)
    parser.add_argument('--test-labels', type=str, required=True)
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
            n_boxes = generate_gt_detection_visualization(img_path, label_file, output_dir)
            processed += 1
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print(f"[SUCCESS] Processed {processed} images")
    return 0


if __name__ == "__main__":
    exit(main())
