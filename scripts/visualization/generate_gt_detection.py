#!/usr/bin/env python3
"""
Generate ONLY ground truth detection visualizations (blue boxes with 'parasite' label)
"""

import os
import sys
import argparse
from pathlib import Path
import cv2


def yolo_to_absolute(box_yolo, img_width, img_height):
    """Convert YOLO format to absolute coordinates"""
    x_center, y_center, w, h = box_yolo[1:]
    x1 = int((x_center - w / 2) * img_width)
    y1 = int((y_center - h / 2) * img_height)
    x2 = int((x_center + w / 2) * img_width)
    y2 = int((y_center + h / 2) * img_height)
    return [x1, y1, x2, y2]


def load_gt_annotations(label_file):
    """Load ground truth annotations"""
    annotations = []
    if not Path(label_file).exists():
        return annotations

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                box = [int(parts[0])] + [float(x) for x in parts[1:5]]
                annotations.append(box)
    return annotations


def draw_boxes(image, boxes, labels, colors, thickness=4, font_scale=0.9):
    """Draw bounding boxes with labels"""
    img_copy = image.copy()

    for box, label, color in zip(boxes, labels, colors):
        x1, y1, x2, y2 = [int(c) for c in box]

        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        if label:
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )

            text_y_top = y1 - text_height - baseline - 8
            text_y_bottom = y1

            if text_y_top < 0:
                text_y_top = y1
                text_y_bottom = y1 + text_height + baseline + 8
                text_pos_y = y1 + text_height + baseline
            else:
                text_pos_y = y1 - baseline - 5

            cv2.rectangle(img_copy, (x1 - 8, text_y_top), (x1 + text_width + 8, text_y_bottom), color, -1)
            cv2.putText(img_copy, label, (x1, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    return img_copy


def generate_gt_detection_visualization(image_path, label_file, output_dir):
    """Generate GT detection visualization"""
    image = cv2.imread(str(image_path))
    img_height, img_width = image.shape[:2]
    image_name = Path(image_path).stem

    # Load GT boxes
    gt_annotations = load_gt_annotations(label_file)
    gt_boxes = [yolo_to_absolute(ann, img_width, img_height) for ann in gt_annotations]

    # Draw blue boxes with 'parasite' labels
    colors = [(255, 0, 0)] * len(gt_boxes)
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
