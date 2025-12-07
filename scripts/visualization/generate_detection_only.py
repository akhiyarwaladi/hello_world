#!/usr/bin/env python3
"""
Generate ONLY predicted detection visualizations (green boxes with confidence)
"""

import sys
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.visualization_utils import draw_boxes, COLORS


def generate_detection_visualization(image_path, detection_model, output_dir, conf_threshold=0.25):
    """Generate predicted detection visualization"""
    image = cv2.imread(str(image_path))
    image_name = Path(image_path).stem

    # Run detection
    results = detection_model(str(image_path), conf=conf_threshold, verbose=False)[0]

    pred_boxes = []
    pred_labels = []
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            box_coords = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            pred_boxes.append([int(c) for c in box_coords])
            pred_labels.append(f"{confidence:.2f}")

    # Draw medium green boxes (clear but not too bright like lime green)
    colors = [COLORS['green']] * len(pred_boxes)  # BGR: Medium green (clear correct indication)
    img_with_boxes = draw_boxes(image, pred_boxes, pred_labels, colors)

    # Save
    output_file = Path(output_dir) / f"{image_name}.png"
    cv2.imwrite(str(output_file), img_with_boxes)

    return len(pred_boxes)


def main():
    parser = argparse.ArgumentParser(description='Generate detection visualizations only')
    parser.add_argument('--detection-model', type=str, required=True)
    parser.add_argument('--test-images', type=str, required=True)
    parser.add_argument('--test-labels', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--det-conf-threshold', type=float, default=0.25)
    parser.add_argument('--max-images', type=int, default=None)

    args = parser.parse_args()

    # Load detection model
    detection_model = YOLO(args.detection_model)

    # Get test images
    test_images_dir = Path(args.test_images)
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
        try:
            n_boxes = generate_detection_visualization(img_path, detection_model, output_dir, args.det_conf_threshold)
            processed += 1
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print(f"[SUCCESS] Processed {processed} images")
    return 0


if __name__ == "__main__":
    exit(main())
