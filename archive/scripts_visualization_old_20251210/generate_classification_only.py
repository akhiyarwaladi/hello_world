#!/usr/bin/env python3
"""
Generate ONLY predicted classification visualizations (GT boxes with predicted class labels)
Uses GT boxes to evaluate pure classification performance
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms, models

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.visualization_utils import (
    draw_boxes, yolo_to_absolute, load_gt_annotations, crop_and_classify, COLORS
)


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
            pass

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


def generate_classification_visualization(
    image_path,
    label_file,
    gt_crops_dir,
    classification_model,
    class_names,
    transform,
    device,
    output_dir
):
    """Generate predicted classification visualization"""
    image = cv2.imread(str(image_path))
    img_height, img_width = image.shape[:2]
    image_name = Path(image_path).stem

    # Load GT boxes (utils returns list of dicts with 'class_id' and 'box' keys)
    gt_annotations = load_gt_annotations(label_file)
    gt_boxes = [yolo_to_absolute(ann['box'], img_width, img_height) for ann in gt_annotations]

    # Load GT class mapping for comparison
    gt_class_mapping = load_gt_class_mapping(gt_crops_dir, image_name)

    # Classify each GT box
    pred_labels = []
    pred_colors = []

    for idx, box_abs in enumerate(gt_boxes):
        # Run classification on GT box (utils version takes np.ndarray BGR)
        pred_class, cls_conf = crop_and_classify(
            image, box_abs, classification_model,
            transform, class_names, device
        )

        # Get GT class for comparison
        gt_class = gt_class_mapping.get(idx, 'unknown')

        # Color code: Medium green if correct, Red if wrong
        if pred_class == gt_class:
            color = COLORS['green']  # Medium green - correct
        else:
            color = COLORS['red']  # Red - wrong

        pred_labels.append(pred_class)
        pred_colors.append(color)

    # Draw GT boxes with predicted classification labels
    img_with_boxes = draw_boxes(image, gt_boxes, pred_labels, pred_colors)

    # Save
    output_file = Path(output_dir) / f"{image_name}.png"
    cv2.imwrite(str(output_file), img_with_boxes)

    return len(gt_boxes)


def main():
    parser = argparse.ArgumentParser(description='Generate classification visualizations only')
    parser.add_argument('--classification-model', type=str, required=True)
    parser.add_argument('--test-images', type=str, required=True)
    parser.add_argument('--test-labels', type=str, required=True)
    parser.add_argument('--gt-crops', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max-images', type=int, default=None)

    args = parser.parse_args()

    # Load classification model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get class names
    crops_test_dir = Path(args.gt_crops) / "crops" / "test"
    class_names = sorted([d.name for d in crops_test_dir.iterdir() if d.is_dir()])

    # Load classification model
    checkpoint = torch.load(args.classification_model, map_location=device, weights_only=False)

    # Infer model architecture
    model_name = checkpoint.get('model_name', 'densenet121')

    if 'efficientnet_b1' in model_name.lower():
        classification_model = models.efficientnet_b1(weights=None)
        classification_model.classifier[1] = torch.nn.Linear(
            classification_model.classifier[1].in_features, len(class_names)
        )
    elif 'efficientnet_b0' in model_name.lower():
        classification_model = models.efficientnet_b0(weights=None)
        classification_model.classifier[1] = torch.nn.Linear(
            classification_model.classifier[1].in_features, len(class_names)
        )
    elif 'efficientnet_b2' in model_name.lower():
        classification_model = models.efficientnet_b2(weights=None)
        classification_model.classifier[1] = torch.nn.Linear(
            classification_model.classifier[1].in_features, len(class_names)
        )
    elif 'resnet50' in model_name.lower():
        classification_model = models.resnet50(weights=None)
        classification_model.fc = torch.nn.Linear(
            classification_model.fc.in_features, len(class_names)
        )
    elif 'resnet101' in model_name.lower():
        classification_model = models.resnet101(weights=None)
        classification_model.fc = torch.nn.Linear(
            classification_model.fc.in_features, len(class_names)
        )
    else:  # Default to densenet121
        classification_model = models.densenet121(weights=None)
        classification_model.classifier = torch.nn.Linear(
            classification_model.classifier.in_features, len(class_names)
        )

    classification_model.load_state_dict(checkpoint['model_state_dict'])
    classification_model.to(device)
    classification_model.eval()

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
            n_boxes = generate_classification_visualization(
                img_path, label_file, args.gt_crops,
                classification_model, class_names, transform, device, output_dir
            )
            processed += 1
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print(f"[SUCCESS] Processed {processed} images")
    return 0


if __name__ == "__main__":
    exit(main())
