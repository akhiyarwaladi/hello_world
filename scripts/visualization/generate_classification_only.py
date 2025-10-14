#!/usr/bin/env python3
"""
Generate ONLY predicted classification visualizations (GT boxes with predicted class labels)
Uses GT boxes to evaluate pure classification performance
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms, models


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


def crop_and_classify(image, box_abs, classification_model, transform, class_names, device):
    """Crop and classify detected region"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    x1, y1, x2, y2 = box_abs
    crop = image.crop((x1, y1, x2, y2))
    crop_resized = crop.resize((224, 224))

    img_tensor = transform(crop_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classification_model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return class_names[predicted_idx.item()], confidence.item()


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

    # Load GT boxes
    gt_annotations = load_gt_annotations(label_file)
    gt_boxes = [yolo_to_absolute(ann, img_width, img_height) for ann in gt_annotations]

    # Load GT class mapping for comparison
    gt_class_mapping = load_gt_class_mapping(gt_crops_dir, image_name)

    # Classify each GT box
    pred_labels = []
    pred_colors = []

    img_pil = Image.open(image_path)

    for idx, box_abs in enumerate(gt_boxes):
        # Run classification on GT box
        pred_class, cls_conf = crop_and_classify(
            img_pil, box_abs, classification_model,
            transform, class_names, device
        )

        # Get GT class for comparison
        gt_class = gt_class_mapping.get(idx, 'unknown')

        # Color code: Medium green if correct, Red if wrong
        if pred_class == gt_class:
            color = (0, 180, 0)  # Medium green - correct (clear, not too bright)
        else:
            color = (0, 0, 255)  # Red - wrong

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
    import numpy as np
    exit(main())
