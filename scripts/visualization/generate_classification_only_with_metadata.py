#!/usr/bin/env python3
"""
Enhanced classification visualization with CSV metadata export
Shows which predictions are correct/incorrect with confidence scores
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import shutil
from datetime import datetime


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


def get_model(model_name, num_classes=4, pretrained=False):
    """Get specified model architecture (from torchvision)"""

    if model_name.startswith('resnet'):
        if model_name == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'resnet34':
            model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        elif model_name == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V2' if pretrained else None)
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name.startswith('efficientnet'):
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'efficientnet_b2':
            model = models.efficientnet_b2(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f"Unknown EfficientNet model: {model_name}")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name.startswith('densenet'):
        if model_name == 'densenet121':
            model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'densenet161':
            model = models.densenet161(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'densenet169':
            model = models.densenet169(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f"Unknown DenseNet model: {model_name}")
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name.startswith('mobilenet'):
        if model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        elif model_name == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        else:
            raise ValueError(f"Unknown MobileNet model: {model_name}")

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


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
            print(f"   [WARNING] Could not load metadata CSV: {e}")

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


def clean_output_directory(output_dir, backup=True):
    """
    Clean output directory before generation

    Args:
        output_dir: Output directory path
        backup: If True, rename old folder to _backup_timestamp instead of deleting

    Returns:
        True if cleaned successfully
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        return True  # Nothing to clean

    # Check if folder has content
    files = list(output_path.glob("*"))
    if not files:
        return True  # Empty folder, nothing to clean

    if backup:
        # Rename to backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.parent / f"{output_path.name}_backup_{timestamp}"

        try:
            shutil.move(str(output_path), str(backup_path))
            print(f"   [BACKUP] Old folder renamed to: {backup_path.name}")
            return True
        except Exception as e:
            print(f"   [WARNING] Could not backup folder: {e}")
            # Try to delete old files instead
            pass

    # If backup failed or backup=False, delete old PNG and CSV files
    try:
        png_files = list(output_path.glob("*.png")) + list(output_path.glob("*.PNG"))
        csv_files = list(output_path.glob("*.csv"))

        deleted = 0
        for f in png_files + csv_files:
            f.unlink()
            deleted += 1

        if deleted > 0:
            print(f"   [CLEAN] Deleted {deleted} old files from {output_path.name}")

        return True
    except Exception as e:
        print(f"   [WARNING] Could not clean folder: {e}")
        return False


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


def generate_classification_visualization_with_metadata(
    image_path,
    label_path,
    classification_model,
    transform,
    class_names,
    crops_dir,
    output_dir,
    device
):
    """Generate classification visualization and return metadata"""

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    image_name = Path(image_path).stem
    img_height, img_width = image.shape[:2]

    # Load ground truth boxes
    gt_boxes = load_gt_annotations(label_path)

    # Load GT class mapping from crops metadata
    gt_class_mapping = load_gt_class_mapping(crops_dir, image_name)

    # Classify each GT box
    box_metadata = []
    pred_boxes = []
    pred_labels = []
    colors = []

    for crop_idx, box_data in enumerate(gt_boxes):
        gt_class_id = int(box_data[0])
        box_abs = yolo_to_absolute(box_data, img_width, img_height)

        # Get GT class name from mapping
        gt_class_name = gt_class_mapping.get(crop_idx, class_names[gt_class_id] if gt_class_id < len(class_names) else f"class_{gt_class_id}")

        # Classify
        predicted_class, confidence = crop_and_classify(
            image, box_abs, classification_model, transform, class_names, device
        )

        # Determine if correct
        is_correct = (predicted_class == gt_class_name)

        # Store metadata
        box_metadata.append({
            'crop_idx': crop_idx,
            'gt_class': gt_class_name,
            'pred_class': predicted_class,
            'confidence': f"{confidence:.3f}",
            'correct': 'Yes' if is_correct else 'No'
        })

        # For visualization
        pred_boxes.append(box_abs)
        label_text = f"{predicted_class} ({confidence:.2f})"
        pred_labels.append(label_text)

        # Color: green if correct, red if wrong
        if is_correct:
            colors.append((0, 180, 0))  # Green
        else:
            colors.append((0, 0, 200))  # Red

    # Calculate statistics
    n_total = len(box_metadata)
    n_correct = sum(1 for m in box_metadata if m['correct'] == 'Yes')
    n_incorrect = n_total - n_correct
    accuracy = n_correct / n_total if n_total > 0 else 0

    # Determine status
    if n_total == 0:
        status = "No boxes"
    elif n_correct == n_total:
        status = "All correct"
    elif n_incorrect == n_total:
        status = "All wrong"
    else:
        status = "Mixed (some correct, some wrong)"

    # Average confidence
    confidences = [float(m['confidence']) for m in box_metadata]
    avg_confidence = np.mean(confidences) if confidences else 0.0

    # Paper score
    if status == "All correct" and avg_confidence > 0.85:
        paper_score = 10  # Excellent
    elif status == "All correct" and avg_confidence > 0.75:
        paper_score = 9   # Very good
    elif n_correct > 0 and n_incorrect == 0:
        paper_score = 8   # Good
    elif status == "Mixed (some correct, some wrong)" and avg_confidence > 0.7:
        paper_score = 6   # Challenging case (shows both correct & errors)
    elif n_incorrect > 0 and avg_confidence > 0.75:
        paper_score = 5   # High confidence errors (interesting failure case)
    elif status == "All wrong":
        paper_score = 3   # Complete failure
    else:
        paper_score = 4   # Low confidence

    # Draw boxes with color-coded predictions
    img_with_boxes = draw_boxes(image, pred_boxes, pred_labels, colors)

    # Save image
    output_file = Path(output_dir) / f"{image_name}.png"
    cv2.imwrite(str(output_file), img_with_boxes)

    # Return image-level metadata
    image_metadata = {
        'image_name': image_name,
        'image_file': str(output_file),
        'n_boxes': n_total,
        'n_correct': n_correct,
        'n_incorrect': n_incorrect,
        'accuracy': f"{accuracy:.3f}",
        'avg_confidence': f"{avg_confidence:.3f}",
        'status': status,
        'paper_score': paper_score,
        'recommendation': 'Best for paper' if paper_score >= 9 else ('Good for paper' if paper_score >= 8 else ('Challenging case' if paper_score >= 5 else 'Not recommended'))
    }

    return image_metadata, box_metadata


def main():
    parser = argparse.ArgumentParser(description='Generate classification visualizations with CSV metadata')
    parser.add_argument('--classification-model', type=str, required=True)
    parser.add_argument('--test-images', type=str, required=True)
    parser.add_argument('--test-labels', type=str, required=True)
    parser.add_argument('--crops-dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--no-backup', action='store_true',
                       help='Delete old files instead of backing up (default: backup to _backup_timestamp)')
    parser.add_argument('--keep-old', action='store_true',
                       help='Keep old files (WARNING: may cause orphaned files!)')

    args = parser.parse_args()

    # Clean output directory BEFORE generation (prevent orphaned files)
    output_dir = Path(args.output)

    if args.keep_old:
        print(f"   [KEEP] Keeping old files (may overwrite with same names)")
    else:
        backup_mode = not args.no_backup
        clean_output_directory(output_dir, backup=backup_mode)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}")

    # Load classification model
    print(f"[LOAD] Loading classification model: {args.classification_model}")
    checkpoint = torch.load(args.classification_model, map_location=device)

    # Get model architecture from checkpoint
    model_name = checkpoint.get('model_name', 'efficientnet_b0')
    num_classes = args.num_classes

    # Create model
    classification_model = get_model(model_name, num_classes=num_classes, pretrained=False)
    classification_model.load_state_dict(checkpoint['model_state_dict'])
    classification_model = classification_model.to(device)
    classification_model.eval()

    # Class names
    class_names = checkpoint.get('class_names', [f'class_{i}' for i in range(num_classes)])
    print(f"[CLASSES] {len(class_names)} classes: {class_names}")

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get test images
    test_images_dir = Path(args.test_images)
    test_labels_dir = Path(args.test_labels)
    crops_dir = Path(args.crops_dir)

    image_files = sorted(
        list(test_images_dir.glob("*.jpg")) +
        list(test_images_dir.glob("*.JPG")) +
        list(test_images_dir.glob("*.png")) +
        list(test_images_dir.glob("*.PNG"))
    )

    if args.max_images:
        image_files = image_files[:args.max_images]

    # Ensure output directory exists (already cleaned above if needed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process images and collect metadata
    print(f"[PROCESS] Processing {len(image_files)} images...")
    all_image_metadata = []
    all_box_metadata = []
    processed = 0

    for img_path in image_files:
        label_path = test_labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"   [SKIP] No label file: {img_path.name}")
            continue

        try:
            image_meta, box_meta = generate_classification_visualization_with_metadata(
                img_path,
                label_path,
                classification_model,
                transform,
                class_names,
                crops_dir,
                output_dir,
                device
            )

            if image_meta and box_meta:
                all_image_metadata.append(image_meta)

                # Add image name to each box metadata
                for bm in box_meta:
                    bm['image_name'] = image_meta['image_name']
                all_box_metadata.extend(box_meta)

                processed += 1

                # Print status for interesting cases
                if image_meta['paper_score'] >= 9:
                    print(f"   ✓ [BEST] {image_meta['image_name']}: {image_meta['status']}")
                elif image_meta['status'] == "Mixed (some correct, some wrong)":
                    print(f"   ! [MIXED] {image_meta['image_name']}: {image_meta['n_correct']}/{image_meta['n_boxes']} correct")

        except Exception as e:
            print(f"   [ERROR] {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Save metadata to CSV
    if all_image_metadata:
        # Image-level summary
        csv_file_images = output_dir / "classification_metadata_images.csv"
        df_images = pd.DataFrame(all_image_metadata)
        df_images = df_images.sort_values('paper_score', ascending=False)
        df_images.to_csv(csv_file_images, index=False)

        # Box-level details
        csv_file_boxes = output_dir / "classification_metadata_boxes.csv"
        df_boxes = pd.DataFrame(all_box_metadata)
        df_boxes.to_csv(csv_file_boxes, index=False)

        print(f"\n[SUCCESS] Processed {processed} images")
        print(f"[CSV] Image metadata: {csv_file_images}")
        print(f"[CSV] Box metadata: {csv_file_boxes}")

        # Statistics
        print(f"\n[STATS] Classification Summary:")
        print(f"   All correct: {len(df_images[df_images['status'] == 'All correct'])}")
        print(f"   Mixed results: {len(df_images[df_images['status'] == 'Mixed (some correct, some wrong)'])}")
        print(f"   All wrong: {len(df_images[df_images['status'] == 'All wrong'])}")
        print(f"   Best for paper (score ≥9): {len(df_images[df_images['paper_score'] >= 9])}")
        print(f"   Good for paper (score ≥8): {len(df_images[df_images['paper_score'] >= 8])}")

        # Box-level stats
        total_boxes = len(df_boxes)
        correct_boxes = len(df_boxes[df_boxes['correct'] == 'Yes'])
        print(f"\n[BOXES] Total: {total_boxes}, Correct: {correct_boxes} ({correct_boxes/total_boxes*100:.1f}%)")

        # Per-class accuracy
        print(f"\n[PER-CLASS] Accuracy by class:")
        for class_name in class_names:
            class_boxes = df_boxes[df_boxes['gt_class'] == class_name]
            if len(class_boxes) > 0:
                class_correct = len(class_boxes[class_boxes['correct'] == 'Yes'])
                class_acc = class_correct / len(class_boxes) * 100
                print(f"   {class_name}: {class_correct}/{len(class_boxes)} ({class_acc:.1f}%)")

        # Show top 5 best images
        print(f"\n[TOP 5] Best images for paper:")
        for idx, row in df_images.head(5).iterrows():
            print(f"   {idx+1}. {row['image_name']}: {row['status']} ({row['n_correct']}/{row['n_boxes']} correct, avg_conf={row['avg_confidence']}, score={row['paper_score']})")

    return 0


if __name__ == "__main__":
    exit(main())
