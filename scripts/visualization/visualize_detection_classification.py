#!/usr/bin/env python3
"""
End-to-End Detection + Classification Visualization
Visualize detection boxes with classification labels on original images

Ground Truth vs Model Predictions (Detection + Classification)
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    Boxes are in format [x1, y1, x2, y2] (absolute coordinates)
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def yolo_to_absolute(box_yolo, img_width, img_height):
    """
    Convert YOLO format [class, x_center, y_center, width, height] (normalized)
    to absolute [x1, y1, x2, y2]
    """
    x_center, y_center, w, h = box_yolo[1:]

    x1 = (x_center - w / 2) * img_width
    y1 = (y_center - h / 2) * img_height
    x2 = (x_center + w / 2) * img_width
    y2 = (y_center + h / 2) * img_height

    return [int(x1), int(y1), int(x2), int(y2)]


def load_ground_truth_annotations(label_file, crops_dir):
    """
    Load ground truth boxes and their class labels from crops directory structure

    Args:
        label_file: Path to YOLO format label file (detection boxes only)
        crops_dir: Path to ground truth crops directory with class subfolders

    Returns:
        List of dicts with 'box' (yolo format) and 'class_name'
    """
    annotations = []

    # Read detection boxes
    if not Path(label_file).exists():
        return annotations

    with open(label_file, 'r') as f:
        lines = f.readlines()

    # Parse YOLO format: class x_center y_center width height
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            box = [cls_id, x_center, y_center, width, height]

            # Class name will be assigned when matching with crops
            annotations.append({
                'box': box,
                'class_name': 'parasite',  # Default, will be updated
                'matched': False
            })

    return annotations


def load_gt_class_mapping(crops_dir, image_name):
    """
    Create mapping of GT boxes to class names based on saved crops

    Args:
        crops_dir: Path to ground truth crops with structure: crops/test/{class}/{image_name}_crop_XXX.jpg
        image_name: Original image name (without extension)

    Returns:
        Dict mapping crop_id to class_name
    """
    mapping = {}
    crops_test_dir = Path(crops_dir) / "crops" / "test"

    if not crops_test_dir.exists():
        return mapping

    # Iterate through class folders
    for class_dir in crops_test_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        # Find crops from this image
        pattern = f"{image_name}_crop_*.jpg"
        for crop_file in class_dir.glob(pattern):
            # Extract crop ID from filename: image_crop_000.jpg -> 0
            crop_id_str = crop_file.stem.split('_crop_')[-1]
            crop_id = int(crop_id_str)
            mapping[crop_id] = class_name

    return mapping


def run_detection(image_path, detection_model, conf_threshold=0.25):
    """Run YOLO detection on image with configurable confidence threshold"""
    results = detection_model(str(image_path), conf=conf_threshold, verbose=False)
    return results[0]  # First image result


def crop_and_classify(image, box_abs, classification_model, transform, class_names, device):
    """
    Crop detected region and run classification

    Args:
        image: PIL Image or numpy array
        box_abs: Absolute box coordinates [x1, y1, x2, y2]
        classification_model: PyTorch classification model
        transform: Image transform for classification
        class_names: List of class names
        device: torch device

    Returns:
        predicted_class_name, confidence
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Crop region
    x1, y1, x2, y2 = box_abs
    crop = image.crop((x1, y1, x2, y2))

    # Resize to 224x224
    crop_resized = crop.resize((224, 224))

    # Transform and predict
    img_tensor = transform(crop_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classification_model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score


def draw_boxes_on_image(image, boxes, labels, colors, thickness=3, font_scale=0.6):
    """
    Draw bounding boxes with labels on image

    Args:
        image: numpy array (BGR)
        boxes: List of [x1, y1, x2, y2] in absolute coordinates
        labels: List of label strings
        colors: List of (B, G, R) tuples
        thickness: Box line thickness
        font_scale: Text font scale

    Returns:
        Image with boxes drawn
    """
    img_copy = image.copy()

    for box, label, color in zip(boxes, labels, colors):
        x1, y1, x2, y2 = [int(c) for c in box]

        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

        # Prepare label text
        label_text = label

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=2
        )

        # Draw label background
        cv2.rectangle(
            img_copy,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1  # Filled
        )

        # Draw label text
        cv2.putText(
            img_copy,
            label_text,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # White text
            2
        )

    return img_copy


def visualize_detection_classification(
    image_path,
    detection_model,
    classification_model,
    class_names,
    transform,
    device,
    gt_label_file,
    gt_crops_dir,
    output_path,
    iou_threshold=0.5,
    det_conf_threshold=0.25
):
    """
    Create side-by-side visualization: Ground Truth vs Predicted (Detection + Classification)
    """
    # Load image
    image = cv2.imread(str(image_path))
    img_height, img_width = image.shape[:2]
    image_name = Path(image_path).stem

    print(f"\n[PROCESS] {image_name}")
    print(f"   Image size: {img_width}x{img_height}")

    # ========================================
    # GROUND TRUTH SIDE
    # ========================================

    # Load GT annotations
    gt_annotations = load_ground_truth_annotations(gt_label_file, gt_crops_dir)
    gt_class_mapping = load_gt_class_mapping(gt_crops_dir, image_name)

    # Update GT class names from crop mapping
    for idx, ann in enumerate(gt_annotations):
        if idx in gt_class_mapping:
            ann['class_name'] = gt_class_mapping[idx]

    # Draw GT boxes
    gt_boxes_abs = []
    gt_labels = []
    gt_colors = []

    for ann in gt_annotations:
        box_abs = yolo_to_absolute(ann['box'], img_width, img_height)
        gt_boxes_abs.append(box_abs)
        gt_labels.append(ann['class_name'])
        gt_colors.append((255, 0, 0))  # Blue for GT

    img_gt = draw_boxes_on_image(image, gt_boxes_abs, gt_labels, gt_colors)

    print(f"   GT boxes: {len(gt_boxes_abs)}")

    # ========================================
    # PREDICTION SIDE
    # ========================================

    # Run detection
    det_results = run_detection(image_path, detection_model, det_conf_threshold)

    pred_boxes_abs = []
    pred_labels = []
    pred_colors = []

    # Process each detection
    boxes = det_results.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            # Get box coordinates (xyxy format)
            box_coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(c) for c in box_coords]
            box_abs = [x1, y1, x2, y2]

            # Get detection confidence
            det_conf = float(box.conf[0])

            # Run classification on cropped region
            img_pil = Image.open(image_path)
            pred_class, cls_conf = crop_and_classify(
                img_pil, box_abs, classification_model, transform, class_names, device
            )

            # Match with GT to determine if correct
            matched_gt = None
            max_iou = 0
            for idx, gt_box in enumerate(gt_boxes_abs):
                iou = calculate_iou(box_abs, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    matched_gt = idx

            # Determine color and label
            if max_iou >= iou_threshold:
                # Detection matched with GT
                gt_class = gt_annotations[matched_gt]['class_name']

                if pred_class == gt_class:
                    # Correct classification
                    color = (0, 255, 0)  # Green
                    label = f"{pred_class} {cls_conf:.2f} ✓"
                else:
                    # Misclassification
                    color = (0, 0, 255)  # Red
                    label = f"{pred_class} {cls_conf:.2f} (GT:{gt_class})"
            else:
                # False positive (no matching GT)
                color = (0, 165, 255)  # Orange
                label = f"{pred_class} {cls_conf:.2f} FP"

            pred_boxes_abs.append(box_abs)
            pred_labels.append(label)
            pred_colors.append(color)

    # Add false negatives (GT boxes not detected)
    for idx, gt_box in enumerate(gt_boxes_abs):
        matched = False
        for pred_box in pred_boxes_abs:
            if calculate_iou(gt_box, pred_box) >= iou_threshold:
                matched = True
                break

        if not matched:
            # False negative
            pred_boxes_abs.append(gt_box)
            pred_labels.append(f"MISSED: {gt_annotations[idx]['class_name']}")
            pred_colors.append((0, 255, 255))  # Yellow

    img_pred = draw_boxes_on_image(image, pred_boxes_abs, pred_labels, pred_colors)

    print(f"   Predicted boxes: {len(pred_boxes_abs)}")

    # ========================================
    # CREATE SIDE-BY-SIDE COMPARISON
    # ========================================

    # Create matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left: Ground Truth
    axes[0].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Ground Truth', fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # Right: Prediction
    axes[1].imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Model Prediction (Detection + Classification)', fontsize=16, fontweight='bold')
    axes[1].axis('off')

    # Add legend
    legend_elements = [
        mpatches.Patch(color='blue', label='Ground Truth'),
        mpatches.Patch(color='green', label='Correct (TP)'),
        mpatches.Patch(color='red', label='Misclassified'),
        mpatches.Patch(color='yellow', label='Missed (FN)'),
        mpatches.Patch(color='orange', label='False Positive')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)

    # Save
    output_file = Path(output_path) / f"{image_name}_detection_classification.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   [SAVED] {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Visualize detection + classification predictions')
    parser.add_argument('--detection-model', type=str, required=True,
                       help='Path to YOLO detection model (best.pt)')
    parser.add_argument('--classification-model', type=str, required=True,
                       help='Path to classification model (best.pt)')
    parser.add_argument('--test-images', type=str, required=True,
                       help='Path to test images directory')
    parser.add_argument('--test-labels', type=str, required=True,
                       help='Path to test labels directory')
    parser.add_argument('--gt-crops', type=str, required=True,
                       help='Path to ground truth crops directory')
    parser.add_argument('--output', type=str, default='detection_classification_comparison',
                       help='Output directory')
    parser.add_argument('--max-images', type=int, default=5,
                       help='Maximum number of images to visualize')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching boxes')
    parser.add_argument('--det-conf-threshold', type=float, default=0.25,
                       help='Detection confidence threshold (default: 0.25)')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DETECTION + CLASSIFICATION VISUALIZATION")
    print("="*80)
    print(f"Detection model: {args.detection_model}")
    print(f"Classification model: {args.classification_model}")
    print(f"Test images: {args.test_images}")
    print(f"Output: {args.output}")

    # Load detection model
    print("\n[LOAD] Loading YOLO detection model...")
    from ultralytics import YOLO
    detection_model = YOLO(args.detection_model)

    # Load classification model
    print("[LOAD] Loading classification model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine architecture from model file
    from torchvision import transforms

    # Load class names from crops directory structure
    crops_test_dir = Path(args.gt_crops) / "crops" / "test"
    class_names = sorted([d.name for d in crops_test_dir.iterdir() if d.is_dir()])

    print(f"   Classes: {class_names}")

    # Load classification model
    classification_checkpoint = torch.load(args.classification_model, map_location=device)

    # Get model architecture from checkpoint
    if 'model_name' in classification_checkpoint:
        model_name = classification_checkpoint['model_name']
    else:
        # Try to infer from filename
        model_name = 'densenet121'  # Default

    # Create model
    from torchvision import models

    if 'densenet' in model_name.lower():
        classification_model = models.densenet121(pretrained=False)
        classification_model.classifier = torch.nn.Linear(classification_model.classifier.in_features, len(class_names))
    elif 'efficientnet_b1' in model_name.lower():
        classification_model = models.efficientnet_b1(pretrained=False)
        classification_model.classifier[1] = torch.nn.Linear(classification_model.classifier[1].in_features, len(class_names))
    elif 'efficientnet_b2' in model_name.lower():
        classification_model = models.efficientnet_b2(pretrained=False)
        classification_model.classifier[1] = torch.nn.Linear(classification_model.classifier[1].in_features, len(class_names))
    elif 'resnet' in model_name.lower():
        classification_model = models.resnet101(pretrained=False)
        classification_model.fc = torch.nn.Linear(classification_model.fc.in_features, len(class_names))
    else:
        print(f"[WARNING] Unknown model architecture: {model_name}")
        classification_model = models.densenet121(pretrained=False)
        classification_model.classifier = torch.nn.Linear(classification_model.classifier.in_features, len(class_names))

    classification_model.load_state_dict(classification_checkpoint['model_state_dict'])
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

    image_files = sorted(list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.JPG")))

    if not image_files:
        print("[ERROR] No test images found!")
        return

    print(f"\n[FOUND] {len(image_files)} test images")

    # Process images
    processed = 0
    for img_path in image_files:
        if processed >= args.max_images:
            break

        # Find corresponding label file
        label_file = test_labels_dir / f"{img_path.stem}.txt"

        if not label_file.exists():
            print(f"[SKIP] No label file for {img_path.name}")
            continue

        try:
            visualize_detection_classification(
                img_path,
                detection_model,
                classification_model,
                class_names,
                transform,
                device,
                label_file,
                args.gt_crops,
                output_path,
                args.iou_threshold,
                args.det_conf_threshold
            )
            processed += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"[SUCCESS] Processed {processed} images")
    print(f"[OUTPUT] {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
