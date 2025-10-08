#!/usr/bin/env python3
"""
Generate Detection + Classification Figures for Paper

For each test image, generates 4 separate visualizations:
1. gt_detection/          - Ground truth boxes with 'parasite' labels (blue)
2. pred_detection/        - Predicted detection boxes with confidence scores (green)
3. gt_classification/     - Ground truth boxes with class labels (blue)
4. pred_classification/   - GT boxes with predicted class labels (green=correct, red=wrong)

IMPORTANT: pred_classification uses GT boxes (not predicted boxes) to evaluate
pure classification performance, matching training methodology.

Default: Process ALL test images
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms, models

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
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
    """Convert YOLO format to absolute coordinates"""
    x_center, y_center, w, h = box_yolo[1:]
    x1 = int((x_center - w / 2) * img_width)
    y1 = int((y_center - h / 2) * img_height)
    x2 = int((x_center + w / 2) * img_width)
    y2 = int((y_center + h / 2) * img_height)
    return [x1, y1, x2, y2]


def load_gt_annotations(label_file):
    """Load ground truth detection boxes from YOLO format"""
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
        import pandas as pd
        try:
            df = pd.read_csv(metadata_file)

            # Filter for this image (handle both with/without .JPG extension)
            image_filter = (df['original_image'].str.contains(image_name, case=False, na=False))
            image_rows = df[image_filter]

            # Extract crop_id from filename and map to class_name
            for _, row in image_rows.iterrows():
                crop_filename = row['crop_filename']
                class_name = row['class_name']

                # Extract crop ID: PA171697_crop_000.jpg -> 0
                crop_id_str = crop_filename.split('_crop_')[-1].replace('.jpg', '')
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
        pattern = f"{image_name}_crop_*.jpg"

        for crop_file in class_dir.glob(pattern):
            crop_id_str = crop_file.stem.split('_crop_')[-1]
            crop_id = int(crop_id_str)
            mapping[crop_id] = class_name

    return mapping


def draw_boxes(image, boxes, labels=None, colors=None, thickness=4, font_scale=0.9):
    """Draw bounding boxes with optional labels"""
    img_copy = image.copy()
    img_height, img_width = image.shape[:2]

    if colors is None:
        colors = [(255, 0, 0)] * len(boxes)  # Default blue

    if labels is None:
        labels = [''] * len(boxes)

    for box, label, color in zip(boxes, labels, colors):
        x1, y1, x2, y2 = [int(c) for c in box]

        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

        # Draw label if provided
        if label and label.strip():  # Check if label is not empty
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )

            # Background for text (larger padding)
            padding = 8

            # Calculate text position (above box if space available, otherwise below)
            text_y_top = y1 - text_height - baseline - padding
            text_y_bottom = y1

            # If text would go above image, place it below the box top instead
            if text_y_top < 0:
                text_y_top = y1
                text_y_bottom = y1 + text_height + baseline + padding
                text_pos_y = y1 + text_height + baseline
            else:
                text_pos_y = y1 - baseline - 5

            # Ensure text doesn't go beyond right edge
            text_x_right = min(x1 + text_width + padding, img_width)
            text_x_left = max(x1 - padding, 0)

            # Draw background rectangle
            cv2.rectangle(
                img_copy,
                (text_x_left, text_y_top),
                (text_x_right, text_y_bottom),
                color,
                -1  # Filled
            )

            # Text (white, medium thickness)
            cv2.putText(
                img_copy,
                label,
                (max(x1, 0), text_pos_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White
                2  # Reduced thickness
            )

    return img_copy


def run_detection(image_path, detection_model, conf_threshold=0.25):
    """Run YOLO detection"""
    results = detection_model(str(image_path), conf=conf_threshold, verbose=False)
    return results[0]


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


def generate_gt_detection(image_path, label_file, output_dir):
    """Generate GT detection visualization (boxes with 'parasite' label)"""
    image = cv2.imread(str(image_path))
    img_height, img_width = image.shape[:2]
    image_name = Path(image_path).stem

    # Load GT boxes
    gt_annotations = load_gt_annotations(label_file)
    gt_boxes = [yolo_to_absolute(ann, img_width, img_height) for ann in gt_annotations]

    # Draw blue boxes with 'parasite' labels (generic)
    colors = [(255, 0, 0)] * len(gt_boxes)  # Blue
    labels = ['parasite'] * len(gt_boxes)  # All boxes labeled as 'parasite'
    img_with_boxes = draw_boxes(image, gt_boxes, labels=labels, colors=colors)

    # Save
    output_path = Path(output_dir) / "gt_detection"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{image_name}.png"
    cv2.imwrite(str(output_file), img_with_boxes)

    return len(gt_boxes)


def generate_pred_detection(image_path, detection_model, output_dir, conf_threshold=0.25):
    """Generate predicted detection visualization (boxes with confidence scores)"""
    image = cv2.imread(str(image_path))
    image_name = Path(image_path).stem

    # Run detection
    det_results = run_detection(image_path, detection_model, conf_threshold)

    pred_boxes = []
    pred_labels = []
    if det_results.boxes is not None and len(det_results.boxes) > 0:
        for box in det_results.boxes:
            box_coords = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            pred_boxes.append([int(c) for c in box_coords])
            pred_labels.append("parasite")

    # Draw green boxes with confidence labels
    colors = [(0, 255, 0)] * len(pred_boxes)  # Green
    img_with_boxes = draw_boxes(image, pred_boxes, labels=pred_labels, colors=colors)

    # Save
    output_path = Path(output_dir) / "pred_detection"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{image_name}.png"
    cv2.imwrite(str(output_file), img_with_boxes)

    return len(pred_boxes)


def generate_gt_classification(image_path, label_file, gt_crops_dir, output_dir):
    """Generate GT classification visualization (boxes + GT class labels)"""
    image = cv2.imread(str(image_path))
    img_height, img_width = image.shape[:2]
    image_name = Path(image_path).stem

    # Load GT boxes and class mapping
    gt_annotations = load_gt_annotations(label_file)
    gt_boxes = [yolo_to_absolute(ann, img_width, img_height) for ann in gt_annotations]
    gt_class_mapping = load_gt_class_mapping(gt_crops_dir, image_name)

    # Create labels
    labels = []
    for idx in range(len(gt_boxes)):
        if idx in gt_class_mapping:
            labels.append(gt_class_mapping[idx])
        else:
            labels.append('parasite')

    print(f"   GT class mapping: {gt_class_mapping}")
    print(f"   GT labels: {labels}")

    # Draw blue boxes with labels
    colors = [(255, 0, 0)] * len(gt_boxes)  # Blue
    img_with_boxes = draw_boxes(image, gt_boxes, labels=labels, colors=colors)

    # Save
    output_path = Path(output_dir) / "gt_classification"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{image_name}.png"
    cv2.imwrite(str(output_file), img_with_boxes)

    return len(gt_boxes)


def generate_pred_classification(
    image_path,
    label_file,
    gt_crops_dir,
    classification_model,
    class_names,
    transform,
    device,
    output_dir
):
    """Generate predicted classification visualization (GT boxes + predicted labels)

    IMPORTANT: Uses GROUND TRUTH boxes, NOT detection predictions!
    This evaluates pure classification performance without detection errors.
    """
    image = cv2.imread(str(image_path))
    img_height, img_width = image.shape[:2]
    image_name = Path(image_path).stem

    # Load GT boxes (same as gt_classification!)
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

        # Color code: Green if correct, Red if wrong
        if pred_class == gt_class:
            color = (0, 255, 0)  # Green - correct
            label = pred_class
        else:
            color = (0, 0, 255)  # Red - wrong
            label = pred_class

        pred_labels.append(label)
        pred_colors.append(color)

        print(f"      Box {idx+1}: Pred={pred_class} ({cls_conf:.2f}), GT={gt_class}")

    # Draw GT boxes with predicted classification labels
    print(f"   Drawing {len(gt_boxes)} GT boxes with predicted labels")
    img_with_boxes = draw_boxes(image, gt_boxes, labels=pred_labels, colors=pred_colors)

    # Save
    output_path = Path(output_dir) / "pred_classification"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{image_name}.png"
    cv2.imwrite(str(output_file), img_with_boxes)

    return len(gt_boxes)


def process_image(
    image_path,
    label_file,
    detection_model,
    classification_model,
    class_names,
    transform,
    device,
    gt_crops_dir,
    output_dir,
    conf_threshold=0.25
):
    """Process single image - generate all 4 visualizations"""
    image_name = Path(image_path).stem

    print(f"\n[PROCESS] {image_name}")

    # 1. GT Detection
    n_gt_det = generate_gt_detection(image_path, label_file, output_dir)
    print(f"   [1/4] GT Detection: {n_gt_det} boxes")

    # 2. Predicted Detection
    n_pred_det = generate_pred_detection(image_path, detection_model, output_dir, conf_threshold)
    print(f"   [2/4] Pred Detection: {n_pred_det} boxes")

    # 3. GT Classification
    n_gt_cls = generate_gt_classification(image_path, label_file, gt_crops_dir, output_dir)
    print(f"   [3/4] GT Classification: {n_gt_cls} boxes")

    # 4. Predicted Classification (using GT boxes!)
    n_pred_cls = generate_pred_classification(
        image_path, label_file, gt_crops_dir,
        classification_model, class_names, transform, device, output_dir
    )
    print(f"   [4/4] Pred Classification: {n_pred_cls} boxes")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate detection and classification figures (4 outputs per image)'
    )
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
    parser.add_argument('--output', type=str, default='paper_figures',
                       help='Output base directory')
    parser.add_argument('--det-conf-threshold', type=float, default=0.25,
                       help='Detection confidence threshold')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum images to process (default: all)')

    args = parser.parse_args()

    print("="*80)
    print("DETECTION + CLASSIFICATION FIGURE GENERATION")
    print("="*80)
    print(f"Detection model: {args.detection_model}")
    print(f"Classification model: {args.classification_model}")
    print(f"Output base: {args.output}")
    print(f"\nOutput structure:")
    print(f"  {args.output}/gt_detection/")
    print(f"  {args.output}/pred_detection/")
    print(f"  {args.output}/gt_classification/")
    print(f"  {args.output}/pred_classification/")

    # Load models
    print("\n[LOAD] Loading YOLO detection model...")
    from ultralytics import YOLO
    detection_model = YOLO(args.detection_model)

    print("[LOAD] Loading classification model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get class names
    crops_test_dir = Path(args.gt_crops) / "crops" / "test"
    class_names = sorted([d.name for d in crops_test_dir.iterdir() if d.is_dir()])
    print(f"   Classes: {class_names}")

    # Load classification model
    checkpoint = torch.load(args.classification_model, map_location=device, weights_only=False)

    # Infer model architecture
    model_name = checkpoint.get('model_name', 'densenet121')
    print(f"   Loading architecture: {model_name}")

    if 'efficientnet_b1' in model_name.lower():
        classification_model = models.efficientnet_b1(weights=None)
        classification_model.classifier[1] = torch.nn.Linear(
            classification_model.classifier[1].in_features, len(class_names)
        )
    elif 'efficientnet_b2' in model_name.lower():
        classification_model = models.efficientnet_b2(weights=None)
        classification_model.classifier[1] = torch.nn.Linear(
            classification_model.classifier[1].in_features, len(class_names)
        )
    elif 'convnext' in model_name.lower():
        classification_model = models.convnext_tiny(weights=None)
        classification_model.classifier[2] = torch.nn.Linear(
            classification_model.classifier[2].in_features, len(class_names)
        )
    elif 'mobilenet' in model_name.lower():
        classification_model = models.mobilenet_v3_large(weights=None)
        classification_model.classifier[3] = torch.nn.Linear(
            classification_model.classifier[3].in_features, len(class_names)
        )
    elif 'resnet' in model_name.lower():
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
    image_files = sorted(
        list(test_images_dir.glob("*.jpg")) +
        list(test_images_dir.glob("*.JPG")) +
        list(test_images_dir.glob("*.png")) +
        list(test_images_dir.glob("*.PNG"))
    )

    if not image_files:
        print("[ERROR] No test images found!")
        return

    total_images = len(image_files)
    if args.max_images:
        image_files = image_files[:args.max_images]
        print(f"\n[FOUND] {total_images} total images, processing {len(image_files)}")
    else:
        print(f"\n[FOUND] {total_images} images, processing ALL")

    # Process images
    processed = 0
    failed = 0

    for img_path in image_files:
        label_file = test_labels_dir / f"{img_path.stem}.txt"

        if not label_file.exists():
            print(f"[SKIP] No label file for {img_path.name}")
            continue

        try:
            process_image(
                img_path,
                label_file,
                detection_model,
                classification_model,
                class_names,
                transform,
                device,
                args.gt_crops,
                args.output,
                args.det_conf_threshold
            )
            processed += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {img_path.name}: {e}")
            failed += 1

    print(f"\n{'='*80}")
    print(f"[SUMMARY]")
    print(f"  Processed: {processed} images")
    print(f"  Failed: {failed} images")
    print(f"  Total outputs: {processed * 4} images")
    print(f"\n[OUTPUT] {args.output}/")
    print(f"  ├── gt_detection/ ({processed} images)")
    print(f"  ├── pred_detection/ ({processed} images)")
    print(f"  ├── gt_classification/ ({processed} images)")
    print(f"  └── pred_classification/ ({processed} images)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
