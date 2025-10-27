#!/usr/bin/env python3
"""
Enhanced detection visualization with CSV metadata export
Helps select best images for paper by showing prediction accuracy
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import shutil
from datetime import datetime


def yolo_to_absolute(box_yolo, img_width, img_height):
    """Convert YOLO format (class, x_center, y_center, w, h) to absolute coordinates"""
    if len(box_yolo) < 5:
        return None
    x_center, y_center, w, h = box_yolo[1:5]
    x1 = int((x_center - w / 2) * img_width)
    y1 = int((y_center - h / 2) * img_height)
    x2 = int((x_center + w / 2) * img_width)
    y2 = int((y_center + h / 2) * img_height)
    return [x1, y1, x2, y2]


def load_gt_annotations(label_file, img_width, img_height):
    """Load ground truth annotations from YOLO format"""
    annotations = []
    if not Path(label_file).exists():
        return annotations

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                box_yolo = [class_id] + [float(x) for x in parts[1:5]]
                box_abs = yolo_to_absolute(box_yolo, img_width, img_height)
                if box_abs:
                    annotations.append({
                        'class_id': class_id,
                        'box': box_abs
                    })
    return annotations


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Match predictions to ground truth using IoU
    Returns: matched_pairs, false_positives, false_negatives
    """
    matched_pairs = []
    unmatched_preds = list(range(len(pred_boxes)))
    unmatched_gts = list(range(len(gt_boxes)))

    # Find best matches
    for pred_idx in range(len(pred_boxes)):
        best_iou = 0
        best_gt_idx = None

        for gt_idx in unmatched_gts:
            iou = calculate_iou(pred_boxes[pred_idx]['box'], gt_boxes[gt_idx]['box'])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None:
            matched_pairs.append({
                'pred_idx': pred_idx,
                'gt_idx': best_gt_idx,
                'iou': best_iou
            })
            unmatched_preds.remove(pred_idx)
            unmatched_gts.remove(best_gt_idx)

    false_positives = unmatched_preds
    false_negatives = unmatched_gts

    return matched_pairs, false_positives, false_negatives


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


def generate_detection_visualization_with_metadata(
    image_path,
    label_path,
    detection_model,
    output_dir,
    conf_threshold=0.25,
    iou_threshold=0.5
):
    """Generate detection visualization and return metadata"""

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    image_name = Path(image_path).stem
    img_height, img_width = image.shape[:2]

    # Load ground truth
    gt_boxes = load_gt_annotations(label_path, img_width, img_height)

    # Run detection
    results = detection_model(str(image_path), conf=conf_threshold, verbose=False)[0]

    pred_boxes = []
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            box_coords = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            pred_boxes.append({
                'box': [int(c) for c in box_coords],
                'confidence': float(confidence),
                'class_id': class_id
            })

    # Match predictions to ground truth
    matched_pairs, false_positives, false_negatives = match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold)

    # Calculate statistics
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    n_correct = len(matched_pairs)
    n_fp = len(false_positives)
    n_fn = len(false_negatives)

    # Determine status
    if n_gt == 0 and n_pred == 0:
        status = "Empty (no GT, no predictions)"
    elif n_gt == n_pred == n_correct:
        status = "Perfect"
    elif n_fn > 0 and n_fp > 0:
        status = "Mixed (FP + FN)"
    elif n_fn > 0:
        status = "Missing detections (FN)"
    elif n_fp > 0:
        status = "False positives (FP)"
    elif n_correct > 0:
        status = "Partial correct"
    else:
        status = "Unknown"

    # Average confidence
    avg_confidence = np.mean([p['confidence'] for p in pred_boxes]) if pred_boxes else 0.0

    # Determine if good for paper
    # Perfect matches with high confidence are best
    paper_score = 0
    if status == "Perfect" and avg_confidence > 0.8:
        paper_score = 10  # Excellent
    elif status == "Perfect":
        paper_score = 9   # Good
    elif n_correct > 0 and n_fp == 0 and avg_confidence > 0.7:
        paper_score = 7   # Acceptable
    elif status == "Mixed (FP + FN)":
        paper_score = 5   # Challenging (could be useful to show limitations)
    elif status == "Missing detections (FN)":
        paper_score = 4   # Show missed detections
    elif status == "False positives (FP)":
        paper_score = 3   # Show false alarms
    else:
        paper_score = 2   # Not ideal

    # Draw boxes with color coding: Green (TP), Red (FP), Yellow (FN)
    # True Positives (matched predictions) - GREEN (darker for white text visibility)
    tp_coords = [pred_boxes[pair['pred_idx']]['box'] for pair in matched_pairs]
    tp_labels = [f"TP {pred_boxes[pair['pred_idx']]['confidence']:.2f}" for pair in matched_pairs]
    tp_colors = [(0, 180, 0)] * len(matched_pairs)  # Dark green (readable white text)

    # False Positives (unmatched predictions) - RED (darker for white text visibility)
    fp_coords = [pred_boxes[idx]['box'] for idx in false_positives]
    fp_labels = [f"FP {pred_boxes[idx]['confidence']:.2f}" for idx in false_positives]
    fp_colors = [(0, 0, 200)] * len(false_positives)  # Dark red (readable white text)

    # False Negatives (unmatched GT) - YELLOW/ORANGE (darker for white text visibility)
    fn_coords = [gt_boxes[idx]['box'] for idx in false_negatives]
    fn_labels = ["FN (missed)"] * len(false_negatives)
    fn_colors = [(0, 180, 180)] * len(false_negatives)  # Dark yellow/cyan (readable white text)

    # Combine all boxes
    all_coords = tp_coords + fp_coords + fn_coords
    all_labels = tp_labels + fp_labels + fn_labels
    all_colors = tp_colors + fp_colors + fn_colors

    img_with_boxes = draw_boxes(image, all_coords, all_labels, all_colors)

    # Save image
    output_file = Path(output_dir) / f"{image_name}.png"
    cv2.imwrite(str(output_file), img_with_boxes)

    # Return metadata
    metadata = {
        'image_name': image_name,
        'image_file': str(output_file),
        'n_gt_boxes': n_gt,
        'n_pred_boxes': n_pred,
        'n_correct_matches': n_correct,
        'n_false_positives': n_fp,
        'n_false_negatives': n_fn,
        'avg_confidence': f"{avg_confidence:.3f}",
        'status': status,
        'paper_score': paper_score,
        'recommendation': 'Best for paper' if paper_score >= 9 else ('Good for paper' if paper_score >= 7 else ('Challenging case' if paper_score == 5 else 'Not recommended'))
    }

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Generate detection visualizations with CSV metadata')
    parser.add_argument('--detection-model', type=str, required=True)
    parser.add_argument('--test-images', type=str, required=True)
    parser.add_argument('--test-labels', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--det-conf-threshold', type=float, default=0.25)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--max-images', type=int, default=None)
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

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load detection model
    print(f"[LOAD] Loading detection model: {args.detection_model}")
    detection_model = YOLO(args.detection_model)

    # Get test images
    test_images_dir = Path(args.test_images)
    test_labels_dir = Path(args.test_labels)

    image_files = sorted(
        list(test_images_dir.glob("*.jpg")) +
        list(test_images_dir.glob("*.JPG")) +
        list(test_images_dir.glob("*.png")) +
        list(test_images_dir.glob("*.PNG"))
    )

    if args.max_images:
        image_files = image_files[:args.max_images]

    # Process images and collect metadata
    print(f"[PROCESS] Processing {len(image_files)} images...")
    all_metadata = []
    processed = 0

    for img_path in image_files:
        label_path = test_labels_dir / f"{img_path.stem}.txt"

        try:
            metadata = generate_detection_visualization_with_metadata(
                img_path,
                label_path,
                detection_model,
                output_dir,
                args.det_conf_threshold,
                args.iou_threshold
            )

            if metadata:
                all_metadata.append(metadata)
                processed += 1

                # Print status for interesting cases
                if metadata['paper_score'] >= 9:
                    print(f"   ✓ [BEST] {metadata['image_name']}: {metadata['status']}")
                elif metadata['paper_score'] == 5:
                    print(f"   ! [CHALLENGING] {metadata['image_name']}: {metadata['status']}")

        except Exception as e:
            print(f"   [ERROR] {img_path.name}: {e}")

    # Save metadata to CSV
    if all_metadata:
        csv_file = output_dir / "detection_metadata.csv"
        df = pd.DataFrame(all_metadata)

        # Sort by paper_score descending (best images first)
        df = df.sort_values('paper_score', ascending=False)
        df.to_csv(csv_file, index=False)

        print(f"\n[SUCCESS] Processed {processed} images")
        print(f"[CSV] Metadata saved to: {csv_file}")
        print(f"\n[STATS] Detection Summary:")
        print(f"   Perfect detections: {len(df[df['status'] == 'Perfect'])}")
        print(f"   Best for paper (score ≥9): {len(df[df['paper_score'] >= 9])}")
        print(f"   Good for paper (score ≥7): {len(df[df['paper_score'] >= 7])}")
        print(f"   Challenging cases (score =5): {len(df[df['paper_score'] == 5])}")

        # Show top 5 best images
        print(f"\n[TOP 5] Best images for paper:")
        for idx, row in df.head(5).iterrows():
            print(f"   {idx+1}. {row['image_name']}: {row['status']} (conf={row['avg_confidence']}, score={row['paper_score']})")

    return 0


if __name__ == "__main__":
    exit(main())
