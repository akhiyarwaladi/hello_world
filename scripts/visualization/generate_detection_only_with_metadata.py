#!/usr/bin/env python3
"""
Enhanced detection visualization with CSV metadata export
Helps select best images for paper by showing prediction accuracy
"""

import sys
import argparse
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.visualization_utils import (
    draw_boxes, yolo_to_absolute, load_gt_annotations,
    calculate_iou, clean_output_directory, save_publication_image, COLORS
)


def load_gt_annotations_absolute(label_file, img_width, img_height):
    """Load ground truth annotations and convert to absolute coordinates"""
    annotations = load_gt_annotations(label_file)
    result = []
    for ann in annotations:
        box_abs = yolo_to_absolute(ann['box'], img_width, img_height)
        result.append({
            'class_id': ann['class_id'],
            'box': box_abs
        })
    return result


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

    # Load ground truth (converted to absolute coordinates)
    gt_boxes = load_gt_annotations_absolute(label_path, img_width, img_height)

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
    # True Positives (matched predictions) - GREEN
    tp_coords = [pred_boxes[pair['pred_idx']]['box'] for pair in matched_pairs]
    tp_labels = [f"TP {pred_boxes[pair['pred_idx']]['confidence']:.2f}" for pair in matched_pairs]
    tp_colors = [COLORS['green']] * len(matched_pairs)

    # False Positives (unmatched predictions) - RED
    fp_coords = [pred_boxes[idx]['box'] for idx in false_positives]
    fp_labels = [f"FP {pred_boxes[idx]['confidence']:.2f}" for idx in false_positives]
    fp_colors = [COLORS['red']] * len(false_positives)

    # False Negatives (unmatched GT) - YELLOW
    fn_coords = [gt_boxes[idx]['box'] for idx in false_negatives]
    fn_labels = ["FN (missed)"] * len(false_negatives)
    fn_colors = [COLORS['yellow']] * len(false_negatives)

    # Combine all boxes
    all_coords = tp_coords + fp_coords + fn_coords
    all_labels = tp_labels + fp_labels + fn_labels
    all_colors = tp_colors + fp_colors + fn_colors

    img_with_boxes = draw_boxes(image, all_coords, all_labels, all_colors)

    # Save image with 300 DPI metadata for publication quality
    output_file = Path(output_dir) / f"{image_name}.png"
    save_publication_image(img_with_boxes, output_file, dpi=300)

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
