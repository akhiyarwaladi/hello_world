#!/usr/bin/env python3
"""
Comprehensive crop quality investigation
Analyzes detection model predictions vs ground truth to identify crop generation issues
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd

def load_ground_truth_boxes(label_file):
    """Load ground truth bounding boxes from YOLO format"""
    boxes = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, x_center, y_center, width, height = map(float, parts[:5])
                    boxes.append([class_id, x_center, y_center, width, height])
    return boxes

def yolo_to_xyxy(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO format to xyxy coordinates"""
    x1 = int((x_center - width/2) * img_width)
    y1 = int((y_center - height/2) * img_height)
    x2 = int((x_center + width/2) * img_width)
    y2 = int((y_center + height/2) * img_height)
    return x1, y1, x2, y2

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in xyxy format"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def analyze_crop_quality():
    """Main analysis function"""

    # Configuration
    model_path = "results/exp_multi_pipeline_20250923_104712/detection/yolov10_detection/multi_pipeline_20250923_104712_yolo10_det/weights/best.pt"
    test_images_dir = "data/kaggle_pipeline_ready/test/images"
    test_labels_dir = "data/kaggle_pipeline_ready/test/labels"

    print("Starting comprehensive crop quality investigation...")
    print(f"Model: {model_path}")
    print(f"Test images: {test_images_dir}")
    print(f"Test labels: {test_labels_dir}")

    # Load model
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return

    print("Loading detection model...")
    model = YOLO(model_path)

    # Get test images
    image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} test images")

    results_data = []
    problem_cases = []
    visualization_data = []

    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file}")

        # Load image
        image_path = os.path.join(test_images_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            continue

        img_height, img_width = image.shape[:2]

        # Load ground truth
        label_file = os.path.join(test_labels_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        gt_boxes = load_ground_truth_boxes(label_file)

        if not gt_boxes:
            print(f"[WARNING] No ground truth boxes for {image_file}")
            continue

        # Run detection
        results = model(image_path, conf=0.25)

        if not results or len(results) == 0:
            print(f"[WARNING] No detections for {image_file}")
            continue

        pred_boxes = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
                    conf = boxes.conf[j].cpu().numpy()
                    pred_boxes.append([x1, y1, x2, y2, conf])

        # Convert ground truth to xyxy format
        gt_xyxy = []
        for gt_box in gt_boxes:
            class_id, x_center, y_center, width, height = gt_box
            x1, y1, x2, y2 = yolo_to_xyxy(x_center, y_center, width, height, img_width, img_height)
            gt_xyxy.append([x1, y1, x2, y2])

        # Calculate IoU matrix
        ious = []
        matched_pairs = []

        for pi, pred_box in enumerate(pred_boxes):
            pred_xyxy = pred_box[:4]
            best_iou = 0
            best_gt_idx = -1

            for gi, gt_box in enumerate(gt_xyxy):
                iou = calculate_iou(pred_xyxy, gt_box)
                ious.append(iou)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi

            if best_iou > 0.1:  # Minimum threshold for matching
                matched_pairs.append({
                    'pred_idx': pi,
                    'gt_idx': best_gt_idx,
                    'iou': best_iou,
                    'confidence': pred_box[4]
                })

        # Analyze this image
        image_stats = {
            'image': image_file,
            'img_width': img_width,
            'img_height': img_height,
            'gt_count': len(gt_boxes),
            'pred_count': len(pred_boxes),
            'matched_count': len(matched_pairs),
            'avg_iou': np.mean([p['iou'] for p in matched_pairs]) if matched_pairs else 0,
            'max_iou': max([p['iou'] for p in matched_pairs]) if matched_pairs else 0,
            'min_iou': min([p['iou'] for p in matched_pairs]) if matched_pairs else 0,
            'precision': len(matched_pairs) / len(pred_boxes) if pred_boxes else 0,
            'recall': len(matched_pairs) / len(gt_boxes) if gt_boxes else 0
        }

        results_data.append(image_stats)

        # Identify problem cases
        if image_stats['avg_iou'] < 0.5 or image_stats['recall'] < 0.5:
            problem_cases.append({
                'image': image_file,
                'issue': 'Low IoU or Recall',
                'avg_iou': float(image_stats['avg_iou']),
                'recall': float(image_stats['recall']),
                'gt_count': int(len(gt_boxes)),
                'pred_count': int(len(pred_boxes))
            })

        # Store visualization data for top 5 problem cases
        if len(problem_cases) <= 5:
            visualization_data.append({
                'image_path': image_path,
                'image': image,
                'gt_boxes': gt_xyxy,
                'pred_boxes': pred_boxes,
                'matched_pairs': matched_pairs,
                'stats': image_stats
            })

    # Generate comprehensive report
    df = pd.DataFrame(results_data)

    print("\n" + "="*80)
    print("COMPREHENSIVE CROP QUALITY ANALYSIS RESULTS")
    print("="*80)

    print(f"\nOVERALL STATISTICS:")
    print(f"  Total images analyzed: {len(df)}")
    print(f"  Total ground truth objects: {df['gt_count'].sum()}")
    print(f"  Total predicted objects: {df['pred_count'].sum()}")
    print(f"  Total matched pairs: {df['matched_count'].sum()}")

    print(f"\nDETECTION PERFORMANCE:")
    print(f"  Overall Precision: {df['matched_count'].sum() / df['pred_count'].sum():.3f}")
    print(f"  Overall Recall: {df['matched_count'].sum() / df['gt_count'].sum():.3f}")
    print(f"  Mean IoU: {df['avg_iou'].mean():.3f}")
    print(f"  Images with avg IoU > 0.5: {(df['avg_iou'] > 0.5).sum()}/{len(df)} ({(df['avg_iou'] > 0.5).mean()*100:.1f}%)")
    print(f"  Images with avg IoU > 0.7: {(df['avg_iou'] > 0.7).sum()}/{len(df)} ({(df['avg_iou'] > 0.7).mean()*100:.1f}%)")

    print(f"\n[WARNING] PROBLEM CASES IDENTIFIED: {len(problem_cases)}")
    for i, case in enumerate(problem_cases[:10]):  # Show top 10
        print(f"  {i+1}. {case['image']}: IoU={case['avg_iou']:.3f}, Recall={case['recall']:.3f}, GT={case['gt_count']}, Pred={case['pred_count']}")

    # Save detailed results
    output_dir = "crop_quality_analysis"
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(f"{output_dir}/detailed_results.csv", index=False)

    with open(f"{output_dir}/problem_cases.json", 'w') as f:
        json.dump(problem_cases, f, indent=2)

    # Create visualizations for problem cases
    print(f"\nCreating visualizations for problem cases...")

    for i, vis_data in enumerate(visualization_data):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Display image
        image_rgb = cv2.cvtColor(vis_data['image'], cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)

        # Draw ground truth boxes (green)
        for gt_box in vis_data['gt_boxes']:
            x1, y1, x2, y2 = gt_box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='green', facecolor='none', label='Ground Truth')
            ax.add_patch(rect)

        # Draw predicted boxes (red)
        for pred_box in vis_data['pred_boxes']:
            x1, y1, x2, y2, conf = pred_box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='red', facecolor='none', alpha=0.7, label=f'Pred (conf={conf:.2f})')
            ax.add_patch(rect)

        # Add statistics text
        stats = vis_data['stats']
        info_text = f"Image: {stats['image']}\nAvg IoU: {stats['avg_iou']:.3f}\nPrecision: {stats['precision']:.3f}\nRecall: {stats['recall']:.3f}\nGT: {stats['gt_count']}, Pred: {stats['pred_count']}"
        ax.text(10, 30, info_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(f"Problem Case {i+1}: {stats['image']}")
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/problem_case_{i+1}_{stats['image']}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n[SUCCESS] Analysis complete! Results saved to '{output_dir}/' directory")

    # Specific crop quality insights
    print(f"\nCROP QUALITY INSIGHTS:")

    low_iou_images = df[df['avg_iou'] < 0.5]
    if len(low_iou_images) > 0:
        print(f"  [WARNING] {len(low_iou_images)} images have low average IoU (<0.5)")
        print(f"     This suggests detection accuracy issues that affect crop quality")

    low_recall_images = df[df['recall'] < 0.7]
    if len(low_recall_images) > 0:
        print(f"  [WARNING] {len(low_recall_images)} images have low recall (<0.7)")
        print(f"     Missing detections = missing crops in classification dataset")

    over_detection = df[df['pred_count'] > df['gt_count'] * 1.5]
    if len(over_detection) > 0:
        print(f"  [WARNING] {len(over_detection)} images have significant over-detection")
        print(f"     False positives = noisy/incorrect crops in classification dataset")

    print(f"\nRECOMMENDATIONS:")

    avg_precision = df['matched_count'].sum() / df['pred_count'].sum()
    avg_recall = df['matched_count'].sum() / df['gt_count'].sum()

    if avg_precision < 0.8:
        print(f"  Consider increasing confidence threshold (currently 0.25) to reduce false positives")

    if avg_recall < 0.8:
        print(f"  Consider decreasing confidence threshold or training with more data to improve recall")

    if df['avg_iou'].mean() < 0.6:
        print(f"  Detection localization needs improvement - consider more training epochs or data augmentation")

    return results_data, problem_cases

if __name__ == "__main__":
    analyze_crop_quality()