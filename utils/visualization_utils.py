#!/usr/bin/env python3
"""
Centralized Visualization Utilities for Malaria Detection Pipeline

This module provides common visualization functions used across all visualization scripts.
Import from here instead of defining locally in each script.

Usage:
    from utils.visualization_utils import (
        draw_boxes, yolo_to_absolute, load_gt_annotations,
        load_gt_class_mapping, crop_and_classify
    )
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import torch
from PIL import Image


def draw_boxes(image: np.ndarray,
               boxes: List[List[int]],
               labels: List[str],
               colors: List[Tuple[int, int, int]],
               thickness: int = 4,
               font_scale: float = 0.9) -> np.ndarray:
    """
    Draw bounding boxes with labels on an image

    Args:
        image: Input image (BGR format)
        boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        labels: List of labels for each box
        colors: List of colors (BGR) for each box
        thickness: Line thickness for boxes
        font_scale: Font scale for labels

    Returns:
        Image with drawn boxes
    """
    img_copy = image.copy()

    for box, label, color in zip(boxes, labels, colors):
        x1, y1, x2, y2 = [int(c) for c in box]

        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

        # Draw label if provided
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

            cv2.rectangle(img_copy, (x1 - 8, text_y_top),
                         (x1 + text_width + 8, text_y_bottom), color, -1)
            cv2.putText(img_copy, label, (x1, text_pos_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    return img_copy


def yolo_to_absolute(box: List[float],
                     img_width: int,
                     img_height: int) -> List[int]:
    """
    Convert YOLO format (x_center, y_center, width, height) to absolute coordinates

    Args:
        box: YOLO format box [x_center, y_center, width, height] (normalized 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Absolute coordinates [x1, y1, x2, y2]
    """
    x_center, y_center, w, h = box

    x1 = int((x_center - w/2) * img_width)
    y1 = int((y_center - h/2) * img_height)
    x2 = int((x_center + w/2) * img_width)
    y2 = int((y_center + h/2) * img_height)

    return [x1, y1, x2, y2]


def load_gt_annotations(label_file: Path) -> List[Dict[str, Any]]:
    """
    Load ground truth annotations from YOLO format label file

    Args:
        label_file: Path to YOLO format label file (.txt)

    Returns:
        List of annotation dictionaries with 'class_id' and 'box' keys
    """
    annotations = []

    if not Path(label_file).exists():
        return annotations

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                box = [float(x) for x in parts[1:5]]
                annotations.append({
                    'class_id': class_id,
                    'box': box  # YOLO format: x_center, y_center, w, h
                })

    return annotations


def load_gt_class_mapping(crops_dir: Path, image_name: str) -> Optional[str]:
    """
    Find ground truth class for an image from crops directory structure

    Args:
        crops_dir: Path to ground truth crops directory
        image_name: Name of the image (without extension)

    Returns:
        Class name if found, None otherwise
    """
    crops_path = Path(crops_dir)

    # Search in train, val, test splits
    for split in ['train', 'val', 'test']:
        split_path = crops_path / split
        if not split_path.exists():
            continue

        # Search in each class folder
        for class_folder in split_path.iterdir():
            if class_folder.is_dir():
                # Look for matching image
                for img_file in class_folder.glob(f"{image_name}*"):
                    return class_folder.name

    return None


def crop_and_classify(image: np.ndarray,
                      box_abs: List[int],
                      classification_model: torch.nn.Module,
                      transform: Any,
                      class_names: List[str],
                      device: torch.device) -> Tuple[str, float]:
    """
    Crop region from image and classify using model

    Args:
        image: Input image (BGR format)
        box_abs: Absolute coordinates [x1, y1, x2, y2]
        classification_model: PyTorch classification model
        transform: Image transform to apply
        class_names: List of class names
        device: PyTorch device

    Returns:
        Tuple of (predicted_class_name, confidence)
    """
    x1, y1, x2, y2 = [max(0, int(c)) for c in box_abs]

    # Ensure valid crop
    h, w = image.shape[:2]
    x2 = min(x2, w)
    y2 = min(y2, h)

    if x2 <= x1 or y2 <= y1:
        return "unknown", 0.0

    # Crop and convert to PIL
    crop = image[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)

    # Transform and predict
    input_tensor = transform(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classification_model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    pred_class = class_names[predicted.item()]
    conf = confidence.item()

    return pred_class, conf


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union between two boxes

    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]

    Returns:
        IoU value (0.0 to 1.0)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def match_predictions_to_gt(pred_boxes: List[List[int]],
                            gt_boxes: List[List[int]],
                            iou_threshold: float = 0.5) -> List[Tuple[int, int, float]]:
    """
    Match predicted boxes to ground truth boxes based on IoU

    Args:
        pred_boxes: List of predicted boxes [[x1,y1,x2,y2], ...]
        gt_boxes: List of ground truth boxes
        iou_threshold: Minimum IoU for a match

    Returns:
        List of (pred_idx, gt_idx, iou) tuples for matches
    """
    matches = []
    used_gt = set()

    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matches.append((pred_idx, best_gt_idx, best_iou))
            used_gt.add(best_gt_idx)

    return matches


def clean_output_directory(output_dir: Path, backup: bool = True) -> None:
    """
    Clean output directory, optionally backing up existing content

    Args:
        output_dir: Directory to clean
        backup: Whether to backup existing content
    """
    import shutil
    from datetime import datetime

    output_path = Path(output_dir)

    if output_path.exists() and any(output_path.iterdir()):
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = output_path.parent / f"{output_path.name}_backup_{timestamp}"
            shutil.move(str(output_path), str(backup_path))
            print(f"Backed up existing output to: {backup_path}")
        else:
            shutil.rmtree(str(output_path))

    output_path.mkdir(parents=True, exist_ok=True)


# Color constants for visualization
COLORS = {
    'green': (0, 180, 0),      # Correct predictions
    'red': (0, 0, 255),        # Wrong predictions
    'blue': (255, 0, 0),       # Ground truth
    'yellow': (0, 255, 255),   # Warnings
    'cyan': (255, 255, 0),     # Info
    'magenta': (255, 0, 255),  # Special
    'white': (255, 255, 255),
    'black': (0, 0, 0),
}
