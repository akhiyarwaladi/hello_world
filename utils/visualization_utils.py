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


def _rects_overlap(r1, r2, margin=2):
    """Check if two rectangles (x1,y1,x2,y2) overlap with optional margin."""
    return not (r1[2] + margin <= r2[0] or r2[2] + margin <= r1[0] or
                r1[3] + margin <= r2[1] or r2[3] + margin <= r1[1])


def _draw_arrow_line(img, pt1, pt2, color, thickness=2, scale=1.0):
    """Draw a line with a small arrowhead at pt2."""
    cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
    # Arrowhead
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = max(1, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length
    arrow_len = int(10 * scale)
    # Perpendicular
    px, py = -uy, ux
    tip = pt2
    left = (int(tip[0] - arrow_len * ux + arrow_len * 0.4 * px),
            int(tip[1] - arrow_len * uy + arrow_len * 0.4 * py))
    right = (int(tip[0] - arrow_len * ux - arrow_len * 0.4 * px),
             int(tip[1] - arrow_len * uy - arrow_len * 0.4 * py))
    cv2.fillPoly(img, [np.array([tip, left, right])], color, lineType=cv2.LINE_AA)


def draw_boxes(image: np.ndarray,
               boxes: List[List[int]],
               labels: List[str],
               colors: List[Tuple[int, int, int]],
               thickness: int = 4,
               font_scale: float = 0.9) -> np.ndarray:
    """
    Draw bounding boxes with labels on an image.
    Uses collision detection: overlapping labels are offset with an arrow
    pointing back to the box. Font size, thickness, and padding auto-scale
    based on image width (reference: 1400px) so labels appear the same
    size across different resolution images.

    Args:
        image: Input image (BGR format)
        boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        labels: List of labels for each box
        colors: List of colors (BGR) for each box
        thickness: Ignored (auto-scaled)
        font_scale: Ignored (auto-scaled)

    Returns:
        Image with drawn boxes
    """
    import math

    img_copy = image.copy()
    img_h, img_w = img_copy.shape[:2]

    # Auto-scale font, thickness and padding based on image size
    # Reference: 1400px wide image uses font_scale=1.4, thickness=5, pad=12
    REF_WIDTH = 1400
    scale_factor = img_w / REF_WIDTH
    font_scale = 1.1 * scale_factor
    thickness = max(3, int(5 * scale_factor + 0.5))
    font_thickness = max(2, int(2.5 * scale_factor + 0.5))
    pad = max(5, int(10 * scale_factor + 0.5))
    arrow_thickness = max(2, int(4 * scale_factor + 0.5))
    label_margin = max(3, int(4 * scale_factor + 0.5))  # margin between labels

    # Phase 1: Draw all bounding box rectangles
    int_boxes = []
    for box, color in zip(boxes, colors):
        x1, y1, x2, y2 = [int(c) for c in box]
        int_boxes.append((x1, y1, x2, y2))
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    # Phase 2: Compute label sizes
    label_info = []
    for label in labels:
        if not label:
            label_info.append(None)
        else:
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            label_info.append((tw, th, bl))

    # Phase 3: Place labels one by one, avoiding collisions
    # Strategy: 3-pass approach
    #   Pass 1: avoid both other labels AND bounding boxes (strict)
    #   Pass 2: avoid other labels only, allow overlap with boxes (relaxed)
    #   Pass 3: minimize overlap count (best-effort fallback)
    placed_rects = []  # list of (lx1, ly1, lx2, ly2) for all placed labels
    final_draw = [None] * len(labels)  # (lx1, ly1, lx2, ly2, text_pos_y)
    needs_arrow = [None] * len(labels)  # (arrow_start_pt, arrow_end_pt)

    def _check_labels_only(rect, margin=None):
        """Check if rect doesn't overlap any already-placed label."""
        m = label_margin if margin is None else margin
        for pr in placed_rects:
            if _rects_overlap(rect, pr, m):
                return False
        return True

    def _check_strict(rect, own_box_idx):
        """Check if rect doesn't overlap any placed label or any OTHER box."""
        if not _check_labels_only(rect):
            return False
        for bi, bo in enumerate(int_boxes):
            if bi == own_box_idx:
                continue  # skip own box
            if _rects_overlap(rect, bo, label_margin):
                return False
        return True

    def _count_label_overlaps(rect):
        """Count how many placed labels this rect overlaps."""
        count = 0
        for pr in placed_rects:
            if _rects_overlap(rect, pr, 0):
                count += 1
        return count

    def _clamp(ox, oy, lw, lh):
        ox = max(0, min(ox, img_w - lw))
        oy = max(0, min(oy, img_h - lh))
        return ox, oy

    def _get_candidate_positions(bx1, by1, bx2, by2, lw, lh):
        """Generate candidate label positions around a bounding box."""
        gap = label_margin + 2
        positions = [
            (bx1, by1 - lh - gap),          # above-left
            (bx1, by2 + gap),               # below-left
            (bx2 + gap, by1),               # right-top
            (bx1 - lw - gap, by1),          # left-top
            (bx2 - lw, by1 - lh - gap),     # above-right
            (bx2 + gap, by2 - lh),          # right-bottom
            (bx1 - lw - gap, by2 - lh),     # left-bottom
            ((bx1+bx2)//2 - lw//2, by1 - lh - gap),  # above-center
            ((bx1+bx2)//2 - lw//2, by2 + gap),        # below-center
        ]
        return [_clamp(x, y, lw, lh) for x, y in positions]

    def _get_radial_positions(bcx, bcy, lw, lh):
        """Generate radial candidate positions at increasing distances."""
        base_dists = [40, 65, 90, 115, 140, 170, 200, 240, 280, 330, 390]
        scaled_dists = [max(20, int(d * scale_factor)) for d in base_dists]
        positions = []
        for dist in scaled_dists:
            for angle_deg in range(0, 360, 10):  # 36 directions
                angle = math.radians(angle_deg)
                ox = int(bcx + dist * math.cos(angle) - lw / 2)
                oy = int(bcy + dist * math.sin(angle) - lh / 2)
                positions.append(_clamp(ox, oy, lw, lh))
        return positions

    for i in range(len(labels)):
        if label_info[i] is None:
            continue

        tw, th, bl = label_info[i]
        bx1, by1, bx2, by2 = int_boxes[i]
        lw = tw + 2 * pad
        lh = th + bl + pad
        bcx = (bx1 + bx2) // 2
        bcy = (by1 + by2) // 2

        def make_rect(ox, oy):
            return (ox, oy, ox + lw, oy + lh)

        # --- Pass 1: Strict (avoid labels + other boxes) ---
        direct_positions = _get_candidate_positions(bx1, by1, bx2, by2, lw, lh)
        placed = False
        for ox, oy in direct_positions:
            rect = make_rect(ox, oy)
            if _check_strict(rect, i):
                placed_rects.append(rect)
                tpy = oy + th + bl
                final_draw[i] = (ox, oy, ox + lw, oy + lh, tpy)
                placed = True
                break
        if placed:
            continue

        # Strict radial search
        radial_positions = _get_radial_positions(bcx, bcy, lw, lh)
        for ox, oy in radial_positions:
            rect = make_rect(ox, oy)
            if _check_strict(rect, i):
                placed_rects.append(rect)
                tpy = oy + th + bl
                final_draw[i] = (ox, oy, ox + lw, oy + lh, tpy)
                lcx, lcy = ox + lw // 2, oy + lh // 2
                cx = max(bx1, min(bcx, bx2))
                cy = max(by1, min(bcy, by2))
                needs_arrow[i] = ((lcx, lcy), (cx, cy))
                placed = True
                break
        if placed:
            continue

        # --- Pass 2: Relaxed (avoid labels only, allow box overlap) ---
        for ox, oy in direct_positions:
            rect = make_rect(ox, oy)
            if _check_labels_only(rect):
                placed_rects.append(rect)
                tpy = oy + th + bl
                final_draw[i] = (ox, oy, ox + lw, oy + lh, tpy)
                placed = True
                break
        if placed:
            continue

        for ox, oy in radial_positions:
            rect = make_rect(ox, oy)
            if _check_labels_only(rect):
                placed_rects.append(rect)
                tpy = oy + th + bl
                final_draw[i] = (ox, oy, ox + lw, oy + lh, tpy)
                lcx, lcy = ox + lw // 2, oy + lh // 2
                cx = max(bx1, min(bcx, bx2))
                cy = max(by1, min(bcy, by2))
                needs_arrow[i] = ((lcx, lcy), (cx, cy))
                placed = True
                break
        if placed:
            continue

        # --- Pass 3: Best-effort (minimize label overlaps) ---
        all_candidates = direct_positions + radial_positions
        best_pos = None
        best_overlaps = float('inf')
        for ox, oy in all_candidates:
            rect = make_rect(ox, oy)
            n_overlaps = _count_label_overlaps(rect)
            if n_overlaps < best_overlaps:
                best_overlaps = n_overlaps
                best_pos = (ox, oy)
                if n_overlaps == 0:
                    break

        if best_pos:
            ox, oy = best_pos
        else:
            ox, oy = _clamp(bx1, by1 - lh, lw, lh)

        rect = make_rect(ox, oy)
        placed_rects.append(rect)
        tpy = oy + th + bl
        final_draw[i] = (ox, oy, ox + lw, oy + lh, tpy)
        # Add arrow if label center is far from box center
        lcx, lcy = ox + lw // 2, oy + lh // 2
        dist_to_box = math.hypot(lcx - bcx, lcy - bcy)
        box_diag = math.hypot(bx2 - bx1, by2 - by1)
        if dist_to_box > box_diag * 0.8:
            cx = max(bx1, min(bcx, bx2))
            cy = max(by1, min(bcy, by2))
            needs_arrow[i] = ((lcx, lcy), (cx, cy))

    # Phase 4: Draw arrows (behind labels)
    for i in range(len(labels)):
        if needs_arrow[i] is not None:
            arr_start, arr_end = needs_arrow[i]
            _draw_arrow_line(img_copy, arr_start, arr_end, colors[i], arrow_thickness, scale_factor)

    # Phase 5: Draw label backgrounds and text
    for i in range(len(labels)):
        if final_draw[i] is None:
            continue

        label = labels[i]
        color = colors[i]
        lx1, ly1, lx2, ly2, tpy = final_draw[i]

        luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (0, 0, 0) if luminance > 150 else (255, 255, 255)

        cv2.rectangle(img_copy, (lx1, ly1), (lx2, ly2), color, -1, lineType=cv2.LINE_AA)
        cv2.putText(img_copy, label, (lx1 + pad, tpy),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

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


def save_publication_image(image: np.ndarray, output_path: Path, dpi: int = 300) -> None:
    """
    Save image with embedded DPI metadata for publication quality.

    Converts OpenCV BGR image to RGB PIL Image, sets DPI metadata,
    and saves as PNG for lossless quality.

    Args:
        image: OpenCV image (BGR format)
        output_path: Path to save the image
        dpi: DPI value to embed (default 300 for journal requirements)
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.save(str(output_path), format='PNG', dpi=(dpi, dpi))


# Color constants for visualization
COLORS = {
    'green': (0, 180, 0),      # Correct predictions (TP)
    'red': (0, 0, 220),        # Wrong predictions (FP)
    'blue': (255, 0, 0),       # Ground truth
    'yellow': (0, 190, 255),   # Warnings / FN - amber-orange, readable with black text
    'cyan': (255, 255, 0),     # Info
    'magenta': (255, 0, 255),  # Special
    'white': (255, 255, 255),
    'black': (0, 0, 0),
}
