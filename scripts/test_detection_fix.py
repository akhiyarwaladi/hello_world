#!/usr/bin/env python3
"""
Test the corrected MP-IDB detection parsing
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def test_detection_fix():
    """Test the corrected MP-IDB detection parsing"""

    # Test image
    test_img = '1305121398-0001-R_S.jpg'

    # Load CSV annotations
    csv_path = 'data/raw/mp_idb/Falciparum/mp-idb-falciparum.csv'
    df = pd.read_csv(csv_path)
    img_annotations = df[df['filename'] == test_img]

    # Load image and ground truth
    img_path = Path('data/raw/mp_idb/Falciparum/img') / test_img
    gt_path = Path('data/raw/mp_idb/Falciparum/gt') / test_img

    img = cv2.imread(str(img_path))
    gt_mask = cv2.imread(str(gt_path), 0)

    if img is None or gt_mask is None:
        print("❌ Cannot load image or mask")
        return

    print(f"Testing {test_img}")
    print(f"Image dimensions: {img.shape}")
    print(f"CSV annotations: {len(img_annotations)}")

    # Find connected components
    _, labels = cv2.connectedComponents(gt_mask)

    # Get the largest components
    component_boxes = []
    for label in range(1, labels.max() + 1):
        mask = (labels == label).astype(np.uint8) * 255
        area = np.sum(labels == label)

        if area < 100:  # Filter noise
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            component_boxes.append({
                'bbox': (x, y, x + w, y + h),
                'area': area,
                'center': (x + w//2, y + h//2)
            })

    # Sort by area (largest first)
    component_boxes.sort(key=lambda x: x['area'], reverse=True)

    # Create visualization
    vis_img = img.copy()

    # Draw ground truth parasites in green
    vis_img[gt_mask > 128] = [0, 255, 0]

    # Draw bounding boxes for the top 3 components
    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]  # Yellow, Magenta, Cyan

    print("\nTop 3 parasites with bounding boxes:")
    for i, component in enumerate(component_boxes[:3]):
        if i >= len(img_annotations):
            break

        x1, y1, x2, y2 = component['bbox']
        color = colors[i % len(colors)]

        # Draw bounding box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)

        # Add label
        parasite_type = img_annotations.iloc[i]['parasite_type']
        cv2.putText(vis_img, f"{parasite_type} {i+1}",
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        print(f"  {i+1}. {parasite_type}: bbox=({x1},{y1},{x2},{y2}), area={component['area']}")

    # Save visualization
    output_dir = Path('results/debug_boxes_fixed')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{Path(test_img).stem}_fixed.jpg"
    success = cv2.imwrite(str(output_path), vis_img)

    if success:
        print(f"\n✅ Corrected visualization saved to: {output_path}")
    else:
        print(f"❌ Failed to save visualization")

    # Compare with CSV coordinates
    print(f"\nCSV vs Ground Truth comparison:")
    for i, (_, row) in enumerate(img_annotations.iterrows()):
        if i < len(component_boxes):
            component = component_boxes[i]
            x1, y1, x2, y2 = component['bbox']
            csv_x = (row['xmin'] + row['xmax']) / 2
            csv_y = (row['ymin'] + row['ymax']) / 2
            gt_center = component['center']

            print(f"  {row['parasite_type']}: CSV_center=({csv_x:.0f},{csv_y:.0f}), GT_bbox=({x1},{y1},{x2},{y2}), GT_center={gt_center}")

if __name__ == "__main__":
    test_detection_fix()