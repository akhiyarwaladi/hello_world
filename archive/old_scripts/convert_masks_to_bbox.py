#!/usr/bin/env python3
"""
Convert MP-IDB segmentation masks to YOLO bounding boxes
Extract bounding boxes from binary masks for all 4 species
"""

import cv2
import numpy as np
import csv
from pathlib import Path
from typing import List, Tuple

MP_IDB_ROOT = Path("data/raw/mp_idb")
SPECIES = ["Falciparum", "Vivax", "Malariae", "Ovale"]

def extract_bounding_boxes_from_mask(mask_path: Path) -> List[Tuple[float, float, float, float]]:
    """Extract YOLO format bounding boxes from binary mask"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    bboxes = []

    for contour in contours:
        # Get bounding rectangle
        x, y, box_w, box_h = cv2.boundingRect(contour)

        # Convert to YOLO format (normalized center coordinates + width/height)
        center_x = (x + box_w / 2) / w
        center_y = (y + box_h / 2) / h
        norm_w = box_w / w
        norm_h = box_h / h

        # Only include reasonable sized boxes
        if norm_w > 0.01 and norm_h > 0.01:
            bboxes.append((center_x, center_y, norm_w, norm_h))

    return bboxes

def convert_species_masks_to_csv(species: str):
    """Convert all masks for a species to CSV with bounding boxes"""
    print(f"\n🔄 Processing {species}...")

    species_dir = MP_IDB_ROOT / species
    gt_dir = species_dir / "gt"
    img_dir = species_dir / "img"

    if not gt_dir.exists():
        print(f"❌ Ground truth directory not found: {gt_dir}")
        return

    # Create CSV file
    csv_path = species_dir / f"mp-idb-{species.lower()}-from-masks.csv"

    results = []
    mask_files = list(gt_dir.glob("*.jpg"))

    print(f"Found {len(mask_files)} mask files")

    for mask_path in mask_files:
        # Find corresponding original image
        img_path = img_dir / mask_path.name
        if not img_path.exists():
            print(f"⚠️  Original image not found: {img_path}")
            continue

        # Extract bounding boxes from mask
        bboxes = extract_bounding_boxes_from_mask(mask_path)

        if bboxes:
            # Add to results (use relative path like original CSV)
            relative_img_path = f"img/{mask_path.name}"
            for bbox in bboxes:
                results.append({
                    'filename': relative_img_path,
                    'center_x': bbox[0],
                    'center_y': bbox[1],
                    'width': bbox[2],
                    'height': bbox[3],
                    'species': species.lower()
                })

    # Write CSV
    if results:
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['filename', 'center_x', 'center_y', 'width', 'height', 'species']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"✅ Created {csv_path} with {len(results)} bounding boxes")
    else:
        print(f"❌ No bounding boxes found for {species}")

def main():
    print("🚀 CONVERTING MP-IDB SEGMENTATION MASKS TO BOUNDING BOXES")
    print("=" * 60)

    if not MP_IDB_ROOT.exists():
        print(f"❌ MP-IDB root not found: {MP_IDB_ROOT}")
        return

    total_boxes = 0

    for species in SPECIES:
        convert_species_masks_to_csv(species)

    print(f"\n📊 SUMMARY:")
    print("Now we have bounding box data for all 4 species!")
    print("\nNext steps:")
    print("1. Update crop_detections.py to use mask-derived bounding boxes")
    print("2. Create proper multi-species classification dataset")
    print("3. Re-train with 6-class system (4 species + Mixed + Uninfected)")

if __name__ == "__main__":
    main()