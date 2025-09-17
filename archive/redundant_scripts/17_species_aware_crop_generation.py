#!/usr/bin/env python3
"""
Species-Aware Crop Generation from Detection Results
Combines detection model results with ground truth species annotations
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
from ultralytics import YOLO

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.results_manager import ResultsManager

def parse_yolo_annotation(annotation_file, image_width, image_height):
    """Parse YOLO format annotation file with species info"""
    boxes = []
    if not annotation_file.exists():
        return boxes

    # Species mapping from class_id to species name
    species_map = {
        0: "P_falciparum",
        1: "P_vivax",
        2: "P_malariae",
        3: "P_ovale",
        4: "Mixed_infection",
        5: "Uninfected"
    }

    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert YOLO format to pixel coordinates
                x_center_px = x_center * image_width
                y_center_px = y_center * image_height
                width_px = width * image_width
                height_px = height * image_height

                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)

                boxes.append({
                    'class_id': class_id,
                    'species': species_map.get(class_id, "unknown"),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'gt_box': True
                })

    return boxes

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])

    if x2 <= x1 or y2 <= y1:
        return 0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def match_detection_with_ground_truth(detection_boxes, gt_boxes, iou_threshold=0.5):
    """Match detection results with ground truth species annotations"""
    matched_crops = []

    for det_box in detection_boxes:
        best_match = None
        best_iou = 0

        # Find best matching ground truth box
        for gt_box in gt_boxes:
            iou = calculate_iou(det_box, gt_box)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = gt_box

        if best_match:
            # Create matched crop with species info
            matched_crop = {
                'x1': det_box['x1'],
                'y1': det_box['y1'],
                'x2': det_box['x2'],
                'y2': det_box['y2'],
                'detection_confidence': det_box['confidence'],
                'species': best_match['species'],
                'class_id': best_match['class_id'],
                'iou': best_iou,
                'source': 'detection_matched'
            }
            matched_crops.append(matched_crop)

    return matched_crops

def generate_species_aware_crops(model_path, input_dir, output_dir, confidence=0.25, crop_size=128, iou_threshold=0.5):
    """Generate crops from detection results with species information"""

    # Load detection model
    model = YOLO(model_path)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    crops_dir = output_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Create species-specific subdirectories
    species_dirs = {}
    for split in ['train', 'val', 'test']:
        split_path = input_path / split
        if split_path.exists():
            (crops_dir / split).mkdir(exist_ok=True)
            for species in ["P_falciparum", "P_vivax", "P_malariae", "P_ovale", "Mixed_infection", "Uninfected"]:
                species_dir = crops_dir / split / species
                species_dir.mkdir(exist_ok=True)
                species_dirs[f"{split}_{species}"] = species_dir

    # Process each split
    metadata = []
    processed_count = 0
    crop_count = 0
    species_count = {}

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

    for split in ['train', 'val', 'test']:
        images_path = input_path / split / "images"
        labels_path = input_path / split / "labels"

        if not images_path.exists() or not labels_path.exists():
            continue

        print(f"Processing {split} split...")

        # Get all images in this split
        split_images = []
        for ext in image_extensions:
            split_images.extend(images_path.glob(ext))

        for image_path in tqdm(split_images, desc=f"Processing {split}"):
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                image_height, image_width = image.shape[:2]

                # Get detection results
                results = model.predict(str(image_path), conf=confidence, verbose=False)
                detection_boxes = []

                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()

                        for box, conf in zip(boxes, confidences):
                            x1, y1, x2, y2 = map(int, box)
                            detection_boxes.append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'confidence': float(conf)
                            })

                # Get ground truth annotations
                annotation_file = labels_path / f"{image_path.stem}.txt"
                gt_boxes = parse_yolo_annotation(annotation_file, image_width, image_height)

                # Match detection with ground truth
                matched_crops = match_detection_with_ground_truth(detection_boxes, gt_boxes, iou_threshold)

                # Generate crops
                for i, crop_data in enumerate(matched_crops):
                    if crop_data['species'] == "Uninfected":
                        continue  # Skip uninfected samples

                    # Extract crop
                    x1, y1, x2, y2 = crop_data['x1'], crop_data['y1'], crop_data['x2'], crop_data['y2']

                    # Create square crop around center
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    half_size = crop_size // 2
                    crop_x1 = max(0, center_x - half_size)
                    crop_y1 = max(0, center_y - half_size)
                    crop_x2 = min(image_width, center_x + half_size)
                    crop_y2 = min(image_height, center_y + half_size)

                    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

                    # Resize to exact crop size if needed
                    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                        crop = cv2.resize(crop, (crop_size, crop_size))

                    # Save crop to species-specific directory
                    species = crop_data['species']
                    crop_filename = f"{image_path.stem}_det_crop_{i:03d}.jpg"
                    species_dir = crops_dir / split / species
                    crop_output_path = species_dir / crop_filename

                    cv2.imwrite(str(crop_output_path), crop)

                    # Update statistics
                    species_key = f"{split}_{species}"
                    species_count[species_key] = species_count.get(species_key, 0) + 1
                    crop_count += 1

                    # Add metadata
                    metadata.append({
                        'original_image': str(image_path.relative_to(input_path)),
                        'crop_filename': crop_filename,
                        'split': split,
                        'species': species,
                        'class_id': crop_data['class_id'],
                        'detection_confidence': crop_data['detection_confidence'],
                        'iou_with_gt': crop_data['iou'],
                        'bbox_x1': x1, 'bbox_y1': y1, 'bbox_x2': x2, 'bbox_y2': y2,
                        'crop_x1': crop_x1, 'crop_y1': crop_y1,
                        'crop_x2': crop_x2, 'crop_y2': crop_y2,
                        'dataset_source': 'detection_species_matched'
                    })

                processed_count += 1

            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                continue

    # Save metadata
    if metadata:
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(output_path / 'species_aware_crop_metadata.csv', index=False)

        print(f"\n✅ Species-aware crop generation completed:")
        print(f"   📸 Images processed: {processed_count}")
        print(f"   ✂️  Crops generated: {crop_count}")

        # Show species distribution
        print(f"   🧬 Species distribution:")
        for species_key, count in sorted(species_count.items()):
            split, species = species_key.split('_', 1)
            print(f"      {split}/{species}: {count} crops")

        return metadata_df
    else:
        print("❌ No species-matched crops were generated!")
        return None

def create_yolo_classification_structure_species(crops_dir, metadata_df, output_dir):
    """Create YOLO classification directory structure with species labels"""
    yolo_dir = Path(output_dir) / "yolo_classification"

    for split in ['train', 'val', 'test']:
        split_crops = metadata_df[metadata_df['split'] == split]
        if len(split_crops) > 0:
            # Group by species
            for species in split_crops['species'].unique():
                species_crops = split_crops[split_crops['species'] == species]

                # Create species directory
                class_dir = yolo_dir / split / species
                class_dir.mkdir(parents=True, exist_ok=True)

                # Copy crops to species directory
                for _, row in species_crops.iterrows():
                    src_path = Path(crops_dir) / split / row['species'] / row['crop_filename']
                    dst_path = class_dir / row['crop_filename']

                    if src_path.exists():
                        import shutil
                        shutil.copy2(src_path, dst_path)

    print(f"✅ YOLO classification structure (species-aware) created at: {yolo_dir}")
    return yolo_dir

def main():
    parser = argparse.ArgumentParser(description="Generate species-aware crops from detection results")
    parser.add_argument("--model", required=True, help="Path to trained detection model")
    parser.add_argument("--input", required=True, help="Input dataset directory")
    parser.add_argument("--output", required=True, help="Output directory for crops")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--crop_size", type=int, default=128, help="Size of generated crops")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for matching detection with GT")
    parser.add_argument("--create_yolo_structure", action="store_true", help="Create YOLO classification structure")

    args = parser.parse_args()

    print("=" * 80)
    print("SPECIES-AWARE CROP GENERATION FROM DETECTION RESULTS")
    print("=" * 80)

    # Validate inputs
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return

    if not Path(args.input).exists():
        print(f"❌ Input directory not found: {args.input}")
        return

    print(f"🤖 Detection model: {args.model}")
    print(f"📁 Input dataset: {args.input}")
    print(f"📂 Output directory: {args.output}")
    print(f"🎯 Detection confidence: {args.confidence}")
    print(f"📏 Crop size: {args.crop_size}x{args.crop_size}")
    print(f"🎯 IoU threshold: {args.iou_threshold}")

    try:
        # Generate species-aware crops
        metadata = generate_species_aware_crops(
            model_path=args.model,
            input_dir=args.input,
            output_dir=args.output,
            confidence=args.confidence,
            crop_size=args.crop_size,
            iou_threshold=args.iou_threshold
        )

        if metadata is not None and args.create_yolo_structure:
            # Create YOLO classification structure
            crops_dir = Path(args.output) / "crops"
            yolo_dir = create_yolo_classification_structure_species(
                crops_dir, metadata, args.output
            )

        print(f"\n🎉 Species-aware crop generation completed successfully!")
        print(f"📊 Results saved to: {args.output}")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return

if __name__ == "__main__":
    main()