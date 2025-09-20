#!/usr/bin/env python3
"""
Create multi-species malaria detection and classification datasets
Uses all 4 species from MP-IDB with bounding box annotations
"""

import csv
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
import cv2

MP_IDB_ROOT = Path("data/raw/mp_idb")
SPECIES = ["Falciparum", "Vivax", "Malariae", "Ovale"]
SPECIES_CANON = {
    "Falciparum": "falciparum",
    "Vivax": "vivax",
    "Malariae": "malariae",
    "Ovale": "ovale"
}

def load_all_bounding_boxes() -> List[Dict]:
    """Load bounding boxes from all species CSV files"""
    all_boxes = []

    for species in SPECIES:
        species_dir = MP_IDB_ROOT / species

        # Try mask-derived CSV first (our new format)
        csv_path = species_dir / f"mp-idb-{species.lower()}-from-masks.csv"
        if not csv_path.exists():
            # Fall back to original CSV
            csv_path = species_dir / f"mp-idb-{species.lower()}.csv"
            if not csv_path.exists():
                csv_path = species_dir / f"mp-idb-{species.lower()}-abspath.csv"

        if not csv_path.exists():
            print(f"⚠️  No CSV found for {species}")
            continue

        print(f"📁 Loading {species} from {csv_path.name}")

        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                # Build full image path
                img_filename = Path(row.get('filename', '')).name
                if not img_filename:
                    continue

                img_path = species_dir / "img" / img_filename

                if img_path.exists():
                    box_data = {
                        'species': SPECIES_CANON[species],
                        'image_path': str(img_path),
                        'filename': img_filename,
                        'center_x': float(row.get('center_x', 0)),
                        'center_y': float(row.get('center_y', 0)),
                        'width': float(row.get('width', 0)),
                        'height': float(row.get('height', 0))
                    }
                    all_boxes.append(box_data)
                    count += 1

        print(f"   ✅ {count} bounding boxes loaded")

    return all_boxes

def create_detection_dataset(all_boxes: List[Dict], output_dir: Path):
    """Create YOLO detection dataset with all species"""
    print(f"\n🎯 Creating detection dataset: {output_dir}")

    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Group by image filename to avoid splitting same image
    images_by_name = {}
    for box in all_boxes:
        img_name = box['filename']
        if img_name not in images_by_name:
            images_by_name[img_name] = []
        images_by_name[img_name].append(box)

    # Split images (70% train, 15% val, 15% test)
    image_names = list(images_by_name.keys())
    random.shuffle(image_names)

    n_train = int(len(image_names) * 0.7)
    n_val = int(len(image_names) * 0.15)

    splits = {
        'train': image_names[:n_train],
        'val': image_names[n_train:n_train+n_val],
        'test': image_names[n_train+n_val:]
    }

    print(f"Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Process each split
    for split_name, img_names in splits.items():
        print(f"Processing {split_name} split...")

        for img_name in img_names:
            boxes_for_img = images_by_name[img_name]

            # Copy image (use first box's path)
            src_img = Path(boxes_for_img[0]['image_path'])
            dst_img = output_dir / split_name / 'images' / img_name
            if src_img.exists():
                shutil.copy2(src_img, dst_img)

                # Create YOLO label file
                label_path = output_dir / split_name / 'labels' / f"{Path(img_name).stem}.txt"
                with open(label_path, 'w') as f:
                    for box in boxes_for_img:
                        # Class 0 for all parasites (detection only, not classification)
                        f.write(f"0 {box['center_x']:.6f} {box['center_y']:.6f} {box['width']:.6f} {box['height']:.6f}\n")

    # Create dataset.yaml
    yaml_content = f"""
path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

names:
  0: parasite

nc: 1
"""

    with open(output_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content.strip())

    print(f"✅ Detection dataset created at {output_dir}")

def create_classification_crops(all_boxes: List[Dict], output_dir: Path, crop_size: int = 96):
    """Create classification dataset by cropping parasite regions"""
    print(f"\n🔬 Creating classification crops: {output_dir}")

    # Create species directories
    for split in ['train', 'val', 'test']:
        for species in SPECIES_CANON.values():
            (output_dir / split / species).mkdir(parents=True, exist_ok=True)

    # Group by image and species
    images_by_species = {}
    for box in all_boxes:
        species = box['species']
        if species not in images_by_species:
            images_by_species[species] = {}

        img_name = box['filename']
        if img_name not in images_by_species[species]:
            images_by_species[species][img_name] = []
        images_by_species[species][img_name].append(box)

    index = {"train": [], "val": [], "test": []}

    # Process each species
    for species, species_images in images_by_species.items():
        print(f"Processing {species} ({len(species_images)} images)...")

        # Split images for this species
        img_names = list(species_images.keys())
        random.shuffle(img_names)

        n_train = max(1, int(len(img_names) * 0.7))
        n_val = max(1, int(len(img_names) * 0.15)) if len(img_names) > 2 else 0

        splits = {
            'train': img_names[:n_train],
            'val': img_names[n_train:n_train+n_val] if n_val > 0 else [],
            'test': img_names[n_train+n_val:] if len(img_names) > n_train + n_val else []
        }

        # Process each split
        for split_name, img_names in splits.items():
            if not img_names:
                continue

            for img_name in img_names:
                boxes_for_img = species_images[img_name]

                # Load image
                img_path = Path(boxes_for_img[0]['image_path'])
                if not img_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h, w = img.shape[:2]

                # Crop each bounding box
                for i, box in enumerate(boxes_for_img):
                    # Convert normalized coords to pixel coords
                    cx = int(box['center_x'] * w)
                    cy = int(box['center_y'] * h)
                    bw = int(box['width'] * w)
                    bh = int(box['height'] * h)

                    x1 = max(0, cx - bw//2)
                    y1 = max(0, cy - bh//2)
                    x2 = min(w, cx + bw//2)
                    y2 = min(h, cy + bh//2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Crop and resize
                    crop = img[y1:y2, x1:x2]
                    if crop_size > 0:
                        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

                    # Save crop
                    crop_name = f"{Path(img_name).stem}_{i}.jpg"
                    crop_path = output_dir / split_name / species / crop_name
                    cv2.imwrite(str(crop_path), crop)

                    index[split_name].append({
                        'src': str(img_path),
                        'label': species,
                        'bbox': [x1, y1, x2, y2],
                        'out': str(crop_path)
                    })

    # Save index
    with open(output_dir / 'crops_index.json', 'w') as f:
        json.dump(index, f, indent=2)

    # Print summary
    for split in ['train', 'val', 'test']:
        split_counts = {}
        for item in index[split]:
            species = item['label']
            split_counts[species] = split_counts.get(species, 0) + 1
        print(f"{split}: {split_counts}")

    print(f"✅ Classification crops created at {output_dir}")

def main():
    print("🚀 CREATING MULTI-SPECIES MALARIA DATASETS")
    print("=" * 60)

    # Set random seed for reproducible splits
    random.seed(42)

    # Load all bounding boxes
    all_boxes = load_all_bounding_boxes()

    if not all_boxes:
        print("❌ No bounding boxes found!")
        return

    print(f"\n📊 SUMMARY: {len(all_boxes)} total bounding boxes")

    # Count by species
    species_counts = {}
    for box in all_boxes:
        species = box['species']
        species_counts[species] = species_counts.get(species, 0) + 1

    print("Species distribution:")
    for species, count in species_counts.items():
        print(f"  - {species}: {count}")

    # Create datasets
    det_output = Path("data/detection_multispecies")
    cls_output = Path("data/classification_multispecies")

    create_detection_dataset(all_boxes, det_output)
    create_classification_crops(all_boxes, cls_output)

    print(f"\n✅ DATASETS CREATED!")
    print(f"Detection dataset: {det_output}")
    print(f"Classification dataset: {cls_output}")

    print("\nNext steps:")
    print("1. Train detection model:")
    print(
        "   python pipeline.py train yolov8_detection "
        f"--data {det_output}/dataset.yaml --name your_detection_run"
    )
    print("2. Train classification model:")
    print(
        "   python pipeline.py train yolov8_classification "
        f"--data {cls_output} --name your_classification_run"
    )

if __name__ == "__main__":
    main()
