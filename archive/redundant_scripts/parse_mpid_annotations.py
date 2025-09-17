#!/usr/bin/env python3
"""
Parse MP-IDB Dataset for Parasite Detection
Extract bounding boxes from CSV annotations for YOLO detection training
"""

import os
import sys
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional

class MPIDBDetectionParser:
    """Parse MP-IDB dataset for parasite detection (bounding boxes)"""

    def __init__(self,
                 mp_idb_path: str = "data/raw/mp_idb",
                 output_path: str = "data/detection",
                 min_box: int = 32,
                 max_box: int = 128,
                 length_scale: float = 1.0):

        self.mp_idb_path = Path(mp_idb_path)
        self.output_path = Path(output_path)

        # Species directories in MP-IDB
        self.species_dirs = ['Falciparum', 'Vivax', 'Malariae', 'Ovale']

        # Create output directories
        self.setup_output_directories()

        # Detection classes (simplified - just parasite vs background)
        self.classes = {
            'parasite': 0  # All parasite types unified as single class
        }

        # Box synthesis params: CSV encodes line spans (x1,x2,y1,y2), not boxes.
        # We convert each span to a square box centered at its midpoint with side
        # clamped to [min_box, max_box] and optionally scaled by line length.
        self.min_box = int(min_box)
        self.max_box = int(max_box)
        self.length_scale = float(length_scale)

        print(f"MP-IDB Parser initialized")
        print(f"Input path: {self.mp_idb_path}")
        print(f"Output path: {self.output_path}")
        print(f"Box synthesis: min={self.min_box}, max={self.max_box}, scale={self.length_scale}")

    def setup_output_directories(self):
        """Create output directory structure"""

        dirs_to_create = [
            self.output_path,
            self.output_path / "images",
            self.output_path / "labels",
            self.output_path / "annotations"
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"✓ Created output directories")

    def load_species_annotations(self, species_name: str) -> Optional[pd.DataFrame]:
        """Load CSV annotations for a species"""

        species_dir = self.mp_idb_path / species_name
        csv_file = species_dir / f"mp-idb-{species_name.lower()}.csv"

        if not csv_file.exists():
            print(f"⚠️ CSV file not found: {csv_file}")
            return None

        try:
            df = pd.read_csv(csv_file)
            print(f"✓ Loaded {len(df)} annotations for {species_name}")
            return df

        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")
            return None

    def validate_bounding_box(self, xmin: int, xmax: int, ymin: int, ymax: int,
                              img_width: int, img_height: int) -> bool:
        """Validate bounding box coordinates"""
        if xmin >= xmax or ymin >= ymax:
            return False
        # within bounds (allow touching edge)
        if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
            return False
        width = xmax - xmin
        height = ymax - ymin
        if width < 8 or height < 8:
            return False
        return True

    def convert_to_yolo_format(self, xmin: int, xmax: int, ymin: int, ymax: int,
                              img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert bounding box to YOLO format (normalized)"""

        # Calculate center coordinates and dimensions
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        # Normalize to [0, 1]
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height

        return center_x, center_y, width, height

    def process_image(self, species_name: str, image_filename: str,
                     image_annotations: pd.DataFrame) -> Optional[Dict]:
        """Process single image and its annotations using ground truth masks"""

        # Paths
        species_dir = self.mp_idb_path / species_name
        image_path = species_dir / "img" / image_filename
        gt_path = species_dir / "gt" / image_filename

        if not image_path.exists():
            print(f"⚠️ Image not found: {image_path}")
            return None

        if not gt_path.exists():
            print(f"⚠️ Ground truth mask not found: {gt_path}")
            return None

        # Load image and ground truth mask
        try:
            img = cv2.imread(str(image_path))
            gt_mask = cv2.imread(str(gt_path), 0)

            if img is None:
                print(f"❌ Cannot load image: {image_path}")
                return None

            if gt_mask is None:
                print(f"❌ Cannot load ground truth mask: {gt_path}")
                return None

            img_height, img_width = img.shape[:2]

        except Exception as e:
            print(f"❌ Error loading image/mask {image_filename}: {e}")
            return None

        # Extract bounding boxes from ground truth mask
        valid_boxes = []

        # Find connected components in ground truth mask
        _, labels = cv2.connectedComponents(gt_mask)

        # Get parasite types from CSV annotations
        parasite_types = {}
        csv_parasites = []

        for _, row in image_annotations.iterrows():
            # CSV coordinates seem to represent parasite centers/regions
            # We'll match them to the largest connected components
            rx1, rx2 = float(row['xmin']), float(row['xmax'])
            ry1, ry2 = float(row['ymin']), float(row['ymax'])

            # Take the coordinate range midpoint as approximate center
            cx = (rx1 + rx2) / 2.0
            cy = (ry1 + ry2) / 2.0

            csv_parasites.append({
                'center': (cx, cy),
                'type': row.get('parasite_type', 'unknown'),
                'coords': (rx1, rx2, ry1, ry2)
            })

        # Process each connected component (parasite region)
        component_boxes = []
        for label in range(1, labels.max() + 1):
            # Create mask for this component
            component_mask = (labels == label).astype(np.uint8) * 255

            # Find contours
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contours[0])
            area = cv2.contourArea(contours[0])

            # Filter out tiny noise pixels (keep only substantial parasites)
            if area < 100:  # minimum area threshold
                continue

            # Calculate center
            center_x_px = x + w // 2
            center_y_px = y + h // 2

            component_boxes.append({
                'bbox': (x, y, x + w, y + h),
                'center': (center_x_px, center_y_px),
                'area': area,
                'label': label
            })

        # Sort by area (largest first) to match with CSV annotations
        component_boxes.sort(key=lambda x: x['area'], reverse=True)

        # Match CSV annotations to the largest components
        for i, csv_parasite in enumerate(csv_parasites):
            if i < len(component_boxes):
                component = component_boxes[i]
                xmin, ymin, xmax, ymax = component['bbox']

                # Validate bounding box
                if not self.validate_bounding_box(xmin, xmax, ymin, ymax, img_width, img_height):
                    continue

                # Convert to YOLO format
                center_x, center_y, width, height = self.convert_to_yolo_format(
                    xmin, xmax, ymin, ymax, img_width, img_height
                )

                valid_boxes.append({
                    'class_id': 0,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'original_coords': [xmin, ymin, xmax, ymax],
                    'parasite_type': csv_parasite['type'],
                    'csv_coords': csv_parasite['coords'],
                    'gt_area': component['area'],
                    'gt_center': component['center']
                })

        if not valid_boxes:
            print(f"⚠️ No valid bounding boxes found for {image_filename}")
            return None

        return {
            'image_filename': image_filename,
            'image_path': str(image_path),
            'species': species_name,
            'image_width': img_width,
            'image_height': img_height,
            'bounding_boxes': valid_boxes,
            'num_parasites': len(valid_boxes)
        }

    def save_yolo_annotations(self, processed_data: List[Dict]) -> bool:
        """Save annotations in YOLO detection format"""

        print("Saving YOLO detection format...")

        for data in tqdm(processed_data, desc="Saving annotations"):
            image_filename = data['image_filename']
            base_name = Path(image_filename).stem

            # Copy image to output directory
            src_image = Path(data['image_path'])
            dst_image = self.output_path / "images" / f"{base_name}.jpg"

            try:
                # Create a symlink to avoid duplicating data on disk
                if not dst_image.exists():
                    try:
                        dst_image.symlink_to(src_image.resolve())
                    except Exception:
                        # Fallback to hardlink if symlink fails
                        import os
                        try:
                            os.link(src_image.resolve(), dst_image)
                        except Exception:
                            # As a last resort, copy
                            import shutil
                            shutil.copy2(src_image, dst_image)

                # Create YOLO label file
                label_file = self.output_path / "labels" / f"{base_name}.txt"

                with open(label_file, 'w') as f:
                    for bbox in data['bounding_boxes']:
                        # YOLO format: class_id center_x center_y width height
                        f.write(f"{bbox['class_id']} {bbox['center_x']:.6f} {bbox['center_y']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}\n")

            except Exception as e:
                print(f"❌ Error saving {base_name}: {e}")
                return False

        print(f"✓ Saved {len(processed_data)} images and labels")
        return True

    def create_dataset_yaml(self, processed_data: List[Dict]) -> bool:
        """Create YOLO dataset configuration"""

        yaml_content = f"""# MP-IDB Parasite Detection Dataset
path: {self.output_path.absolute()}
train: images
val: images  # Using same for now, should split later
test: images

nc: 1
names:
  0: parasite

# Dataset info
total_images: {len(processed_data)}
total_parasites: {sum(data['num_parasites'] for data in processed_data)}
species: {list(set(data['species'] for data in processed_data))}
"""

        yaml_path = self.output_path / "dataset.yaml"

        try:
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)
            print(f"✓ Created dataset config: {yaml_path}")
            return True

        except Exception as e:
            print(f"❌ Error creating dataset.yaml: {e}")
            return False

    def create_summary_report(self, processed_data: List[Dict]) -> bool:
        """Create summary report"""

        # Calculate statistics
        total_images = len(processed_data)
        total_parasites = sum(data['num_parasites'] for data in processed_data)
        species_counts = {}
        parasite_type_counts = {}

        for data in processed_data:
            species = data['species']
            species_counts[species] = species_counts.get(species, 0) + 1

            for bbox in data['bounding_boxes']:
                parasite_type = bbox['parasite_type']
                parasite_type_counts[parasite_type] = parasite_type_counts.get(parasite_type, 0) + 1

        # Create report
        report = {
            'dataset_type': 'parasite_detection',
            'total_images': total_images,
            'total_parasites': total_parasites,
            'avg_parasites_per_image': total_parasites / total_images if total_images > 0 else 0,
            'species_distribution': species_counts,
            'parasite_type_distribution': parasite_type_counts,
            'image_dimensions': '2592x1944',
            'annotation_format': 'YOLO (normalized bounding boxes)',
            'classes': self.classes
        }

        # Save report
        report_path = self.output_path / "annotations" / "detection_report.json"

        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            # Print summary
            print(f"\n{'='*50}")
            print("MP-IDB DETECTION DATASET SUMMARY")
            print(f"{'='*50}")
            print(f"Total images: {total_images}")
            print(f"Total parasites: {total_parasites}")
            print(f"Avg parasites per image: {report['avg_parasites_per_image']:.1f}")
            print(f"Species distribution: {species_counts}")
            print(f"Output directory: {self.output_path}")
            print(f"{'='*50}\n")

            return True

        except Exception as e:
            print(f"❌ Error creating summary report: {e}")
            return False

    def parse_dataset(self) -> bool:
        """Main method to parse MP-IDB dataset"""

        print("Starting MP-IDB detection dataset parsing...")

        all_processed_data = []

        # Process each species
        for species_name in self.species_dirs:
            print(f"\nProcessing {species_name}...")

            # Load annotations
            annotations_df = self.load_species_annotations(species_name)
            if annotations_df is None:
                continue

            # Group by filename
            grouped = annotations_df.groupby('filename')

            # Process each image
            for image_filename, image_annotations in tqdm(grouped, desc=f"Processing {species_name}"):
                processed_data = self.process_image(species_name, image_filename, image_annotations)

                if processed_data:
                    all_processed_data.append(processed_data)

        if not all_processed_data:
            print("❌ No valid data processed!")
            return False

        print(f"\n✓ Processed {len(all_processed_data)} images total")

        # Save in YOLO format
        if not self.save_yolo_annotations(all_processed_data):
            return False

        # Create dataset configuration
        if not self.create_dataset_yaml(all_processed_data):
            return False

        # Create summary report
        if not self.create_summary_report(all_processed_data):
            return False

        print("🎉 MP-IDB detection dataset parsing completed successfully!")
        return True

def main():
    """Main function"""
    import argparse as _argparse

    ap = _argparse.ArgumentParser(description="Parse MP-IDB spans into YOLO detection dataset")
    ap.add_argument('--mp-idb-path', default='data/raw/mp_idb', help='MP-IDB root path')
    ap.add_argument('--output-path', default='data/detection', help='Output detection root')
    ap.add_argument('--min-box', type=int, default=32, help='Minimum square box side (px)')
    ap.add_argument('--max-box', type=int, default=128, help='Maximum square box side (px)')
    ap.add_argument('--length-scale', type=float, default=1.0, help='Box side scales with span length * scale')
    args = ap.parse_args()

    parser = MPIDBDetectionParser(
        mp_idb_path=args.mp_idb_path,
        output_path=args.output_path,
        min_box=args.min_box,
        max_box=args.max_box,
        length_scale=args.length_scale,
    )
    success = parser.parse_dataset()

    if success:
        print("✅ Detection dataset ready for YOLO training!")
        print("Next steps:")
        print("1. python scripts/09_train_detection.py")
        print("2. Evaluate detection performance")
    else:
        print("❌ Failed to parse detection dataset")
        sys.exit(1)

if __name__ == "__main__":
    main()
