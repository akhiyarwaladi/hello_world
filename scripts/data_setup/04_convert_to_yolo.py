#!/usr/bin/env python3
"""
Script to convert integrated dataset to YOLO format
Author: Malaria Detection Team
Date: 2024
"""

import os
import yaml
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import argparse
from collections import defaultdict


class MalariaYOLOConverter:
    """Converts integrated malaria dataset to YOLO training format"""
    
    def __init__(self, 
                 integrated_data_dir: str = "data/integrated",
                 yolo_output_dir: str = "data/yolo"):
        """Initialize converter"""
        self.integrated_data_dir = Path(integrated_data_dir)
        self.yolo_output_dir = Path(yolo_output_dir)
        self.yolo_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create YOLO directory structure
        for split in ['train', 'val', 'test']:
            (self.yolo_output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.yolo_output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.conversion_stats = {
            'total_images': 0,
            'splits_processed': defaultdict(int),
            'class_distribution': defaultdict(int)
        }
        
        # Class names (should match integration script)
        self.class_names = {
            0: 'P_falciparum',
            1: 'P_vivax',
            2: 'P_malariae',
            3: 'P_ovale',
            4: 'Mixed_infection',
            5: 'Uninfected'
        }
    
    def load_integrated_annotations(self) -> Dict[str, pd.DataFrame]:
        """Load integrated annotations for each split"""
        annotations = {}
        
        for split in ['train', 'val', 'test']:
            annotation_file = self.integrated_data_dir / "annotations" / f"{split}_annotations.csv"
            
            if not annotation_file.exists():
                print(f"Warning: {annotation_file} not found")
                continue
            
            df = pd.read_csv(annotation_file)
            annotations[split] = df
            print(f"Loaded {len(df)} {split} annotations")
        
        return annotations
    
    def convert_single_image_annotation(self, row: pd.Series, 
                                      detection_mode: bool = False) -> str:
        """Convert single image annotation to YOLO format"""
        class_id = int(row['unified_class'])
        
        if detection_mode:
            # For detection: class_id x_center y_center width height (normalized)
            # Since we're dealing with cell classification, assume full image
            x_center, y_center = 0.5, 0.5
            width, height = 1.0, 1.0
            return f"{class_id} {x_center} {y_center} {width} {height}"
        else:
            # For classification: just class_id
            return str(class_id)
    
    def create_detection_boxes_from_cells(self, image_path: Path,
                                        class_id: int,
                                        min_box_size: int = 50) -> List[str]:
        """
        Create detection boxes from cell images
        For cell-level datasets, we can either:
        1. Use full image as single detection
        2. Try to detect individual cells if multiple are present
        """
        
        # Load image to get dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        
        height, width = img.shape[:2]
        
        # For simplicity, create single detection covering most of the image
        # In a real scenario, you might want to detect individual cells
        
        # Normalize coordinates
        x_center = 0.5
        y_center = 0.5
        box_width = 0.8  # Cover 80% of image width
        box_height = 0.8  # Cover 80% of image height
        
        return [f"{class_id} {x_center} {y_center} {box_width} {box_height}"]
    
    def convert_split_to_yolo(self, split_name: str, split_df: pd.DataFrame,
                             detection_mode: bool = False,
                             create_detection_boxes: bool = False):
        """Convert single split to YOLO format"""
        print(f"Converting {split_name} split to YOLO format...")
        
        split_dir = self.yolo_output_dir / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        processed_count = 0
        
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), 
                            desc=f"Converting {split_name}"):
            
            # Find source image in integrated dataset
            unified_image_path = row.get('unified_image_path')
            if pd.isna(unified_image_path):
                continue
            
            src_image_path = self.integrated_data_dir / unified_image_path
            
            if not src_image_path.exists():
                print(f"Warning: Source image not found: {src_image_path}")
                continue
            
            # Create YOLO-style filename
            image_id = row['image_id']
            yolo_image_name = f"{image_id:06d}.jpg"
            yolo_label_name = f"{image_id:06d}.txt"
            
            dst_image_path = images_dir / yolo_image_name
            dst_label_path = labels_dir / yolo_label_name
            
            # Copy image
            shutil.copy2(src_image_path, dst_image_path)
            
            # Create label file
            if create_detection_boxes and detection_mode:
                # Create detection boxes
                label_lines = self.create_detection_boxes_from_cells(
                    src_image_path, int(row['unified_class'])
                )
            else:
                # Simple conversion
                label_line = self.convert_single_image_annotation(row, detection_mode)
                label_lines = [label_line] if label_line else []
            
            # Write label file
            with open(dst_label_path, 'w') as f:
                for line in label_lines:
                    f.write(line + "\\n")
            
            # Update statistics
            processed_count += 1
            self.conversion_stats['class_distribution'][int(row['unified_class'])] += 1
        
        self.conversion_stats['splits_processed'][split_name] = processed_count
        print(f"Converted {processed_count} images for {split_name} split")
    
    def create_yolo_config_files(self, task_type: str = "classify"):
        """Create YOLO configuration files"""
        print("Creating YOLO configuration files...")
        
        # Create data.yaml with relative paths for cross-machine compatibility
        # Use output directory path instead of hardcoded path
        relative_output_path = str(self.yolo_output_dir).replace('\\', '/')

        if task_type == "detect":
            data_config = {
                'path': relative_output_path,
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': len(self.class_names),
                'names': list(self.class_names.values())
            }
        else:  # classify
            data_config = {
                'path': relative_output_path,
                'train': 'train',
                'val': 'val',
                'test': 'test',
                'nc': len(self.class_names),
                'names': list(self.class_names.values())
            }
        
        # Save data.yaml
        with open(self.yolo_output_dir / "data.yaml", 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        # Create class names file
        with open(self.yolo_output_dir / "classes.txt", 'w') as f:
            for class_name in self.class_names.values():
                f.write(f"{class_name}\\n")
        
        # Create training config template
        training_config = {
            'task': task_type,
            'mode': 'train',
            'model': 'yolov8m.pt',  # Base model
            'data': str(self.yolo_output_dir / "data.yaml"),
            'epochs': 100,
            'patience': 50,
            'batch': 16,
            'imgsz': 640,
            'save': True,
            'save_period': 10,
            'cache': False,
            'device': '',  # Auto-detect
            'workers': 8,
            'project': 'malaria_detection',
            'name': f'yolov8_{task_type}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
        }
        
        with open(self.yolo_output_dir / f"train_{task_type}_config.yaml", 'w') as f:
            yaml.dump(training_config, f, default_flow_style=False)
        
        print(f"YOLO configuration files created in {self.yolo_output_dir}")
    
    def create_conversion_report(self, annotations: Dict[str, pd.DataFrame]):
        """Create conversion report"""
        print("Creating conversion report...")
        
        report = {
            'conversion_summary': {
                'total_images_converted': sum(self.conversion_stats['splits_processed'].values()),
                'conversion_date': pd.Timestamp.now().isoformat(),
                'yolo_output_dir': str(self.yolo_output_dir.absolute())
            },
            'split_sizes': dict(self.conversion_stats['splits_processed']),
            'class_distribution': {
                self.class_names[class_id]: count
                for class_id, count in self.conversion_stats['class_distribution'].items()
            },
            'yolo_structure': {
                'data_yaml': 'data.yaml',
                'class_names': 'classes.txt',
                'training_config': 'train_*_config.yaml',
                'splits': {
                    'train': {'images': 'train/images/', 'labels': 'train/labels/'},
                    'val': {'images': 'val/images/', 'labels': 'val/labels/'},
                    'test': {'images': 'test/images/', 'labels': 'test/labels/'}
                }
            }
        }
        
        # Save report
        report_path = self.yolo_output_dir / "conversion_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Conversion report saved to {report_path}")
        
        # Print summary
        self.print_conversion_summary(report)
    
    def print_conversion_summary(self, report: Dict):
        """Print conversion summary"""
        print("\\n" + "="*60)
        print(" YOLO CONVERSION SUMMARY ")
        print("="*60)
        print(f"Total images converted: {report['conversion_summary']['total_images_converted']}")
        print(f"Output directory: {report['conversion_summary']['yolo_output_dir']}")
        
        print("\\nSplit sizes:")
        for split, count in report['split_sizes'].items():
            print(f"  {split}: {count} images")
        
        print("\\nClass distribution:")
        for class_name, count in report['class_distribution'].items():
            percentage = (count / report['conversion_summary']['total_images_converted']) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print("\\nYOLO Training Commands:")
        print("  # Classification:")
        print(f"  yolo classify train data={self.yolo_output_dir}/data.yaml model=yolov8m-cls.pt epochs=100")
        print("  # Detection:")
        print(f"  yolo detect train data={self.yolo_output_dir}/data.yaml model=yolov8m.pt epochs=100")
        
        print("="*60)
    
    def create_sample_visualization(self, annotations: Dict[str, pd.DataFrame], 
                                  num_samples: int = 9):
        """Create sample visualization of converted data"""
        print("Creating sample visualization...")
        
        # Collect samples from train set
        train_df = annotations.get('train', pd.DataFrame())
        if len(train_df) == 0:
            return
        
        # Sample from each class
        samples = []
        for class_id, class_name in self.class_names.items():
            class_samples = train_df[train_df['unified_class'] == class_id]
            if len(class_samples) > 0:
                samples.append(class_samples.iloc[0])
            if len(samples) >= num_samples:
                break
        
        if len(samples) == 0:
            return
        
        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle('YOLO Dataset Samples', fontsize=16, fontweight='bold')
        
        for idx, (ax, sample) in enumerate(zip(axes.flat, samples)):
            if idx >= len(samples):
                ax.axis('off')
                continue
            
            # Load and display image
            image_id = sample['image_id']
            yolo_image_path = self.yolo_output_dir / "train" / "images" / f"{image_id:06d}.jpg"
            
            if yolo_image_path.exists():
                img = cv2.imread(str(yolo_image_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                ax.imshow(img_rgb)
                ax.set_title(f"{sample['unified_class_name']}\\nID: {image_id:06d}", 
                           fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.yolo_output_dir / "sample_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sample visualization saved to {viz_path}")

    def convert_lifecycle_json_to_yolo(self,
                                     lifecycle_data_dir: str,
                                     split_ratios: Dict[str, float] = None):
        """Convert lifecycle JSON annotations to YOLO format"""
        lifecycle_path = Path(lifecycle_data_dir)
        annotations_file = lifecycle_path / "annotations.json"
        images_dir = lifecycle_path / "IML_Malaria"

        if not annotations_file.exists():
            print(f"[ERROR] Annotations file not found: {annotations_file}")
            return

        if not images_dir.exists():
            print(f"[ERROR] Images directory not found: {images_dir}")
            return

        print(f"Converting lifecycle dataset from {lifecycle_path}")
        print(f"Images: {images_dir}")
        print(f"Annotations: {annotations_file}")

        # Load JSON annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        # Lifecycle class mapping
        lifecycle_classes = {
            "red blood cell": 0,
            "ring": 1,
            "gametocyte": 2,
            "trophozoite": 3,
            "schizont": 4
        }

        # Update class names for lifecycle
        self.class_names = {
            0: 'red_blood_cell',
            1: 'ring',
            2: 'gametocyte',
            3: 'trophozoite',
            4: 'schizont'
        }

        # Default split ratios if not provided
        if split_ratios is None:
            split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

        # Process annotations
        converted_data = []
        for img_data in annotations:
            image_name = img_data['image_name']
            image_path = images_dir / image_name

            if not image_path.exists():
                print(f"[WARNING] Image not found: {image_path}")
                continue

            # Load image to get dimensions
            img = cv2.imread(str(image_path))
            if img is None:
                continue

            img_height, img_width = img.shape[:2]

            # Process each object in the image
            for obj in img_data.get('objects', []):
                obj_type = obj.get('type', '').lower()
                bbox = obj.get('bbox', {})

                if obj_type not in lifecycle_classes:
                    continue

                # Convert bbox to YOLO format (normalized x_center, y_center, width, height)
                x = float(bbox.get('x', 0))
                y = float(bbox.get('y', 0))
                w = float(bbox.get('w', 0))
                h = float(bbox.get('h', 0))

                # Convert to center coordinates and normalize
                x_center = (x + w/2) / img_width
                y_center = (y + h/2) / img_height
                width_norm = w / img_width
                height_norm = h / img_height

                # Ensure normalized coordinates are in [0,1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width_norm = max(0, min(1, width_norm))
                height_norm = max(0, min(1, height_norm))

                converted_data.append({
                    'image_name': image_name,
                    'image_path': image_path,
                    'class_id': lifecycle_classes[obj_type],
                    'class_name': obj_type,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width_norm,
                    'height': height_norm
                })

        print(f"[SUCCESS] Converted {len(converted_data)} annotations from {len(annotations)} images")

        # Split data
        import random
        random.shuffle(converted_data)

        total_annotations = len(converted_data)
        train_end = int(total_annotations * split_ratios['train'])
        val_end = train_end + int(total_annotations * split_ratios['val'])

        splits = {
            'train': converted_data[:train_end],
            'val': converted_data[train_end:val_end],
            'test': converted_data[val_end:]
        }

        # Write YOLO format files
        for split_name, split_data in splits.items():
            split_images_dir = self.yolo_output_dir / split_name / "images"
            split_labels_dir = self.yolo_output_dir / split_name / "labels"

            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)

            print(f"Processing {split_name} split: {len(split_data)} annotations")

            # Group by image
            image_annotations = defaultdict(list)
            for item in split_data:
                image_annotations[item['image_name']].append(item)

            for image_name, annotations in image_annotations.items():
                # Copy image
                src_image_path = annotations[0]['image_path']
                dst_image_path = split_images_dir / image_name
                shutil.copy2(src_image_path, dst_image_path)

                # Create label file
                label_path = split_labels_dir / f"{Path(image_name).stem}.txt"
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        # YOLO detection format: class_id x_center y_center width height
                        line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                        f.write(line)

                        # Update statistics
                        self.conversion_stats['class_distribution'][ann['class_id']] += 1

            self.conversion_stats['splits_processed'][split_name] = len(image_annotations)
            print(f"[SUCCESS] {split_name}: {len(image_annotations)} images, {len(split_data)} annotations")

        self.conversion_stats['total_images'] = sum(self.conversion_stats['splits_processed'].values())

        return splits

    def convert_kaggle_stage_to_yolo(self,
                                   kaggle_data_dir: str,
                                   split_ratios: Dict[str, float] = None):
        """Convert Kaggle 16-class dataset to 4 stage classes"""
        kaggle_path = Path(kaggle_data_dir)
        data_yaml_file = kaggle_path / "data.yaml"
        images_dir = kaggle_path / "images"
        labels_dir = kaggle_path / "labels"

        if not data_yaml_file.exists():
            print(f"[ERROR] Data YAML file not found: {data_yaml_file}")
            return

        if not images_dir.exists() or not labels_dir.exists():
            print(f"[ERROR] Images or labels directory not found: {images_dir}, {labels_dir}")
            return

        print(f"Converting Kaggle stage dataset from {kaggle_path}")

        # Load original class names from data.yaml
        with open(data_yaml_file, 'r') as f:
            original_data = yaml.safe_load(f)

        original_classes = original_data.get('names', [])

        # Map 16 classes to 4 stages
        # Original: falciparum_R, falciparum_S, falciparum_T, falciparum_G, vivax_R, etc.
        # Target: ring(0), schizont(1), trophozoite(2), gametocyte(3)
        stage_mapping = {
            # Ring stage (suffix _R)
            0: 0, 4: 0, 8: 0, 12: 0,   # falciparum_R, vivax_R, ovale_R, malariae_R
            # Schizont stage (suffix _S)
            1: 1, 5: 1, 9: 1, 13: 1,   # falciparum_S, vivax_S, ovale_S, malariae_S
            # Trophozoite stage (suffix _T)
            2: 2, 6: 2, 10: 2, 14: 2,  # falciparum_T, vivax_T, ovale_T, malariae_T
            # Gametocyte stage (suffix _G)
            3: 3, 7: 3, 11: 3, 15: 3   # falciparum_G, vivax_G, ovale_G, malariae_G
        }

        # Update class names for stages
        self.class_names = {
            0: 'ring',
            1: 'schizont',
            2: 'trophozoite',
            3: 'gametocyte'
        }

        # Default split ratios if not provided
        if split_ratios is None:
            split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

        # Process all label files
        converted_data = []
        label_files = list(labels_dir.glob("*.txt"))

        for label_file in label_files:
            image_name = label_file.stem + ".jpg"  # Assuming JPG images
            image_path = images_dir / image_name

            if not image_path.exists():
                print(f"[WARNING] Image not found: {image_path}")
                continue

            # Read YOLO labels (handle both detection and segmentation formats)
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                original_class_id = int(parts[0])

                # Handle segmentation format (polygon coordinates)
                if len(parts) > 5:
                    # Extract polygon coordinates
                    coords = [float(x) for x in parts[1:]]
                    if len(coords) % 2 != 0:
                        continue  # Skip malformed polygons

                    # Convert polygon to bounding box
                    x_coords = coords[0::2]  # Every even index (x coordinates)
                    y_coords = coords[1::2]  # Every odd index (y coordinates)

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    # Convert to YOLO detection format
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                else:
                    # Handle detection format (already in correct format)
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                # Convert to stage class
                if original_class_id in stage_mapping:
                    stage_class_id = stage_mapping[original_class_id]

                    converted_data.append({
                        'image_name': image_name,
                        'image_path': image_path,
                        'class_id': stage_class_id,
                        'class_name': self.class_names[stage_class_id],
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })

        print(f"[SUCCESS] Converted {len(converted_data)} annotations from {len(label_files)} label files")

        # Split data
        import random
        random.shuffle(converted_data)

        total_annotations = len(converted_data)
        train_end = int(total_annotations * split_ratios['train'])
        val_end = train_end + int(total_annotations * split_ratios['val'])

        splits = {
            'train': converted_data[:train_end],
            'val': converted_data[train_end:val_end],
            'test': converted_data[val_end:]
        }

        # Write YOLO format files
        for split_name, split_data in splits.items():
            split_images_dir = self.yolo_output_dir / split_name / "images"
            split_labels_dir = self.yolo_output_dir / split_name / "labels"

            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)

            print(f"Processing {split_name} split: {len(split_data)} annotations")

            # Group by image
            image_annotations = defaultdict(list)
            for item in split_data:
                image_annotations[item['image_name']].append(item)

            for image_name, annotations in image_annotations.items():
                # Copy image
                src_image_path = annotations[0]['image_path']
                dst_image_path = split_images_dir / image_name
                shutil.copy2(src_image_path, dst_image_path)

                # Create label file
                label_path = split_labels_dir / f"{Path(image_name).stem}.txt"
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        # YOLO detection format: class_id x_center y_center width height
                        line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                        f.write(line)

                        # Update statistics
                        self.conversion_stats['class_distribution'][ann['class_id']] += 1

            self.conversion_stats['splits_processed'][split_name] = len(image_annotations)
            print(f"[SUCCESS] {split_name}: {len(image_annotations)} images, {len(split_data)} annotations")

        self.conversion_stats['total_images'] = sum(self.conversion_stats['splits_processed'].values())

        return splits

    def convert_to_yolo_format(self,
                              task_type: str = "classify",
                              detection_mode: bool = False,
                              create_detection_boxes: bool = False,
                              dataset_type: str = "species"):
        """Main conversion function"""
        print("\\n" + "="*60)
        print(" MALARIA DATASET TO YOLO CONVERSION ")
        print("="*60)
        print(f"Dataset type: {dataset_type}")
        print(f"Task type: {task_type}")
        print(f"Detection mode: {detection_mode}")

        if dataset_type == "lifecycle":
            # Handle lifecycle dataset with JSON annotations
            print("[INFO] Processing lifecycle dataset with JSON annotations")
            lifecycle_data_dir = "data/raw/malaria_lifecycle"

            if not Path(lifecycle_data_dir).exists():
                print(f"[ERROR] Lifecycle dataset not found at {lifecycle_data_dir}")
                print("[TIP] Download first: python scripts/data_setup/01_download_datasets.py --dataset malaria_lifecycle")
                return

            # Convert lifecycle JSON to YOLO
            splits = self.convert_lifecycle_json_to_yolo(lifecycle_data_dir)

            if not splits:
                print("[ERROR] Failed to convert lifecycle dataset")
                return

            # Create YOLO config files for lifecycle
            self.create_yolo_config_files(task_type)

            print(f"\\n[SUCCESS] Lifecycle dataset conversion completed!")
            print(f"Output directory: {self.yolo_output_dir}")
            print(f"Classes: {list(self.class_names.values())}")

        elif dataset_type == "stage":
            # Handle Kaggle stage dataset with 16-class to 4-stage conversion
            print("[INFO] Processing Kaggle stage dataset with 16-class to 4-stage conversion")
            kaggle_data_dir = "data/kaggle_dataset/MP-IDB-YOLO"

            if not Path(kaggle_data_dir).exists():
                print(f"[ERROR] Kaggle dataset not found at {kaggle_data_dir}")
                print("[TIP] Download first: python scripts/data_setup/01_download_datasets.py --dataset kaggle_mp_idb")
                return

            # Convert Kaggle 16-class to 4 stages
            splits = self.convert_kaggle_stage_to_yolo(kaggle_data_dir)

            if not splits:
                print("[ERROR] Failed to convert Kaggle stage dataset")
                return

            # Create YOLO config files for stages
            self.create_yolo_config_files(task_type)

            print(f"\\n[SUCCESS] Kaggle stage dataset conversion completed!")
            print(f"Output directory: {self.yolo_output_dir}")
            print(f"Classes: {list(self.class_names.values())}")

        else:
            # Handle original species dataset with integrated annotations
            print("[INFO] Processing species dataset with integrated annotations")

            # Load integrated annotations
            annotations = self.load_integrated_annotations()

            if not annotations:
                print("No annotations found. Please run integration step first.")
                return

            # Convert each split
            for split_name, split_df in annotations.items():
                self.convert_split_to_yolo(
                    split_name, split_df,
                    detection_mode=detection_mode,
                    create_detection_boxes=create_detection_boxes
                )

            # Create YOLO config files
            self.create_yolo_config_files(task_type)

            # Create visualization
            self.create_sample_visualization(annotations)

            # Create report
            self.create_conversion_report(annotations)
        
        print("\nYOLO format conversion completed successfully!")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Convert integrated dataset to YOLO format")
    parser.add_argument("--integrated-dir", default="data/integrated",
                       help="Integrated data directory")
    parser.add_argument("--output-dir", default="data/yolo",
                       help="YOLO output directory")
    parser.add_argument("--task", choices=["classify", "detect"], default="classify",
                       help="YOLO task type")
    parser.add_argument("--detection-mode", action="store_true",
                       help="Create detection-style annotations")
    parser.add_argument("--create-boxes", action="store_true",
                       help="Create detection boxes for cells")
    parser.add_argument("--dataset", choices=["species", "lifecycle", "stage"], default="species",
                       help="Dataset type: species classification, lifecycle classification, or stage classification")

    args = parser.parse_args()

    # Initialize converter with appropriate output directory
    output_dir = args.output_dir
    if args.dataset == "lifecycle":
        output_dir = output_dir.replace("/yolo", "/lifecycle_yolo")
    elif args.dataset == "stage":
        output_dir = output_dir.replace("/yolo", "/stage_yolo")

    converter = MalariaYOLOConverter(
        integrated_data_dir=args.integrated_dir,
        yolo_output_dir=output_dir
    )

    # Run conversion
    converter.convert_to_yolo_format(
        task_type=args.task,
        detection_mode=args.detection_mode,
        create_detection_boxes=args.create_boxes,
        dataset_type=args.dataset
    )


if __name__ == "__main__":
    main()