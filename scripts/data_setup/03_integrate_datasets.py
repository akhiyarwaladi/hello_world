#!/usr/bin/env python3
"""
Script to integrate multiple malaria datasets into a unified format
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
import seaborn as sns
from collections import defaultdict, Counter
import cv2
from tqdm import tqdm
import argparse


class MalariaDatasetIntegrator:
    """Integrates preprocessed malaria datasets into unified format"""
    
    def __init__(self, 
                 processed_data_dir: str = "data/processed", 
                 integrated_data_dir: str = "data/integrated"):
        """Initialize integrator"""
        self.processed_data_dir = Path(processed_data_dir)
        self.integrated_data_dir = Path(integrated_data_dir)
        self.integrated_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.integrated_data_dir / "images").mkdir(exist_ok=True)
        (self.integrated_data_dir / "annotations").mkdir(exist_ok=True)
        (self.integrated_data_dir / "metadata").mkdir(exist_ok=True)
        
        # Class mapping configuration
        self.class_mapping = {
            'infected': {
                'P_falciparum': 0,
                'P_vivax': 1,
                'P_malariae': 2,
                'P_ovale': 3,
                'mixed': 4,
                'unknown': 4  # Unknown infected -> mixed
            },
            'uninfected': 5
        }
        
        # Reverse mapping for class names
        self.class_names = {
            0: 'P_falciparum',
            1: 'P_vivax',
            2: 'P_malariae',
            3: 'P_ovale',
            4: 'Mixed_infection',
            5: 'Uninfected'
        }
        
        self.integration_stats = {
            'total_images': 0,
            'class_distribution': defaultdict(int),
            'species_distribution': defaultdict(int),
            'dataset_distribution': defaultdict(int),
            'quality_stats': {}
        }
    
    def load_processed_samples(self) -> pd.DataFrame:
        """Load processed samples from preprocessing stage"""
        samples_file = self.processed_data_dir / "processed_samples.csv"
        
        if not samples_file.exists():
            print(f"No processed samples found at {samples_file}")
            print("Please run preprocessing step first (02_preprocess_data.py)")
            return pd.DataFrame()
        
        df = pd.read_csv(samples_file)
        print(f"Loaded {len(df)} processed samples")
        return df
    
    def map_species_to_class(self, class_label: str, species: str) -> int:
        """Map species and class to unified class ID"""
        if class_label == 'uninfected' or species == 'none':
            return self.class_mapping['uninfected']
        
        # Handle infected cases
        species_lower = species.lower()
        
        if species_lower in ['p_falciparum', 'pf', 'falciparum']:
            return self.class_mapping['infected']['P_falciparum']
        elif species_lower in ['p_vivax', 'pv', 'vivax']:
            return self.class_mapping['infected']['P_vivax']
        elif species_lower in ['p_malariae', 'pm', 'malariae']:
            return self.class_mapping['infected']['P_malariae']
        elif species_lower in ['p_ovale', 'po', 'ovale']:
            return self.class_mapping['infected']['P_ovale']
        else:
            # Unknown species or mixed -> Mixed_infection
            return self.class_mapping['infected']['mixed']
    
    def create_unified_annotations(self, df: pd.DataFrame) -> List[Dict]:
        """Create unified annotation format"""
        unified_annotations = []
        
        print("Creating unified annotations...")
        
        def _infer_species_from_path(row) -> Optional[str]:
            try:
                path_str = str(row.get('original_path', '')).lower()
                # Infer for MP-IDB by folder name in path
                if row.get('dataset') == 'mp_idb':
                    if 'falciparum' in path_str:
                        return 'P_falciparum'
                    if 'vivax' in path_str:
                        return 'P_vivax'
                    if 'malariae' in path_str:
                        return 'P_malariae'
                    if 'ovale' in path_str:
                        return 'P_ovale'
                # Infer for NIH thick datasets from dataset name
                dname = str(row.get('dataset', '')).lower()
                if dname == 'nih_thick_pf':
                    return 'P_falciparum'
                if dname == 'nih_thick_pv':
                    return 'P_vivax'
                if dname == 'nih_thick_uninfected':
                    return 'none'
            except Exception:
                pass
            return None

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Map to unified class
            species_val = row.get('species')
            # Fallback: try to infer species from original path/dataset if unknown
            if isinstance(species_val, str) and species_val.lower() == 'unknown':
                inferred = _infer_species_from_path(row)
                if inferred is not None:
                    species_val = inferred
            unified_class = self.map_species_to_class(row['class'], species_val)
            
            annotation = {
                'image_id': idx,
                'image_path': row['image_path'],
                'original_path': row['original_path'],
                'dataset': row['dataset'],
                'original_class': row['class'],
                'original_species': row['species'],
                'unified_class': unified_class,
                'unified_class_name': self.class_names[unified_class],
                'quality_score': float(row['quality_score']),
                'original_size': row['original_size'],
                'processed_size': row['processed_size']
            }
            
            # Add quality metrics if available
            quality_metrics = ['sharpness', 'contrast', 'brightness', 'dark_ratio', 'bright_ratio']
            for metric in quality_metrics:
                if metric in row and pd.notna(row[metric]):
                    annotation[metric] = float(row[metric])

            # Add bounding box information if available (from MP-IDB processing)
            bbox_fields = ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'original_bbox_x', 'original_bbox_y', 'original_bbox_w', 'original_bbox_h', 'object_id']
            for field in bbox_fields:
                if field in row and pd.notna(row[field]):
                    if field.startswith('bbox_') or field.startswith('original_bbox_'):
                        annotation[field] = float(row[field])
                    else:
                        annotation[field] = row[field]
            
            unified_annotations.append(annotation)
            
            # Update statistics
            self.integration_stats['total_images'] += 1
            self.integration_stats['class_distribution'][unified_class] += 1
            self.integration_stats['species_distribution'][row['species']] += 1
            self.integration_stats['dataset_distribution'][row['dataset']] += 1
        
        return unified_annotations
    
    def copy_images_to_unified_structure(self, annotations: List[Dict]):
        """Copy images to unified directory structure"""
        print("Copying images to unified structure...")
        
        # Create class-based subdirectories
        for class_id, class_name in self.class_names.items():
            class_dir = self.integrated_data_dir / "images" / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        for annotation in tqdm(annotations, desc="Copying images"):
            src_path = self.processed_data_dir / annotation['image_path']
            
            if not src_path.exists():
                print(f"Warning: Source image not found: {src_path}")
                continue
            
            # Create new filename with unified naming
            class_name = annotation['unified_class_name']
            dataset = annotation['dataset']
            image_id = annotation['image_id']
            
            new_filename = f"{dataset}_{class_name}_{image_id:06d}.jpg"
            dst_path = self.integrated_data_dir / "images" / class_name / new_filename
            
            # Copy image
            shutil.copy2(src_path, dst_path)
            
            # Update annotation with new path
            annotation['unified_image_path'] = str(dst_path.relative_to(self.integrated_data_dir))
    
    def create_dataset_splits(self, annotations: List[Dict], 
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15) -> Dict[str, List[Dict]]:
        """Create train/val/test splits with class balancing"""
        print(f"Creating dataset splits (train:{train_ratio}, val:{val_ratio}, test:{test_ratio})...")
        
        # Verify ratios sum to 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Group annotations by class
        class_annotations = defaultdict(list)
        for ann in annotations:
            class_annotations[ann['unified_class']].append(ann)
        
        splits = {
            'train': [],
            'val': [],
            'test': []
        }
        
        # Split each class separately to maintain balance
        for class_id, class_anns in class_annotations.items():
            # Shuffle annotations
            np.random.shuffle(class_anns)
            
            n_samples = len(class_anns)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            n_test = n_samples - n_train - n_val
            
            # Split
            splits['train'].extend(class_anns[:n_train])
            splits['val'].extend(class_anns[n_train:n_train + n_val])
            splits['test'].extend(class_anns[n_train + n_val:])
            
            print(f"  {self.class_names[class_id]}: {n_train} train, {n_val} val, {n_test} test")
        
        # Shuffle each split
        for split in splits.values():
            np.random.shuffle(split)
        
        print(f"Total splits - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def save_annotations(self, annotations: List[Dict], splits: Dict[str, List[Dict]]):
        """Save annotations in multiple formats"""
        print("Saving annotations...")
        
        # Save complete annotations
        annotations_df = pd.DataFrame(annotations)
        annotations_df.to_csv(self.integrated_data_dir / "annotations" / "unified_annotations.csv", index=False)
        
        # Save annotations as JSON
        with open(self.integrated_data_dir / "annotations" / "unified_annotations.json", 'w') as f:
            json.dump(annotations, f, indent=2)
        
        # Save split annotations
        for split_name, split_annotations in splits.items():
            split_df = pd.DataFrame(split_annotations)
            split_df.to_csv(self.integrated_data_dir / "annotations" / f"{split_name}_annotations.csv", index=False)
            
            with open(self.integrated_data_dir / "annotations" / f"{split_name}_annotations.json", 'w') as f:
                json.dump(split_annotations, f, indent=2)
    
    def create_yolo_format_data(self, splits: Dict[str, List[Dict]]):
        """Create YOLO format dataset structure"""
        print("Creating YOLO format data...")
        
        yolo_dir = self.integrated_data_dir / "yolo"
        yolo_dir.mkdir(exist_ok=True)
        
        # Create YOLO directory structure
        for split in ['train', 'val', 'test']:
            (yolo_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (yolo_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Copy images and create label files for each split
        for split_name, split_annotations in splits.items():
            print(f"  Creating {split_name} split...")
            
            for annotation in tqdm(split_annotations, desc=f"Processing {split_name}"):
                # Copy image
                src_image = self.integrated_data_dir / annotation['unified_image_path']
                dst_image = yolo_dir / split_name / "images" / f"{annotation['image_id']:06d}.jpg"
                
                if src_image.exists():
                    shutil.copy2(src_image, dst_image)
                    
                    # Create YOLO format label for object detection
                    label_file = yolo_dir / split_name / "labels" / f"{annotation['image_id']:06d}.txt"
                    with open(label_file, 'w') as f:
                        # Check if this annotation has bounding box info (from MP-IDB processing)
                        if 'bbox_x' in annotation and 'bbox_y' in annotation:
                            # Convert absolute coordinates to YOLO normalized format
                            img_width = img_height = self.processed_data_dir.parent / "integrated" / "yolo" / "target_size"  # 640
                            target_size = 640  # From preprocessing

                            x_center = (annotation['bbox_x'] + annotation['bbox_w'] / 2) / target_size
                            y_center = (annotation['bbox_y'] + annotation['bbox_h'] / 2) / target_size
                            width = annotation['bbox_w'] / target_size
                            height = annotation['bbox_h'] / target_size

                            # Ensure values are within [0,1] range
                            x_center = max(0.0, min(1.0, x_center))
                            y_center = max(0.0, min(1.0, y_center))
                            width = max(0.001, min(1.0, width))
                            height = max(0.001, min(1.0, height))

                            # Write YOLO detection format: class_id x_center y_center width height
                            f.write(f"{annotation['unified_class']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        else:
                            # Fallback for other datasets without bbox info - use full image
                            f.write(f"{annotation['unified_class']} 0.5 0.5 1.0 1.0\n")
        
        # Create data.yaml file
        data_yaml = {
            'path': str(yolo_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': list(self.class_names.values())
        }
        
        with open(yolo_dir / "data.yaml", 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"✓ YOLO format data saved to {yolo_dir}")
    
    def create_integration_report(self, annotations: List[Dict], splits: Dict[str, List[Dict]]):
        """Create comprehensive integration report"""
        print("Creating integration report...")
        
        # Calculate additional statistics
        annotations_df = pd.DataFrame(annotations)
        
        if len(annotations_df) > 0:
            self.integration_stats['quality_stats'] = {
                'mean_quality_score': float(annotations_df['quality_score'].mean()),
                'std_quality_score': float(annotations_df['quality_score'].std()),
                'min_quality_score': float(annotations_df['quality_score'].min()),
                'max_quality_score': float(annotations_df['quality_score'].max())
            }
        
        # Create report
        report = {
            'integration_summary': {
                'total_images': self.integration_stats['total_images'],
                'num_classes': len(self.class_names),
                'class_names': list(self.class_names.values()),
                'integration_date': pd.Timestamp.now().isoformat()
            },
            'class_distribution': dict(self.integration_stats['class_distribution']),
            'dataset_distribution': dict(self.integration_stats['dataset_distribution']),
            'split_sizes': {split: len(anns) for split, anns in splits.items()},
            'quality_statistics': self.integration_stats['quality_stats']
        }
        
        # Add class name mapping to distribution
        report['class_distribution_named'] = {
            self.class_names[class_id]: count 
            for class_id, count in self.integration_stats['class_distribution'].items()
        }
        
        # Save report
        report_path = self.integrated_data_dir / "metadata" / "integration_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Integration report saved to {report_path}")
        
        # Create visualization
        self.create_visualizations(annotations_df, splits)
        
        # Print summary
        self.print_integration_summary(report)
    
    def create_visualizations(self, annotations_df: pd.DataFrame, splits: Dict[str, List[Dict]]):
        """Create visualization plots"""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Malaria Dataset Integration Analysis', fontsize=16, fontweight='bold')
        
        # 1. Class Distribution
        ax = axes[0, 0]
        class_counts = annotations_df['unified_class_name'].value_counts()
        bars = ax.bar(range(len(class_counts)), class_counts.values)
        ax.set_title('Class Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(class_counts)))
        ax.set_xticklabels(class_counts.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{int(height)}', ha='center', va='bottom')
        
        # 2. Dataset Distribution
        ax = axes[0, 1]
        dataset_counts = annotations_df['dataset'].value_counts()
        ax.pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('Dataset Distribution')
        
        # 3. Quality Score Distribution
        ax = axes[0, 2]
        ax.hist(annotations_df['quality_score'], bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(annotations_df['quality_score'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {annotations_df["quality_score"].mean():.1f}')
        ax.set_title('Quality Score Distribution')
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # 4. Split Sizes
        ax = axes[1, 0]
        split_sizes = {split: len(anns) for split, anns in splits.items()}
        bars = ax.bar(split_sizes.keys(), split_sizes.values())
        ax.set_title('Dataset Split Sizes')
        ax.set_ylabel('Number of Images')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{int(height)}', ha='center', va='bottom')
        
        # 5. Class Balance per Split
        ax = axes[1, 1]
        split_class_data = []
        for split_name, split_annotations in splits.items():
            split_df = pd.DataFrame(split_annotations)
            class_counts = split_df['unified_class_name'].value_counts()
            for class_name, count in class_counts.items():
                split_class_data.append({'Split': split_name, 'Class': class_name, 'Count': count})
        
        split_class_df = pd.DataFrame(split_class_data)
        split_pivot = split_class_df.pivot(index='Class', columns='Split', values='Count').fillna(0)
        split_pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Class Distribution per Split')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.legend(title='Split')
        ax.tick_params(axis='x', rotation=45)
        
        # 6. Quality by Dataset
        ax = axes[1, 2]
        datasets = annotations_df['dataset'].unique()
        quality_data = []
        for dataset in datasets:
            dataset_df = annotations_df[annotations_df['dataset'] == dataset]
            quality_data.append(dataset_df['quality_score'].values)
        
        ax.boxplot(quality_data, labels=datasets)
        ax.set_title('Quality Score by Dataset')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Quality Score')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        viz_path = self.integrated_data_dir / "metadata" / "integration_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualizations saved to {viz_path}")
    
    def print_integration_summary(self, report: Dict):
        """Print integration summary"""
        print("\\n" + "="*70)
        print(" DATASET INTEGRATION SUMMARY ")
        print("="*70)
        print(f"Total images integrated: {report['integration_summary']['total_images']}")
        print(f"Number of classes: {report['integration_summary']['num_classes']}")
        print("\\nClass distribution:")
        for class_name, count in report['class_distribution_named'].items():
            percentage = (count / report['integration_summary']['total_images']) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print("\\nDataset split sizes:")
        for split, size in report['split_sizes'].items():
            percentage = (size / report['integration_summary']['total_images']) * 100
            print(f"  {split}: {size} ({percentage:.1f}%)")
        
        print("\\nQuality statistics:")
        if report['quality_statistics']:
            stats = report['quality_statistics']
            print(f"  Mean quality score: {stats['mean_quality_score']:.2f}")
            print(f"  Std quality score: {stats['std_quality_score']:.2f}")
            print(f"  Quality range: {stats['min_quality_score']:.2f} - {stats['max_quality_score']:.2f}")
        
        print("="*70)
    
    def integrate_all_datasets(self, 
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15, 
                             test_ratio: float = 0.15):
        """Run complete integration pipeline"""
        print("\\n" + "="*70)
        print(" MALARIA DATASET INTEGRATION PIPELINE ")
        print("="*70)
        
        # Load processed samples
        df = self.load_processed_samples()
        if len(df) == 0:
            return
        
        # Create unified annotations
        annotations = self.create_unified_annotations(df)
        
        # Copy images to unified structure
        self.copy_images_to_unified_structure(annotations)
        
        # Create dataset splits
        splits = self.create_dataset_splits(annotations, train_ratio, val_ratio, test_ratio)
        
        # Save annotations
        self.save_annotations(annotations, splits)
        
        # Create YOLO format data
        self.create_yolo_format_data(splits)
        
        # Create comprehensive report
        self.create_integration_report(annotations, splits)
        
        print("\\n✓ Dataset integration completed successfully!")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Integrate malaria datasets")
    parser.add_argument("--processed-dir", default="data/processed", 
                       help="Processed data directory")
    parser.add_argument("--output-dir", default="data/integrated", 
                       help="Output integrated data directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, 
                       help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, 
                       help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, 
                       help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Initialize integrator
    integrator = MalariaDatasetIntegrator(
        processed_data_dir=args.processed_dir,
        integrated_data_dir=args.output_dir
    )
    
    # Run integration
    integrator.integrate_all_datasets(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )


if __name__ == "__main__":
    main()
