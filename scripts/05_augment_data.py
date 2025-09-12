#!/usr/bin/env python3
"""
Script for data augmentation to balance malaria datasets
Author: Malaria Detection Team
Date: 2024
"""

import os
import cv2
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm
import argparse
from collections import defaultdict, Counter
import random


class MalariaDataAugmenter:
    """Data augmenter for balancing malaria datasets"""
    
    def __init__(self, 
                 integrated_data_dir: str = "data/integrated",
                 augmented_output_dir: str = "data/augmented"):
        """Initialize augmenter"""
        self.integrated_data_dir = Path(integrated_data_dir)
        self.augmented_output_dir = Path(augmented_output_dir)
        self.augmented_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory structure
        (self.augmented_output_dir / "images").mkdir(exist_ok=True)
        (self.augmented_output_dir / "annotations").mkdir(exist_ok=True)
        (self.augmented_output_dir / "metadata").mkdir(exist_ok=True)
        
        # Class names
        self.class_names = {
            0: 'P_falciparum',
            1: 'P_vivax',
            2: 'P_malariae',
            3: 'P_ovale',
            4: 'Mixed_infection',
            5: 'Uninfected'
        }
        
        # Augmentation statistics
        self.augmentation_stats = {
            'original_counts': defaultdict(int),
            'augmented_counts': defaultdict(int),
            'total_generated': 0,
            'class_targets': {}
        }
        
        # Create augmentation pipeline
        self.augmentation_transforms = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create albumentations augmentation pipeline"""
        transforms = A.Compose([
            # Geometric transforms
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.8),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=30,
                    p=0.8
                ),
            ], p=0.8),
            
            # Color/intensity transforms
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.8
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.8
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            ], p=0.7),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
            ], p=0.4),
            
            # Advanced transforms
            A.OneOf([
                A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50, p=0.3
                ),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
            ], p=0.3),
            
        ], p=1.0)
        
        return transforms
    
    def load_integrated_annotations(self) -> Dict[str, pd.DataFrame]:
        """Load integrated annotations"""
        annotations = {}
        
        for split in ['train', 'val', 'test']:
            annotation_file = self.integrated_data_dir / "annotations" / f"{split}_annotations.csv"
            
            if annotation_file.exists():
                df = pd.read_csv(annotation_file)
                annotations[split] = df
                print(f"Loaded {len(df)} {split} annotations")
        
        return annotations
    
    def analyze_class_distribution(self, annotations: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Analyze current class distribution"""
        analysis = {}
        
        for split_name, split_df in annotations.items():
            if len(split_df) == 0:
                continue
            
            class_counts = split_df['unified_class'].value_counts().sort_index()
            total_samples = len(split_df)
            
            class_info = {}
            for class_id, count in class_counts.items():
                class_name = self.class_names.get(class_id, f"Class_{class_id}")
                class_info[class_id] = {
                    'name': class_name,
                    'count': int(count),
                    'percentage': (count / total_samples) * 100
                }
            
            analysis[split_name] = {
                'total_samples': total_samples,
                'class_distribution': class_info,
                'imbalance_ratio': max(class_counts) / min(class_counts) if len(class_counts) > 1 else 1.0
            }
        
        return analysis
    
    def calculate_augmentation_targets(self, 
                                     annotations: Dict[str, pd.DataFrame],
                                     target_samples_per_class: Optional[int] = None,
                                     balance_method: str = "oversample_to_max") -> Dict[str, Dict[int, int]]:
        """Calculate how many augmented samples needed per class"""
        targets = {}
        
        for split_name, split_df in annotations.items():
            if len(split_df) == 0:
                continue
            
            class_counts = split_df['unified_class'].value_counts().sort_index()
            
            if balance_method == "oversample_to_max":
                # Augment minority classes to match majority class
                max_count = max(class_counts)
                targets[split_name] = {}
                
                for class_id in class_counts.index:
                    current_count = class_counts[class_id]
                    if target_samples_per_class:
                        target_count = max(target_samples_per_class, current_count)
                    else:
                        target_count = max_count
                    
                    augmented_needed = max(0, target_count - current_count)
                    targets[split_name][class_id] = augmented_needed
                    
            elif balance_method == "uniform":
                # Set uniform target for all classes
                if target_samples_per_class is None:
                    target_samples_per_class = int(np.mean(class_counts))
                
                targets[split_name] = {}
                for class_id in class_counts.index:
                    current_count = class_counts[class_id]
                    augmented_needed = max(0, target_samples_per_class - current_count)
                    targets[split_name][class_id] = augmented_needed
        
        return targets
    
    def augment_single_image(self, image_path: Path, num_augmentations: int = 1) -> List[np.ndarray]:
        """Apply augmentation to single image"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        # Convert BGR to RGB for albumentations
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented_images = []
        
        for _ in range(num_augmentations):
            # Apply augmentation
            augmented = self.augmentation_transforms(image=image_rgb)
            augmented_image = augmented['image']
            
            # Convert back to BGR for saving
            augmented_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            augmented_images.append(augmented_bgr)
        
        return augmented_images
    
    def augment_class_samples(self, 
                            split_df: pd.DataFrame, 
                            class_id: int, 
                            num_augmentations: int,
                            output_dir: Path) -> List[Dict]:
        """Augment samples for a specific class"""
        class_samples = split_df[split_df['unified_class'] == class_id]
        
        if len(class_samples) == 0:
            return []
        
        augmented_annotations = []
        generated_count = 0
        
        # Create class subdirectory
        class_name = self.class_names[class_id]
        class_output_dir = output_dir / "images" / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"    Augmenting {class_name} class: need {num_augmentations} samples")
        
        # Calculate augmentations per original image
        samples_list = class_samples.to_dict('records')
        augmentations_per_sample = max(1, num_augmentations // len(samples_list))
        remaining_augmentations = num_augmentations % len(samples_list)
        
        for idx, sample in enumerate(tqdm(samples_list, desc=f"    Augmenting {class_name}")):
            # Get original image path
            original_image_path = self.integrated_data_dir / sample['unified_image_path']
            
            if not original_image_path.exists():
                continue
            
            # Determine number of augmentations for this sample
            current_augmentations = augmentations_per_sample
            if idx < remaining_augmentations:
                current_augmentations += 1
            
            if current_augmentations == 0:
                continue
            
            # Generate augmented images
            augmented_images = self.augment_single_image(
                original_image_path, current_augmentations
            )
            
            # Save augmented images and create annotations
            for aug_idx, aug_image in enumerate(augmented_images):
                # Create unique filename
                aug_filename = f"aug_{sample['image_id']:06d}_{aug_idx:03d}.jpg"
                aug_path = class_output_dir / aug_filename
                
                # Save augmented image
                cv2.imwrite(str(aug_path), aug_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Create annotation
                aug_annotation = sample.copy()
                aug_annotation.update({
                    'image_id': f"aug_{sample['image_id']:06d}_{aug_idx:03d}",
                    'unified_image_path': str(aug_path.relative_to(output_dir)),
                    'is_augmented': True,
                    'augmentation_source': sample['image_id'],
                    'augmentation_index': aug_idx
                })
                
                augmented_annotations.append(aug_annotation)
                generated_count += 1
                
                if generated_count >= num_augmentations:
                    break
            
            if generated_count >= num_augmentations:
                break
        
        self.augmentation_stats['augmented_counts'][class_id] = generated_count
        
        return augmented_annotations
    
    def augment_split_data(self, 
                          split_name: str, 
                          split_df: pd.DataFrame,
                          augmentation_targets: Dict[int, int]) -> pd.DataFrame:
        """Augment data for a single split"""
        print(f"\\nAugmenting {split_name} split...")
        
        # Copy original annotations
        all_annotations = split_df.to_dict('records')
        
        # Update original counts
        class_counts = split_df['unified_class'].value_counts()
        for class_id, count in class_counts.items():
            self.augmentation_stats['original_counts'][class_id] += count
        
        # Generate augmented samples for each class
        for class_id, num_augmentations in augmentation_targets.items():
            if num_augmentations > 0:
                augmented_annotations = self.augment_class_samples(
                    split_df, class_id, num_augmentations, self.augmented_output_dir
                )
                all_annotations.extend(augmented_annotations)
                self.augmentation_stats['total_generated'] += len(augmented_annotations)
        
        return pd.DataFrame(all_annotations)
    
    def create_augmentation_visualization(self, 
                                        original_annotations: Dict[str, pd.DataFrame],
                                        augmented_annotations: Dict[str, pd.DataFrame]):
        """Create visualization of augmentation results"""
        print("Creating augmentation visualization...")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Augmentation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Original vs Augmented Distribution
        ax = axes[0, 0]
        
        # Collect data for all splits
        orig_class_counts = defaultdict(int)
        aug_class_counts = defaultdict(int)
        
        for split_df in original_annotations.values():
            if len(split_df) > 0:
                counts = split_df['unified_class'].value_counts()
                for class_id, count in counts.items():
                    orig_class_counts[class_id] += count
        
        for split_df in augmented_annotations.values():
            if len(split_df) > 0:
                counts = split_df['unified_class'].value_counts()
                for class_id, count in counts.items():
                    aug_class_counts[class_id] += count
        
        # Plot comparison
        class_ids = sorted(set(orig_class_counts.keys()) | set(aug_class_counts.keys()))
        class_labels = [self.class_names.get(cid, f"Class_{cid}") for cid in class_ids]
        
        orig_counts = [orig_class_counts.get(cid, 0) for cid in class_ids]
        aug_counts = [aug_class_counts.get(cid, 0) for cid in class_ids]
        
        x = np.arange(len(class_ids))
        width = 0.35
        
        ax.bar(x - width/2, orig_counts, width, label='Original', alpha=0.8)
        ax.bar(x + width/2, aug_counts, width, label='After Augmentation', alpha=0.8)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution: Before vs After Augmentation')
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.legend()
        
        # 2. Augmentation Generated per Class
        ax = axes[0, 1]
        generated_counts = []
        for class_id in class_ids:
            generated = aug_counts[class_ids.index(class_id)] - orig_counts[class_ids.index(class_id)]
            generated_counts.append(max(0, generated))
        
        bars = ax.bar(class_labels, generated_counts)
        ax.set_title('Augmented Samples Generated per Class')
        ax.set_xlabel('Class')
        ax.set_ylabel('Generated Samples')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
        
        # 3. Imbalance Ratios
        ax = axes[1, 0]
        
        orig_max = max(orig_counts) if orig_counts else 1
        orig_min = min([c for c in orig_counts if c > 0]) if orig_counts else 1
        orig_ratio = orig_max / orig_min
        
        aug_max = max(aug_counts) if aug_counts else 1
        aug_min = min([c for c in aug_counts if c > 0]) if aug_counts else 1
        aug_ratio = aug_max / aug_min
        
        categories = ['Before Augmentation', 'After Augmentation']
        ratios = [orig_ratio, aug_ratio]
        
        bars = ax.bar(categories, ratios)
        ax.set_title('Class Imbalance Ratio')
        ax.set_ylabel('Max/Min Ratio')
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.1f}x', ha='center', va='bottom')
        
        # 4. Split-wise Distribution
        ax = axes[1, 1]
        
        split_data = []
        for split_name in ['train', 'val', 'test']:
            if split_name in augmented_annotations:
                split_df = augmented_annotations[split_name]
                total_count = len(split_df)
                split_data.append({'Split': split_name, 'Count': total_count})
        
        if split_data:
            split_df_plot = pd.DataFrame(split_data)
            bars = ax.bar(split_df_plot['Split'], split_df_plot['Count'])
            ax.set_title('Final Dataset Split Sizes')
            ax.set_xlabel('Split')
            ax.set_ylabel('Sample Count')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.augmented_output_dir / "metadata" / "augmentation_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Visualization saved to {viz_path}")
    
    def save_augmented_data(self, augmented_annotations: Dict[str, pd.DataFrame]):
        """Save augmented annotations"""
        print("Saving augmented annotations...")
        
        # Save annotations for each split
        for split_name, split_df in augmented_annotations.items():
            # Save CSV
            csv_path = self.augmented_output_dir / "annotations" / f"{split_name}_augmented.csv"
            split_df.to_csv(csv_path, index=False)
            
            # Save JSON
            json_path = self.augmented_output_dir / "annotations" / f"{split_name}_augmented.json"
            with open(json_path, 'w') as f:
                json.dump(split_df.to_dict('records'), f, indent=2)
        
        # Save combined annotations
        combined_df = pd.concat(augmented_annotations.values(), ignore_index=True)
        combined_csv = self.augmented_output_dir / "annotations" / "all_augmented.csv"
        combined_df.to_csv(combined_csv, index=False)
        
        print(f" Augmented annotations saved to {self.augmented_output_dir / 'annotations'}")
    
    def create_augmentation_report(self, 
                                 original_annotations: Dict[str, pd.DataFrame],
                                 augmented_annotations: Dict[str, pd.DataFrame]):
        """Create comprehensive augmentation report"""
        print("Creating augmentation report...")
        
        # Calculate summary statistics
        orig_total = sum(len(df) for df in original_annotations.values())
        aug_total = sum(len(df) for df in augmented_annotations.values())
        
        report = {
            'augmentation_summary': {
                'original_samples': orig_total,
                'augmented_samples': aug_total,
                'generated_samples': aug_total - orig_total,
                'augmentation_factor': aug_total / orig_total if orig_total > 0 else 0,
                'augmentation_date': pd.Timestamp.now().isoformat()
            },
            'original_distribution': dict(self.augmentation_stats['original_counts']),
            'augmented_distribution': dict(self.augmentation_stats['augmented_counts']),
            'split_summary': {},
            'class_summary': {}
        }
        
        # Split-wise summary
        for split_name in augmented_annotations.keys():
            orig_count = len(original_annotations.get(split_name, []))
            aug_count = len(augmented_annotations[split_name])
            
            report['split_summary'][split_name] = {
                'original_count': orig_count,
                'final_count': aug_count,
                'generated_count': aug_count - orig_count
            }
        
        # Class-wise summary
        for class_id, class_name in self.class_names.items():
            orig_count = self.augmentation_stats['original_counts'].get(class_id, 0)
            aug_count = self.augmentation_stats['augmented_counts'].get(class_id, 0)
            
            if orig_count > 0 or aug_count > 0:
                report['class_summary'][class_name] = {
                    'class_id': class_id,
                    'original_count': orig_count,
                    'generated_count': aug_count,
                    'final_count': orig_count + aug_count
                }
        
        # Save report
        report_path = self.augmented_output_dir / "metadata" / "augmentation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f" Augmentation report saved to {report_path}")
        
        # Print summary
        self.print_augmentation_summary(report)
    
    def print_augmentation_summary(self, report: Dict):
        """Print augmentation summary"""
        print("\\n" + "="*60)
        print(" DATA AUGMENTATION SUMMARY ")
        print("="*60)
        
        summary = report['augmentation_summary']
        print(f"Original samples: {summary['original_samples']}")
        print(f"Final samples: {summary['augmented_samples']}")
        print(f"Generated samples: {summary['generated_samples']}")
        print(f"Augmentation factor: {summary['augmentation_factor']:.2f}x")
        
        print("\\nSplit-wise results:")
        for split, stats in report['split_summary'].items():
            print(f"  {split}: {stats['original_count']} ’ {stats['final_count']} "
                  f"(+{stats['generated_count']})")
        
        print("\\nClass-wise results:")
        for class_name, stats in report['class_summary'].items():
            if stats['final_count'] > 0:
                print(f"  {class_name}: {stats['original_count']} ’ {stats['final_count']} "
                      f"(+{stats['generated_count']})")
        
        print("="*60)
    
    def augment_dataset(self, 
                       target_samples_per_class: Optional[int] = None,
                       balance_method: str = "oversample_to_max",
                       augment_splits: List[str] = None):
        """Main augmentation function"""
        print("\\n" + "="*60)
        print(" MALARIA DATASET AUGMENTATION ")
        print("="*60)
        
        if augment_splits is None:
            augment_splits = ['train']  # Usually only augment training data
        
        # Load original annotations
        original_annotations = self.load_integrated_annotations()
        
        if not original_annotations:
            print("No integrated annotations found. Please run integration step first.")
            return
        
        # Analyze current distribution
        distribution_analysis = self.analyze_class_distribution(original_annotations)
        
        print("\\nCurrent class distribution:")
        for split_name, analysis in distribution_analysis.items():
            print(f"\\n{split_name.upper()} split:")
            for class_id, info in analysis['class_distribution'].items():
                print(f"  {info['name']}: {info['count']} ({info['percentage']:.1f}%)")
            print(f"  Imbalance ratio: {analysis['imbalance_ratio']:.1f}x")
        
        # Calculate augmentation targets
        augmentation_targets = self.calculate_augmentation_targets(
            original_annotations, target_samples_per_class, balance_method
        )
        
        print("\\nAugmentation targets:")
        for split_name, targets in augmentation_targets.items():
            if split_name in augment_splits:
                print(f"\\n{split_name.upper()} split:")
                for class_id, num_aug in targets.items():
                    class_name = self.class_names[class_id]
                    print(f"  {class_name}: +{num_aug} samples")
        
        # Perform augmentation
        augmented_annotations = {}
        
        for split_name, split_df in original_annotations.items():
            if split_name in augment_splits and split_name in augmentation_targets:
                # Augment this split
                aug_df = self.augment_split_data(
                    split_name, split_df, augmentation_targets[split_name]
                )
                augmented_annotations[split_name] = aug_df
            else:
                # Keep original data unchanged
                augmented_annotations[split_name] = split_df
        
        # Save augmented data
        self.save_augmented_data(augmented_annotations)
        
        # Create visualization and report
        self.create_augmentation_visualization(original_annotations, augmented_annotations)
        self.create_augmentation_report(original_annotations, augmented_annotations)
        
        print("\\n Data augmentation completed successfully!")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Augment malaria dataset")
    parser.add_argument("--integrated-dir", default="data/integrated",
                       help="Integrated data directory")
    parser.add_argument("--output-dir", default="data/augmented",
                       help="Augmented output directory")
    parser.add_argument("--target-samples", type=int,
                       help="Target samples per class")
    parser.add_argument("--balance-method", 
                       choices=["oversample_to_max", "uniform"],
                       default="oversample_to_max",
                       help="Balancing method")
    parser.add_argument("--augment-splits", nargs="+", 
                       default=["train"],
                       help="Splits to augment")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize augmenter
    augmenter = MalariaDataAugmenter(
        integrated_data_dir=args.integrated_dir,
        augmented_output_dir=args.output_dir
    )
    
    # Run augmentation
    augmenter.augment_dataset(
        target_samples_per_class=args.target_samples,
        balance_method=args.balance_method,
        augment_splits=args.augment_splits
    )


if __name__ == "__main__":
    main()