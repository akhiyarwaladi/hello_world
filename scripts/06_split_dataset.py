#!/usr/bin/env python3
"""
Script to create final train/validation/test splits for malaria dataset
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
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from collections import defaultdict, Counter
import argparse
from tqdm import tqdm


class MalariaDatasetSplitter:
    """Creates final train/validation/test splits for malaria dataset"""
    
    def __init__(self, 
                 input_data_dir: str = "data/augmented",
                 output_data_dir: str = "data/final"):
        """Initialize splitter"""
        self.input_data_dir = Path(input_data_dir)
        self.output_data_dir = Path(output_data_dir)
        self.output_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory structure
        for split in ['train', 'val', 'test']:
            (self.output_data_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_data_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        (self.output_data_dir / "annotations").mkdir(exist_ok=True)
        (self.output_data_dir / "metadata").mkdir(exist_ok=True)
        
        # Class names
        self.class_names = {
            0: 'P_falciparum',
            1: 'P_vivax',
            2: 'P_malariae',
            3: 'P_ovale',
            4: 'Mixed_infection',
            5: 'Uninfected'
        }
        
        # Splitting statistics
        self.split_stats = {
            'total_samples': 0,
            'split_sizes': defaultdict(int),
            'class_distribution': defaultdict(lambda: defaultdict(int))
        }
    
    def load_augmented_data(self) -> pd.DataFrame:
        """Load augmented dataset"""
        # First try to load combined augmented data
        combined_file = self.input_data_dir / "annotations" / "all_augmented.csv"
        
        if combined_file.exists():
            df = pd.read_csv(combined_file)
            print(f"Loaded {len(df)} samples from combined augmented data")
            return df
        
        # Otherwise, load and combine individual split files
        all_data = []
        
        for split in ['train', 'val', 'test']:
            split_file = self.input_data_dir / "annotations" / f"{split}_augmented.csv"
            
            if split_file.exists():
                split_df = pd.read_csv(split_file)
                split_df['original_split'] = split
                all_data.append(split_df)
                print(f"Loaded {len(split_df)} samples from {split} split")
        
        if not all_data:
            print("No augmented data found. Trying to load integrated data...")
            # Fallback to integrated data
            integrated_file = self.input_data_dir.parent / "integrated" / "annotations" / "unified_annotations.csv"
            
            if integrated_file.exists():
                df = pd.read_csv(integrated_file)
                print(f"Loaded {len(df)} samples from integrated data")
                return df
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Combined {len(combined_df)} total samples")
            return combined_df
        
        return pd.DataFrame()
    
    def analyze_current_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze current data distribution"""
        if len(df) == 0:
            return {}
        
        analysis = {
            'total_samples': len(df),
            'class_distribution': {},
            'quality_stats': {},
            'dataset_sources': {},
        }
        
        # Class distribution
        class_counts = df['unified_class'].value_counts().sort_index()
        for class_id, count in class_counts.items():
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            percentage = (count / len(df)) * 100
            analysis['class_distribution'][class_name] = {
                'count': int(count),
                'percentage': percentage
            }
        
        # Quality statistics
        if 'quality_score' in df.columns:
            analysis['quality_stats'] = {
                'mean': float(df['quality_score'].mean()),
                'std': float(df['quality_score'].std()),
                'min': float(df['quality_score'].min()),
                'max': float(df['quality_score'].max())
            }
        
        # Dataset source distribution
        if 'dataset' in df.columns:
            dataset_counts = df['dataset'].value_counts()
            for dataset, count in dataset_counts.items():
                percentage = (count / len(df)) * 100
                analysis['dataset_sources'][dataset] = {
                    'count': int(count),
                    'percentage': percentage
                }
        
        return analysis
    
    def create_stratified_splits(self, 
                               df: pd.DataFrame,
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15,
                               random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """Create stratified train/validation/test splits"""
        
        # Verify ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        print(f"Creating stratified splits: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")
        
        # Use unified_class for stratification
        X = df.drop(['unified_class'], axis=1)
        y = df['unified_class']
        
        # First split: separate test set
        test_size = test_ratio
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        # Reconstruct dataframes
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        # Print split information
        print("\\nSplit sizes and class distributions:")
        for split_name, split_df in splits.items():
            class_dist = split_df['unified_class'].value_counts().sort_index()
            print(f"\\n{split_name.upper()} ({len(split_df)} samples):")
            for class_id, count in class_dist.items():
                class_name = self.class_names.get(class_id, f"Class_{class_id}")
                percentage = (count / len(split_df)) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return splits
    
    def copy_images_to_splits(self, splits: Dict[str, pd.DataFrame]):
        """Copy images to final split directories"""
        print("\\nCopying images to final split directories...")
        
        for split_name, split_df in splits.items():
            print(f"Copying {split_name} images...")
            
            split_images_dir = self.output_data_dir / split_name / "images"
            split_labels_dir = self.output_data_dir / split_name / "labels"
            
            processed_count = 0
            
            for idx, row in tqdm(split_df.iterrows(), total=len(split_df), 
                                desc=f"Copying {split_name}"):
                
                # Get source image path
                if 'unified_image_path' in row and pd.notna(row['unified_image_path']):
                    src_image_path = self.input_data_dir / row['unified_image_path']
                elif 'image_path' in row and pd.notna(row['image_path']):
                    src_image_path = self.input_data_dir / row['image_path']
                else:
                    print(f"Warning: No image path found for sample {idx}")
                    continue
                
                # Check if source exists
                if not src_image_path.exists():
                    # Try alternative paths
                    alt_paths = [
                        self.input_data_dir.parent / "integrated" / row.get('unified_image_path', ''),
                        self.input_data_dir.parent / "processed" / row.get('image_path', '')
                    ]
                    
                    src_found = False
                    for alt_path in alt_paths:
                        if alt_path.exists():
                            src_image_path = alt_path
                            src_found = True
                            break
                    
                    if not src_found:
                        print(f"Warning: Source image not found: {src_image_path}")
                        continue
                
                # Create destination filename
                image_id = row.get('image_id', idx)
                if isinstance(image_id, str) and not image_id.isdigit():
                    # Handle augmented IDs like "aug_000123_001"
                    dst_filename = f"{image_id}.jpg"
                else:
                    dst_filename = f"{image_id:06d}.jpg"
                
                dst_image_path = split_images_dir / dst_filename
                dst_label_path = split_labels_dir / f"{Path(dst_filename).stem}.txt"
                
                # Copy image
                shutil.copy2(src_image_path, dst_image_path)
                
                # Create label file (YOLO format)
                class_id = int(row['unified_class'])
                
                # For classification: just class_id
                # For detection: class_id x_center y_center width height
                # Using classification format here
                with open(dst_label_path, 'w') as f:
                    f.write(f"{class_id}\\n")
                
                processed_count += 1
            
            self.split_stats['split_sizes'][split_name] = processed_count
            print(f" Copied {processed_count} images to {split_name}")
    
    def save_split_annotations(self, splits: Dict[str, pd.DataFrame]):
        """Save split annotations"""
        print("\\nSaving split annotations...")
        
        for split_name, split_df in splits.items():
            # Add split information
            split_df_copy = split_df.copy()
            split_df_copy['final_split'] = split_name
            
            # Save CSV
            csv_path = self.output_data_dir / "annotations" / f"{split_name}_final.csv"
            split_df_copy.to_csv(csv_path, index=False)
            
            # Save JSON
            json_path = self.output_data_dir / "annotations" / f"{split_name}_final.json"
            with open(json_path, 'w') as f:
                json.dump(split_df_copy.to_dict('records'), f, indent=2)
            
            # Update class distribution stats
            class_counts = split_df['unified_class'].value_counts()
            for class_id, count in class_counts.items():
                self.split_stats['class_distribution'][split_name][class_id] = count
        
        # Save combined annotations
        combined_df = pd.concat([
            df.assign(final_split=split_name) 
            for split_name, df in splits.items()
        ], ignore_index=True)
        
        combined_csv = self.output_data_dir / "annotations" / "all_final.csv"
        combined_df.to_csv(combined_csv, index=False)
        
        combined_json = self.output_data_dir / "annotations" / "all_final.json"
        with open(combined_json, 'w') as f:
            json.dump(combined_df.to_dict('records'), f, indent=2)
        
        print(f" Annotations saved to {self.output_data_dir / 'annotations'}")
    
    def create_yolo_data_config(self):
        """Create YOLO data configuration file"""
        print("Creating YOLO data configuration...")
        
        # Create data.yaml for YOLO training
        data_config = {
            'path': str(self.output_data_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': list(self.class_names.values())
        }
        
        # Save data.yaml
        with open(self.output_data_dir / "data.yaml", 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        # Create classes.txt
        with open(self.output_data_dir / "classes.txt", 'w') as f:
            for class_name in self.class_names.values():
                f.write(f"{class_name}\\n")
        
        print(f" YOLO configuration saved to {self.output_data_dir}")
    
    def create_visualization(self, splits: Dict[str, pd.DataFrame], original_analysis: Dict):
        """Create visualization of final splits"""
        print("Creating split visualization...")
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Final Dataset Splits Analysis', fontsize=16, fontweight='bold')
        
        # 1. Split Sizes
        ax = axes[0, 0]
        split_sizes = [len(splits[split]) for split in ['train', 'val', 'test']]
        split_names = ['Train', 'Validation', 'Test']
        
        bars = ax.bar(split_names, split_sizes)
        ax.set_title('Split Sizes')
        ax.set_ylabel('Number of Samples')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        # 2. Class Distribution per Split
        ax = axes[0, 1]
        
        class_data = []
        for split_name, split_df in splits.items():
            class_counts = split_df['unified_class'].value_counts().sort_index()
            for class_id, count in class_counts.items():
                class_name = self.class_names.get(class_id, f"Class_{class_id}")
                class_data.append({
                    'Split': split_name.capitalize(),
                    'Class': class_name,
                    'Count': count
                })
        
        if class_data:
            class_df = pd.DataFrame(class_data)
            class_pivot = class_df.pivot(index='Class', columns='Split', values='Count').fillna(0)
            class_pivot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Class Distribution per Split')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.legend(title='Split')
            ax.tick_params(axis='x', rotation=45)
        
        # 3. Class Distribution Percentages
        ax = axes[0, 2]
        
        # Overall class distribution
        total_class_counts = defaultdict(int)
        for split_df in splits.values():
            class_counts = split_df['unified_class'].value_counts()
            for class_id, count in class_counts.items():
                total_class_counts[class_id] += count
        
        class_names = [self.class_names[cid] for cid in sorted(total_class_counts.keys())]
        class_counts_list = [total_class_counts[cid] for cid in sorted(total_class_counts.keys())]
        
        wedges, texts, autotexts = ax.pie(class_counts_list, labels=class_names, 
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title('Overall Class Distribution')
        
        # 4. Split Ratio Visualization
        ax = axes[1, 0]
        
        total_samples = sum(len(split_df) for split_df in splits.values())
        split_percentages = [len(split_df) / total_samples * 100 for split_df in splits.values()]
        
        wedges, texts, autotexts = ax.pie(split_percentages, labels=split_names,
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title('Split Ratios')
        
        # 5. Quality Distribution (if available)
        ax = axes[1, 1]
        
        has_quality = all('quality_score' in split_df.columns for split_df in splits.values())
        if has_quality:
            quality_data = []
            labels = []
            
            for split_name, split_df in splits.items():
                if 'quality_score' in split_df.columns:
                    quality_data.append(split_df['quality_score'].values)
                    labels.append(split_name.capitalize())
            
            if quality_data:
                ax.boxplot(quality_data, labels=labels)
                ax.set_title('Quality Score Distribution by Split')
                ax.set_ylabel('Quality Score')
        else:
            ax.text(0.5, 0.5, 'Quality scores\\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Quality Score Distribution')
            ax.axis('off')
        
        # 6. Dataset Source Distribution (if available)
        ax = axes[1, 2]
        
        has_dataset = all('dataset' in split_df.columns for split_df in splits.values())
        if has_dataset:
            dataset_counts = defaultdict(int)
            for split_df in splits.values():
                if 'dataset' in split_df.columns:
                    for dataset in split_df['dataset']:
                        dataset_counts[dataset] += 1
            
            if dataset_counts:
                datasets = list(dataset_counts.keys())
                counts = list(dataset_counts.values())
                
                bars = ax.bar(datasets, counts)
                ax.set_title('Dataset Source Distribution')
                ax.set_xlabel('Dataset')
                ax.set_ylabel('Sample Count')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'Dataset info\\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Dataset Source Distribution')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_data_dir / "metadata" / "final_splits_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Visualization saved to {viz_path}")
    
    def create_final_report(self, splits: Dict[str, pd.DataFrame], original_analysis: Dict):
        """Create comprehensive final report"""
        print("Creating final dataset report...")
        
        total_samples = sum(len(split_df) for split_df in splits.values())
        self.split_stats['total_samples'] = total_samples
        
        report = {
            'dataset_summary': {
                'total_samples': total_samples,
                'num_classes': len(self.class_names),
                'class_names': list(self.class_names.values()),
                'creation_date': pd.Timestamp.now().isoformat(),
                'output_directory': str(self.output_data_dir.absolute())
            },
            'split_information': {},
            'class_distribution': {},
            'quality_statistics': {},
            'training_commands': {}
        }
        
        # Split information
        for split_name, split_df in splits.items():
            class_dist = split_df['unified_class'].value_counts().sort_index()
            
            split_info = {
                'sample_count': len(split_df),
                'percentage': (len(split_df) / total_samples) * 100,
                'class_distribution': {
                    self.class_names[class_id]: int(count)
                    for class_id, count in class_dist.items()
                }
            }
            
            # Quality stats if available
            if 'quality_score' in split_df.columns:
                split_info['quality_stats'] = {
                    'mean': float(split_df['quality_score'].mean()),
                    'std': float(split_df['quality_score'].std()),
                    'min': float(split_df['quality_score'].min()),
                    'max': float(split_df['quality_score'].max())
                }
            
            report['split_information'][split_name] = split_info
        
        # Overall class distribution
        total_class_counts = defaultdict(int)
        for split_df in splits.values():
            class_counts = split_df['unified_class'].value_counts()
            for class_id, count in class_counts.items():
                total_class_counts[class_id] += count
        
        for class_id, count in total_class_counts.items():
            class_name = self.class_names[class_id]
            report['class_distribution'][class_name] = {
                'total_count': int(count),
                'percentage': (count / total_samples) * 100
            }
        
        # Training commands
        data_yaml_path = self.output_data_dir / "data.yaml"
        report['training_commands'] = {
            'yolo_classification': f"yolo classify train data={data_yaml_path} model=yolov8m-cls.pt epochs=100 batch=16",
            'yolo_detection': f"yolo detect train data={data_yaml_path} model=yolov8m.pt epochs=100 batch=16",
            'custom_training': f"python train_model.py --data {data_yaml_path} --epochs 100 --batch 16"
        }
        
        # File structure
        report['file_structure'] = {
            'data_config': 'data.yaml',
            'class_names': 'classes.txt',
            'annotations': {
                'train': 'annotations/train_final.csv',
                'val': 'annotations/val_final.csv',
                'test': 'annotations/test_final.csv',
                'combined': 'annotations/all_final.csv'
            },
            'images': {
                'train': 'train/images/',
                'val': 'val/images/',
                'test': 'test/images/'
            },
            'labels': {
                'train': 'train/labels/',
                'val': 'val/labels/',
                'test': 'test/labels/'
            }
        }
        
        # Save report
        report_path = self.output_data_dir / "metadata" / "final_dataset_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f" Final report saved to {report_path}")
        
        # Print summary
        self.print_final_summary(report)
    
    def print_final_summary(self, report: Dict):
        """Print final dataset summary"""
        print("\\n" + "="*70)
        print(" FINAL DATASET SUMMARY ")
        print("="*70)
        
        summary = report['dataset_summary']
        print(f"Total samples: {summary['total_samples']}")
        print(f"Number of classes: {summary['num_classes']}")
        print(f"Output directory: {summary['output_directory']}")
        
        print("\\nSplit distribution:")
        for split_name, split_info in report['split_information'].items():
            print(f"  {split_name}: {split_info['sample_count']} samples ({split_info['percentage']:.1f}%)")
        
        print("\\nClass distribution:")
        for class_name, class_info in report['class_distribution'].items():
            print(f"  {class_name}: {class_info['total_count']} samples ({class_info['percentage']:.1f}%)")
        
        print("\\nTraining commands:")
        for cmd_name, command in report['training_commands'].items():
            print(f"  {cmd_name}:")
            print(f"    {command}")
        
        print("="*70)
    
    def create_final_splits(self,
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15,
                          random_state: int = 42):
        """Main function to create final dataset splits"""
        print("\\n" + "="*70)
        print(" FINAL DATASET SPLITTING ")
        print("="*70)
        
        # Load augmented data
        df = self.load_augmented_data()
        
        if len(df) == 0:
            print("No data found. Please run previous steps first.")
            return
        
        # Analyze current distribution
        original_analysis = self.analyze_current_distribution(df)
        print(f"\\nLoaded {original_analysis['total_samples']} total samples")
        
        print("\\nCurrent class distribution:")
        for class_name, info in original_analysis['class_distribution'].items():
            print(f"  {class_name}: {info['count']} ({info['percentage']:.1f}%)")
        
        # Create stratified splits
        splits = self.create_stratified_splits(
            df, train_ratio, val_ratio, test_ratio, random_state
        )
        
        # Copy images to split directories
        self.copy_images_to_splits(splits)
        
        # Save annotations
        self.save_split_annotations(splits)
        
        # Create YOLO configuration
        self.create_yolo_data_config()
        
        # Create visualization and report
        self.create_visualization(splits, original_analysis)
        self.create_final_report(splits, original_analysis)
        
        print("\\n Final dataset splitting completed successfully!")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Create final dataset splits")
    parser.add_argument("--input-dir", default="data/augmented",
                       help="Input data directory (augmented or integrated)")
    parser.add_argument("--output-dir", default="data/final",
                       help="Final output directory")
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
    
    # Initialize splitter
    splitter = MalariaDatasetSplitter(
        input_data_dir=args.input_dir,
        output_data_dir=args.output_dir
    )
    
    # Create final splits
    splitter.create_final_splits(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )


if __name__ == "__main__":
    main()