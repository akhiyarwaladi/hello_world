#!/usr/bin/env python3
"""
Script to preprocess malaria datasets
Author: Malaria Detection Team
Date: 2024
"""

import os
import cv2
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import shutil
from sklearn.model_selection import train_test_split


class MalariaDataPreprocessor:
    """Preprocessor for malaria dataset images"""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        """Initialize preprocessor"""
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard image size for preprocessing
        self.target_size = 640
        self.quality_threshold = 30  # Minimum quality score
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'total_rejected': 0,
            'datasets_processed': {},
            'resolution_stats': {},
            'quality_stats': {}
        }
    
    def check_image_quality(self, image_path: Path) -> Tuple[bool, float, Dict]:
        """Check image quality and return quality metrics"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False, 0.0, {}
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate various quality metrics
            metrics = {}
            
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['sharpness'] = float(laplacian_var)
            
            # 2. Contrast (standard deviation)
            metrics['contrast'] = float(np.std(gray))
            
            # 3. Brightness (mean intensity)
            metrics['brightness'] = float(np.mean(gray))
            
            # 4. Check for extremely dark or bright images
            dark_pixels = np.sum(gray < 30) / gray.size
            bright_pixels = np.sum(gray > 225) / gray.size
            metrics['dark_ratio'] = float(dark_pixels)
            metrics['bright_ratio'] = float(bright_pixels)
            
            # 5. Calculate overall quality score
            quality_score = 0
            if laplacian_var > 50:  # Sharp enough
                quality_score += 30
            if 20 < metrics['contrast'] < 100:  # Good contrast
                quality_score += 25
            if 50 < metrics['brightness'] < 200:  # Good brightness
                quality_score += 25
            if dark_pixels < 0.1 and bright_pixels < 0.1:  # Not too extreme
                quality_score += 20
            
            metrics['quality_score'] = quality_score
            is_good_quality = quality_score >= self.quality_threshold
            
            return is_good_quality, quality_score, metrics
            
        except Exception as e:
            print(f"Error checking quality for {image_path}: {e}")
            return False, 0.0, {}
    
    def resize_and_pad(self, image: np.ndarray, target_size: int = 640) -> np.ndarray:
        """Resize image while maintaining aspect ratio and pad to square"""
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Calculate padding
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Pad image with gray color
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[114, 114, 114]
        )
        
        return padded
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization and normalization"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Merge channels and convert back
            enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_bgr
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def process_nih_cell_dataset(self) -> List[Dict]:
        """Process NIH cell images dataset"""
        print("\nProcessing NIH Cell Images Dataset...")
        
        dataset_dir = self.raw_data_dir / "nih_cell"
        processed_samples = []
        
        if not dataset_dir.exists():
            print(f"NIH dataset not found at {dataset_dir}")
            return processed_samples
        
        # Find extracted cell_images folder
        cell_images_dir = None
        for item in dataset_dir.iterdir():
            if item.is_dir() and "cell_images" in item.name.lower():
                cell_images_dir = item
                break
        
        if not cell_images_dir:
            print("cell_images directory not found")
            return processed_samples
        
        # Process Parasitized and Uninfected folders
        for class_folder in ["Parasitized", "Uninfected"]:
            class_dir = cell_images_dir / class_folder
            if not class_dir.exists():
                continue
            
            class_label = "infected" if class_folder == "Parasitized" else "uninfected"
            image_files = list(class_dir.glob("*.png"))
            
            print(f"  Processing {len(image_files)} {class_label} images...")
            
            for img_path in tqdm(image_files, desc=f"Processing {class_folder}"):
                # Check quality
                is_good, quality_score, metrics = self.check_image_quality(img_path)
                
                if not is_good:
                    self.stats['total_rejected'] += 1
                    continue
                
                # Load and process image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Resize and normalize
                img_resized = self.resize_and_pad(img, self.target_size)
                img_enhanced = self.normalize_image(img_resized)
                
                # Save processed image
                output_filename = f"nih_cell_{class_label}_{img_path.stem}.jpg"
                output_path = self.processed_data_dir / "images" / output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                cv2.imwrite(str(output_path), img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Record sample info
                sample_info = {
                    'image_path': str(output_path.relative_to(self.processed_data_dir)),
                    'original_path': str(img_path),
                    'dataset': 'nih_cell',
                    'class': class_label,
                    'species': 'none' if class_label == 'uninfected' else 'unknown',
                    'quality_score': quality_score,
                    'original_size': f"{img.shape[1]}x{img.shape[0]}",
                    'processed_size': f"{self.target_size}x{self.target_size}",
                    **metrics
                }
                processed_samples.append(sample_info)
                self.stats['total_processed'] += 1
        
        return processed_samples
    
    def process_mp_idb_dataset(self) -> List[Dict]:
        """Process MP-IDB dataset with correct species mapping.
        Expected structure:
          data/raw/mp_idb/
            ├── Falciparum/{img,gt,...}
            ├── Vivax/{img,gt,...}
            ├── Malariae/{img,gt,...}
            └── Ovale/{img,gt,...}
        We infer species from the top-level directory name, not inner folders like 'img'.
        """
        print("\nProcessing MP-IDB Dataset...")

        dataset_dir = self.raw_data_dir / "mp_idb"
        processed_samples = []

        if not dataset_dir.exists():
            print(f"MP-IDB dataset not found at {dataset_dir}")
            return processed_samples

        # Map folder names to canonical species labels
        species_map = {
            'falciparum': 'P_falciparum',
            'vivax': 'P_vivax',
            'malariae': 'P_malariae',
            'ovale': 'P_ovale',
        }

        # Iterate only top-level species directories
        for sp_dir in sorted([d for d in dataset_dir.iterdir() if d.is_dir()]):
            sp_name_lower = sp_dir.name.lower()
            if sp_name_lower.startswith('.'):
                continue

            species = None
            for key, val in species_map.items():
                if key in sp_name_lower:
                    species = val
                    break

            if species is None:
                # Not a species directory; skip (e.g., .git, scripts)
                continue

            # Prefer images under 'img' subfolder if exists
            img_root = sp_dir / 'img'
            search_root = img_root if img_root.exists() else sp_dir

            image_files: List[Path] = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp', '*.JPG', '*.PNG']:
                image_files.extend(list(search_root.rglob(ext)))

            if not image_files:
                print(f"  No images found under {sp_dir}")
                continue

            print(f"  Processing {len(image_files)} images for {species} from {sp_dir.name}...")

            for img_path in tqdm(image_files, desc=f"Processing {sp_dir.name}"):
                # Check quality
                is_good, quality_score, metrics = self.check_image_quality(img_path)
                if not is_good:
                    self.stats['total_rejected'] += 1
                    continue

                # Load and process image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Resize and normalize
                img_resized = self.resize_and_pad(img, self.target_size)
                img_enhanced = self.normalize_image(img_resized)

                # Save processed image
                output_filename = f"mp_idb_{species}_{img_path.stem}.jpg"
                output_path = self.processed_data_dir / "images" / output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Record sample info
                sample_info = {
                    'image_path': str(output_path.relative_to(self.processed_data_dir)),
                    'original_path': str(img_path),
                    'dataset': 'mp_idb',
                    'class': 'infected',
                    'species': species,
                    'quality_score': quality_score,
                    'original_size': f"{img.shape[1]}x{img.shape[0]}",
                    'processed_size': f"{self.target_size}x{self.target_size}",
                    **metrics
                }
                processed_samples.append(sample_info)
                self.stats['total_processed'] += 1

        return processed_samples
    
    def process_kaggle_dataset(self) -> List[Dict]:
        """Process Kaggle NIH dataset"""
        print("\nProcessing Kaggle NIH Dataset...")
        
        dataset_dir = self.raw_data_dir / "kaggle_nih"
        processed_samples = []
        
        if not dataset_dir.exists():
            print(f"Kaggle dataset not found at {dataset_dir}")
            return processed_samples
        
        # Look for cell_images directory
        cell_images_dirs = list(dataset_dir.glob("**/cell_images"))
        if not cell_images_dirs:
            print("cell_images directory not found in Kaggle dataset")
            return processed_samples
        
        cell_images_dir = cell_images_dirs[0]
        
        # Process Parasitized and Uninfected folders
        for class_folder in ["Parasitized", "Uninfected"]:
            class_dir = cell_images_dir / class_folder
            if not class_dir.exists():
                continue
            
            class_label = "infected" if class_folder == "Parasitized" else "uninfected"
            image_files = list(class_dir.glob("*.png"))
            
            print(f"  Processing {len(image_files)} {class_label} images...")
            
            for img_path in tqdm(image_files, desc=f"Processing {class_folder}"):
                # Check quality
                is_good, quality_score, metrics = self.check_image_quality(img_path)
                
                if not is_good:
                    self.stats['total_rejected'] += 1
                    continue
                
                # Load and process image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Resize and normalize
                img_resized = self.resize_and_pad(img, self.target_size)
                img_enhanced = self.normalize_image(img_resized)
                
                # Save processed image
                output_filename = f"kaggle_nih_{class_label}_{img_path.stem}.jpg"
                output_path = self.processed_data_dir / "images" / output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                cv2.imwrite(str(output_path), img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Record sample info
                sample_info = {
                    'image_path': str(output_path.relative_to(self.processed_data_dir)),
                    'original_path': str(img_path),
                    'dataset': 'kaggle_nih',
                    'class': class_label,
                    'species': 'none' if class_label == 'uninfected' else 'unknown',
                    'quality_score': quality_score,
                    'original_size': f"{img.shape[1]}x{img.shape[0]}",
                    'processed_size': f"{self.target_size}x{self.target_size}",
                    **metrics
                }
                processed_samples.append(sample_info)
                self.stats['total_processed'] += 1
        
        return processed_samples
    
    def process_nih_thick_smear_datasets(self) -> List[Dict]:
        """Process NIH thick smear datasets with species-specific labels"""
        print("\nProcessing NIH Thick Smear Datasets...")
        
        all_samples = []
        
        # Define thick smear datasets with their species
        thick_datasets = {
            'nih_thick_pf': 'P_falciparum',
            'nih_thick_pv': 'P_vivax', 
            'nih_thick_uninfected': 'none'
        }
        
        for dataset_name, species in thick_datasets.items():
            dataset_dir = self.raw_data_dir / dataset_name
            if not dataset_dir.exists():
                print(f"  {dataset_name} not found, skipping...")
                continue
                
            # Find image files in all subdirectories
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']:
                image_files.extend(list(dataset_dir.rglob(ext)))
            
            if not image_files:
                print(f"  No images found in {dataset_name}")
                continue
                
            print(f"  Processing {len(image_files)} images from {dataset_name} ({species})...")
            
            # Determine class label
            class_label = 'uninfected' if species == 'none' else 'infected'
            
            for img_path in tqdm(image_files, desc=f"Processing {dataset_name}"):
                # Check quality
                is_good, quality_score, metrics = self.check_image_quality(img_path)
                
                if not is_good:
                    self.stats['total_rejected'] += 1
                    continue
                
                # Load and process image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Resize and normalize
                img_resized = self.resize_and_pad(img, self.target_size)
                img_enhanced = self.normalize_image(img_resized)
                
                # Save processed image
                output_filename = f"{dataset_name}_{species}_{img_path.stem}.jpg"
                output_path = self.processed_data_dir / "images" / output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                cv2.imwrite(str(output_path), img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Record sample info
                sample_info = {
                    'image_path': str(output_path.relative_to(self.processed_data_dir)),
                    'original_path': str(img_path),
                    'dataset': dataset_name,
                    'class': class_label,
                    'species': species,
                    'quality_score': quality_score,
                    'original_size': f"{img.shape[1]}x{img.shape[0]}",
                    'processed_size': f"{self.target_size}x{self.target_size}",
                    **metrics
                }
                all_samples.append(sample_info)
                self.stats['total_processed'] += 1
        
        return all_samples
    
    def create_processing_report(self, all_samples: List[Dict]):
        """Create comprehensive processing report"""
        print("\nCreating processing report...")
        
        # Create report
        report = {
            'processing_summary': {
                'total_images_processed': len(all_samples),
                'total_images_rejected': self.stats['total_rejected'],
                'target_image_size': f"{self.target_size}x{self.target_size}",
                'quality_threshold': self.quality_threshold,
                'processing_date': pd.Timestamp.now().isoformat()
            },
            'dataset_breakdown': {},
            'class_distribution': {},
            'species_distribution': {},
            'quality_statistics': {
                'mean_quality_score': 0,
                'std_quality_score': 0,
                'min_quality_score': 0,
                'max_quality_score': 0
            }
        }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_samples)
        
        if len(df) > 0:
            # Dataset breakdown
            dataset_counts = df['dataset'].value_counts().to_dict()
            report['dataset_breakdown'] = dataset_counts
            
            # Class distribution
            class_counts = df['class'].value_counts().to_dict()
            report['class_distribution'] = class_counts
            
            # Species distribution
            species_counts = df['species'].value_counts().to_dict()
            report['species_distribution'] = species_counts
            
            # Quality statistics
            quality_scores = df['quality_score'].astype(float)
            report['quality_statistics'] = {
                'mean_quality_score': float(quality_scores.mean()),
                'std_quality_score': float(quality_scores.std()),
                'min_quality_score': float(quality_scores.min()),
                'max_quality_score': float(quality_scores.max())
            }
        
        # Save report
        report_path = self.processed_data_dir / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save sample data
        samples_path = self.processed_data_dir / "processed_samples.csv"
        df.to_csv(samples_path, index=False)
        
        print(f"✓ Processing report saved to {report_path}")
        print(f"✓ Sample data saved to {samples_path}")
        
        # Print summary
        print("\n" + "="*60)
        print(" PREPROCESSING SUMMARY ")
        print("="*60)
        print(f"Total images processed: {len(all_samples)}")
        print(f"Total images rejected: {self.stats['total_rejected']}")
        print(f"Success rate: {len(all_samples)/(len(all_samples)+self.stats['total_rejected'])*100:.1f}%")
        print("\nDataset breakdown:")
        for dataset, count in dataset_counts.items():
            print(f"  {dataset}: {count} images")
        print("\nClass distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")
        print("="*60)
    
    def process_all_datasets(self):
        """Process all available datasets"""
        print("\n" + "="*60)
        print(" MALARIA DATASET PREPROCESSING ")
        print("="*60)
        
        all_samples = []
        
        # Process each dataset
        nih_samples = self.process_nih_cell_dataset()
        all_samples.extend(nih_samples)
        
        # Process species-specific thick smear datasets
        thick_samples = self.process_nih_thick_smear_datasets()
        all_samples.extend(thick_samples)
        
        mp_idb_samples = self.process_mp_idb_dataset()
        all_samples.extend(mp_idb_samples)
        
        kaggle_samples = self.process_kaggle_dataset()
        all_samples.extend(kaggle_samples)
        
        # Create comprehensive report
        self.create_processing_report(all_samples)
        
        return all_samples


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess malaria datasets")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--target-size", type=int, default=640, help="Target image size")
    parser.add_argument("--quality-threshold", type=int, default=30, help="Quality threshold")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = MalariaDataPreprocessor(
        raw_data_dir=args.raw_dir,
        processed_data_dir=args.output_dir
    )
    
    # Set parameters
    preprocessor.target_size = args.target_size
    preprocessor.quality_threshold = args.quality_threshold
    
    # Process all datasets
    preprocessor.process_all_datasets()


if __name__ == "__main__":
    main()
