#!/usr/bin/env python3
"""
Dataset Statistics Analyzer
Membuat tabel perbandingan jumlah data train/val/test sebelum dan sesudah augmentasi
"""

import os
import json
import pandas as pd
from pathlib import Path
import yaml
import argparse
from datetime import datetime

class DatasetStatsAnalyzer:
    """Analyzer untuk statistik dataset sebelum dan sesudah augmentasi"""

    def __init__(self):
        self.datasets = {
            "iml_lifecycle": "data/processed/lifecycle",
            "mp_idb_species": "data/processed/species",
            "mp_idb_stages": "data/processed/stages"
        }

        # Parameter augmentasi yang digunakan dalam training
        self.detection_augmentation_params = {
            "mosaic": 0.5,      # 50% mosaic augmentation
            "mixup": 0.0,       # 0% mixup (disabled)
            "fliplr": 0.5,      # 50% horizontal flip
            "flipud": 0.0,      # 0% vertical flip (disabled for medical)
            "hsv_h": 0.010,     # 1% hue variation
            "hsv_s": 0.5,       # 50% saturation variation
            "hsv_v": 0.3,       # 30% brightness variation
            "degrees": 15,      # ±15° rotation
            "scale": 0.3,       # ±30% scale variation
        }

        self.classification_augmentation_params = {
            "horizontal_flip": 0.5,     # 50% horizontal flip
            "rotation": 15,             # ±15° rotation
            "color_jitter": 0.3,        # 30% color variation
            "gaussian_blur": 0.1,       # 10% gaussian blur
            "crop_scale": (0.8, 1.0),   # 80-100% crop scale
        }

    def get_dataset_splits(self, dataset_path):
        """Menghitung jumlah data di setiap split"""
        stats = {
            "train": 0,
            "val": 0,
            "test": 0
        }

        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            return stats

        # Hitung images di setiap split
        for split in ["train", "val", "test"]:
            split_dir = dataset_dir / split / "images"
            if split_dir.exists():
                stats[split] = len(list(split_dir.glob("*.jpg"))) + len(list(split_dir.glob("*.png")))

        return stats

    def calculate_augmented_data_amount(self, original_count, augmentation_type="detection"):
        """
        Menghitung perkiraan jumlah data setelah augmentasi
        Ini estimasi berdasarkan parameter augmentasi
        """
        if augmentation_type == "detection":
            # Untuk detection, estimasi berdasarkan parameter YOLO
            mosaic_factor = 1 + self.detection_augmentation_params["mosaic"]  # Mosaic menggabungkan 4 images
            flip_factor = 1 + self.detection_augmentation_params["fliplr"]    # Horizontal flip doubles variety
            rotation_factor = 1.5  # Rotation creates variety
            scale_factor = 1.3     # Scale creates variety

            # Estimasi total variasi (conservative estimate)
            total_factor = mosaic_factor * flip_factor * rotation_factor * scale_factor
            return int(original_count * total_factor)

        elif augmentation_type == "classification":
            # Untuk classification PyTorch
            flip_factor = 1 + self.classification_augmentation_params["horizontal_flip"]
            rotation_factor = 1.5   # Rotation varieties
            color_factor = 1.3      # Color jitter varieties
            crop_factor = 1.2       # Random crop varieties

            total_factor = flip_factor * rotation_factor * color_factor * crop_factor
            return int(original_count * total_factor)

        return original_count

    def analyze_all_datasets(self, output_dir="analysis_dataset_stats"):
        """Analisa semua dataset dan buat tabel perbandingan"""

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("=== DATASET STATISTICS ANALYZER ===")
        print("Analyzing train/val/test splits for all datasets")
        print("Calculating augmentation effects (on-the-fly augmentation)\n")

        all_stats = []

        for dataset_name, dataset_path in self.datasets.items():
            print(f"[{dataset_name.upper()}] Analyzing dataset...")

            # Get original splits
            original_stats = self.get_dataset_splits(dataset_path)

            if sum(original_stats.values()) == 0:
                print(f"   [WARNING] No data found for {dataset_name}")
                continue

            # Calculate augmented amounts (only for train data)
            detection_augmented_train = self.calculate_augmented_data_amount(
                original_stats["train"], "detection"
            )

            classification_augmented_train = self.calculate_augmented_data_amount(
                original_stats["train"], "classification"
            )

            # Val and test are NOT augmented (for fair evaluation)
            val_count = original_stats["val"]
            test_count = original_stats["test"]

            print(f"   [ORIGINAL] Train: {original_stats['train']}, Val: {val_count}, Test: {test_count}")
            print(f"   [AUG_DET] Train variations: ~{detection_augmented_train} (detection)")
            print(f"   [AUG_CLS] Train variations: ~{classification_augmented_train} (classification)")
            print(f"   [NOTE] Val/Test not augmented for fair evaluation\n")

            # Store stats for table
            dataset_stats = {
                "Dataset": dataset_name,
                "Original_Train": original_stats["train"],
                "Original_Val": val_count,
                "Original_Test": test_count,
                "Original_Total": sum(original_stats.values()),
                "Detection_Aug_Train": detection_augmented_train,
                "Classification_Aug_Train": classification_augmented_train,
                "Aug_Val": val_count,  # Same as original
                "Aug_Test": test_count,  # Same as original
                "Detection_Aug_Total": detection_augmented_train + val_count + test_count,
                "Classification_Aug_Total": classification_augmented_train + val_count + test_count,
                "Detection_Multiplier": f"{detection_augmented_train / max(original_stats['train'], 1):.1f}x",
                "Classification_Multiplier": f"{classification_augmented_train / max(original_stats['train'], 1):.1f}x"
            }

            all_stats.append(dataset_stats)

        if not all_stats:
            print("[ERROR] No datasets found to analyze")
            return None

        # Create comparison table
        df = pd.DataFrame(all_stats)

        # Save detailed CSV
        csv_path = Path(output_dir) / "dataset_statistics_detailed.csv"
        df.to_csv(csv_path, index=False)

        # Create DETECTION table
        detection_df = df[["Dataset", "Original_Train", "Original_Val", "Original_Test", "Original_Total",
                          "Detection_Aug_Train", "Aug_Val", "Aug_Test", "Detection_Aug_Total", "Detection_Multiplier"]].copy()
        detection_df.columns = ["Dataset", "Original_Train", "Original_Val", "Original_Test", "Original_Total",
                               "Augmented_Train", "Augmented_Val", "Augmented_Test", "Augmented_Total", "Multiplier"]

        detection_csv = Path(output_dir) / "dataset_statistics_detection.csv"
        detection_df.to_csv(detection_csv, index=False)

        # Create CLASSIFICATION table
        classification_df = df[["Dataset", "Original_Train", "Original_Val", "Original_Test", "Original_Total",
                               "Classification_Aug_Train", "Aug_Val", "Aug_Test", "Classification_Aug_Total", "Classification_Multiplier"]].copy()
        classification_df.columns = ["Dataset", "Original_Train", "Original_Val", "Original_Test", "Original_Total",
                                    "Augmented_Train", "Augmented_Val", "Augmented_Test", "Augmented_Total", "Multiplier"]

        classification_csv = Path(output_dir) / "dataset_statistics_classification.csv"
        classification_df.to_csv(classification_csv, index=False)

        # Create combined summary for quick reference
        summary_df = df[["Dataset", "Original_Train", "Original_Val", "Original_Test",
                        "Detection_Aug_Train", "Classification_Aug_Train",
                        "Detection_Multiplier", "Classification_Multiplier"]].copy()

        summary_csv = Path(output_dir) / "dataset_statistics_summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        # Create markdown report
        self.create_markdown_report(df, output_dir)

        # Print DETECTION table
        print("=== DETECTION MODEL DATASET STATISTICS ===")
        print(detection_df.to_string(index=False))

        print("\n=== CLASSIFICATION MODEL DATASET STATISTICS ===")
        print(classification_df.to_string(index=False))

        print(f"\n[SUCCESS] Analysis complete!")
        print(f"[SAVE] Detection table: {detection_csv}")
        print(f"[SAVE] Classification table: {classification_csv}")
        print(f"[SAVE] Combined summary: {summary_csv}")
        print(f"[SAVE] Detailed results: {csv_path}")

        return df

    def create_markdown_report(self, df, output_dir):
        """Create comprehensive markdown report"""

        # Create separate tables for detection and classification
        detection_df = df[["Dataset", "Original_Train", "Original_Val", "Original_Test", "Original_Total",
                          "Detection_Aug_Train", "Aug_Val", "Aug_Test", "Detection_Aug_Total", "Detection_Multiplier"]].copy()

        classification_df = df[["Dataset", "Original_Train", "Original_Val", "Original_Test", "Original_Total",
                               "Classification_Aug_Train", "Aug_Val", "Aug_Test", "Classification_Aug_Total", "Classification_Multiplier"]].copy()

        report_content = f"""# Dataset Statistics Analysis Report

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview

This report analyzes the train/validation/test splits for malaria detection datasets and estimates the effect of data augmentation during training.

## Key Points

- **Augmentation**: Only applied to TRAINING data during model training (on-the-fly)
- **Validation/Test**: Never augmented to ensure fair evaluation
- **Detection vs Classification**: Different augmentation strategies for different tasks

## Dataset Statistics

### Detection Model Training

| Dataset | Original Train | Original Val | Original Test | Original Total | Augmented Train | Augmented Val | Augmented Test | Augmented Total | Multiplier |
|---------|----------------|---------------|---------------|----------------|-----------------|---------------|----------------|-----------------|------------|
"""

        for _, row in detection_df.iterrows():
            report_content += f"| {row['Dataset']} | {row['Original_Train']} | {row['Original_Val']} | {row['Original_Test']} | {row['Original_Total']} | {row['Detection_Aug_Train']} | {row['Aug_Val']} | {row['Aug_Test']} | {row['Detection_Aug_Total']} | {row['Detection_Multiplier']} |\n"

        report_content += f"""
### Classification Model Training

| Dataset | Original Train | Original Val | Original Test | Original Total | Augmented Train | Augmented Val | Augmented Test | Augmented Total | Multiplier |
|---------|----------------|---------------|---------------|----------------|-----------------|---------------|----------------|-----------------|------------|
"""

        for _, row in classification_df.iterrows():
            report_content += f"| {row['Dataset']} | {row['Original_Train']} | {row['Original_Val']} | {row['Original_Test']} | {row['Original_Total']} | {row['Classification_Aug_Train']} | {row['Aug_Val']} | {row['Aug_Test']} | {row['Classification_Aug_Total']} | {row['Classification_Multiplier']} |\n"

        report_content += f"""
## Augmentation Parameters

### Detection Training (YOLO)
- **Mosaic**: 50% (combines 4 images)
- **Horizontal Flip**: 50%
- **Rotation**: ±15°
- **Scale**: ±30%
- **HSV Color**: Small variations (medical-safe)
- **Vertical Flip**: 0% (disabled for medical data)

### Classification Training (PyTorch)
- **Horizontal Flip**: 50%
- **Rotation**: ±15°
- **Color Jitter**: 30%
- **Random Crop**: 80-100% scale
- **Gaussian Blur**: 10%

## Important Notes

1. **On-the-Fly Augmentation**: Augmentation happens during training, not as separate files
2. **Conservative Medical Augmentation**: Parameters chosen to preserve medical diagnostic features
3. **Fair Evaluation**: Val/test sets never augmented for unbiased performance measurement
4. **Effective Dataset Size**: Training sees much more variety due to augmentation
5. **Memory Efficient**: No additional storage required for augmented data

## Implications

- **Reduced Overfitting**: Augmentation provides training variety
- **Better Generalization**: Models see more data variations
- **Consistent Evaluation**: Clean val/test sets ensure reliable metrics
- **Medical Safety**: Conservative augmentation preserves diagnostic information

## Generated Files

This analysis produces the following files:

1. **`dataset_statistics_detection.csv`** - Detection model dataset statistics
2. **`dataset_statistics_classification.csv`** - Classification model dataset statistics
3. **`dataset_statistics_summary.csv`** - Combined summary for quick reference
4. **`dataset_statistics_detailed.csv`** - All metrics in one detailed table
5. **`dataset_statistics_report.md`** - This comprehensive report

## Usage in Pipeline

This analysis is automatically run during the pipeline's analysis stage (STAGE 4D) and provides insights into:
- Effective dataset sizes for training
- Augmentation impact on model training
- Fair evaluation practices for medical AI

---
*Generated by Dataset Statistics Analyzer*
"""

        report_path = Path(output_dir) / "dataset_statistics_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"[SAVE] Full report: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze dataset statistics and augmentation effects")
    parser.add_argument("--output", default="analysis_dataset_stats",
                       help="Output directory for analysis results")

    args = parser.parse_args()

    analyzer = DatasetStatsAnalyzer()
    results = analyzer.analyze_all_datasets(args.output)

    if results is not None:
        print(f"\n[CONCLUSION] Dataset analysis completed successfully!")
        print(f"[INSIGHT] Augmentation significantly increases training data variety")
        print(f"[ADVANTAGE] Val/test remain clean for fair evaluation")
    else:
        print(f"\n[ERROR] Dataset analysis failed!")

if __name__ == "__main__":
    main()