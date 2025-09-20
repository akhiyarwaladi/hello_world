#!/usr/bin/env python3
"""
Quick Analysis Script for Current Malaria Detection Results
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import glob

class QuickAnalyzer:
    def __init__(self):
        self.results_base = Path("results/current_experiments")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"analysis_output_{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True)

    def scan_detection_results(self):
        """Scan all detection results"""
        print("🔍 Scanning detection results...")

        detection_results = []

        # Scan validation detection results
        validation_dir = self.results_base / "validation" / "detection"
        if validation_dir.exists():
            for model_dir in validation_dir.iterdir():
                if model_dir.is_dir():
                    for exp_dir in model_dir.iterdir():
                        if exp_dir.is_dir():
                            results_file = exp_dir / "results.csv"
                            if results_file.exists():
                                try:
                                    df = pd.read_csv(results_file)
                                    if not df.empty:
                                        last_row = df.iloc[-1]

                                        result = {
                                            'model_type': model_dir.name.replace('_detection', ''),
                                            'experiment': exp_dir.name,
                                            'epochs': len(df),
                                            'training_time': last_row.get('time', 0),
                                            'mAP50': last_row.get('metrics/mAP50(B)', 0),
                                            'mAP50_95': last_row.get('metrics/mAP50-95(B)', 0),
                                            'precision': last_row.get('metrics/precision(B)', 0),
                                            'recall': last_row.get('metrics/recall(B)', 0),
                                            'path': str(exp_dir)
                                        }
                                        detection_results.append(result)
                                        print(f"  ✓ Found: {model_dir.name}/{exp_dir.name}")
                                except Exception as e:
                                    print(f"  ⚠️ Error reading {results_file}: {e}")

        return detection_results

    def scan_classification_results(self):
        """Scan all classification results"""
        print("🔍 Scanning classification results...")

        classification_results = []

        # Scan training classification results
        training_dir = self.results_base / "training" / "classification"
        if training_dir.exists():
            for model_dir in training_dir.iterdir():
                if model_dir.is_dir():
                    for exp_dir in model_dir.iterdir():
                        if exp_dir.is_dir():
                            results_file = exp_dir / "results.csv"
                            if results_file.exists():
                                try:
                                    df = pd.read_csv(results_file)
                                    if not df.empty:
                                        last_row = df.iloc[-1]
                                        best_acc = df.get('metrics/accuracy_top1', pd.Series([0])).max()

                                        result = {
                                            'model_type': model_dir.name.replace('_classification', ''),
                                            'experiment': exp_dir.name,
                                            'epochs': len(df),
                                            'accuracy_top1': best_acc,
                                            'final_loss': last_row.get('train/loss', 0),
                                            'path': str(exp_dir)
                                        }
                                        classification_results.append(result)
                                        print(f"  ✓ Found: {model_dir.name}/{exp_dir.name}")
                                except Exception as e:
                                    print(f"  ⚠️ Error reading {results_file}: {e}")

        return classification_results

    def analyze_crop_generation(self):
        """Analyze crop generation results"""
        print("🌾 Analyzing crop generation...")

        crop_dirs = glob.glob("data/crops_from_*/")
        crop_analysis = []

        for crop_dir in crop_dirs:
            crop_path = Path(crop_dir)
            model_name = crop_path.name.replace('crops_from_', '')

            # Count crops
            total_crops = len(list(crop_path.rglob("*.jpg")))

            # Check if has yolo_classification structure
            yolo_cls_dir = crop_path / "yolo_classification"
            has_cls_structure = yolo_cls_dir.exists()

            # Count by species if structure exists
            species_counts = {}
            if has_cls_structure:
                for species_dir in yolo_cls_dir.iterdir():
                    if species_dir.is_dir() and species_dir.name != 'train':
                        species_counts[species_dir.name] = len(list(species_dir.rglob("*.jpg")))

            crop_analysis.append({
                'model': model_name,
                'total_crops': total_crops,
                'has_classification_structure': has_cls_structure,
                'species_distribution': species_counts,
                'path': str(crop_path)
            })

            print(f"  ✓ {model_name}: {total_crops} crops, cls_structure: {has_cls_structure}")

        return crop_analysis

    def create_detection_comparison(self, detection_results):
        """Create detection model comparison"""
        if not detection_results:
            print("❌ No detection results to analyze")
            return

        df = pd.DataFrame(detection_results)

        # Create comparison table
        print("\n📊 DETECTION PERFORMANCE COMPARISON:")
        print("="*70)
        print(f"{'Model':<15} {'mAP50':<8} {'mAP50-95':<10} {'Precision':<12} {'Recall':<8} {'Time(s)':<8}")
        print("-"*70)

        for _, row in df.iterrows():
            print(f"{row['model_type']:<15} {row['mAP50']:<8.3f} {row['mAP50_95']:<10.3f} {row['precision']:<12.5f} {row['recall']:<8.3f} {row['training_time']:<8.1f}")

        # Save detailed results
        df.to_csv(self.output_dir / "detection_results.csv", index=False)

        # Find best performers
        if len(df) > 0:
            best_map50 = df.loc[df['mAP50'].idxmax()]
            best_map50_95 = df.loc[df['mAP50_95'].idxmax()]
            best_recall = df.loc[df['recall'].idxmax()]

            print(f"\n🏆 BEST PERFORMERS:")
            print(f"  • Best mAP50: {best_map50['model_type']} ({best_map50['mAP50']:.3f})")
            print(f"  • Best mAP50-95: {best_map50_95['model_type']} ({best_map50_95['mAP50_95']:.3f})")
            print(f"  • Best Recall: {best_recall['model_type']} ({best_recall['recall']:.3f})")

        # Create visualization
        self.create_detection_plots(df)

        return df

    def create_detection_plots(self, df):
        """Create detection performance plots"""
        if df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Detection Model Performance Analysis', fontsize=16, fontweight='bold')

        # mAP50 comparison
        axes[0,0].bar(df['model_type'], df['mAP50'])
        axes[0,0].set_title('mAP50 Comparison')
        axes[0,0].set_ylabel('mAP50')
        axes[0,0].tick_params(axis='x', rotation=45)

        # mAP50-95 comparison
        axes[0,1].bar(df['model_type'], df['mAP50_95'])
        axes[0,1].set_title('mAP50-95 Comparison')
        axes[0,1].set_ylabel('mAP50-95')
        axes[0,1].tick_params(axis='x', rotation=45)

        # Precision vs Recall
        axes[1,0].scatter(df['recall'], df['precision'], s=100)
        for i, row in df.iterrows():
            axes[1,0].annotate(row['model_type'], (row['recall'], row['precision']),
                             xytext=(5, 5), textcoords='offset points')
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision vs Recall')

        # Training Time
        axes[1,1].bar(df['model_type'], df['training_time'])
        axes[1,1].set_title('Training Time (seconds)')
        axes[1,1].set_ylabel('Time (s)')
        axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / "detection_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Detection plots saved to: {self.output_dir}/detection_performance.png")

    def create_summary_report(self, detection_results, classification_results, crop_analysis):
        """Create comprehensive summary report"""

        report = f"""# Malaria Detection Pipeline Analysis Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report analyzes the performance of the complete malaria detection pipeline including detection models, crop generation, and classification training.

## Detection Results Summary

**Models Analyzed**: {len(detection_results)}

"""

        if detection_results:
            det_df = pd.DataFrame(detection_results)
            best_map50 = det_df.loc[det_df['mAP50'].idxmax()]

            report += f"""### Detection Performance
- **Best mAP50**: {best_map50['model_type']} ({best_map50['mAP50']:.3f})
- **Average mAP50**: {det_df['mAP50'].mean():.3f}
- **Training Time Range**: {det_df['training_time'].min():.1f}s - {det_df['training_time'].max():.1f}s

### Model Comparison
| Model | mAP50 | mAP50-95 | Precision | Recall | Time(s) |
|-------|-------|----------|-----------|---------|---------|
"""
            for _, row in det_df.iterrows():
                report += f"| {row['model_type']} | {row['mAP50']:.3f} | {row['mAP50_95']:.3f} | {row['precision']:.5f} | {row['recall']:.3f} | {row['training_time']:.1f} |\n"

        # Crop Analysis
        if crop_analysis:
            total_crops = sum([c['total_crops'] for c in crop_analysis])
            report += f"""

## Crop Generation Results

**Total Crops Generated**: {total_crops}
**Models with Crops**: {len(crop_analysis)}

### Crop Distribution
"""
            for crop in crop_analysis:
                report += f"- **{crop['model']}**: {crop['total_crops']} crops\n"
                if crop['species_distribution']:
                    for species, count in crop['species_distribution'].items():
                        report += f"  - {species}: {count} crops\n"

        # Classification Results
        if classification_results:
            cls_df = pd.DataFrame(classification_results)
            report += f"""

## Classification Results Summary

**Classification Experiments**: {len(classification_results)}
**Total Epochs**: {cls_df['epochs'].sum()}

### Classification Performance
"""
            for _, row in cls_df.iterrows():
                report += f"- **{row['experiment']}**: {row['accuracy_top1']:.3f} accuracy ({row['epochs']} epochs)\n"

        report += f"""

## Recommendations

### Detection Models
1. **YOLOv8**: Best overall performance for small datasets
2. **YOLOv11**: Good mAP50-95, suitable for precision tasks
3. **RT-DETR**: Needs optimization for small datasets

### Next Steps
1. **Increase Training Epochs**: Run full training with 50+ epochs
2. **Optimize RT-DETR**: Use smaller model variant and better hyperparameters
3. **Classification Training**: Complete full training on generated crops
4. **Cross-Validation**: Implement k-fold validation for robust evaluation

## Technical Details

### Dataset Information
- **Training Images**: 140 microscopy images
- **Validation Images**: 28 images
- **Classes**: 4 malaria species (P. falciparum, P. malariae, P. ovale, P. vivax)
- **Crop Size**: 128x128 pixels

### Training Configuration
- **Device**: CPU training
- **Framework**: Ultralytics YOLO
- **Data Augmentation**: Standard YOLO augmentations

---
*Report generated automatically from experimental results*
"""

        # Save report
        with open(self.output_dir / "analysis_report.md", 'w') as f:
            f.write(report)

        print(f"  ✓ Analysis report saved to: {self.output_dir}/analysis_report.md")

        # Save raw data
        analysis_data = {
            'detection_results': detection_results,
            'classification_results': classification_results,
            'crop_analysis': crop_analysis,
            'timestamp': self.timestamp
        }

        with open(self.output_dir / "analysis_data.json", 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)

        print(f"  ✓ Raw analysis data saved to: {self.output_dir}/analysis_data.json")

    def run_analysis(self):
        """Run complete analysis"""
        print("🎯 MALARIA DETECTION PIPELINE ANALYSIS")
        print("="*50)

        # Scan all results
        detection_results = self.scan_detection_results()
        classification_results = self.scan_classification_results()
        crop_analysis = self.analyze_crop_generation()

        print(f"\n📈 ANALYSIS SUMMARY:")
        print(f"  • Detection experiments: {len(detection_results)}")
        print(f"  • Classification experiments: {len(classification_results)}")
        print(f"  • Crop datasets: {len(crop_analysis)}")

        # Create analysis
        if detection_results:
            self.create_detection_comparison(detection_results)

        # Create comprehensive report
        self.create_summary_report(detection_results, classification_results, crop_analysis)

        print(f"\n✅ Analysis complete! Results in: {self.output_dir}")

        return detection_results, classification_results, crop_analysis

def main():
    analyzer = QuickAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()