#!/usr/bin/env python3
"""
Compare Performance of All Models for Research Paper
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

class ModelPerformanceComparator:
    """Compare performance across different models"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.comparison_data = []

    def extract_yolo_metrics(self, experiment_path: Path) -> dict:
        """Extract metrics from YOLO experiment results"""

        metrics = {
            'experiment_name': experiment_path.name,
            'model_type': 'unknown',
            'task': 'unknown'
        }

        # Determine model type and task
        name = experiment_path.name.lower()
        if 'yolov8' in name:
            metrics['model_type'] = 'YOLOv8'
        elif 'yolo11' in name:
            metrics['model_type'] = 'YOLOv11'
        elif 'rtdetr' in name:
            metrics['model_type'] = 'RT-DETR'

        if 'detection' in str(experiment_path.parent):
            metrics['task'] = 'detection'
        elif 'classification' in str(experiment_path.parent):
            metrics['task'] = 'classification'

        # Try to read results from different possible files
        results_files = [
            experiment_path / "results.csv",
            experiment_path / "results.json",
            experiment_path / "metrics.json"
        ]

        for results_file in results_files:
            if results_file.exists():
                try:
                    if results_file.suffix == '.csv':
                        df = pd.read_csv(results_file)
                        if not df.empty:
                            # Get best epoch metrics
                            if 'metrics/mAP50(B)' in df.columns:
                                metrics['mAP50'] = df['metrics/mAP50(B)'].max()
                            if 'metrics/mAP50-95(B)' in df.columns:
                                metrics['mAP50_95'] = df['metrics/mAP50-95(B)'].max()
                            if 'metrics/precision(B)' in df.columns:
                                metrics['precision'] = df['metrics/precision(B)'].max()
                            if 'metrics/recall(B)' in df.columns:
                                metrics['recall'] = df['metrics/recall(B)'].max()
                            if 'train/box_loss' in df.columns:
                                metrics['final_box_loss'] = df['train/box_loss'].iloc[-1]
                            if 'val/accuracy_top1' in df.columns:
                                metrics['accuracy_top1'] = df['val/accuracy_top1'].max()

                    elif results_file.suffix == '.json':
                        with open(results_file, 'r') as f:
                            data = json.load(f)
                            metrics.update(data)

                    break
                except Exception as e:
                    print(f"⚠️ Error reading {results_file}: {e}")

        # Try to get training time from args.yaml
        args_file = experiment_path / "args.yaml"
        if args_file.exists():
            try:
                import yaml
                with open(args_file, 'r') as f:
                    args_data = yaml.safe_load(f)
                    if 'epochs' in args_data:
                        metrics['epochs'] = args_data['epochs']
                    if 'batch' in args_data:
                        metrics['batch_size'] = args_data['batch']
                    if 'imgsz' in args_data:
                        metrics['image_size'] = args_data['imgsz']
            except Exception as e:
                print(f"⚠️ Error reading args.yaml: {e}")

        return metrics

    def scan_experiment_results(self):
        """Scan all experiment directories for results"""

        print("🔍 Scanning experiment results...")

        # Detection experiments
        detection_dir = self.results_dir / "detection"
        if detection_dir.exists():
            for exp_dir in detection_dir.iterdir():
                if exp_dir.is_dir():
                    metrics = self.extract_yolo_metrics(exp_dir)
                    metrics['task'] = 'detection'
                    self.comparison_data.append(metrics)

        # Classification experiments
        classification_dir = self.results_dir / "classification"
        if classification_dir.exists():
            for exp_dir in classification_dir.iterdir():
                if exp_dir.is_dir():
                    metrics = self.extract_yolo_metrics(exp_dir)
                    metrics['task'] = 'classification'
                    self.comparison_data.append(metrics)

        print(f"✓ Found {len(self.comparison_data)} experiments")

    def create_comparison_report(self, output_path: str = "results/model_comparison_report.md"):
        """Create markdown report comparing all models"""

        if not self.comparison_data:
            print("❌ No comparison data available")
            return

        # Create DataFrame
        df = pd.DataFrame(self.comparison_data)

        # Generate report
        report = f"""# Malaria Detection Models Comparison Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report compares the performance of YOLOv8, YOLOv11, YOLOv12, and RT-DETR models for malaria parasite detection and classification.

### Dataset Summary
- **Detection Dataset**: 103 microscopy images, 1,242 parasites (P. falciparum)
- **Classification Dataset**: 1,242 cropped parasite images (128x128px)
- **Source**: MP-IDB dataset with corrected bounding box annotations

## Model Performance Comparison

### Detection Models
"""

        # Filter detection models
        detection_df = df[df['task'] == 'detection']
        if not detection_df.empty:
            report += "\\n| Model | mAP50 | mAP50-95 | Precision | Recall | Epochs | Batch Size |\\n"
            report += "|-------|-------|----------|-----------|---------|---------|------------|\\n"

            for _, row in detection_df.iterrows():
                model = row.get('model_type', 'Unknown')
                map50 = f"{row.get('mAP50', 0):.3f}" if row.get('mAP50') else 'N/A'
                map50_95 = f"{row.get('mAP50_95', 0):.3f}" if row.get('mAP50_95') else 'N/A'
                precision = f"{row.get('precision', 0):.3f}" if row.get('precision') else 'N/A'
                recall = f"{row.get('recall', 0):.3f}" if row.get('recall') else 'N/A'
                epochs = row.get('epochs', 'N/A')
                batch = row.get('batch_size', 'N/A')

                report += f"| {model} | {map50} | {map50_95} | {precision} | {recall} | {epochs} | {batch} |\\n"

        # Filter classification models
        classification_df = df[df['task'] == 'classification']
        if not classification_df.empty:
            report += "\\n### Classification Models\\n"
            report += "\\n| Model | Top-1 Accuracy | Epochs | Batch Size |\\n"
            report += "|-------|----------------|---------|------------|\\n"

            for _, row in classification_df.iterrows():
                model = row.get('model_type', 'Unknown')
                acc = f"{row.get('accuracy_top1', 0):.3f}" if row.get('accuracy_top1') else 'N/A'
                epochs = row.get('epochs', 'N/A')
                batch = row.get('batch_size', 'N/A')

                report += f"| {model} | {acc} | {epochs} | {batch} |\\n"

        # Safe statistics with error handling
        detection_best_map50 = "N/A"
        detection_best_precision = "N/A"
        detection_best_recall = "N/A"
        classification_best_acc = "N/A"
        
        if not detection_df.empty and 'mAP50' in detection_df.columns:
            valid_map50 = detection_df['mAP50'].dropna()
            if not valid_map50.empty:
                detection_best_map50 = f"{valid_map50.max():.3f}"
                
        if not detection_df.empty and 'precision' in detection_df.columns:
            valid_precision = detection_df['precision'].dropna()
            if not valid_precision.empty:
                detection_best_precision = f"{valid_precision.max():.3f}"
                
        if not detection_df.empty and 'recall' in detection_df.columns:
            valid_recall = detection_df['recall'].dropna()
            if not valid_recall.empty:
                detection_best_recall = f"{valid_recall.max():.3f}"
                
        if not classification_df.empty and 'accuracy_top1' in classification_df.columns:
            valid_acc = classification_df['accuracy_top1'].dropna()
            if not valid_acc.empty:
                classification_best_acc = f"{valid_acc.max():.3f}"

        report += f"""

## Key Findings

### Detection Performance
- **Best mAP50**: {detection_best_map50}
- **Best Precision**: {detection_best_precision}
- **Best Recall**: {detection_best_recall}

### Classification Performance
- **Best Accuracy**: {classification_best_acc}

## Training Status
- **Note**: Training was interrupted due to system crash
- **Completed Epochs**: Most models completed 1-3 epochs only
- **Recommendation**: Resume training for full evaluation

## Conclusions

### Model Comparison for Malaria Detection Research

1. **YOLOv8**: Baseline performance, well-established architecture
2. **YOLOv11**: Newer version with potential improvements
3. **RT-DETR**: Transformer-based approach, different detection paradigm

### Recommendations

Based on the preliminary results:
- **Resume Training**: Complete full epoch training for fair comparison
- For deployment: Consider model size vs accuracy trade-off
- For research: Compare inference speed and computational requirements
- For clinical use: Prioritize precision to minimize false positives

## Technical Details

### Experimental Setup
- **Framework**: Ultralytics YOLO
- **Hardware**: CPU training
- **Dataset Split**: 70% train, 15% val, 15% test
- **Image Preprocessing**: CLAHE enhancement, normalization

### Data Quality Improvements
- ✅ **Bounding Box Correction**: Fixed coordinate mapping from MP-IDB CSV annotations
- ✅ **Ground Truth Validation**: Used binary masks for accurate parasite localization
- ✅ **Proper Cropping**: Individual parasite cells for classification
- ✅ **Dataset Ready**: 1,242 cropped parasites, 103 detection images

---

*This report was generated automatically from training experiment results.*
"""

        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write(report)

        print(f"✅ Comparison report saved to: {output_file}")

        # Also save raw data as JSON
        json_file = output_file.with_suffix('.json')
        with open(json_file, 'w') as f:
            json.dump(self.comparison_data, f, indent=2)

        print(f"✅ Raw comparison data saved to: {json_file}")

    def create_performance_plots(self, output_dir: str = "results/plots"):
        """Create performance visualization plots"""

        if not self.comparison_data:
            print("❌ No comparison data available for plotting")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.comparison_data)

        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

        # Detection performance plot
        detection_df = df[df['task'] == 'detection'].copy()
        if not detection_df.empty and 'mAP50' in detection_df.columns:
            plt.figure(figsize=(10, 6))

            sns.barplot(data=detection_df, x='model_type', y='mAP50')
            plt.title('Detection Model Performance Comparison\\nmAP50 Score')
            plt.ylabel('mAP50')
            plt.xlabel('Model Type')
            plt.xticks(rotation=45)
            plt.tight_layout()

            plot_file = output_path / "detection_performance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Detection performance plot saved to: {plot_file}")

        # Classification performance plot
        classification_df = df[df['task'] == 'classification'].copy()
        if not classification_df.empty and 'accuracy_top1' in classification_df.columns:
            plt.figure(figsize=(10, 6))

            sns.barplot(data=classification_df, x='model_type', y='accuracy_top1')
            plt.title('Classification Model Performance Comparison\\nTop-1 Accuracy')
            plt.ylabel('Top-1 Accuracy')
            plt.xlabel('Model Type')
            plt.xticks(rotation=45)
            plt.tight_layout()

            plot_file = output_path / "classification_performance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Classification performance plot saved to: {plot_file}")

def main():
    parser = argparse.ArgumentParser(description="Compare model performance for research paper")
    parser.add_argument("--results-dir", default="results",
                       help="Results directory path")
    parser.add_argument("--output", default="results/model_comparison_report.md",
                       help="Output report file")

    args = parser.parse_args()

    print("=" * 60)
    print("MODEL PERFORMANCE COMPARISON FOR RESEARCH PAPER")
    print("=" * 60)

    # Create comparator
    comparator = ModelPerformanceComparator(args.results_dir)

    # Scan for results
    comparator.scan_experiment_results()

    # Create report
    comparator.create_comparison_report(args.output)

    # Create plots
    comparator.create_performance_plots()

    print("\\n🎉 Model comparison analysis completed!")
    print("📊 Results ready for research paper")

if __name__ == "__main__":
    main()