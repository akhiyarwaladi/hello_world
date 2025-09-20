#!/usr/bin/env python3
"""
UNIFIED JOURNAL ANALYSIS - YOLOv8 vs YOLOv11 Malaria Detection
Comprehensive analysis for research publication following IEEE methodology
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
from sklearn.metrics import confusion_matrix, classification_report
import yaml

class UnifiedJournalAnalyzer:
    def __init__(self):
        self.results_base = Path("results/current_experiments")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"journal_analysis_{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True)

        # Model mappings
        self.model_info = {
            'yolo8': {
                'name': 'YOLOv8n',
                'detection_dir': 'yolov8_detection',
                'classification_dir': 'yolov8_classification',
                'color': '#2E86AB',
                'marker': 'o'
            },
            'yolo11': {
                'name': 'YOLOv11n',
                'detection_dir': 'yolo11_detection',
                'classification_dir': 'yolov8_classification',  # Uses YOLOv8 classification
                'color': '#A23B72',
                'marker': 's'
            }
        }

        self.analysis_results = {}
        print(f"📊 Unified Journal Analyzer Initialized")
        print(f"📂 Output: {self.output_dir}")

    def find_pipeline_experiments(self, timestamp_pattern="multi_pipeline_20250920_131500"):
        """Find completed pipeline experiments"""
        experiments = {}

        for model_key, model_config in self.model_info.items():
            # Find detection results
            det_pattern = f"{timestamp_pattern}_{model_key}_det"
            det_path = self.results_base / "training" / "detection" / model_config['detection_dir'] / det_pattern

            # Find classification results
            cls_pattern = f"{timestamp_pattern}_{model_key}_cls"
            cls_path = self.results_base / "training" / "classification" / model_config['classification_dir'] / cls_pattern

            if det_path.exists() and cls_path.exists():
                experiments[model_key] = {
                    'detection_path': det_path,
                    'classification_path': cls_path,
                    'detection_results': det_path / "results.csv",
                    'classification_results': cls_path / "results.csv",
                    'detection_args': det_path / "args.yaml",
                    'classification_args': cls_path / "args.yaml"
                }
                print(f"✅ Found complete pipeline for {model_config['name']}")
            else:
                print(f"❌ Incomplete pipeline for {model_config['name']}")
                if not det_path.exists():
                    print(f"   Missing detection: {det_path}")
                if not cls_path.exists():
                    print(f"   Missing classification: {cls_path}")

        return experiments

    def analyze_detection_performance(self, experiments):
        """Analyze detection performance for both models"""
        detection_results = {}

        plt.figure(figsize=(15, 5))

        for i, (model_key, exp_data) in enumerate(experiments.items()):
            model_config = self.model_info[model_key]

            # Read detection results
            if exp_data['detection_results'].exists():
                df = pd.read_csv(exp_data['detection_results'])

                # Get final metrics
                final_metrics = df.iloc[-1]

                detection_results[model_key] = {
                    'model_name': model_config['name'],
                    'epochs': len(df),
                    'final_mAP50': final_metrics.get('metrics/mAP50(B)', 0),
                    'final_mAP50_95': final_metrics.get('metrics/mAP50-95(B)', 0),
                    'final_precision': final_metrics.get('metrics/precision(B)', 0),
                    'final_recall': final_metrics.get('metrics/recall(B)', 0),
                    'training_time': final_metrics.get('time', 0),
                    'box_loss': final_metrics.get('train/box_loss', 0),
                    'cls_loss': final_metrics.get('train/cls_loss', 0),
                    'val_box_loss': final_metrics.get('val/box_loss', 0),
                    'val_cls_loss': final_metrics.get('val/cls_loss', 0)
                }

                # Plot training curves
                plt.subplot(1, 3, i+1)
                plt.plot(df['epoch'], df['metrics/mAP50(B)'],
                        color=model_config['color'], linewidth=2, label='mAP50')
                plt.plot(df['epoch'], df['metrics/mAP50-95(B)'],
                        color=model_config['color'], linewidth=2, linestyle='--', label='mAP50-95')
                plt.title(f'{model_config["name"]} Detection Training')
                plt.xlabel('Epoch')
                plt.ylabel('mAP')
                plt.legend()
                plt.grid(True, alpha=0.3)

        # Comparison plot
        if len(detection_results) > 1:
            plt.subplot(1, 3, 3)
            models = list(detection_results.keys())
            map50_values = [detection_results[m]['final_mAP50'] for m in models]
            map50_95_values = [detection_results[m]['final_mAP50_95'] for m in models]

            x = np.arange(len(models))
            width = 0.35

            plt.bar(x - width/2, map50_values, width, label='mAP50', alpha=0.8)
            plt.bar(x + width/2, map50_95_values, width, label='mAP50-95', alpha=0.8)

            plt.xlabel('Models')
            plt.ylabel('mAP')
            plt.title('Detection Performance Comparison')
            plt.xticks(x, [self.model_info[m]['name'] for m in models])
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "detection_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

        return detection_results

    def analyze_classification_performance(self, experiments):
        """Analyze classification performance for both models"""
        classification_results = {}

        plt.figure(figsize=(15, 10))

        for i, (model_key, exp_data) in enumerate(experiments.items()):
            model_config = self.model_info[model_key]

            # Read classification results
            if exp_data['classification_results'].exists():
                df = pd.read_csv(exp_data['classification_results'])

                # Get final metrics
                final_metrics = df.iloc[-1]

                classification_results[model_key] = {
                    'model_name': model_config['name'],
                    'epochs': len(df),
                    'final_top1_acc': final_metrics.get('metrics/accuracy_top1', 0),
                    'final_top5_acc': final_metrics.get('metrics/accuracy_top5', 0),
                    'training_time': final_metrics.get('time', 0),
                    'train_loss': final_metrics.get('train/loss', 0),
                    'val_loss': final_metrics.get('val/loss', 0)
                }

                # Plot training curves - Top subplot
                plt.subplot(2, 2, i+1)
                plt.plot(df['epoch'], df['metrics/accuracy_top1'],
                        color=model_config['color'], linewidth=2, label='Top-1 Accuracy')
                plt.plot(df['epoch'], df['train/loss'],
                        color=model_config['color'], linewidth=2, linestyle='--', alpha=0.7, label='Train Loss')
                plt.title(f'{model_config["name"]} Classification Training')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy / Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)

        # Comparison plots
        if len(classification_results) > 1:
            models = list(classification_results.keys())

            # Accuracy comparison
            plt.subplot(2, 2, 3)
            top1_values = [classification_results[m]['final_top1_acc'] for m in models]
            colors = [self.model_info[m]['color'] for m in models]

            bars = plt.bar([self.model_info[m]['name'] for m in models], top1_values,
                          color=colors, alpha=0.8)
            plt.ylabel('Top-1 Accuracy')
            plt.title('Classification Accuracy Comparison')
            plt.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, top1_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

            # Training time comparison
            plt.subplot(2, 2, 4)
            det_times = [self.analysis_results['detection'][m]['training_time'] for m in models]
            cls_times = [classification_results[m]['training_time'] for m in models]

            x = np.arange(len(models))
            width = 0.35

            plt.bar(x - width/2, det_times, width, label='Detection', alpha=0.8)
            plt.bar(x + width/2, cls_times, width, label='Classification', alpha=0.8)

            plt.xlabel('Models')
            plt.ylabel('Training Time (minutes)')
            plt.title('Training Time Comparison')
            plt.xticks(x, [self.model_info[m]['name'] for m in models])
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "classification_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

        return classification_results

    def create_journal_comparison_table(self):
        """Create IEEE journal style comparison table"""
        if not hasattr(self, 'analysis_results') or not self.analysis_results:
            print("❌ No analysis results available")
            return

        # Create comprehensive comparison table
        comparison_data = []

        for model_key in self.analysis_results['detection'].keys():
            det_results = self.analysis_results['detection'][model_key]
            cls_results = self.analysis_results['classification'][model_key]

            comparison_data.append({
                'Model': det_results['model_name'],
                'Detection mAP50': f"{det_results['final_mAP50']:.3f}",
                'Detection mAP50-95': f"{det_results['final_mAP50_95']:.3f}",
                'Detection Precision': f"{det_results['final_precision']:.3f}",
                'Detection Recall': f"{det_results['final_recall']:.3f}",
                'Classification Top-1': f"{cls_results['final_top1_acc']:.3f}",
                'Classification Top-5': f"{cls_results['final_top5_acc']:.3f}",
                'Total Training Time (min)': f"{(det_results['training_time'] + cls_results['training_time'])/60:.1f}",
                'Detection Epochs': det_results['epochs'],
                'Classification Epochs': cls_results['epochs']
            })

        df_comparison = pd.DataFrame(comparison_data)

        # Save as CSV and formatted text
        df_comparison.to_csv(self.output_dir / "journal_comparison_table.csv", index=False)

        # Create LaTeX-style table
        latex_table = df_comparison.to_latex(index=False, float_format="%.3f")
        with open(self.output_dir / "journal_comparison_table.tex", 'w') as f:
            f.write(latex_table)

        print("📊 Journal comparison table created")
        return df_comparison

    def create_journal_report(self):
        """Create comprehensive journal-style analysis report"""
        report_path = self.output_dir / "unified_journal_analysis.md"

        with open(report_path, 'w') as f:
            f.write("# Unified Journal Analysis: YOLOv8 vs YOLOv11 Malaria Detection\n\n")
            f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**Task**: Automated Identification of Malaria-Infected Cells\n")
            f.write("**Methodology**: Two-Stage Deep Learning (Detection → Classification)\n\n")

            f.write("---\n\n")

            f.write("## 🎯 Executive Summary\n\n")
            f.write("This analysis compares YOLOv8 and YOLOv11 performance for malaria parasite detection and species classification following IEEE journal methodology.\n\n")

            # Detection Results
            if 'detection' in self.analysis_results:
                f.write("## 📊 Detection Performance\n\n")
                for model_key, results in self.analysis_results['detection'].items():
                    f.write(f"### {results['model_name']}\n")
                    f.write(f"- **mAP50**: {results['final_mAP50']:.3f}\n")
                    f.write(f"- **mAP50-95**: {results['final_mAP50_95']:.3f}\n")
                    f.write(f"- **Precision**: {results['final_precision']:.3f}\n")
                    f.write(f"- **Recall**: {results['final_recall']:.3f}\n")
                    f.write(f"- **Training Time**: {results['training_time']/60:.1f} minutes\n")
                    f.write(f"- **Epochs**: {results['epochs']}\n\n")

            # Classification Results
            if 'classification' in self.analysis_results:
                f.write("## 🧬 Classification Performance\n\n")
                for model_key, results in self.analysis_results['classification'].items():
                    f.write(f"### {results['model_name']}\n")
                    f.write(f"- **Top-1 Accuracy**: {results['final_top1_acc']:.3f}\n")
                    f.write(f"- **Top-5 Accuracy**: {results['final_top5_acc']:.3f} (Note: Meaningless with 4 classes)\n")
                    f.write(f"- **Training Time**: {results['training_time']/60:.1f} minutes\n")
                    f.write(f"- **Epochs**: {results['epochs']}\n\n")

            f.write("## 🔬 Key Findings\n\n")
            f.write("1. **Top-5 Accuracy Artifact**: 100% top-5 accuracy is expected with only 4 malaria species classes\n")
            f.write("2. **Focus on Top-1 Accuracy**: This is the meaningful metric for 4-class classification\n")
            f.write("3. **Two-Stage Pipeline**: Detection followed by classification shows effective results\n")
            f.write("4. **Model Comparison**: Direct performance comparison between YOLOv8 and YOLOv11\n\n")

            f.write("## 📈 Generated Visualizations\n\n")
            f.write("- `detection_performance.png`: Detection training curves and comparison\n")
            f.write("- `classification_performance.png`: Classification training curves and comparison\n")
            f.write("- `journal_comparison_table.csv`: IEEE-style comparison table\n")
            f.write("- `journal_comparison_table.tex`: LaTeX table for publication\n\n")

            f.write("## 🎯 Conclusions\n\n")
            f.write("The analysis provides comprehensive performance evaluation suitable for academic publication, ")
            f.write("addressing the suspicious 100% top-5 accuracy and providing meaningful comparison metrics.\n\n")

            f.write("---\n\n")
            f.write("*Generated by Unified Journal Analyzer*\n")

        print(f"📄 Journal report created: {report_path}")

    def run_complete_analysis(self):
        """Run complete unified analysis"""
        print("🚀 Starting Unified Journal Analysis...")

        # Find completed experiments
        experiments = self.find_pipeline_experiments()

        if not experiments:
            print("❌ No completed pipeline experiments found!")
            return

        print(f"✅ Found {len(experiments)} completed pipelines")

        # Analyze detection performance
        print("📊 Analyzing detection performance...")
        detection_results = self.analyze_detection_performance(experiments)
        self.analysis_results['detection'] = detection_results

        # Analyze classification performance
        print("🧬 Analyzing classification performance...")
        classification_results = self.analyze_classification_performance(experiments)
        self.analysis_results['classification'] = classification_results

        # Create comparison table
        print("📋 Creating journal comparison table...")
        comparison_table = self.create_journal_comparison_table()

        # Create journal report
        print("📄 Creating journal analysis report...")
        self.create_journal_report()

        # Save complete results
        results_json = self.output_dir / "complete_analysis_results.json"
        with open(results_json, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = {}
            for key, value in self.analysis_results.items():
                serializable_results[key] = {}
                for model, metrics in value.items():
                    serializable_results[key][model] = {}
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (np.integer, np.floating)):
                            serializable_results[key][model][metric_name] = metric_value.item()
                        else:
                            serializable_results[key][model][metric_name] = metric_value

            json.dump(serializable_results, f, indent=2)

        print(f"🎉 Unified Analysis Complete!")
        print(f"📂 Results saved to: {self.output_dir}")

        return self.analysis_results

def main():
    analyzer = UnifiedJournalAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()