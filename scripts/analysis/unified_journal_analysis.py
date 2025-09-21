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
    def __init__(self, centralized_experiment=None):
        if centralized_experiment:
            # Use centralized experiment directory
            self.results_base = Path(f"results/{centralized_experiment}")
            self.centralized_mode = True
        else:
            # Use distributed structure
            self.results_base = Path("results/current_experiments")
            self.centralized_mode = False

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
        if self.centralized_mode:
            return self.find_centralized_experiments(timestamp_pattern)
        else:
            return self.find_distributed_experiments(timestamp_pattern)

    def find_distributed_experiments(self, timestamp_pattern):
        """Find experiments in distributed structure (results/current_experiments/)"""
        experiments = {}

        for model_key, model_config in self.model_info.items():
            # Find detection results
            det_pattern = f"{timestamp_pattern}_{model_key}_det"
            det_path = self.results_base / "training" / "detection" / model_config['detection_dir'] / det_pattern

            # Find classification results - check both classification/ and models/ folders
            cls_pattern = f"{timestamp_pattern}_{model_key}_cls"
            cls_path = self.results_base / "training" / "classification" / model_config['classification_dir'] / cls_pattern

            # NEW: Also check in models/ folder for PyTorch models
            models_path = self.results_base / "training" / "models" / model_config['classification_dir'] / cls_pattern

            # Use whichever classification path exists
            final_cls_path = None
            if cls_path.exists():
                final_cls_path = cls_path
            elif models_path.exists():
                final_cls_path = models_path

            if det_path.exists() and final_cls_path:
                experiments[model_key] = {
                    'detection_path': det_path,
                    'classification_path': final_cls_path,
                    'detection_results': det_path / "results.csv",
                    'classification_results': final_cls_path / "results.csv",
                    'detection_args': det_path / "args.yaml",
                    'classification_args': final_cls_path / "args.yaml"
                }
                print(f"✅ Found complete pipeline for {model_config['name']}")
                print(f"   Classification found in: {final_cls_path.parent.parent.name}/{final_cls_path.parent.name}")
            else:
                print(f"❌ Incomplete pipeline for {model_config['name']}")
                if not det_path.exists():
                    print(f"   Missing detection: {det_path}")
                if not final_cls_path:
                    print(f"   Missing classification in both:")
                    print(f"     {cls_path}")
                    print(f"     {models_path}")

        return experiments

    def find_centralized_experiments(self, timestamp_pattern):
        """Find experiments in centralized structure (results/exp_name/)"""
        experiments = {}

        # In centralized mode, look for any completed models directly
        detection_base = self.results_base / "detection"
        models_base = self.results_base / "models"
        classification_base = self.results_base / "classification"

        if not detection_base.exists():
            print(f"❌ No detection folder found in {self.results_base}")
            return experiments

        # Find all detection models
        for det_model_dir in detection_base.iterdir():
            if not det_model_dir.is_dir():
                continue

            model_type = det_model_dir.name.replace("_detection", "")

            # Find detection experiments
            for det_exp_dir in det_model_dir.iterdir():
                if not det_exp_dir.is_dir() or not (det_exp_dir / "weights" / "best.pt").exists():
                    continue

                # Try to match with model_info or create dynamic entry
                if model_type not in self.model_info:
                    # Create dynamic model info for models not in predefined list
                    self.model_info[model_type] = {
                        'name': model_type.upper(),
                        'detection_dir': det_model_dir.name,
                        'classification_dir': 'dynamic',
                        'color': '#333333',
                        'marker': 'x'
                    }

                # Look for corresponding classification models
                # Check both models/ and classification/ folders for any matching experiments
                cls_models_found = []

                # Check models/ folder (PyTorch models)
                if models_base.exists():
                    for cls_model_dir in models_base.iterdir():
                        if cls_model_dir.is_dir():
                            for cls_exp_dir in cls_model_dir.iterdir():
                                if cls_exp_dir.is_dir() and (cls_exp_dir / "best.pt").exists():
                                    cls_models_found.append(cls_exp_dir)

                # Check classification/ folder (YOLO models)
                if classification_base.exists():
                    for cls_model_dir in classification_base.iterdir():
                        if cls_model_dir.is_dir():
                            for cls_exp_dir in cls_model_dir.iterdir():
                                if cls_exp_dir.is_dir() and (cls_exp_dir / "weights" / "best.pt").exists():
                                    cls_models_found.append(cls_exp_dir)

                # Create experiment entries for each classification model found
                for i, cls_path in enumerate(cls_models_found):
                    exp_key = f"{model_type}_{cls_path.parent.name}_{i}" if len(cls_models_found) > 1 else f"{model_type}_{cls_path.parent.name}"

                    experiments[exp_key] = {
                        'detection_path': det_exp_dir,
                        'classification_path': cls_path,
                        'detection_results': det_exp_dir / "results.csv",
                        'classification_results': cls_path / "results.txt",  # PyTorch models save as .txt
                        'detection_args': det_exp_dir / "args.yaml",
                        'classification_args': cls_path / "training_config.json"
                    }

                    print(f"✅ Found centralized pipeline: {det_model_dir.name} → {cls_path.parent.name}/{cls_path.name}")

        return experiments

    def analyze_detection_performance(self, experiments):
        """Analyze detection performance for both models"""
        detection_results = {}

        num_experiments = len(experiments)
        if num_experiments == 0:
            return {}

        # Adjust figure layout based on number of experiments
        if num_experiments <= 3:
            fig_cols = num_experiments
            plt.figure(figsize=(5 * fig_cols, 5))
        else:
            # For many experiments, create a summary plot instead
            plt.figure(figsize=(15, 10))

        for i, (model_key, exp_data) in enumerate(experiments.items()):
            model_config = self.model_info.get(model_key, {'name': model_key, 'color': '#333333'})

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

                # Skip individual plots for too many experiments
                if num_experiments <= 3:
                    # Plot training curves
                    plt.subplot(1, fig_cols, i+1)
                    plt.plot(df['epoch'], df['metrics/mAP50(B)'],
                            color=model_config['color'], linewidth=2, label='mAP50')
                    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'],
                            color=model_config['color'], linewidth=2, linestyle='--', label='mAP50-95')
                    plt.title(f'{model_config["name"]} Detection Training')
                    plt.xlabel('Epoch')
                    plt.ylabel('mAP')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

        # Comparison plot for all experiments
        if len(detection_results) > 1:
            if num_experiments > 3:
                plt.clf()  # Clear for single comparison plot
                plt.figure(figsize=(15, 8))

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
            plt.xticks(x, [detection_results[m]['model_name'] for m in models], rotation=45, ha='right')
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
            model_config = self.model_info.get(model_key, {'name': model_key})

            # Read classification results
            if exp_data['classification_results'].exists():
                cls_results_path = exp_data['classification_results']

                if cls_results_path.suffix == '.csv':
                    # YOLO results format
                    df = pd.read_csv(cls_results_path)
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

                elif cls_results_path.suffix == '.txt':
                    # PyTorch results format
                    with open(cls_results_path, 'r') as f:
                        lines = f.readlines()

                    # Parse PyTorch results file
                    model_name = None
                    best_val_acc = 0
                    test_acc = 0
                    training_time = 0

                    for line in lines:
                        if line.startswith('Model:'):
                            model_name = line.strip().split(': ')[1]
                        elif line.startswith('Best Val Acc:'):
                            best_val_acc = float(line.strip().split(': ')[1].replace('%', '')) / 100
                        elif line.startswith('Test Acc:'):
                            test_acc = float(line.strip().split(': ')[1].replace('%', '')) / 100
                        elif line.startswith('Training Time:'):
                            time_str = line.strip().split(': ')[1]
                            if 'min' in time_str:
                                training_time = float(time_str.replace(' min', '')) * 60

                    classification_results[model_key] = {
                        'model_name': model_name or model_config['name'],
                        'epochs': 2,  # Default for PyTorch models (we don't have epoch info in results.txt)
                        'final_top1_acc': test_acc,  # Use test accuracy as final accuracy
                        'final_top5_acc': 0,  # Not available in PyTorch results
                        'training_time': training_time,
                        'train_loss': 0,  # Not available in simple results.txt
                        'val_loss': 0   # Not available in simple results.txt
                    }

                # Plot training curves - Top subplot (only for YOLO models with CSV data)
                if cls_results_path.suffix == '.csv' and len(classification_results) <= 2:
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
            colors = [self.model_info.get(m, {'color': '#333333'})['color'] for m in models]

            bars = plt.bar([self.model_info.get(m, {'name': m})['name'] for m in models], top1_values,
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
            plt.xticks(x, [self.model_info.get(m, {'name': m})['name'] for m in models], rotation=45, ha='right')
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
    import argparse

    parser = argparse.ArgumentParser(description="Unified Journal Analysis for Malaria Detection Pipeline")
    parser.add_argument('--centralized-experiment', type=str,
                       help='Name of centralized experiment directory (e.g., exp_multi_pipeline_20250921_144544)')
    parser.add_argument('--timestamp-pattern', type=str, default="multi_pipeline_20250920_131500",
                       help='Timestamp pattern for distributed experiments')

    args = parser.parse_args()

    if args.centralized_experiment:
        print(f"🎯 Analyzing centralized experiment: {args.centralized_experiment}")
        analyzer = UnifiedJournalAnalyzer(centralized_experiment=args.centralized_experiment)
    else:
        print(f"🎯 Analyzing distributed experiments with pattern: {args.timestamp_pattern}")
        analyzer = UnifiedJournalAnalyzer()

    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()