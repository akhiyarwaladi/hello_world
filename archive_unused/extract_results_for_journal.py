#!/usr/bin/env python3
"""
Extract and organize training results for journal publication
Generates comprehensive tables and metrics from training outputs
"""

import pandas as pd
import json
import yaml
from pathlib import Path
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class JournalResultsExtractor:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.journal_output_dir = Path("journal_results")
        self.journal_output_dir.mkdir(exist_ok=True)

    def extract_detection_results(self, experiment_path):
        """Extract detection metrics from results.csv"""
        results_csv = experiment_path / "results.csv"
        if not results_csv.exists():
            return None

        df = pd.read_csv(results_csv)

        # Get final epoch metrics
        final_metrics = df.iloc[-1] if len(df) > 0 else None
        if final_metrics is None:
            return None

        # Get best mAP50 epoch
        best_map50_idx = df['metrics/mAP50(B)'].idxmax()
        best_metrics = df.iloc[best_map50_idx]

        return {
            'final_epoch': int(final_metrics['epoch']),
            'final_map50': float(final_metrics['metrics/mAP50(B)']),
            'final_map50_95': float(final_metrics['metrics/mAP50-95(B)']),
            'final_precision': float(final_metrics['metrics/precision(B)']),
            'final_recall': float(final_metrics['metrics/recall(B)']),
            'best_map50': float(best_metrics['metrics/mAP50(B)']),
            'best_map50_epoch': int(best_metrics['epoch']),
            'training_time': float(final_metrics['time']),
            'convergence_epochs': self._detect_convergence(df),
            'learning_stability': self._calculate_stability(df)
        }

    def extract_classification_results(self, experiment_path):
        """Extract classification metrics from results.csv"""
        results_csv = experiment_path / "results.csv"
        if not results_csv.exists():
            return None

        df = pd.read_csv(results_csv)

        # Get final and best metrics
        final_metrics = df.iloc[-1] if len(df) > 0 else None
        if final_metrics is None:
            return None

        # Find best validation accuracy
        val_acc_col = 'metrics/accuracy_top1'
        if val_acc_col not in df.columns:
            # Try alternative column names
            possible_cols = [col for col in df.columns if 'acc' in col.lower()]
            val_acc_col = possible_cols[0] if possible_cols else None

        if val_acc_col:
            best_acc_idx = df[val_acc_col].idxmax()
            best_metrics = df.iloc[best_acc_idx]

            return {
                'final_epoch': int(final_metrics['epoch']),
                'final_accuracy': float(final_metrics[val_acc_col]),
                'final_loss': float(final_metrics.get('val/loss', final_metrics.get('train/loss', 0))),
                'best_accuracy': float(best_metrics[val_acc_col]),
                'best_accuracy_epoch': int(best_metrics['epoch']),
                'training_time': float(final_metrics['time']),
                'convergence_epochs': self._detect_convergence(df, val_acc_col),
                'learning_stability': self._calculate_stability(df, val_acc_col)
            }
        return None

    def _detect_convergence(self, df, metric_col='metrics/mAP50(B)'):
        """Detect convergence epoch (when improvement plateaus)"""
        if metric_col not in df.columns:
            return None

        values = df[metric_col].values
        # Find where improvement becomes < 1% over 5 epochs
        window = 5
        threshold = 0.01

        for i in range(window, len(values)):
            recent_max = np.max(values[i-window:i])
            recent_min = np.min(values[i-window:i])
            if (recent_max - recent_min) < threshold:
                return i - window + 1
        return len(values)

    def _calculate_stability(self, df, metric_col='metrics/mAP50(B)'):
        """Calculate learning stability (lower std = more stable)"""
        if metric_col not in df.columns:
            return None

        values = df[metric_col].values
        # Calculate coefficient of variation for last 10 epochs
        last_epochs = values[-10:] if len(values) >= 10 else values
        return float(np.std(last_epochs) / np.mean(last_epochs)) if np.mean(last_epochs) > 0 else 0

    def scan_all_results(self):
        """Scan all result directories and extract metrics"""
        all_results = {}

        # Scan results directories
        for results_type in ['detection', 'classification', 'pipeline_final']:
            type_dir = self.results_dir / results_type
            if not type_dir.exists():
                continue

            for experiment_dir in type_dir.iterdir():
                if not experiment_dir.is_dir():
                    continue

                experiment_name = experiment_dir.name

                # Determine if detection or classification
                if 'detection' in results_type or 'detect' in experiment_name.lower():
                    metrics = self.extract_detection_results(experiment_dir)
                    result_type = 'detection'
                else:
                    metrics = self.extract_classification_results(experiment_dir)
                    result_type = 'classification'

                if metrics:
                    # Get dataset info
                    args_file = experiment_dir / "args.yaml"
                    dataset_info = self._extract_dataset_info(args_file)

                    all_results[f"{results_type}_{experiment_name}"] = {
                        'type': result_type,
                        'experiment': experiment_name,
                        'category': results_type,
                        'metrics': metrics,
                        'dataset': dataset_info
                    }

        return all_results

    def _extract_dataset_info(self, args_file):
        """Extract dataset information from args.yaml"""
        if not args_file.exists():
            return {}

        try:
            with open(args_file, 'r') as f:
                args = yaml.safe_load(f)

            return {
                'data_path': args.get('data', ''),
                'epochs': args.get('epochs', 0),
                'batch_size': args.get('batch', 0),
                'image_size': args.get('imgsz', 640),
                'device': args.get('device', 'unknown')
            }
        except:
            return {}

    def generate_journal_tables(self):
        """Generate comprehensive tables for journal publication"""
        results = self.scan_all_results()

        # 1. Detection Performance Table
        detection_results = {k: v for k, v in results.items() if v['type'] == 'detection'}
        detection_table = self._create_detection_table(detection_results)

        # 2. Classification Performance Table
        classification_results = {k: v for k, v in results.items() if v['type'] == 'classification'}
        classification_table = self._create_classification_table(classification_results)

        # 3. Dataset Statistics Table
        dataset_table = self._create_dataset_table(results)

        # 4. Training Efficiency Table
        efficiency_table = self._create_efficiency_table(results)

        # Save all tables
        self._save_tables({
            'detection_performance': detection_table,
            'classification_performance': classification_table,
            'dataset_statistics': dataset_table,
            'training_efficiency': efficiency_table
        })

        # Generate summary statistics
        self._generate_summary_stats(results)

        return {
            'total_experiments': len(results),
            'detection_experiments': len(detection_results),
            'classification_experiments': len(classification_results),
            'tables_saved': self.journal_output_dir
        }

    def _create_detection_table(self, results):
        """Create detection performance table"""
        table_data = []

        for experiment, data in results.items():
            metrics = data['metrics']
            dataset = data['dataset']

            table_data.append({
                'Experiment': data['experiment'],
                'Dataset': self._format_dataset_name(dataset['data_path']),
                'mAP@0.5': f"{metrics['final_map50']:.3f}",
                'mAP@0.5:0.95': f"{metrics['final_map50_95']:.3f}",
                'Precision': f"{metrics['final_precision']:.3f}",
                'Recall': f"{metrics['final_recall']:.3f}",
                'Best mAP@0.5': f"{metrics['best_map50']:.3f}",
                'Epochs': metrics['final_epoch'],
                'Convergence': metrics.get('convergence_epochs', 'N/A'),
                'Stability': f"{metrics.get('learning_stability', 0):.4f}",
                'Training Time (min)': f"{metrics['training_time']/60:.1f}"
            })

        return pd.DataFrame(table_data)

    def _create_classification_table(self, results):
        """Create classification performance table"""
        table_data = []

        for experiment, data in results.items():
            metrics = data['metrics']
            dataset = data['dataset']

            table_data.append({
                'Experiment': data['experiment'],
                'Dataset': self._format_dataset_name(dataset['data_path']),
                'Final Accuracy': f"{metrics['final_accuracy']:.3f}",
                'Best Accuracy': f"{metrics['best_accuracy']:.3f}",
                'Final Loss': f"{metrics['final_loss']:.4f}",
                'Epochs': metrics['final_epoch'],
                'Convergence': metrics.get('convergence_epochs', 'N/A'),
                'Stability': f"{metrics.get('learning_stability', 0):.4f}",
                'Training Time (min)': f"{metrics['training_time']/60:.1f}"
            })

        return pd.DataFrame(table_data)

    def _create_dataset_table(self, results):
        """Create dataset statistics table"""
        dataset_stats = defaultdict(list)

        for experiment, data in results.items():
            dataset = data['dataset']
            dataset_name = self._format_dataset_name(dataset['data_path'])
            dataset_stats[dataset_name].append({
                'type': data['type'],
                'batch_size': dataset['batch_size'],
                'epochs': dataset['epochs'],
                'image_size': dataset['image_size']
            })

        table_data = []
        for dataset, experiments in dataset_stats.items():
            table_data.append({
                'Dataset': dataset,
                'Experiments': len(experiments),
                'Types': ', '.join(set(exp['type'] for exp in experiments)),
                'Avg Epochs': int(np.mean([exp['epochs'] for exp in experiments])),
                'Avg Batch Size': int(np.mean([exp['batch_size'] for exp in experiments])),
                'Image Size': experiments[0]['image_size']
            })

        return pd.DataFrame(table_data)

    def _create_efficiency_table(self, results):
        """Create training efficiency comparison table"""
        table_data = []

        for experiment, data in results.items():
            metrics = data['metrics']
            dataset = data['dataset']

            # Calculate efficiency metrics
            epochs_to_convergence = metrics.get('convergence_epochs', metrics['final_epoch'])
            efficiency_score = self._calculate_efficiency_score(metrics, dataset)

            table_data.append({
                'Experiment': data['experiment'],
                'Type': data['type'].capitalize(),
                'Total Epochs': metrics['final_epoch'],
                'Convergence Epochs': epochs_to_convergence,
                'Convergence %': f"{epochs_to_convergence/metrics['final_epoch']*100:.1f}%",
                'Efficiency Score': f"{efficiency_score:.3f}",
                'Time/Epoch (min)': f"{metrics['training_time']/metrics['final_epoch']/60:.2f}",
                'Stability': f"{metrics.get('learning_stability', 0):.4f}"
            })

        return pd.DataFrame(table_data)

    def _calculate_efficiency_score(self, metrics, dataset):
        """Calculate efficiency score (performance/time/epochs)"""
        if metrics.get('final_map50'):
            performance = metrics['final_map50']
        else:
            performance = metrics.get('final_accuracy', 0)

        time_factor = 1 / (metrics['training_time'] / 3600)  # inverse hours
        epoch_factor = 1 / metrics['final_epoch']  # inverse epochs

        return performance * time_factor * epoch_factor * 1000  # scaled

    def _format_dataset_name(self, data_path):
        """Format dataset path to readable name"""
        if 'multispecies' in data_path:
            return 'Multi-Species'
        elif 'detection_fixed' in data_path:
            return 'Detection Fixed'
        elif 'classification_crops' in data_path:
            return 'Classification Crops'
        else:
            return Path(data_path).name.title()

    def _save_tables(self, tables):
        """Save all tables to files"""
        for name, table in tables.items():
            # Save as CSV
            csv_path = self.journal_output_dir / f"{name}.csv"
            table.to_csv(csv_path, index=False)

            # Save as formatted text for LaTeX
            latex_path = self.journal_output_dir / f"{name}.tex"
            with open(latex_path, 'w') as f:
                f.write(table.to_latex(index=False, float_format="%.3f"))

            # Save as pretty formatted text
            txt_path = self.journal_output_dir / f"{name}.txt"
            with open(txt_path, 'w') as f:
                f.write(table.to_string(index=False))

    def _generate_summary_stats(self, results):
        """Generate summary statistics for the paper"""
        detection_results = [v for v in results.values() if v['type'] == 'detection']
        classification_results = [v for v in results.values() if v['type'] == 'classification']

        summary = {
            'total_experiments': len(results),
            'detection_experiments': len(detection_results),
            'classification_experiments': len(classification_results),
            'detection_stats': self._calculate_detection_stats(detection_results),
            'classification_stats': self._calculate_classification_stats(classification_results),
            'efficiency_stats': self._calculate_efficiency_stats(results)
        }

        # Save summary
        summary_path = self.journal_output_dir / "summary_statistics.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Create readable summary report
        self._create_summary_report(summary)

    def _calculate_detection_stats(self, results):
        """Calculate detection summary statistics"""
        if not results:
            return {}

        map50_values = [r['metrics']['final_map50'] for r in results]
        map50_95_values = [r['metrics']['final_map50_95'] for r in results]

        return {
            'mean_map50': float(np.mean(map50_values)),
            'std_map50': float(np.std(map50_values)),
            'best_map50': float(np.max(map50_values)),
            'mean_map50_95': float(np.mean(map50_95_values)),
            'std_map50_95': float(np.std(map50_95_values)),
            'best_map50_95': float(np.max(map50_95_values))
        }

    def _calculate_classification_stats(self, results):
        """Calculate classification summary statistics"""
        if not results:
            return {}

        acc_values = [r['metrics']['final_accuracy'] for r in results]

        return {
            'mean_accuracy': float(np.mean(acc_values)),
            'std_accuracy': float(np.std(acc_values)),
            'best_accuracy': float(np.max(acc_values)),
            'min_accuracy': float(np.min(acc_values))
        }

    def _calculate_efficiency_stats(self, results):
        """Calculate efficiency summary statistics"""
        training_times = [r['metrics']['training_time'] for r in results.values()]
        epochs = [r['metrics']['final_epoch'] for r in results.values()]

        return {
            'mean_training_time_hours': float(np.mean(training_times) / 3600),
            'mean_epochs': float(np.mean(epochs)),
            'total_training_time_hours': float(np.sum(training_times) / 3600)
        }

    def _create_summary_report(self, summary):
        """Create human-readable summary report"""
        report_path = self.journal_output_dir / "summary_report.md"

        with open(report_path, 'w') as f:
            f.write("# Malaria Detection Pipeline - Journal Results Summary\n\n")

            f.write(f"## Experiment Overview\n")
            f.write(f"- Total Experiments: {summary['total_experiments']}\n")
            f.write(f"- Detection Experiments: {summary['detection_experiments']}\n")
            f.write(f"- Classification Experiments: {summary['classification_experiments']}\n\n")

            if summary['detection_stats']:
                f.write(f"## Detection Performance\n")
                det = summary['detection_stats']
                f.write(f"- Mean mAP@0.5: {det['mean_map50']:.3f} ± {det['std_map50']:.3f}\n")
                f.write(f"- Best mAP@0.5: {det['best_map50']:.3f}\n")
                f.write(f"- Mean mAP@0.5:0.95: {det['mean_map50_95']:.3f} ± {det['std_map50_95']:.3f}\n\n")

            if summary['classification_stats']:
                f.write(f"## Classification Performance\n")
                cls = summary['classification_stats']
                f.write(f"- Mean Accuracy: {cls['mean_accuracy']:.3f} ± {cls['std_accuracy']:.3f}\n")
                f.write(f"- Best Accuracy: {cls['best_accuracy']:.3f}\n")
                f.write(f"- Range: {cls['min_accuracy']:.3f} - {cls['best_accuracy']:.3f}\n\n")

            if summary['efficiency_stats']:
                f.write(f"## Training Efficiency\n")
                eff = summary['efficiency_stats']
                f.write(f"- Mean Training Time: {eff['mean_training_time_hours']:.1f} hours\n")
                f.write(f"- Mean Epochs: {eff['mean_epochs']:.0f}\n")
                f.write(f"- Total Compute Time: {eff['total_training_time_hours']:.1f} hours\n\n")

def main():
    """Main function to extract all results"""
    print("🔍 Extracting results for journal publication...")

    extractor = JournalResultsExtractor()
    results = extractor.generate_journal_tables()

    print(f"✅ Results extraction completed!")
    print(f"📊 Total experiments: {results['total_experiments']}")
    print(f"🎯 Detection experiments: {results['detection_experiments']}")
    print(f"📈 Classification experiments: {results['classification_experiments']}")
    print(f"📁 Tables saved to: {results['tables_saved']}")

    print("\n📋 Generated files:")
    journal_dir = Path("journal_results")
    for file in sorted(journal_dir.glob("*")):
        print(f"   - {file.name}")

if __name__ == "__main__":
    main()