#!/usr/bin/env python3
"""
Real-time Training Progress Monitor and Visualization
Monitors active training processes and generates live progress reports
"""

import os
import sys
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime, timedelta
import subprocess
import re
import argparse
from collections import defaultdict

class TrainingMonitor:
    """Monitor and visualize training progress across all experiments"""

    def __init__(self, refresh_interval=60):
        self.refresh_interval = refresh_interval
        self.start_time = datetime.now()
        self.training_data = defaultdict(list)
        self.experiment_status = {}

        # Create output directory
        self.output_dir = Path("training_monitor")
        self.output_dir.mkdir(exist_ok=True)

        print(f"🔍 Training Monitor initialized")
        print(f"📂 Output directory: {self.output_dir}")
        print(f"⏱️  Refresh interval: {refresh_interval}s")

    def get_active_processes(self):
        """Get all active Python training processes"""
        active_processes = []

        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['name'] == 'python':
                    cmdline = ' '.join(proc.info['cmdline'])

                    # Check if it's a training process
                    if any(keyword in cmdline for keyword in [
                        'train_classification', 'train_detection', 'train_yolo',
                        'train_pytorch', 'rtdetr', 'pipeline.py'
                    ]):

                        # Extract experiment details
                        experiment_info = self._extract_experiment_info(cmdline)
                        experiment_info.update({
                            'pid': proc.info['pid'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_percent': proc.info['memory_percent'],
                            'runtime': datetime.now() - datetime.fromtimestamp(proc.info['create_time']),
                            'command': cmdline
                        })

                        active_processes.append(experiment_info)

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        return active_processes

    def _extract_experiment_info(self, cmdline):
        """Extract experiment information from command line"""
        info = {
            'experiment_name': 'unknown',
            'model_type': 'unknown',
            'dataset': 'unknown',
            'epochs': 'unknown',
            'batch_size': 'unknown',
            'device': 'cpu'
        }

        # Extract name
        name_match = re.search(r'--name\s+([^\s]+)', cmdline)
        if name_match:
            info['experiment_name'] = name_match.group(1)

        # Extract model type from script name or parameters
        if 'yolo8' in cmdline.lower():
            info['model_type'] = 'YOLOv8'
        elif 'yolo11' in cmdline.lower():
            info['model_type'] = 'YOLOv11'
        elif 'rtdetr' in cmdline.lower():
            info['model_type'] = 'RT-DETR'
        elif 'resnet' in cmdline.lower():
            info['model_type'] = 'ResNet'
        elif 'efficientnet' in cmdline.lower():
            info['model_type'] = 'EfficientNet'
        elif 'densenet' in cmdline.lower():
            info['model_type'] = 'DenseNet'
        elif 'mobilenet' in cmdline.lower():
            info['model_type'] = 'MobileNet'

        # Extract epochs
        epochs_match = re.search(r'--epochs\s+(\d+)', cmdline)
        if epochs_match:
            info['epochs'] = int(epochs_match.group(1))

        # Extract batch size
        batch_match = re.search(r'--batch\s+(\d+)', cmdline)
        if batch_match:
            info['batch_size'] = int(batch_match.group(1))

        # Extract dataset
        data_match = re.search(r'--data\s+([^\s]+)', cmdline)
        if data_match:
            info['dataset'] = Path(data_match.group(1)).name

        # Extract device
        device_match = re.search(r'--device\s+([^\s]+)', cmdline)
        if device_match:
            info['device'] = device_match.group(1)

        return info

    def scan_training_results(self):
        """Scan for training results and progress"""
        results = []

        # Scan results directories
        results_dirs = [
            "results",
            "results/current_experiments/training",
            "results/current_experiments/validation"
        ]

        for results_dir in results_dirs:
            if not Path(results_dir).exists():
                continue

            # Find all results.csv files
            csv_files = list(Path(results_dir).rglob("results.csv"))

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) > 0:
                        experiment_path = csv_file.parent
                        experiment_name = experiment_path.name

                        # Extract latest metrics
                        latest = df.iloc[-1]

                        result_info = {
                            'experiment_name': experiment_name,
                            'experiment_path': str(experiment_path),
                            'current_epoch': len(df),
                            'latest_metrics': dict(latest),
                            'progress_data': df.to_dict('records')
                        }

                        results.append(result_info)

                except Exception as e:
                    continue

        return results

    def generate_progress_visualizations(self, processes, results):
        """Generate training progress visualizations"""

        # 1. System Resource Usage
        self._plot_system_resources(processes)

        # 2. Training Progress Charts
        self._plot_training_progress(results)

        # 3. Experiment Status Dashboard
        self._create_status_dashboard(processes, results)

        # 4. Performance Comparison
        self._create_performance_comparison(results)

    def _plot_system_resources(self, processes):
        """Plot system resource usage"""
        if not processes:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('System Resource Usage - Training Processes', fontsize=16, fontweight='bold')

        # CPU usage by process
        names = [p['experiment_name'][:20] + '...' if len(p['experiment_name']) > 20 else p['experiment_name'] for p in processes]
        cpu_usage = [p['cpu_percent'] for p in processes]
        memory_usage = [p['memory_percent'] for p in processes]

        axes[0,0].bar(range(len(names)), cpu_usage)
        axes[0,0].set_title('CPU Usage by Process (%)')
        axes[0,0].set_xticks(range(len(names)))
        axes[0,0].set_xticklabels(names, rotation=45)
        axes[0,0].set_ylabel('CPU %')

        # Memory usage by process
        axes[0,1].bar(range(len(names)), memory_usage, color='orange')
        axes[0,1].set_title('Memory Usage by Process (%)')
        axes[0,1].set_xticks(range(len(names)))
        axes[0,1].set_xticklabels(names, rotation=45)
        axes[0,1].set_ylabel('Memory %')

        # Runtime distribution
        runtimes = [(p['runtime'].total_seconds() / 3600) for p in processes]  # Convert to hours
        model_types = [p['model_type'] for p in processes]

        axes[1,0].bar(range(len(names)), runtimes, color='green')
        axes[1,0].set_title('Training Runtime (hours)')
        axes[1,0].set_xticks(range(len(names)))
        axes[1,0].set_xticklabels(names, rotation=45)
        axes[1,0].set_ylabel('Hours')

        # Model type distribution
        from collections import Counter
        type_counts = Counter(model_types)
        axes[1,1].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        axes[1,1].set_title('Active Model Types')

        plt.tight_layout()
        plt.savefig(self.output_dir / "system_resources.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_training_progress(self, results):
        """Plot training progress for experiments"""
        if not results:
            return

        # Create subplots for different metrics
        metrics_of_interest = [
            'train/loss', 'val/loss', 'metrics/accuracy_top1',
            'metrics/mAP50(B)', 'metrics/precision(B)', 'metrics/recall(B)'
        ]

        available_metrics = set()
        for result in results:
            if result['progress_data']:
                available_metrics.update(result['progress_data'][0].keys())

        # Filter to available metrics
        available_metrics = [m for m in metrics_of_interest if m in available_metrics]

        if not available_metrics:
            return

        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        fig.suptitle('Training Progress - All Experiments', fontsize=16, fontweight='bold')

        colors = plt.cm.tab10(range(len(results)))

        for i, metric in enumerate(available_metrics):
            ax = axes[i] if len(available_metrics) > 1 else axes

            for j, result in enumerate(results):
                progress_df = pd.DataFrame(result['progress_data'])
                if metric in progress_df.columns:
                    epochs = range(1, len(progress_df) + 1)
                    values = progress_df[metric].values

                    # Remove NaN values
                    mask = ~pd.isna(values)
                    epochs_clean = [e for e, m in zip(epochs, mask) if m]
                    values_clean = values[mask]

                    if len(values_clean) > 0:
                        ax.plot(epochs_clean, values_clean,
                               color=colors[j],
                               label=result['experiment_name'][:15],
                               linewidth=2)

            ax.set_title(metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.split('/')[-1])
            ax.grid(True, alpha=0.3)

            if i == 0:  # Only add legend to first subplot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Hide extra subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / "training_progress.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_status_dashboard(self, processes, results):
        """Create a status dashboard"""
        # Create HTML dashboard
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Malaria Detection Training Monitor</title>
    <meta http-equiv="refresh" content="{self.refresh_interval}">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .panel {{ background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .metric {{ display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #eee; }}
        .metric:last-child {{ border-bottom: none; }}
        .status-active {{ color: #27ae60; font-weight: bold; }}
        .status-completed {{ color: #3498db; font-weight: bold; }}
        .process-table {{ width: 100%; border-collapse: collapse; }}
        .process-table th, .process-table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        .process-table th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 Malaria Detection Training Monitor</h1>
        <p>Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Monitor started: {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Uptime: {str(datetime.now() - self.start_time).split('.')[0]}</p>
    </div>

    <div class="dashboard">
        <div class="panel">
            <h3>📊 System Overview</h3>
            <div class="metric">
                <span>Active Training Processes:</span>
                <span class="status-active">{len(processes)}</span>
            </div>
            <div class="metric">
                <span>Experiments with Results:</span>
                <span class="status-completed">{len(results)}</span>
            </div>
            <div class="metric">
                <span>Total CPU Usage:</span>
                <span>{sum(p['cpu_percent'] for p in processes):.1f}%</span>
            </div>
            <div class="metric">
                <span>Total Memory Usage:</span>
                <span>{sum(p['memory_percent'] for p in processes):.1f}%</span>
            </div>
        </div>

        <div class="panel">
            <h3>🎯 Model Types in Training</h3>
"""

        # Count model types
        model_counts = defaultdict(int)
        for p in processes:
            model_counts[p['model_type']] += 1

        for model_type, count in model_counts.items():
            html_content += f"""
            <div class="metric">
                <span>{model_type}:</span>
                <span class="status-active">{count}</span>
            </div>
"""

        html_content += f"""
        </div>
    </div>

    <div class="panel" style="margin-top: 20px;">
        <h3>🔄 Active Training Processes</h3>
        <table class="process-table">
            <tr>
                <th>Experiment</th>
                <th>Model Type</th>
                <th>Dataset</th>
                <th>Epochs</th>
                <th>Batch Size</th>
                <th>Runtime</th>
                <th>CPU %</th>
                <th>Memory %</th>
            </tr>
"""

        for p in processes:
            runtime_str = str(p['runtime']).split('.')[0]
            html_content += f"""
            <tr>
                <td>{p['experiment_name']}</td>
                <td>{p['model_type']}</td>
                <td>{p['dataset']}</td>
                <td>{p['epochs']}</td>
                <td>{p['batch_size']}</td>
                <td>{runtime_str}</td>
                <td>{p['cpu_percent']:.1f}%</td>
                <td>{p['memory_percent']:.1f}%</td>
            </tr>
"""

        html_content += """
        </table>
    </div>

    <div class="panel" style="margin-top: 20px;">
        <h3>📈 Recent Results</h3>
"""

        for result in results[-10:]:  # Show last 10 results
            current_epoch = result['current_epoch']
            html_content += f"""
        <div class="metric">
            <span>{result['experiment_name']} (Epoch {current_epoch})</span>
            <span>📊</span>
        </div>
"""

        html_content += """
    </div>

    <div style="margin-top: 20px; text-align: center; color: #7f8c8d;">
        <p>🔄 Auto-refreshing every """ + str(self.refresh_interval) + """ seconds</p>
        <p>Generated by Malaria Detection Training Monitor</p>
    </div>
</body>
</html>
"""

        # Save HTML dashboard
        with open(self.output_dir / "dashboard.html", 'w') as f:
            f.write(html_content)

    def _create_performance_comparison(self, results):
        """Create performance comparison chart"""
        if len(results) < 2:
            return

        # Extract performance metrics
        performance_data = []

        for result in results:
            if result['progress_data']:
                latest_metrics = result['latest_metrics']

                perf_record = {
                    'experiment': result['experiment_name'],
                    'epoch': result['current_epoch']
                }

                # Extract relevant metrics
                for key, value in latest_metrics.items():
                    if key in ['metrics/accuracy_top1', 'metrics/mAP50(B)', 'train/loss', 'val/loss']:
                        perf_record[key] = value

                performance_data.append(perf_record)

        if not performance_data:
            return

        perf_df = pd.DataFrame(performance_data)

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Comparison Across Experiments', fontsize=16, fontweight='bold')

        # Accuracy comparison (if available)
        if 'metrics/accuracy_top1' in perf_df.columns:
            acc_data = perf_df[['experiment', 'metrics/accuracy_top1']].dropna()
            if not acc_data.empty:
                axes[0,0].bar(range(len(acc_data)), acc_data['metrics/accuracy_top1'])
                axes[0,0].set_title('Classification Accuracy')
                axes[0,0].set_xticks(range(len(acc_data)))
                axes[0,0].set_xticklabels(acc_data['experiment'], rotation=45)
                axes[0,0].set_ylabel('Accuracy')

        # mAP comparison (if available)
        if 'metrics/mAP50(B)' in perf_df.columns:
            map_data = perf_df[['experiment', 'metrics/mAP50(B)']].dropna()
            if not map_data.empty:
                axes[0,1].bar(range(len(map_data)), map_data['metrics/mAP50(B)'], color='orange')
                axes[0,1].set_title('Detection mAP@50')
                axes[0,1].set_xticks(range(len(map_data)))
                axes[0,1].set_xticklabels(map_data['experiment'], rotation=45)
                axes[0,1].set_ylabel('mAP@50')

        # Training loss comparison
        if 'train/loss' in perf_df.columns:
            loss_data = perf_df[['experiment', 'train/loss']].dropna()
            if not loss_data.empty:
                axes[1,0].bar(range(len(loss_data)), loss_data['train/loss'], color='red')
                axes[1,0].set_title('Training Loss')
                axes[1,0].set_xticks(range(len(loss_data)))
                axes[1,0].set_xticklabels(loss_data['experiment'], rotation=45)
                axes[1,0].set_ylabel('Loss')

        # Epoch progression
        axes[1,1].bar(range(len(perf_df)), perf_df['epoch'], color='green')
        axes[1,1].set_title('Training Progress (Epochs)')
        axes[1,1].set_xticks(range(len(perf_df)))
        axes[1,1].set_xticklabels(perf_df['experiment'], rotation=45)
        axes[1,1].set_ylabel('Current Epoch')

        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_summary_report(self, processes, results):
        """Generate summary report"""
        report_content = f"""# Training Monitor Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Monitor Uptime:** {str(datetime.now() - self.start_time).split('.')[0]}

## System Status

- **Active Training Processes:** {len(processes)}
- **Experiments with Results:** {len(results)}
- **Total CPU Usage:** {sum(p['cpu_percent'] for p in processes):.1f}%
- **Total Memory Usage:** {sum(p['memory_percent'] for p in processes):.1f}%

## Active Experiments

"""

        for p in processes:
            runtime = str(p['runtime']).split('.')[0]
            report_content += f"""### {p['experiment_name']}
- **Model Type:** {p['model_type']}
- **Dataset:** {p['dataset']}
- **Configuration:** {p['epochs']} epochs, batch size {p['batch_size']}, {p['device']}
- **Runtime:** {runtime}
- **Resource Usage:** CPU {p['cpu_percent']:.1f}%, Memory {p['memory_percent']:.1f}%

"""

        report_content += f"""
## Recent Results

"""

        for result in results[-5:]:  # Last 5 results
            report_content += f"""### {result['experiment_name']}
- **Current Epoch:** {result['current_epoch']}
- **Status:** In Progress

"""

        # Save report
        with open(self.output_dir / "summary_report.md", 'w') as f:
            f.write(report_content)

    def run_monitoring_loop(self):
        """Main monitoring loop"""
        print(f"🔄 Starting monitoring loop (refresh every {self.refresh_interval}s)")
        print("Press Ctrl+C to stop")

        iteration = 0

        try:
            while True:
                iteration += 1
                print(f"\n{'='*60}")
                print(f"Monitor Update #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*60}")

                # Get current status
                processes = self.get_active_processes()
                results = self.scan_training_results()

                print(f"📊 Active processes: {len(processes)}")
                print(f"📋 Experiments with results: {len(results)}")

                if processes:
                    print("🔄 Running experiments:")
                    for p in processes:
                        runtime = str(p['runtime']).split('.')[0]
                        print(f"  • {p['experiment_name']} ({p['model_type']}) - {runtime}")

                # Generate visualizations and reports
                self.generate_progress_visualizations(processes, results)
                self.generate_summary_report(processes, results)

                print(f"💾 Reports updated in {self.output_dir}")
                print(f"🌐 Dashboard available at {self.output_dir}/dashboard.html")

                # Wait for next iteration
                print(f"⏳ Next update in {self.refresh_interval}s...")
                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user")
            print(f"📊 Final reports available in {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Training Progress Monitor")
    parser.add_argument("--refresh", type=int, default=60, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")

    args = parser.parse_args()

    print("=" * 60)
    print("MALARIA DETECTION TRAINING MONITOR")
    print("=" * 60)

    monitor = TrainingMonitor(refresh_interval=args.refresh)

    if args.once:
        # Single run
        processes = monitor.get_active_processes()
        results = monitor.scan_training_results()

        monitor.generate_progress_visualizations(processes, results)
        monitor.generate_summary_report(processes, results)

        print(f"📊 Single report generated in {monitor.output_dir}")
    else:
        # Continuous monitoring
        monitor.run_monitoring_loop()

if __name__ == "__main__":
    main()