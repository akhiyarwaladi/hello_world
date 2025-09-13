#!/usr/bin/env python3
"""
Training Monitor Script
Monitors YOLOv8 classification training progress and provides real-time updates
"""

import time
import os
import psutil
from pathlib import Path
import re

class TrainingMonitor:
    def __init__(self):
        self.project_dir = Path.cwd()
        self.log_file = self.project_dir / "training_log.txt"
        self.results_dir = self.project_dir / "results" / "classification" / "malaria_background"
        
    def check_process_status(self):
        """Check if training process is still running"""
        for proc in psutil.process_iter(['pid', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                cmdline_list = proc.info['cmdline']
                if cmdline_list:
                    cmdline = ' '.join(str(x) for x in cmdline_list)
                    if 'ultralytics' in cmdline and 'malaria_background' in cmdline:
                        return {
                            'pid': proc.info['pid'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024
                        }
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                continue
        return None
    
    def parse_training_log(self):
        """Parse training log for progress information"""
        if not self.log_file.exists():
            return None
            
        try:
            with open(self.log_file, 'r') as f:
                content = f.read()
            
            # Extract epoch information
            epoch_pattern = r'(\d+)/(\d+)\s+.*?(\d+\.\d+)\s+.*?(\d+)\s+.*?(\d+):'
            matches = re.findall(epoch_pattern, content)
            
            if matches:
                last_match = matches[-1]
                current_epoch = int(last_match[0])
                total_epochs = int(last_match[1])
                loss = float(last_match[2])
                
                return {
                    'current_epoch': current_epoch,
                    'total_epochs': total_epochs,
                    'loss': loss,
                    'progress_percent': (current_epoch / total_epochs) * 100
                }
        except Exception as e:
            print(f"Error parsing log: {e}")
        
        return None
    
    def check_results(self):
        """Check for training results and weights"""
        results = {}
        
        if self.results_dir.exists():
            # Check for weights
            weights_dir = self.results_dir / "weights"
            if weights_dir.exists():
                weight_files = list(weights_dir.glob("*.pt"))
                results['weights'] = [w.name for w in weight_files]
            
            # Check for plots/results
            plot_files = list(self.results_dir.glob("*.png")) + list(self.results_dir.glob("*.jpg"))
            results['plots'] = [p.name for p in plot_files]
            
            # Check for results CSV
            csv_files = list(self.results_dir.glob("*.csv"))
            results['csv_files'] = [c.name for c in csv_files]
        
        return results
    
    def get_dataset_info(self):
        """Get dataset information"""
        data_dir = Path("data/classification")
        if not data_dir.exists():
            return None
        
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        
        info = {}
        
        if train_dir.exists():
            classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
            train_counts = {}
            for cls in classes:
                count = len(list((train_dir / cls).glob("*.jpg")))
                train_counts[cls] = count
            info['train'] = train_counts
        
        if val_dir.exists():
            classes = [d.name for d in val_dir.iterdir() if d.is_dir()]
            val_counts = {}
            for cls in classes:
                count = len(list((val_dir / cls).glob("*.jpg")))
                val_counts[cls] = count
            info['val'] = val_counts
        
        return info
    
    def display_status(self):
        """Display comprehensive training status"""
        print("=" * 80)
        print("MALARIA DETECTION - YOLOv8 CLASSIFICATION TRAINING MONITOR")
        print("=" * 80)
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Process status
        proc_info = self.check_process_status()
        if proc_info:
            print("🟢 TRAINING STATUS: RUNNING")
            print(f"   PID: {proc_info['pid']}")
            print(f"   CPU Usage: {proc_info['cpu_percent']:.1f}%")
            print(f"   Memory Usage: {proc_info['memory_mb']:.1f} MB")
        else:
            print("🔴 TRAINING STATUS: NOT RUNNING")
        
        print()
        
        # Training progress
        log_info = self.parse_training_log()
        if log_info:
            print("📊 TRAINING PROGRESS:")
            print(f"   Epoch: {log_info['current_epoch']}/{log_info['total_epochs']}")
            print(f"   Progress: {log_info['progress_percent']:.1f}%")
            print(f"   Current Loss: {log_info['loss']:.4f}")
        else:
            print("📊 TRAINING PROGRESS: No progress data available")
        
        print()
        
        # Dataset info
        dataset_info = self.get_dataset_info()
        if dataset_info:
            print("📁 DATASET INFORMATION:")
            if 'train' in dataset_info:
                total_train = sum(dataset_info['train'].values())
                print(f"   Training Images: {total_train}")
                for cls, count in dataset_info['train'].items():
                    print(f"     {cls}: {count}")
            
            if 'val' in dataset_info:
                total_val = sum(dataset_info['val'].values())
                print(f"   Validation Images: {total_val}")
        
        print()
        
        # Results
        results = self.check_results()
        if results:
            print("💾 TRAINING RESULTS:")
            if results.get('weights'):
                print(f"   Weights: {', '.join(results['weights'])}")
            if results.get('plots'):
                print(f"   Plots: {len(results['plots'])} files")
            if results.get('csv_files'):
                print(f"   Results: {', '.join(results['csv_files'])}")
        else:
            print("💾 TRAINING RESULTS: No results generated yet")
        
        print("=" * 80)

def main():
    monitor = TrainingMonitor()
    
    print("Starting Training Monitor...")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            monitor.display_status()
            time.sleep(30)  # Update every 30 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()