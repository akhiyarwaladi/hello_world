#!/usr/bin/env python3
"""
Simple Training Monitor - Concise status checker without consuming context
"""
import os
import json
import time
from pathlib import Path
from datetime import datetime

class SimpleMonitor:
    def __init__(self):
        self.results_dir = Path('results/current_experiments/training')

    def get_training_status(self):
        """Get concise training status"""
        status = {
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'detection_models': 0,
            'classification_models': 0,
            'combination_models': 0,
            'completed_models': 0,
            'latest_results': []
        }

        if not self.results_dir.exists():
            return status

        # Count models by category
        detection_dir = self.results_dir / 'detection'
        classification_dir = self.results_dir / 'classification'

        if detection_dir.exists():
            status['detection_models'] = len(list(detection_dir.rglob('best.pt')))

        if classification_dir.exists():
            status['classification_models'] = len(list(classification_dir.rglob('best.pt')))

        # Find recent results (last 10 minutes)
        recent_cutoff = time.time() - 600  # 10 minutes
        recent_models = []

        for weights_file in self.results_dir.rglob('best.pt'):
            if weights_file.stat().st_mtime > recent_cutoff:
                model_name = weights_file.parent.parent.name
                recent_models.append(model_name)

        status['latest_results'] = recent_models[:5]  # Latest 5
        status['combination_models'] = len([m for m in recent_models if '_to_' in m])
        status['completed_models'] = len(recent_models)

        return status

    def check_key_processes(self):
        """Check status of key background processes"""
        key_processes = []

        # Check for ground truth -> efficientnet (our champion)
        gt_efficientnet = self.results_dir / 'classification/pytorch_classification/ground_truth_to_efficientnet'
        if gt_efficientnet.exists():
            key_processes.append("✅ Ground Truth → EfficientNet: COMPLETED")
        else:
            key_processes.append("🔄 Ground Truth → EfficientNet: Training...")

        # Check detection models
        detection_count = len(list(self.results_dir.glob('detection/*/*/weights/best.pt')))
        key_processes.append(f"🎯 Detection Models: {detection_count} completed")

        # Check classification combinations
        combo_count = len([p for p in self.results_dir.rglob('*_to_*/weights/best.pt')])
        key_processes.append(f"🔗 Combination Models: {combo_count} completed")

        return key_processes

    def print_status(self):
        """Print concise status"""
        status = self.get_training_status()
        processes = self.check_key_processes()

        print(f"\n🚀 MALARIA DETECTION - Training Status [{status['timestamp']}]")
        print("=" * 60)

        # Key metrics
        total_models = status['detection_models'] + status['classification_models'] + status['combination_models']
        print(f"📊 Total Models Trained: {total_models}")
        print(f"🎯 Detection: {status['detection_models']} | 🔬 Classification: {status['classification_models']} | 🔗 Combinations: {status['combination_models']}")

        # Recent activity
        if status['latest_results']:
            print(f"\n⏱️  Recently Completed (last 10min):")
            for model in status['latest_results']:
                print(f"   ✅ {model}")
        else:
            print(f"\n⏱️  No models completed in last 10 minutes")

        # Key processes
        print(f"\n🔑 Key Process Status:")
        for process in processes:
            print(f"   {process}")

        print("=" * 60)

if __name__ == "__main__":
    monitor = SimpleMonitor()
    monitor.print_status()