#!/usr/bin/env python3
"""
Monitor Training Progress
Real-time monitoring untuk semua training yang sedang berjalan
"""

import os
import time
import subprocess
from pathlib import Path
from datetime import datetime

def check_training_status():
    """Check status semua model yang sedang training"""

    print("\n" + "="*60)
    print(f"🔍 TRAINING MONITOR - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)

    # Check running processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout

        training_processes = []
        for line in processes.split('\n'):
            if 'python' in line and ('train' in line or 'pipeline.py' in line):
                training_processes.append(line)

        if training_processes:
            print(f"🚀 Active Training Processes: {len(training_processes)}")
            for i, proc in enumerate(training_processes[:5], 1):
                parts = proc.split()
                if len(parts) > 10:
                    cpu = parts[2]
                    mem = parts[3]
                    command = ' '.join(parts[10:])
                    print(f"   {i}. CPU: {cpu}% | MEM: {mem}% | {command[:80]}...")
        else:
            print("✅ No active training processes found")

    except Exception as e:
        print(f"❌ Error checking processes: {e}")

    # Check results directories
    print("\n📊 RESULTS OVERVIEW:")
    results_path = Path("results/current_experiments/training")

    if results_path.exists():
        detection_models = list(results_path.glob("detection/*/"))
        classification_models = list(results_path.glob("classification/*/"))

        print(f"   🎯 Detection Models: {len(detection_models)} experiments")
        for model_dir in detection_models:
            if (model_dir / "weights" / "best.pt").exists():
                print(f"      ✅ {model_dir.name} - Training completed")
            elif list(model_dir.glob("*.log")):
                print(f"      🔄 {model_dir.name} - In progress")
            else:
                print(f"      ⏳ {model_dir.name} - Starting...")

        print(f"   🏷️  Classification Models: {len(classification_models)} experiments")
        for model_dir in classification_models:
            if (model_dir / "weights" / "best.pt").exists():
                print(f"      ✅ {model_dir.name} - Training completed")
            elif list(model_dir.glob("*.log")):
                print(f"      🔄 {model_dir.name} - In progress")
            else:
                print(f"      ⏳ {model_dir.name} - Starting...")
    else:
        print("   📁 Results directory not found")

    # Check latest logs
    print("\n📜 LATEST TRAINING LOGS:")
    logs_path = Path("results/experiment_logs")
    if logs_path.exists():
        log_files = sorted(logs_path.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        for log_file in log_files[:3]:
            print(f"   📄 {log_file.name}")
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        print(f"      Last: {last_line[:100]}...")
            except:
                print(f"      Unable to read log")

    print("\n" + "="*60)

def main():
    """Main monitoring loop"""
    print("🔍 MALARIA DETECTION TRAINING MONITOR")
    print("Press Ctrl+C to stop monitoring")

    try:
        while True:
            check_training_status()
            print("\n⏰ Next update in 30 seconds...")
            time.sleep(30)

    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

if __name__ == "__main__":
    main()
