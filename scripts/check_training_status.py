#!/usr/bin/env python3
"""
Clear Training Status Checker
Eliminates ambiguity in training status by checking actual files and processes
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime

def check_model_completion(model_path):
    """Check if a model training is completed by looking at files"""
    weights_path = Path(model_path) / "weights"

    if not weights_path.exists():
        return {"status": "not_started", "files": []}

    files = list(weights_path.glob("*.pt"))
    file_info = []

    for f in files:
        stat = f.stat()
        file_info.append({
            "name": f.name,
            "size_mb": round(stat.st_size / (1024*1024), 1),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        })

    # Check if training is complete
    has_best = any("best.pt" in f["name"] for f in file_info)
    has_last = any("last.pt" in f["name"] for f in file_info)

    if has_best and has_last:
        return {"status": "completed", "files": file_info}
    elif file_info:
        return {"status": "in_progress", "files": file_info}
    else:
        return {"status": "not_started", "files": []}

def check_active_processes():
    """Check what training processes are actually running"""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )

        training_processes = []
        for line in result.stdout.split('\n'):
            if 'python' in line and ('train' in line or 'run_' in line):
                if 'grep' not in line:
                    # Extract key info
                    parts = line.split()
                    if len(parts) > 10:
                        cpu_time = parts[9]  # CPU time
                        command = ' '.join(parts[10:])[:100]  # Command (truncated)
                        training_processes.append({
                            "cpu_time": cpu_time,
                            "command": command
                        })

        return training_processes
    except subprocess.CalledProcessError:
        return []

def main():
    print("🔍 CLEAR TRAINING STATUS CHECK")
    print("=" * 50)

    # Check production models
    production_models = [
        ("YOLOv8", "results/completed_models/detection/yolov8_detection/production_full_yolo8_yolo8_det"),
        ("YOLOv11", "results/completed_models/detection/yolo11_detection/production_full_yolo11_yolo11_det"),
        ("YOLOv12", "results/completed_models/detection/yolo12_detection/production_full_yolo12_yolo12_det"),
        ("RT-DETR", "results/completed_models/detection/rtdetr_detection/production_full_rtdetr_rtdetr_det")
    ]

    print("\n📊 PRODUCTION MODEL STATUS:")
    for model_name, model_path in production_models:
        status_info = check_model_completion(model_path)
        status = status_info["status"]

        if status == "completed":
            print(f"✅ {model_name}: COMPLETED")
            best_file = next((f for f in status_info["files"] if "best" in f["name"]), None)
            if best_file:
                print(f"   📁 Best model: {best_file['size_mb']}MB, saved {best_file['modified']}")

        elif status == "in_progress":
            print(f"🔄 {model_name}: IN PROGRESS")
            latest_file = max(status_info["files"], key=lambda x: x["modified"])
            print(f"   📁 Latest: {latest_file['name']}, saved {latest_file['modified']}")

        elif status == "not_started":
            print(f"⭕ {model_name}: NOT STARTED")

        print()

    print("\n🔥 ACTIVE TRAINING PROCESSES:")
    active_processes = check_active_processes()

    if not active_processes:
        print("   No active training processes")
    else:
        for i, proc in enumerate(active_processes, 1):
            print(f"   {i}. CPU Time: {proc['cpu_time']}")
            print(f"      Command: {proc['command']}")
            print()

    print("=" * 50)
    print("✅ Status check complete - No ambiguity!")

if __name__ == "__main__":
    main()