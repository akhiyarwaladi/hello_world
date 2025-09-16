#!/usr/bin/env python3
"""
Malaria Detection Pipeline - Wrapper Script
Provides unified interface to all training scripts
"""

import sys
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Malaria Detection Pipeline")
    parser.add_argument("command", choices=["train"], help="Command to run")
    parser.add_argument("model", choices=["yolov8_detection", "yolov8_classification", "yolov11_detection"],
                       help="Model type to train")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Training device")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--background", action="store_true", help="Run in background")
    parser.add_argument("--data", help="Dataset path")

    args = parser.parse_args()

    if args.command == "train":
        if args.model == "yolov8_detection":
            script = "scripts/07_train_yolo_detection.py"
            data = args.data or "data/detection_fixed/dataset.yaml"
        elif args.model == "yolov8_classification":
            script = "scripts/11_train_classification_crops.py"
            data = args.data or "data/classification_crops"
        elif args.model == "yolov11_detection":
            script = "scripts/08_train_yolo11_detection.py"
            data = args.data or "data/detection_multispecies/dataset.yaml"

        cmd = f"python {script} --data {data} --epochs {args.epochs} --batch {args.batch} --device {args.device} --name {args.name}"

        print(f"Running: {cmd}")
        os.system(cmd)

if __name__ == "__main__":
    main()