#!/usr/bin/env python3
"""
Enhanced Malaria Detection Pipeline
Clean and organized pipeline with descriptive script names
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class MalariaPipeline:
    """Enhanced Malaria Detection Pipeline with organized outputs"""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"pipeline_run_{self.timestamp}"
        self.run_dir = self.base_dir / "pipeline_runs" / self.run_id

        # Create run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline stages with new clean script names
        self.stages = {
            1: {
                "name": "Dataset Download",
                "script": "download_datasets.py",
                "description": "Download all required datasets"
            },
            2: {
                "name": "Image Preprocessing",
                "script": "preprocess_images.py",
                "description": "Preprocess and enhance image quality"
            },
            3: {
                "name": "Dataset Integration",
                "script": "integrate_datasets.py",
                "description": "Integrate multiple datasets into unified format"
            },
            4: {
                "name": "YOLO Format Conversion",
                "script": "convert_to_yolo.py",
                "description": "Convert annotations to YOLO format"
            },
            5: {
                "name": "Data Augmentation",
                "script": "augment_data.py",
                "description": "Generate augmented training data"
            },
            6: {
                "name": "Dataset Splitting",
                "script": "split_dataset.py",
                "description": "Split data into train/val/test sets"
            },
            7: {
                "name": "Parasite Cropping",
                "script": "crop_detections.py",
                "description": "Create classification dataset from detections"
            },
            8: {
                "name": "YOLOv8 Detection Training",
                "script": "train_yolo_detection.py",
                "description": "Train YOLOv8 parasite detection model"
            },
            9: {
                "name": "Classification Training",
                "script": "11_train_classification_crops.py",  # Will rename when process stops
                "description": "Train parasite classification model"
            }
        }

        self.checkpoint_file = self.run_dir / "checkpoint.json"
        self.log_file = self.run_dir / "pipeline.log"

    def print_status(self, message: str, status: str = "info"):
        """Print formatted status message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "info": "\033[96m",      # Cyan
            "success": "\033[92m",   # Green
            "warning": "\033[93m",   # Yellow
            "error": "\033[91m",     # Red
            "reset": "\033[0m"       # Reset
        }

        color = colors.get(status, colors["info"])
        reset = colors["reset"]

        formatted_message = f"{color}[{timestamp}] {message}{reset}"
        print(formatted_message)

        # Also log to file
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] [{status.upper()}] {message}\n")

    def save_checkpoint(self, stage: int, status: str, details: Dict[str, Any] = None):
        """Save pipeline checkpoint"""
        checkpoint_data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "current_stage": stage,
            "stage_status": status,
            "stage_name": self.stages[stage]["name"],
            "completed_stages": [i for i in range(1, stage) if i <= stage-1],
            "details": details or {}
        }

        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load existing checkpoint"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.print_status(f"Error loading checkpoint: {e}", "error")
        return None

    def run_stage(self, stage_num: int, args: List[str] = None) -> bool:
        """Run a specific pipeline stage"""
        stage = self.stages[stage_num]
        script_name = stage["script"]

        self.print_status(f"Running Stage {stage_num}: {stage['name']}")
        self.print_status(f"Description: {stage['description']}")

        # Build command - scripts are in current directory (copied from numbered versions)
        cmd = [sys.executable, script_name]
        if args:
            cmd.extend(args)

        # Special handling for stages that need output directories
        if stage_num == 8:  # YOLOv8 Detection Training
            results_dir = f"detection_training_{self.timestamp}"
            cmd.extend(["--name", results_dir])
        elif stage_num == 9:  # Classification Training
            results_dir = f"classification_training_{self.timestamp}"
            cmd.extend(["--name", results_dir])

        try:
            self.save_checkpoint(stage_num, "running")

            # Run the script
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0:
                self.print_status(f"✅ Stage {stage_num} completed successfully", "success")
                self.save_checkpoint(stage_num, "completed", {"output": result.stdout})
                return True
            else:
                self.print_status(f"❌ Stage {stage_num} failed", "error")
                self.print_status(f"Error: {result.stderr}", "error")
                self.save_checkpoint(stage_num, "failed", {"error": result.stderr})
                return False

        except subprocess.TimeoutExpired:
            self.print_status(f"⏱️ Stage {stage_num} timed out", "error")
            self.save_checkpoint(stage_num, "timeout")
            return False
        except Exception as e:
            self.print_status(f"❌ Stage {stage_num} error: {e}", "error")
            self.save_checkpoint(stage_num, "error", {"error": str(e)})
            return False

    def run_pipeline(self, start_stage: int = 1, end_stage: int = 9):
        """Run the complete pipeline"""
        self.print_status("🚀 Starting Malaria Detection Pipeline")
        self.print_status(f"Run ID: {self.run_id}")
        self.print_status(f"Output Directory: {self.run_dir}")

        success_count = 0
        total_stages = end_stage - start_stage + 1

        for stage_num in range(start_stage, end_stage + 1):
            if not self.run_stage(stage_num):
                self.print_status(f"Pipeline stopped at stage {stage_num}", "error")
                break
            success_count += 1

        # Final summary
        self.print_status("\n" + "="*50)
        if success_count == total_stages:
            self.print_status("🎉 Pipeline completed successfully!", "success")
        else:
            self.print_status(f"⚠️ Pipeline completed {success_count}/{total_stages} stages", "warning")

        self.print_status(f"📁 Results saved in: {self.run_dir}")
        self.print_status("="*50)


def main():
    parser = argparse.ArgumentParser(description="Enhanced Malaria Detection Pipeline")
    parser.add_argument("--start-stage", type=int, default=1,
                       help="Starting stage (1-9)")
    parser.add_argument("--end-stage", type=int, default=9,
                       help="Ending stage (1-9)")
    parser.add_argument("--stage", type=int,
                       help="Run single stage")

    args = parser.parse_args()

    pipeline = MalariaPipeline()

    if args.stage:
        # Run single stage
        pipeline.run_stage(args.stage)
    else:
        # Run pipeline range
        pipeline.run_pipeline(args.start_stage, args.end_stage)


if __name__ == "__main__":
    main()