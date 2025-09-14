#!/usr/bin/env python3
"""
Malaria Detection Pipeline Manager dengan Checkpoint System
Menggantikan quick_setup_new_machine.sh dengan sistem Python yang mendukung resume

Author: Pipeline Automation Team
Date: September 2024
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

class PipelineCheckpoint:
    """Manages pipeline stage checkpoints"""

    def __init__(self, checkpoint_file: str = ".pipeline_checkpoint.json"):
        self.checkpoint_file = Path(checkpoint_file)
        self.data = self._load_checkpoint()

    def _load_checkpoint(self) -> Dict:
        """Load existing checkpoint or create new one"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        return {
            "stages": {},
            "last_run": None,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "base"),
            "python_path": sys.executable
        }

    def save_checkpoint(self):
        """Save current checkpoint state"""
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def mark_stage_completed(self, stage_name: str, metadata: Optional[Dict] = None):
        """Mark a stage as completed with optional metadata"""
        self.data["stages"][stage_name] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.save_checkpoint()

    def mark_stage_failed(self, stage_name: str, error_msg: str):
        """Mark a stage as failed"""
        self.data["stages"][stage_name] = {
            "status": "failed",
            "failed_at": datetime.now().isoformat(),
            "error": error_msg
        }
        self.save_checkpoint()

    def mark_stage_running(self, stage_name: str):
        """Mark a stage as currently running"""
        self.data["stages"][stage_name] = {
            "status": "running",
            "started_at": datetime.now().isoformat()
        }
        self.save_checkpoint()

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage is completed"""
        return (stage_name in self.data["stages"] and
                self.data["stages"][stage_name]["status"] == "completed")

    def get_stage_status(self, stage_name: str) -> str:
        """Get stage status"""
        if stage_name not in self.data["stages"]:
            return "not_started"
        return self.data["stages"][stage_name]["status"]

    def get_completed_stages(self) -> List[str]:
        """Get list of completed stages"""
        return [stage for stage, data in self.data["stages"].items()
                if data["status"] == "completed"]

    def reset_stage(self, stage_name: str):
        """Reset a stage (remove from checkpoint)"""
        if stage_name in self.data["stages"]:
            del self.data["stages"][stage_name]
            self.save_checkpoint()


class MalariaPipelineManager:
    """Complete pipeline manager with checkpoint support"""

    def __init__(self):
        self.checkpoint = PipelineCheckpoint()
        self.setup_logging()

        # Pipeline stages definition
        self.stages = [
            {
                "name": "environment_check",
                "description": "Environment & Dependencies Check",
                "function": self.check_environment,
                "skip_if_exists": False
            },
            {
                "name": "dataset_download",
                "description": "Dataset Download",
                "function": self.download_datasets,
                "skip_if_exists": True,
                "check_path": "data/raw/mp_idb"
            },
            {
                "name": "detection_preparation",
                "description": "Detection Dataset Preparation",
                "function": self.prepare_detection_dataset,
                "skip_if_exists": True,
                "check_path": "data/detection_fixed/images"
            },
            {
                "name": "parasite_cropping",
                "description": "Parasite Cropping",
                "function": self.crop_parasites,
                "skip_if_exists": True,
                "check_path": "data/classification_crops/train"
            },
            {
                "name": "training_verification",
                "description": "Training System Verification",
                "function": self.verify_training,
                "skip_if_exists": False
            }
        ]

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def print_status(self, message: str, level: str = "info"):
        """Print colored status messages"""
        colors = {
            "info": "\033[0;34m",     # Blue
            "success": "\033[0;32m",  # Green
            "warning": "\033[1;33m",  # Yellow
            "error": "\033[0;31m",    # Red
            "reset": "\033[0m"        # Reset
        }

        symbols = {
            "info": "ℹ️ ",
            "success": "✅",
            "warning": "⚠️ ",
            "error": "❌"
        }

        color = colors.get(level, colors["info"])
        symbol = symbols.get(level, "")

        print(f"{color}{symbol} {message}{colors['reset']}")

        # Also log to file
        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def run_script(self, script_path: str, description: str,
                   args: List[str] = None, timeout: int = 300) -> Tuple[bool, str]:
        """Run a Python script with error handling and timeout"""
        if not Path(script_path).exists():
            return False, f"Script not found: {script_path}"

        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)

        try:
            self.print_status(f"Running: {description}")
            self.logger.info(f"Executing command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout  # Add timeout support
            )
            
            self.logger.info(f"Script {script_path} completed successfully")
            return True, result.stdout

        except subprocess.TimeoutExpired:
            error_msg = f"Script timeout after {timeout} seconds"
            self.logger.error(error_msg)
            return False, error_msg
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Script failed with return code {e.returncode}\n"
            error_msg += f"Stdout: {e.stdout[-500:]}\n" if e.stdout else ""
            error_msg += f"Stderr: {e.stderr[-500:]}" if e.stderr else ""
            self.logger.error(f"Script {script_path} failed: {error_msg}")
            return False, error_msg

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def check_path_exists(self, path: str, expected_count: int = None) -> bool:
        """Check if path exists with optional file count verification"""
        path_obj = Path(path)
        if not path_obj.exists():
            return False

        if expected_count and path_obj.is_dir():
            file_count = len(list(path_obj.rglob("*")))
            return file_count >= expected_count

        return True

    def check_environment(self) -> bool:
        """Check Python environment and dependencies"""
        self.print_status("Checking Python environment...")

        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.print_status(f"Python version: {python_version}")

        # Check conda environment
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "base")
        self.print_status(f"Conda environment: {conda_env}")

        # Check critical imports
        critical_packages = [
            ("torch", "PyTorch"),
            ("ultralytics", "Ultralytics"),
            ("cv2", "OpenCV"),
            ("pandas", "Pandas"),
            ("numpy", "NumPy")
        ]

        for package, name in critical_packages:
            try:
                __import__(package)
                self.print_status(f"{name}: OK", "success")
            except ImportError:
                self.print_status(f"{name}: Missing!", "error")
                return False

        return True

    def download_datasets(self) -> bool:
        """Download required datasets"""
        self.print_status("Checking dataset requirements...")

        # Check if MP-IDB already exists
        if self.check_path_exists("data/raw/mp_idb"):
            self.print_status("MP-IDB dataset already exists", "success")
            return True

        success, output = self.run_script(
            "scripts/01_download_datasets.py",
            "Dataset Download",
            ["--dataset", "mp_idb"]
        )

        if success:
            self.print_status("Dataset download completed", "success")
        else:
            self.print_status(f"Dataset download failed: {output}", "error")

        return success

    def prepare_detection_dataset(self) -> bool:
        """Prepare detection dataset from MP-IDB"""

        # Check if already exists
        if (self.check_path_exists("data/detection_fixed/images", 100) and
            self.check_path_exists("data/detection_fixed/labels", 100)):
            self.print_status("Detection dataset already prepared", "success")
            return True

        success, output = self.run_script(
            "scripts/08_parse_mpid_detection.py",
            "Detection Dataset Preparation",
            ["--output-path", "data/detection_fixed"]
        )

        if success:
            # Verify results
            image_count = len(list(Path("data/detection_fixed/images").glob("*.jpg")))
            label_count = len(list(Path("data/detection_fixed/labels").glob("*.txt")))

            self.print_status(f"Detection preparation completed: {image_count} images, {label_count} labels", "success")
            return image_count >= 100 and label_count >= 100
        else:
            self.print_status(f"Detection preparation failed: {output}", "error")
            return False

    def crop_parasites(self) -> bool:
        """Crop parasites for classification"""

        # Check if already exists
        if self.check_path_exists("data/classification_crops/train/parasite", 800):
            crop_count = len(list(Path("data/classification_crops").rglob("*.jpg")))
            if crop_count >= 1200:
                self.print_status(f"Parasite cropping already completed: {crop_count} crops", "success")
                return True

        success, output = self.run_script(
            "scripts/09_crop_parasites_from_detection.py",
            "Parasite Cropping",
            ["--detection-path", "data/detection_fixed",
             "--output-path", "data/classification_crops"]
        )

        if success:
            crop_count = len(list(Path("data/classification_crops").rglob("*.jpg")))
            self.print_status(f"Parasite cropping completed: {crop_count} crops", "success")
            return crop_count >= 1200
        else:
            self.print_status(f"Parasite cropping failed: {output}", "error")
            return False

    def verify_training(self) -> bool:
        """Verify training system with quick test"""
        self.print_status("Verifying training system...")

        # Quick detection training test (1 epoch) - longer timeout for training
        success, output = self.run_script(
            "scripts/10_train_yolo_detection.py",
            "Detection Training Test",
            ["--data", "data/detection_fixed/dataset.yaml",
             "--epochs", "1", "--batch", "4", "--device", "cpu",
             "--name", "pipeline_test_detection"],
            timeout=600  # 10 minutes timeout for training
        )

        if not success:
            self.print_status(f"Detection training test failed: {output}", "error")
            return False

        # Quick classification training test (1 epoch)
        success, output = self.run_script(
            "scripts/11_train_classification_crops.py",
            "Classification Training Test",
            ["--data", "data/classification_crops",
             "--epochs", "1", "--batch", "4", "--device", "cpu",
             "--name", "pipeline_test_classification"],
            timeout=600  # 10 minutes timeout for training
        )

        if success:
            self.print_status("Training system verification completed", "success")
        else:
            self.print_status(f"Classification training test failed: {output}", "error")

        return success

    def run_stage(self, stage: Dict) -> bool:
        """Run a single pipeline stage"""
        stage_name = stage["name"]
        description = stage["description"]

        self.print_status(f"\n{'='*60}")
        self.print_status(f"Stage: {description}")
        self.print_status(f"{'='*60}")

        # Check if already completed
        if self.checkpoint.is_stage_completed(stage_name):
            self.print_status(f"Stage '{description}' already completed - skipping", "success")
            return True

        # Check if we can skip based on existing files
        if (stage.get("skip_if_exists", False) and
            "check_path" in stage and
            self.check_path_exists(stage["check_path"])):
            self.print_status(f"Stage '{description}' data exists - marking as completed", "success")
            self.checkpoint.mark_stage_completed(stage_name, {"skipped": True})
            return True

        # Run the stage
        self.checkpoint.mark_stage_running(stage_name)

        try:
            success = stage["function"]()

            if success:
                self.checkpoint.mark_stage_completed(stage_name)
                self.print_status(f"Stage '{description}' completed successfully", "success")
                return True
            else:
                self.checkpoint.mark_stage_failed(stage_name, "Stage function returned False")
                self.print_status(f"Stage '{description}' failed", "error")
                return False

        except Exception as e:
            error_msg = f"Exception in stage: {str(e)}"
            self.checkpoint.mark_stage_failed(stage_name, error_msg)
            self.print_status(f"Stage '{description}' failed with exception: {e}", "error")
            return False

    def show_pipeline_status(self):
        """Show current pipeline status"""
        self.print_status(f"\n{'='*60}")
        self.print_status("🔍 PIPELINE STATUS")
        self.print_status(f"{'='*60}")

        completed_stages = self.checkpoint.get_completed_stages()

        for i, stage in enumerate(self.stages, 1):
            status = self.checkpoint.get_stage_status(stage["name"])

            if status == "completed":
                self.print_status(f"{i}. {stage['description']}: ✅ COMPLETED", "success")
            elif status == "running":
                self.print_status(f"{i}. {stage['description']}: 🔄 RUNNING", "warning")
            elif status == "failed":
                self.print_status(f"{i}. {stage['description']}: ❌ FAILED", "error")
            else:
                self.print_status(f"{i}. {stage['description']}: ⭕ PENDING")

        self.print_status(f"\nCompleted: {len(completed_stages)}/{len(self.stages)} stages")

    def run_pipeline(self, force_restart: bool = False, start_from: str = None):
        """Run the complete pipeline with checkpoint support"""

        if force_restart:
            self.print_status("Force restart requested - clearing all checkpoints", "warning")
            self.checkpoint.data["stages"] = {}
            self.checkpoint.save_checkpoint()

        self.print_status("🚀 MALARIA DETECTION PIPELINE MANAGER")
        self.print_status("Using conda environment: " + os.environ.get("CONDA_DEFAULT_ENV", "base"))

        # Show current status
        self.show_pipeline_status()

        # Find starting point
        start_index = 0
        if start_from:
            for i, stage in enumerate(self.stages):
                if stage["name"] == start_from:
                    start_index = i
                    break
            self.print_status(f"Starting from stage: {start_from}")

        # Run stages
        for stage in self.stages[start_index:]:
            if not self.run_stage(stage):
                self.print_status("❌ Pipeline stopped due to stage failure", "error")
                return False

        # Success summary
        self.print_status(f"\n{'='*60}")
        self.print_status("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        self.print_status(f"{'='*60}")

        self.print_status("✨ All stages completed:")
        for stage in self.stages:
            self.print_status(f"  ✅ {stage['description']}")

        self.print_status("\n🚀 Ready for production training:")
        self.print_status("  • Detection: python scripts/10_train_yolo_detection.py --epochs 30")
        self.print_status("  • Classification: python scripts/11_train_classification_crops.py --epochs 25")

        return True


def main():
    """Main pipeline execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Malaria Detection Pipeline Manager")
    parser.add_argument("--force-restart", action="store_true",
                       help="Force restart pipeline from beginning")
    parser.add_argument("--start-from", type=str,
                       help="Start from specific stage")
    parser.add_argument("--status", action="store_true",
                       help="Show pipeline status only")
    parser.add_argument("--reset-stage", type=str,
                       help="Reset specific stage")

    args = parser.parse_args()

    pipeline = MalariaPipelineManager()

    if args.status:
        pipeline.show_pipeline_status()
        return

    if args.reset_stage:
        pipeline.checkpoint.reset_stage(args.reset_stage)
        pipeline.print_status(f"Reset stage: {args.reset_stage}", "success")
        return

    # Run pipeline
    success = pipeline.run_pipeline(
        force_restart=args.force_restart,
        start_from=args.start_from
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()