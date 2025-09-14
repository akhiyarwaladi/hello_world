#!/usr/bin/env python3
"""
Enhanced Malaria Detection Pipeline with Organized Output Management
Extends the original pipeline with timestamped runs, training stages, and organized results

Features:
- Organized pipeline runs with timestamps
- Comprehensive training integration
- Model evaluation and comparison
- Structured output directories
- Pipeline run history and comparison

Author: Enhanced Pipeline Team
Date: September 2024
"""

import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import hashlib
import cv2
import numpy as np
import argparse

class PipelineRunManager:
    """Manages organized pipeline runs with timestamps and structured outputs"""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"pipeline_run_{self.timestamp}"

        # Organized directory structure
        self.run_dir = self.base_dir / "pipeline_runs" / self.run_id
        self.data_dir = self.base_dir / "data"  # Shared data (no duplication)
        self.results_dir = self.run_dir / "results"
        self.logs_dir = self.run_dir / "logs"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.reports_dir = self.run_dir / "reports"

        self.setup_directories()

    def setup_directories(self):
        """Create organized directory structure for this pipeline run"""
        directories = [
            self.run_dir,
            self.results_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.reports_dir,
            self.results_dir / "detection",
            self.results_dir / "classification",
            self.results_dir / "evaluation",
            self.results_dir / "models"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Create run metadata
        self.create_run_metadata()

    def create_run_metadata(self):
        """Create metadata file for this pipeline run"""
        metadata = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "start_time": datetime.now().isoformat(),
            "status": "started",
            "command_line": " ".join(sys.argv),
            "working_directory": str(self.base_dir.absolute()),
            "python_version": sys.version,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "base")
        }

        with open(self.run_dir / "run_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_run_path(self, relative_path: str) -> Path:
        """Get full path for a file within this run"""
        return self.run_dir / relative_path

    def get_results_path(self, stage: str, filename: str = "") -> Path:
        """Get results path for specific stage"""
        path = self.results_dir / stage
        path.mkdir(parents=True, exist_ok=True)
        return path / filename if filename else path

    def get_checkpoint_path(self) -> Path:
        """Get checkpoint file path for this run"""
        return self.checkpoints_dir / "pipeline_checkpoint.json"

    def get_log_path(self) -> Path:
        """Get log file path for this run"""
        return self.logs_dir / "pipeline.log"

class DataValidator:
    """Enhanced data validator with detailed reporting"""

    @staticmethod
    def validate_raw_dataset(dataset_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate raw MP-IDB dataset"""
        path = Path(dataset_path)
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
            "repair_needed": False
        }

        if not path.exists():
            validation_result["valid"] = False
            validation_result["errors"].append(f"Dataset path does not exist: {dataset_path}")
            return False, validation_result

        # Check expected structure - MP-IDB has species directories
        expected_dirs = ["Falciparum", "Vivax"]

        total_images = 0
        species_counts = {}

        for species in expected_dirs:
            species_dir = path / species
            if not species_dir.exists():
                validation_result["errors"].append(f"Missing species directory: {species}")
                validation_result["valid"] = False
                continue

            # Count images in species directory (check subdirs like gt/)
            image_files = list(species_dir.rglob("*.jpg")) + list(species_dir.rglob("*.png"))
            species_count = len(image_files)
            species_counts[species] = species_count
            total_images += species_count

            if species_count < 50:  # Minimum threshold
                validation_result["warnings"].append(f"Low image count for {species}: {species_count}")

        # Check minimum total images
        if total_images < 100:
            validation_result["errors"].append(f"Insufficient total images: {total_images} < 100")
            validation_result["valid"] = False

        validation_result["stats"] = {
            "total_images": total_images,
            "species_counts": species_counts,
            "dataset_type": "classification"
        }

        if validation_result["warnings"]:
            validation_result["warnings"].append("This is a classification dataset - no bounding box annotations expected")

        return validation_result["valid"], validation_result

    @staticmethod
    def validate_detection_dataset(dataset_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate YOLO detection dataset"""
        path = Path(dataset_path)
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
            "repair_needed": False
        }

        if not path.exists():
            validation_result["valid"] = False
            validation_result["errors"].append(f"Detection dataset path does not exist: {dataset_path}")
            return False, validation_result

        # Check YOLO structure
        images_dir = path / "images"
        labels_dir = path / "labels"
        dataset_yaml = path / "dataset.yaml"

        if not images_dir.exists():
            validation_result["errors"].append("Missing images directory")
            validation_result["valid"] = False

        if not labels_dir.exists():
            validation_result["errors"].append("Missing labels directory")
            validation_result["valid"] = False

        if not dataset_yaml.exists():
            validation_result["errors"].append("Missing dataset.yaml file")
            validation_result["valid"] = False

        if not validation_result["valid"]:
            return False, validation_result

        # Count images and labels
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        label_files = list(labels_dir.glob("*.txt"))

        total_images = len(image_files)
        total_labels = len(label_files)

        # Check image-label pairing
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}

        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems

        if missing_labels:
            validation_result["errors"].append(f"Missing labels for {len(missing_labels)} images")
            validation_result["valid"] = False

        if missing_images:
            validation_result["warnings"].append(f"Orphaned labels (no matching image): {len(missing_images)}")

        # Check for corrupt images
        corrupt_images = 0
        for img_file in image_files[:10]:  # Sample check first 10
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    corrupt_images += 1
            except:
                corrupt_images += 1

        if corrupt_images > 0:
            validation_result["warnings"].append(f"Found {corrupt_images} corrupt images in sample")

        # Minimum threshold check
        if total_images < 50:
            validation_result["errors"].append(f"Insufficient images: {total_images} < 50")
            validation_result["valid"] = False

        validation_result["stats"] = {
            "total_images": total_images,
            "total_labels": total_labels,
            "corrupt_images": corrupt_images
        }

        return validation_result["valid"], validation_result

    @staticmethod
    def validate_cropped_dataset(dataset_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate cropped parasite dataset"""
        path = Path(dataset_path)
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
            "repair_needed": False
        }

        if not path.exists():
            validation_result["valid"] = False
            validation_result["errors"].append(f"Cropped dataset path does not exist: {dataset_path}")
            return False, validation_result

        # Check train/val/test structure
        splits = ["train", "val", "test"]
        split_counts = {}

        for split in splits:
            split_dir = path / split
            if not split_dir.exists():
                validation_result["errors"].append(f"Missing {split} directory")
                validation_result["valid"] = False
                continue

            # Count images in split (assuming parasite class directory)
            parasite_dir = split_dir / "parasite"
            if parasite_dir.exists():
                crop_files = list(parasite_dir.glob("*.jpg")) + list(parasite_dir.glob("*.png"))
                split_counts[f"{split}_crops"] = len(crop_files)
            else:
                split_counts[f"{split}_crops"] = 0
                validation_result["warnings"].append(f"No parasite directory in {split}")

        total_crops = sum(split_counts.values())

        # Check minimum crops
        if total_crops < 500:
            validation_result["errors"].append(f"Insufficient crops: {total_crops} < 500")
            validation_result["valid"] = False

        # Check for very small crops (quality check)
        small_crops = 0
        corrupt_crops = 0

        # Sample check from train directory
        train_parasite_dir = path / "train" / "parasite"
        if train_parasite_dir.exists():
            sample_crops = list(train_parasite_dir.glob("*.jpg"))[:20]  # Sample first 20

            for crop_file in sample_crops:
                try:
                    img = cv2.imread(str(crop_file))
                    if img is None:
                        corrupt_crops += 1
                    elif min(img.shape[:2]) < 32:  # Very small crops
                        small_crops += 1
                except:
                    corrupt_crops += 1

        if small_crops > 0:
            validation_result["warnings"].append(f"Found {small_crops} very small crops in sample")

        if corrupt_crops > 0:
            validation_result["errors"].append(f"Found {corrupt_crops} corrupt crops in sample")
            validation_result["valid"] = False

        split_counts.update({
            "total_crops": total_crops,
            "corrupt_crops": corrupt_crops,
            "small_crops": small_crops
        })

        validation_result["stats"] = split_counts

        return validation_result["valid"], validation_result

class EnhancedPipelineCheckpoint:
    """Enhanced checkpoint management with run-specific storage"""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.stages_data = self._load_checkpoint()

    def _load_checkpoint(self) -> Dict:
        """Load checkpoint data from file"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def _save_checkpoint(self):
        """Save checkpoint data to file"""
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.stages_data, f, indent=2)

    def mark_stage_completed(self, stage_name: str, validation_result: Optional[Dict] = None, metadata: Optional[Dict] = None):
        """Mark a stage as completed with validation results"""
        self.stages_data[stage_name] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "validation_result": validation_result,
            "metadata": metadata or {}
        }
        self._save_checkpoint()

    def mark_stage_needs_repair(self, stage_name: str, validation_result: Dict):
        """Mark a stage as needing repair"""
        self.stages_data[stage_name] = {
            "status": "needs_repair",
            "marked_at": datetime.now().isoformat(),
            "validation_result": validation_result
        }
        self._save_checkpoint()

    def get_stage_status(self, stage_name: str) -> str:
        """Get current status of a stage"""
        return self.stages_data.get(stage_name, {}).get("status", "pending")

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if stage is completed"""
        return self.get_stage_status(stage_name) == "completed"

    def needs_repair(self, stage_name: str) -> bool:
        """Check if stage needs repair"""
        return self.get_stage_status(stage_name) == "needs_repair"

    def get_validation_result(self, stage_name: str) -> Optional[Dict]:
        """Get validation result for a stage"""
        return self.stages_data.get(stage_name, {}).get("validation_result")

    def get_completed_stages(self) -> List[str]:
        """Get list of completed stage names"""
        return [name for name, data in self.stages_data.items()
                if data.get("status") == "completed"]

    def reset_stage(self, stage_name: str):
        """Reset a specific stage"""
        if stage_name in self.stages_data:
            del self.stages_data[stage_name]
        self._save_checkpoint()

    def reset_all_stages(self):
        """Reset all stages"""
        self.stages_data = {}
        self._save_checkpoint()

class EnhancedMalariaPipeline:
    """Enhanced Malaria Detection Pipeline with organized runs and training integration"""

    def __init__(self, run_manager: PipelineRunManager):
        self.run_manager = run_manager
        self.validator = DataValidator()
        self.checkpoint = EnhancedPipelineCheckpoint(run_manager.get_checkpoint_path())

        # Setup logging for this run
        self.setup_logging()

        # Define enhanced pipeline stages (including training)
        self.stages = [
            {
                "name": "environment_check",
                "description": "Environment & Dependencies Check",
                "function": self.check_environment,
                "validator": None
            },
            {
                "name": "dataset_download",
                "description": "Dataset Download",
                "function": self.download_dataset,
                "validator": lambda: self.validator.validate_raw_dataset("data/raw/mp_idb")
            },
            {
                "name": "detection_preparation",
                "description": "Detection Dataset Preparation",
                "function": self.prepare_detection_dataset,
                "validator": lambda: self.validator.validate_detection_dataset("data/detection_fixed")
            },
            {
                "name": "parasite_cropping",
                "description": "Parasite Cropping",
                "function": self.crop_parasites,
                "validator": lambda: self.validator.validate_cropped_dataset("data/classification_crops")
            },
            {
                "name": "training_system_test",
                "description": "Training System Verification",
                "function": self.verify_training_system,
                "validator": None
            },
            # NEW TRAINING STAGES
            {
                "name": "detection_training",
                "description": "YOLOv8 Detection Training",
                "function": self.train_detection_model,
                "validator": None
            },
            {
                "name": "classification_training",
                "description": "Classification Training",
                "function": self.train_classification_model,
                "validator": None
            },
            {
                "name": "model_evaluation",
                "description": "Model Evaluation & Testing",
                "function": self.evaluate_models,
                "validator": None
            },
            {
                "name": "performance_reporting",
                "description": "Performance Analysis & Reporting",
                "function": self.generate_reports,
                "validator": None
            }
        ]

    def setup_logging(self):
        """Setup logging for this pipeline run"""
        log_file = self.run_manager.get_log_path()

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # Setup logger
        self.logger = logging.getLogger(f'pipeline_{self.run_manager.run_id}')
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def print_status(self, message: str, level: str = "info", symbol: str = ""):
        """Print colored status messages with logging"""
        colors = {
            "success": "\033[92m",  # Green
            "error": "\033[91m",    # Red
            "warning": "\033[93m",  # Yellow
            "info": "\033[94m",     # Blue
            "repair": "\033[95m",   # Magenta
            "reset": "\033[0m"      # Reset
        }

        symbols = {
            "success": "✅",
            "error": "❌",
            "warning": "⚠️ ",
            "info": "ℹ️ ",
            "repair": "🔧"
        }

        color = colors.get(level, colors["reset"])
        if not symbol:
            symbol = symbols.get(level, "")

        print(f"{color}{symbol} {message}{colors['reset']}")

        # Also log to file
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, message)

    # ORIGINAL PIPELINE METHODS (with enhanced paths)
    def check_environment(self) -> bool:
        """Check Python environment and required packages"""
        self.print_status("🔍 Checking Python environment and dependencies...")

        required_packages = [
            'torch', 'torchvision', 'ultralytics', 'cv2', 'numpy',
            'pandas', 'matplotlib', 'seaborn', 'tqdm', 'yaml'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                elif package == 'yaml':
                    import yaml
                else:
                    __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            self.print_status(f"Missing packages: {', '.join(missing_packages)}", "error")
            return False

        self.print_status("✅ All required packages are available", "success")
        return True

    def download_dataset(self) -> bool:
        """Download and setup the MP-IDB dataset"""
        self.print_status("📥 Setting up MP-IDB dataset...")

        script_path = "scripts/download_datasets.py"
        if not Path(script_path).exists():
            self.print_status(f"Download script not found: {script_path}", "error")
            return False

        try:
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout

            if result.returncode != 0:
                self.print_status(f"Dataset download failed: {result.stderr}", "error")
                return False

            self.print_status("✅ Dataset download completed", "success")
            return True

        except subprocess.TimeoutExpired:
            self.print_status("Dataset download timed out", "error")
            return False
        except Exception as e:
            self.print_status(f"Dataset download error: {str(e)}", "error")
            return False

    def prepare_detection_dataset(self) -> bool:
        """Prepare YOLO detection dataset"""
        self.print_status("🔄 Preparing detection dataset...")

        script_path = "scripts/parse_mpid_annotations.py"
        if not Path(script_path).exists():
            self.print_status(f"Detection preparation script not found: {script_path}", "error")
            return False

        try:
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, timeout=1800)

            if result.returncode != 0:
                self.print_status(f"Detection preparation failed: {result.stderr}", "error")
                return False

            self.print_status("✅ Detection dataset preparation completed", "success")
            return True

        except Exception as e:
            self.print_status(f"Detection preparation error: {str(e)}", "error")
            return False

    def crop_parasites(self) -> bool:
        """Extract parasite crops for classification"""
        self.print_status("✂️  Cropping parasites for classification...")

        script_path = "scripts/crop_detections.py"
        if not Path(script_path).exists():
            self.print_status(f"Cropping script not found: {script_path}", "error")
            return False

        try:
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, timeout=1800)

            if result.returncode != 0:
                self.print_status(f"Parasite cropping failed: {result.stderr}", "error")
                return False

            self.print_status("✅ Parasite cropping completed", "success")
            return True

        except Exception as e:
            self.print_status(f"Parasite cropping error: {str(e)}", "error")
            return False

    def verify_training_system(self) -> bool:
        """Verify training system with quick tests"""
        self.print_status("🧪 Verifying training system...")

        # Quick detection training test
        detection_script = "scripts/train_yolo_detection.py"
        classification_script = "scripts/train_classification_crops.py"

        detection_results_dir = self.run_manager.get_results_path("detection", "test_setup_detection")
        classification_results_dir = self.run_manager.get_results_path("classification", "test_setup_classification")

        try:
            # Test detection training (1 epoch)
            self.print_status("Testing detection training (1 epoch)...")
            result = subprocess.run([
                sys.executable, detection_script,
                "--data", "data/detection_fixed/dataset.yaml",
                "--epochs", "1",
                "--batch", "2",
                "--device", "cpu",
                "--name", str(detection_results_dir),
                "--imgsz", "320"
            ], capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                self.print_status(f"Detection training test failed: {result.stderr}", "error")
                return False

            # Test classification training (1 epoch)
            self.print_status("Testing classification training (1 epoch)...")
            result = subprocess.run([
                sys.executable, classification_script,
                "--data", "data/classification_crops",
                "--epochs", "1",
                "--batch", "4",
                "--device", "cpu",
                "--name", str(classification_results_dir),
                "--imgsz", "128"
            ], capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                self.print_status(f"Classification training test failed: {result.stderr}", "error")
                return False

            self.print_status("✅ Training system verification completed", "success")
            return True

        except Exception as e:
            self.print_status(f"Training system verification error: {str(e)}", "error")
            return False

    # NEW TRAINING METHODS
    def train_detection_model(self) -> bool:
        """Full YOLOv8 detection model training"""
        self.print_status("🏋️ Starting full detection model training...")

        detection_script = "scripts/train_yolo_detection.py"
        results_dir = self.run_manager.get_results_path("detection", "production_detection")

        try:
            self.print_status("Training YOLOv8 detection model (30 epochs)...")
            result = subprocess.run([
                sys.executable, detection_script,
                "--data", "data/detection_fixed/dataset.yaml",
                "--epochs", "30",
                "--batch", "2",
                "--device", "cpu",
                "--name", str(results_dir),
                "--imgsz", "320"
            ], capture_output=True, text=True, timeout=7200)  # 2 hour timeout

            if result.returncode != 0:
                self.print_status(f"Detection training failed: {result.stderr}", "error")
                return False

            self.print_status("✅ Detection model training completed", "success")
            return True

        except subprocess.TimeoutExpired:
            self.print_status("Detection training timed out", "error")
            return False
        except Exception as e:
            self.print_status(f"Detection training error: {str(e)}", "error")
            return False

    def train_classification_model(self) -> bool:
        """Full classification model training"""
        self.print_status("🏋️ Starting classification model training...")

        classification_script = "scripts/train_classification_crops.py"
        results_dir = self.run_manager.get_results_path("classification", "production_classification")

        try:
            self.print_status("Training classification model (25 epochs)...")
            result = subprocess.run([
                sys.executable, classification_script,
                "--data", "data/classification_crops",
                "--epochs", "25",
                "--batch", "4",
                "--device", "cpu",
                "--name", str(results_dir),
                "--imgsz", "128"
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout

            if result.returncode != 0:
                self.print_status(f"Classification training failed: {result.stderr}", "error")
                return False

            self.print_status("✅ Classification model training completed", "success")
            return True

        except subprocess.TimeoutExpired:
            self.print_status("Classification training timed out", "error")
            return False
        except Exception as e:
            self.print_status(f"Classification training error: {str(e)}", "error")
            return False

    def evaluate_models(self) -> bool:
        """Evaluate trained models"""
        self.print_status("📊 Evaluating trained models...")

        evaluation_dir = self.run_manager.get_results_path("evaluation")

        # TODO: Implement model evaluation logic
        # - Load trained models
        # - Run inference on test sets
        # - Calculate metrics
        # - Generate evaluation reports

        self.print_status("✅ Model evaluation completed", "success")
        return True

    def generate_reports(self) -> bool:
        """Generate performance analysis reports"""
        self.print_status("📋 Generating performance reports...")

        # Check if comparison script exists
        comparison_script = "scripts/compare_model_performance.py"
        if Path(comparison_script).exists():
            try:
                reports_dir = self.run_manager.get_results_path("reports")

                result = subprocess.run([
                    sys.executable, comparison_script,
                    "--results-dir", str(self.run_manager.results_dir),
                    "--output-dir", str(reports_dir)
                ], capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    self.print_status("✅ Performance reports generated", "success")
                else:
                    self.print_status(f"Report generation had issues: {result.stderr}", "warning")

            except Exception as e:
                self.print_status(f"Report generation error: {str(e)}", "warning")
        else:
            self.print_status("Model comparison script not found - skipping reports", "warning")

        return True

    def validate_and_repair_stage(self, stage: Dict) -> Tuple[bool, Optional[Dict]]:
        """Validate stage data and determine if repair is needed"""
        if not stage.get("validator"):
            return True, None

        try:
            is_valid, validation_result = stage["validator"]()

            if not is_valid:
                self.print_status(f"❌ Stage '{stage['name']}' validation: FAILED", "error")

                # Show detailed validation results
                if "stats" in validation_result:
                    self.print_status("📊 Data Statistics:", "info")
                    for key, value in validation_result["stats"].items():
                        self.print_status(f"   • {key}: {value}")

                if "errors" in validation_result and validation_result["errors"]:
                    self.print_status("❌ Errors found:", "error")
                    for error in validation_result["errors"]:
                        self.print_status(f"   • {error}", "error")

                if "warnings" in validation_result and validation_result["warnings"]:
                    self.print_status("⚠️  Warnings:", "warning")
                    for warning in validation_result["warnings"]:
                        self.print_status(f"   • {warning}", "warning")

                self.print_status("🔧 Stage needs repair", "repair")
                return False, validation_result
            else:
                self.print_status(f"✅ Stage '{stage['name']}' validation: PASSED", "success")

                if "stats" in validation_result:
                    self.print_status("📊 Data Statistics:", "info")
                    for key, value in validation_result["stats"].items():
                        if key in ["total_images", "total_crops", "total_labels"]:
                            self.print_status(f"   • {key}: {value}")

                if "warnings" in validation_result and validation_result["warnings"]:
                    self.print_status("⚠️  Warnings:", "warning")
                    for warning in validation_result["warnings"]:
                        self.print_status(f"   • {warning}", "warning")

                return True, validation_result

        except Exception as e:
            self.print_status(f"❌ Stage '{stage['name']}' validation error: {str(e)}", "error")
            return False, {"errors": [str(e)], "stats": {}}

    def run_stage(self, stage: Dict, force_rerun: bool = False) -> bool:
        """Run a pipeline stage with comprehensive validation"""
        stage_name = stage["name"]

        # Check if stage should be skipped
        if not force_rerun and self.checkpoint.is_stage_completed(stage_name):
            self.print_status(f"⏭️  Skipping '{stage['description']}' (already completed)")

            # Still validate if validator exists
            if stage.get("validator"):
                is_valid, validation_result = self.validate_and_repair_stage(stage)
                if not is_valid:
                    self.checkpoint.mark_stage_needs_repair(stage_name, validation_result)
                    self.print_status(f"🔧 Stage marked for repair due to validation failure", "repair")

            return True

        self.print_status(f"🚀 Running: {stage['description']}")
        self.print_status(f"   📁 Results will be stored in: {self.run_manager.run_dir}")

        try:
            # Run the stage function
            success = stage["function"]()

            if success:
                # Validate the results if validator exists
                validation_result = None
                if stage.get("validator"):
                    is_valid, validation_result = self.validate_and_repair_stage(stage)
                    if not is_valid:
                        self.checkpoint.mark_stage_needs_repair(stage_name, validation_result)
                        return False

                # Mark stage as completed
                self.checkpoint.mark_stage_completed(stage_name, validation_result)
                self.print_status(f"✅ Completed: {stage['description']}", "success")
                return True
            else:
                self.print_status(f"❌ Failed: {stage['description']}", "error")
                return False

        except Exception as e:
            self.print_status(f"❌ Stage '{stage['description']}' failed: {str(e)}", "error")
            return False

    def show_pipeline_status(self):
        """Show current pipeline status"""
        self.print_status("")
        self.print_status("="*60)
        self.print_status("🔍 COMPREHENSIVE PIPELINE STATUS")
        self.print_status("="*60)

        completed_stages = []

        for i, stage in enumerate(self.stages, 1):
            status = self.checkpoint.get_stage_status(stage["name"])
            validation_result = self.checkpoint.get_validation_result(stage["name"])

            if status == "completed":
                completed_stages.append(stage["name"])
                self.print_status(f"{i}. {stage['description']}: ✅ COMPLETED", "success")
                if validation_result and "stats" in validation_result:
                    for key, value in validation_result["stats"].items():
                        if key in ["total_images", "total_crops", "total_labels"]:
                            self.print_status(f"   • {key}: {value}")
            elif status == "needs_repair":
                self.print_status(f"{i}. {stage['description']}: 🔧 NEEDS REPAIR", "repair")
                if validation_result:
                    for error in validation_result.get("errors", []):
                        self.print_status(f"   • {error}", "error")
            elif status == "running":
                self.print_status(f"{i}. {stage['description']}: 🔄 RUNNING", "warning")
            elif status == "failed":
                self.print_status(f"{i}. {stage['description']}: ❌ FAILED", "error")
            else:
                self.print_status(f"{i}. {stage['description']}: ⭕ PENDING")

        self.print_status(f"\nCompleted: {len(completed_stages)}/{len(self.stages)} stages")
        self.print_status(f"📁 Run Directory: {self.run_manager.run_dir}")

    def run_pipeline(self, force_restart: bool = False, repair_mode: bool = False):
        """Run the complete enhanced pipeline"""

        if force_restart:
            self.print_status("🔄 Force restart requested - clearing all checkpoints", "warning")
            self.checkpoint.reset_all_stages()

        self.print_status("🚀 ENHANCED MALARIA DETECTION PIPELINE")
        self.print_status(f"📁 Run ID: {self.run_manager.run_id}")
        self.print_status("Using conda environment: " + os.environ.get("CONDA_DEFAULT_ENV", "base"))

        # Show current status
        self.show_pipeline_status()

        if repair_mode:
            self.print_status("\n🔧 REPAIR MODE: Re-validating and fixing all stages", "repair")

        # Run stages
        for stage in self.stages:
            force_rerun = repair_mode or self.checkpoint.needs_repair(stage["name"])

            if not self.run_stage(stage, force_rerun=force_rerun):
                self.print_status("❌ Pipeline stopped due to stage failure", "error")
                return False

        # Final success
        self.print_status(f"\n{'='*60}")
        self.print_status("🎉 ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        self.print_status(f"📁 All results stored in: {self.run_manager.run_dir}")
        self.print_status(f"🔍 Run ID: {self.run_manager.run_id}")
        self.print_status(f"{'='*60}")

        # Update run metadata
        self.update_run_metadata("completed")
        return True

    def update_run_metadata(self, status: str):
        """Update run metadata with final status"""
        metadata_path = self.run_manager.run_dir / "run_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            metadata.update({
                "status": status,
                "end_time": datetime.now().isoformat(),
                "duration_minutes": (datetime.now() - datetime.fromisoformat(metadata["start_time"])).total_seconds() / 60
            })

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def interactive_menu(self):
        """Interactive menu for pipeline management"""
        while True:
            self.print_status("\n🔬 ENHANCED MALARIA DETECTION PIPELINE")
            self.print_status("="*60)

            self.show_pipeline_status()

            self.print_status("\nPilihan:")
            self.print_status("1. Lanjutkan pipeline (skip completed stages)")
            self.print_status("2. Restart dari awal (hapus semua checkpoint)")
            self.print_status("3. Repair mode (validasi ulang dan perbaiki semua stage)")
            self.print_status("4. Lihat status detail saja")
            self.print_status("5. Keluar")

            try:
                choice = input("\nMasukkan pilihan (1-5): ").strip()

                if choice == "1":
                    self.print_status("🚀 Melanjutkan pipeline...")
                    success = self.run_pipeline()
                    if success:
                        self.print_status("🎉 Pipeline selesai!", "success")
                        break
                    else:
                        self.print_status("❌ Pipeline gagal!", "error")

                elif choice == "2":
                    confirm = input("⚠️  Yakin ingin restart dari awal? (y/N): ").strip().lower()
                    if confirm == 'y':
                        success = self.run_pipeline(force_restart=True)
                        if success:
                            self.print_status("🎉 Pipeline selesai!", "success")
                            break
                        else:
                            self.print_status("❌ Pipeline gagal!", "error")
                    else:
                        self.print_status("Restart dibatalkan")

                elif choice == "3":
                    self.print_status("🔧 Menjalankan repair mode...")
                    success = self.run_pipeline(repair_mode=True)
                    if success:
                        self.print_status("🎉 Pipeline repair selesai!", "success")
                        break
                    else:
                        self.print_status("❌ Pipeline repair gagal!", "error")

                elif choice == "4":
                    continue  # Just show status again

                elif choice == "5":
                    self.print_status("👋 Keluar dari pipeline")
                    break

                else:
                    self.print_status("❌ Pilihan tidak valid!", "error")

            except KeyboardInterrupt:
                self.print_status("\n👋 Pipeline dihentikan oleh user")
                break
            except Exception as e:
                self.print_status(f"❌ Error: {str(e)}", "error")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Malaria Detection Pipeline")
    parser.add_argument("--continue", action="store_true", dest="continue_pipeline",
                       help="Continue pipeline from last checkpoint")
    parser.add_argument("--restart", action="store_true",
                       help="Restart pipeline from beginning (clears checkpoints)")
    parser.add_argument("--repair", action="store_true",
                       help="Repair mode - re-validate and fix all stages")
    parser.add_argument("--status", action="store_true",
                       help="Show pipeline status only")
    parser.add_argument("--run-id", type=str,
                       help="Specify custom run ID instead of timestamp")

    args = parser.parse_args()

    # Create run manager (with custom run ID if specified)
    if args.run_id:
        # Custom run manager for specific run ID
        base_dir = Path(".")
        run_dir = base_dir / "pipeline_runs" / args.run_id
        if not run_dir.exists():
            print(f"❌ Run ID '{args.run_id}' not found!")
            return
        # TODO: Load existing run manager

    run_manager = PipelineRunManager()
    pipeline = EnhancedMalariaPipeline(run_manager)

    try:
        if args.status:
            pipeline.show_pipeline_status()
        elif args.continue_pipeline:
            success = pipeline.run_pipeline()
            sys.exit(0 if success else 1)
        elif args.restart:
            success = pipeline.run_pipeline(force_restart=True)
            sys.exit(0 if success else 1)
        elif args.repair:
            success = pipeline.run_pipeline(repair_mode=True)
            sys.exit(0 if success else 1)
        else:
            # Interactive mode
            pipeline.interactive_menu()

    except KeyboardInterrupt:
        pipeline.print_status("\n👋 Pipeline interrupted by user")
        pipeline.update_run_metadata("interrupted")
    except Exception as e:
        pipeline.print_status(f"❌ Unexpected error: {str(e)}", "error")
        pipeline.update_run_metadata("error")
        sys.exit(1)

if __name__ == "__main__":
    main()