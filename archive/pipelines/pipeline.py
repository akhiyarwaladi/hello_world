#!/usr/bin/env python3
"""
Malaria Detection Pipeline - Comprehensive System with Data Validation
Menggantikan quick_setup_new_machine.sh dengan sistem Python yang robust

Features:
- Comprehensive data validation at each stage
- Automatic data repair and re-processing
- Intelligent checkpoint management
- Detailed logging and progress tracking

Author: Pipeline Automation Team
Date: September 2024
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import hashlib
import cv2
import numpy as np

class DataValidator:
    """Validates data integrity at each pipeline stage"""

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
        expected_dirs = ["Falciparum", "Vivax"]  # Main species in MP-IDB
        actual_dirs = [d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith('.')]

        found_species = [d for d in expected_dirs if d in actual_dirs]
        if len(found_species) < 1:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Missing species directories. Expected at least one of: {expected_dirs}, found: {actual_dirs}")
            validation_result["repair_needed"] = True

        # Count total images (this is a classification dataset, not detection)
        total_images = 0
        species_counts = {}

        for species in found_species:
            species_path = path / species
            species_images = list(species_path.rglob("*.jpg"))
            species_count = len(species_images)
            species_counts[species] = species_count
            total_images += species_count

        validation_result["stats"] = {
            "total_images": total_images,
            "species_directories": actual_dirs,
            "species_counts": species_counts,
            "dataset_type": "classification"
        }

        # Minimum thresholds for classification dataset
        if total_images < 100:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Insufficient images: {total_images} < 100")
            validation_result["repair_needed"] = True

        # This is a classification dataset, so no bounding box annotations expected
        validation_result["warnings"].append("This is a classification dataset - no bounding box annotations expected")

        return validation_result["valid"], validation_result

    @staticmethod
    def validate_detection_dataset(dataset_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate prepared detection dataset"""
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

        # Check required structure
        images_dir = path / "images"
        labels_dir = path / "labels"
        dataset_yaml = path / "dataset.yaml"

        for required_path in [images_dir, labels_dir, dataset_yaml]:
            if not required_path.exists():
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required path: {required_path}")
                validation_result["repair_needed"] = True

        if not validation_result["valid"]:
            return False, validation_result

        # Count and validate files
        image_files = list(images_dir.glob("*.jpg"))
        label_files = list(labels_dir.glob("*.txt"))

        validation_result["stats"] = {
            "total_images": len(image_files),
            "total_labels": len(label_files),
        }

        # Check image-label pairs
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}

        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems

        if missing_labels:
            validation_result["warnings"].append(f"Images without labels: {len(missing_labels)}")
            validation_result["stats"]["missing_labels"] = len(missing_labels)

        if missing_images:
            validation_result["warnings"].append(f"Labels without images: {len(missing_images)}")
            validation_result["stats"]["missing_images"] = len(missing_images)

        # Validate some random images
        corrupt_images = []
        sample_images = image_files[:10]  # Check first 10 images

        for img_path in sample_images:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    corrupt_images.append(img_path.name)
            except Exception as e:
                corrupt_images.append(f"{img_path.name}: {str(e)}")

        if corrupt_images:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Corrupt images found: {corrupt_images}")
            validation_result["repair_needed"] = True

        validation_result["stats"]["corrupt_images"] = len(corrupt_images)

        # Check minimum thresholds
        if len(image_files) < 100:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Insufficient images: {len(image_files)} < 100")
            validation_result["repair_needed"] = True

        # Validate YAML content
        try:
            import yaml
            with open(dataset_yaml, 'r') as f:
                yaml_content = yaml.safe_load(f)

            required_keys = ['path', 'train', 'val', 'nc', 'names']
            missing_keys = [k for k in required_keys if k not in yaml_content]

            if missing_keys:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing YAML keys: {missing_keys}")
                validation_result["repair_needed"] = True

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Invalid YAML file: {str(e)}")
            validation_result["repair_needed"] = True

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

        # Check required structure
        required_dirs = [
            path / "train" / "parasite",
            path / "val" / "parasite",
            path / "test" / "parasite"
        ]

        for required_dir in required_dirs:
            if not required_dir.exists():
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing directory: {required_dir}")
                validation_result["repair_needed"] = True

        if not validation_result["valid"]:
            return False, validation_result

        # Count files in each split
        train_count = len(list(required_dirs[0].glob("*.jpg")))
        val_count = len(list(required_dirs[1].glob("*.jpg")))
        test_count = len(list(required_dirs[2].glob("*.jpg")))
        total_count = train_count + val_count + test_count

        validation_result["stats"] = {
            "train_crops": train_count,
            "val_crops": val_count,
            "test_crops": test_count,
            "total_crops": total_count
        }

        # Check minimum thresholds
        if total_count < 1200:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Insufficient crops: {total_count} < 1200")
            validation_result["repair_needed"] = True

        if train_count < 800:
            validation_result["warnings"].append(f"Low train count: {train_count} < 800")

        # Check image quality (sample)
        corrupt_crops = []
        invalid_sizes = []
        sample_crops = list(required_dirs[0].glob("*.jpg"))[:10]

        for crop_path in sample_crops:
            try:
                img = cv2.imread(str(crop_path))
                if img is None:
                    corrupt_crops.append(crop_path.name)
                    continue

                h, w = img.shape[:2]
                if h < 32 or w < 32:  # Too small
                    invalid_sizes.append(f"{crop_path.name}: {w}x{h}")

            except Exception as e:
                corrupt_crops.append(f"{crop_path.name}: {str(e)}")

        if corrupt_crops:
            validation_result["errors"].append(f"Corrupt crops: {corrupt_crops}")
            validation_result["repair_needed"] = True

        if invalid_sizes:
            validation_result["warnings"].append(f"Small crops: {invalid_sizes}")

        validation_result["stats"]["corrupt_crops"] = len(corrupt_crops)
        validation_result["stats"]["small_crops"] = len(invalid_sizes)

        # Check metadata file
        metadata_file = path / "crop_metadata.json"
        if not metadata_file.exists():
            validation_result["warnings"].append("Missing crop_metadata.json")
        else:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                validation_result["stats"]["metadata_entries"] = len(metadata)
            except Exception as e:
                validation_result["warnings"].append(f"Invalid metadata: {str(e)}")

        return validation_result["valid"], validation_result


class PipelineCheckpoint:
    """Enhanced checkpoint management with validation results"""

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
            "python_path": sys.executable,
            "pipeline_version": "2.0_comprehensive"
        }

    def save_checkpoint(self):
        """Save current checkpoint state"""
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def mark_stage_completed(self, stage_name: str, validation_result: Optional[Dict] = None, metadata: Optional[Dict] = None):
        """Mark a stage as completed with validation results"""
        self.data["stages"][stage_name] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "validation_result": validation_result,
            "metadata": metadata or {}
        }
        self.save_checkpoint()

    def mark_stage_failed(self, stage_name: str, error_msg: str, validation_result: Optional[Dict] = None):
        """Mark a stage as failed"""
        self.data["stages"][stage_name] = {
            "status": "failed",
            "failed_at": datetime.now().isoformat(),
            "error": error_msg,
            "validation_result": validation_result
        }
        self.save_checkpoint()

    def mark_stage_needs_repair(self, stage_name: str, validation_result: Dict):
        """Mark a stage as needing repair due to validation failure"""
        self.data["stages"][stage_name] = {
            "status": "needs_repair",
            "marked_at": datetime.now().isoformat(),
            "validation_result": validation_result
        }
        self.save_checkpoint()

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage is completed and validated"""
        return (stage_name in self.data["stages"] and
                self.data["stages"][stage_name]["status"] == "completed")

    def needs_repair(self, stage_name: str) -> bool:
        """Check if a stage needs repair"""
        return (stage_name in self.data["stages"] and
                self.data["stages"][stage_name]["status"] == "needs_repair")

    def get_stage_status(self, stage_name: str) -> str:
        """Get stage status"""
        if stage_name not in self.data["stages"]:
            return "not_started"
        return self.data["stages"][stage_name]["status"]

    def get_validation_result(self, stage_name: str) -> Optional[Dict]:
        """Get validation result for a stage"""
        if stage_name in self.data["stages"]:
            return self.data["stages"][stage_name].get("validation_result")
        return None

    def get_completed_stages(self) -> List[str]:
        """Get list of completed stages"""
        return [stage for stage, data in self.data["stages"].items()
                if data["status"] == "completed"]

    def reset_stage(self, stage_name: str):
        """Reset a specific stage"""
        if stage_name in self.data["stages"]:
            del self.data["stages"][stage_name]
            self.save_checkpoint()

    def reset_all_stages(self):
        """Reset all stages"""
        self.data["stages"] = {}
        self.save_checkpoint()


class MalariaPipeline:
    """Comprehensive malaria detection pipeline with data validation"""

    def __init__(self):
        self.checkpoint = PipelineCheckpoint()
        self.validator = DataValidator()
        self.setup_logging()

        # Pipeline stages with validation
        self.stages = [
            {
                "name": "environment_check",
                "description": "Environment & Dependencies Check",
                "function": self.check_environment,
                "validator": None,  # No data validation needed
                "skip_if_exists": False
            },
            {
                "name": "dataset_download",
                "description": "Dataset Download",
                "function": self.download_datasets,
                "validator": lambda: self.validator.validate_raw_dataset("data/raw/mp_idb"),
                "check_path": "data/raw/mp_idb",
                "skip_if_exists": True
            },
            {
                "name": "detection_preparation",
                "description": "Detection Dataset Preparation",
                "function": self.prepare_detection_dataset,
                "validator": lambda: self.validator.validate_detection_dataset("data/detection_fixed"),
                "check_path": "data/detection_fixed",
                "skip_if_exists": True
            },
            {
                "name": "parasite_cropping",
                "description": "Parasite Cropping",
                "function": self.crop_parasites,
                "validator": lambda: self.validator.validate_cropped_dataset("data/classification_crops"),
                "check_path": "data/classification_crops",
                "skip_if_exists": True
            },
            {
                "name": "training_verification",
                "description": "Training System Verification",
                "function": self.verify_training,
                "validator": None,  # Training verification handles its own validation
                "skip_if_exists": False
            }
        ]

    def setup_logging(self):
        """Setup comprehensive logging"""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Setup logging with both file and console handlers
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / 'pipeline.log'),
                logging.FileHandler(logs_dir / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def print_status(self, message: str, level: str = "info"):
        """Print colored status messages with logging"""
        colors = {
            "info": "\033[0;34m",     # Blue
            "success": "\033[0;32m",  # Green
            "warning": "\033[1;33m",  # Yellow
            "error": "\033[0;31m",    # Red
            "repair": "\033[1;35m",   # Magenta
            "reset": "\033[0m"        # Reset
        }

        symbols = {
            "info": "ℹ️ ",
            "success": "✅",
            "warning": "⚠️ ",
            "error": "❌",
            "repair": "🔧"
        }

        color = colors.get(level, colors["info"])
        symbol = symbols.get(level, "")

        print(f"{color}{symbol} {message}{colors['reset']}")

        # Log to file
        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def print_validation_result(self, validation_result: Dict[str, Any], stage_name: str):
        """Print detailed validation results"""
        if validation_result["valid"]:
            self.print_status(f"✅ {stage_name} validation: PASSED", "success")
        else:
            self.print_status(f"❌ {stage_name} validation: FAILED", "error")

        # Print stats
        if "stats" in validation_result and validation_result["stats"]:
            self.print_status("📊 Data Statistics:")
            for key, value in validation_result["stats"].items():
                self.print_status(f"   • {key}: {value}")

        # Print errors
        if validation_result["errors"]:
            self.print_status("❌ Errors found:", "error")
            for error in validation_result["errors"]:
                self.print_status(f"   • {error}", "error")

        # Print warnings
        if validation_result["warnings"]:
            self.print_status("⚠️  Warnings:", "warning")
            for warning in validation_result["warnings"]:
                self.print_status(f"   • {warning}", "warning")

    def run_script(self, script_path: str, description: str,
                   args: List[str] = None, timeout: int = 300) -> Tuple[bool, str]:
        """Run a Python script with comprehensive error handling"""
        if not Path(script_path).exists():
            return False, f"Script not found: {script_path}"

        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)

        try:
            self.print_status(f"🚀 Running: {description}")
            self.logger.info(f"Executing command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout
            )

            self.logger.info(f"Script {script_path} completed successfully")
            return True, result.stdout

        except subprocess.TimeoutExpired:
            error_msg = f"Script timeout after {timeout} seconds"
            self.logger.error(error_msg)
            return False, error_msg

        except subprocess.CalledProcessError as e:
            error_msg = f"Script failed with return code {e.returncode}\n"
            error_msg += f"Stdout: {e.stdout[-1000:]}\n" if e.stdout else ""
            error_msg += f"Stderr: {e.stderr[-1000:]}" if e.stderr else ""
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
        self.print_status("🔍 Checking Python environment...")

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
            ("numpy", "NumPy"),
            ("yaml", "PyYAML")
        ]

        all_ok = True
        for package, name in critical_packages:
            try:
                __import__(package)
                self.print_status(f"{name}: OK", "success")
            except ImportError:
                self.print_status(f"{name}: Missing!", "error")
                all_ok = False

        return all_ok

    def download_datasets(self) -> bool:
        """Download and validate datasets"""
        self.print_status("📥 Checking dataset requirements...")

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
        """Prepare and validate detection dataset"""
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
        """Crop parasites with validation"""
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
        """Verify training system with comprehensive tests"""
        self.print_status("🧪 Verifying training system...")

        # Test detection training
        success, output = self.run_script(
            "scripts/10_train_yolo_detection.py",
            "Detection Training Test",
            ["--data", "data/detection_fixed/dataset.yaml",
             "--epochs", "1", "--batch", "4", "--device", "cpu",
             "--name", "pipeline_test_detection"],
            timeout=600
        )

        if not success:
            self.print_status(f"Detection training test failed: {output}", "error")
            return False

        # Test classification training
        success, output = self.run_script(
            "scripts/11_train_classification_crops.py",
            "Classification Training Test",
            ["--data", "data/classification_crops",
             "--epochs", "1", "--batch", "4", "--device", "cpu",
             "--name", "pipeline_test_classification"],
            timeout=600
        )

        if success:
            self.print_status("Training system verification completed", "success")
        else:
            self.print_status(f"Classification training test failed: {output}", "error")

        return success

    def validate_and_repair_stage(self, stage: Dict) -> Tuple[bool, Optional[Dict]]:
        """Validate stage data and attempt repair if needed"""
        stage_name = stage["name"]

        # Skip stages without validators
        if not stage.get("validator"):
            return True, None

        self.print_status(f"🔍 Validating {stage['description']}...")

        try:
            is_valid, validation_result = stage["validator"]()
            self.print_validation_result(validation_result, stage["description"])

            if not is_valid and validation_result.get("repair_needed", False):
                self.print_status(f"🔧 Stage '{stage['description']}' needs repair", "repair")
                self.checkpoint.mark_stage_needs_repair(stage_name, validation_result)
                return False, validation_result
            elif not is_valid:
                self.print_status(f"❌ Stage '{stage['description']}' validation failed", "error")
                return False, validation_result
            else:
                self.print_status(f"✅ Stage '{stage['description']}' validation passed", "success")
                return True, validation_result

        except Exception as e:
            self.print_status(f"❌ Validation error for {stage['description']}: {str(e)}", "error")
            return False, {"valid": False, "error": str(e)}

    def run_stage(self, stage: Dict, force_rerun: bool = False) -> bool:
        """Run a single pipeline stage with comprehensive validation"""
        stage_name = stage["name"]
        description = stage["description"]

        self.print_status(f"\n{'='*60}")
        self.print_status(f"Stage: {description}")
        self.print_status(f"{'='*60}")

        # Check if stage needs repair
        if self.checkpoint.needs_repair(stage_name) or force_rerun:
            self.print_status(f"🔧 Stage '{description}' marked for repair/rerun", "repair")
        # Check if already completed (and not forced)
        elif self.checkpoint.is_stage_completed(stage_name):
            self.print_status(f"✅ Stage '{description}' already completed", "success")

            # Still validate if validator exists
            if stage.get("validator"):
                is_valid, validation_result = self.validate_and_repair_stage(stage)
                if not is_valid:
                    self.print_status(f"🔧 Re-running due to validation failure", "repair")
                    self.checkpoint.reset_stage(stage_name)
                else:
                    return True
            else:
                return True

        # Check if we can skip based on existing files (but still validate)
        if (stage.get("skip_if_exists", False) and
            "check_path" in stage and
            self.check_path_exists(stage["check_path"]) and
            not force_rerun):

            # Validate existing data
            if stage.get("validator"):
                is_valid, validation_result = self.validate_and_repair_stage(stage)
                if is_valid:
                    self.print_status(f"✅ Stage '{description}' data exists and valid - marking as completed", "success")
                    self.checkpoint.mark_stage_completed(stage_name, validation_result)
                    return True
                else:
                    self.print_status(f"🔧 Existing data invalid - re-processing needed", "repair")
            else:
                self.print_status(f"✅ Stage '{description}' data exists - marking as completed", "success")
                self.checkpoint.mark_stage_completed(stage_name)
                return True

        # Run the stage
        try:
            success = stage["function"]()

            if success:
                # Validate the output if validator exists
                validation_result = None
                if stage.get("validator"):
                    is_valid, validation_result = self.validate_and_repair_stage(stage)
                    if not is_valid:
                        self.print_status(f"❌ Stage '{description}' completed but validation failed", "error")
                        self.checkpoint.mark_stage_failed(stage_name, "Post-execution validation failed", validation_result)
                        return False

                self.checkpoint.mark_stage_completed(stage_name, validation_result)
                self.print_status(f"✅ Stage '{description}' completed successfully", "success")
                return True
            else:
                self.checkpoint.mark_stage_failed(stage_name, "Stage function returned False")
                self.print_status(f"❌ Stage '{description}' failed", "error")
                return False

        except Exception as e:
            error_msg = f"Exception in stage: {str(e)}"
            self.checkpoint.mark_stage_failed(stage_name, error_msg)
            self.print_status(f"❌ Stage '{description}' failed with exception: {e}", "error")
            return False

    def show_pipeline_status(self):
        """Show comprehensive pipeline status with validation results"""
        self.print_status(f"\n{'='*60}")
        self.print_status("🔍 COMPREHENSIVE PIPELINE STATUS")
        self.print_status(f"{'='*60}")

        completed_stages = self.checkpoint.get_completed_stages()

        for i, stage in enumerate(self.stages, 1):
            status = self.checkpoint.get_stage_status(stage["name"])
            validation_result = self.checkpoint.get_validation_result(stage["name"])

            if status == "completed":
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

    def run_pipeline(self, force_restart: bool = False, repair_mode: bool = False):
        """Run the complete pipeline with comprehensive validation and repair"""

        if force_restart:
            self.print_status("🔄 Force restart requested - clearing all checkpoints", "warning")
            self.checkpoint.reset_all_stages()

        self.print_status("🚀 COMPREHENSIVE MALARIA DETECTION PIPELINE")
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

        # Final validation summary
        self.print_status(f"\n{'='*60}")
        self.print_status("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        self.print_status(f"{'='*60}")

        self.print_status("✨ All stages completed and validated:")
        for stage in self.stages:
            validation_result = self.checkpoint.get_validation_result(stage["name"])
            if validation_result and "stats" in validation_result:
                key_stats = {k: v for k, v in validation_result["stats"].items()
                           if k in ["total_images", "total_crops", "total_labels", "total_annotations"]}
                if key_stats:
                    stats_str = ", ".join(f"{k}: {v}" for k, v in key_stats.items())
                    self.print_status(f"  ✅ {stage['description']} ({stats_str})")
                else:
                    self.print_status(f"  ✅ {stage['description']}")
            else:
                self.print_status(f"  ✅ {stage['description']}")

        self.print_status("\n🚀 Ready for production training:")
        self.print_status("  • Detection: python scripts/10_train_yolo_detection.py --epochs 30")
        self.print_status("  • Classification: python scripts/11_train_classification_crops.py --epochs 25")

        return True

    def interactive_menu(self):
        """Enhanced interactive menu with validation and repair options"""

        print("🔬 COMPREHENSIVE MALARIA DETECTION PIPELINE")
        print("="*70)
        print("Sistem pipeline dengan validasi data menyeluruh dan perbaikan otomatis.")
        print("Setiap tahap akan divalidasi untuk memastikan kualitas data.\n")

        # Show current status first
        self.show_pipeline_status()

        print("\nPilihan:")
        print("1. Lanjutkan pipeline (skip completed stages)")
        print("2. Restart dari awal (hapus semua checkpoint)")
        print("3. Repair mode (validasi ulang dan perbaiki semua stage)")
        print("4. Lihat status detail saja")
        print("5. Keluar")

        try:
            choice = input("\nPilih (1-5) [1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar.")
            return

        if choice == "1":
            print("\n🚀 Melanjutkan pipeline...")
            success = self.run_pipeline()

        elif choice == "2":
            print("\n⚠️  Restart dari awal - semua checkpoint akan dihapus!")
            try:
                confirm = input("Yakin? (y/N): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nDibatalkan.")
                return

            if confirm == 'y':
                success = self.run_pipeline(force_restart=True)
            else:
                print("Dibatalkan.")
                return

        elif choice == "3":
            print("\n🔧 Repair mode - validasi ulang dan perbaiki semua data...")
            try:
                confirm = input("Lanjutkan repair mode? (y/N): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nDibatalkan.")
                return

            if confirm == 'y':
                success = self.run_pipeline(repair_mode=True)
            else:
                print("Dibatalkan.")
                return

        elif choice == "4":
            self.show_pipeline_status()
            return

        elif choice == "5":
            print("Keluar.")
            return

        else:
            print("Pilihan tidak valid.")
            return

        if success:
            print("\n🎉 Pipeline selesai! Semua data sudah divalidasi dan siap untuk training.")
        else:
            print("\n❌ Pipeline berhenti karena error. Periksa log untuk detail.")


def main():
    """Main function with comprehensive CLI support"""
    import argparse

    # Check if running with arguments (CLI mode) or without (interactive mode)
    if len(sys.argv) > 1:
        # CLI Mode
        parser = argparse.ArgumentParser(description="Comprehensive Malaria Detection Pipeline")
        parser.add_argument("--continue", action="store_true", dest="continue_pipeline",
                           help="Continue pipeline (skip completed stages)")
        parser.add_argument("--restart", action="store_true",
                           help="Restart pipeline from beginning")
        parser.add_argument("--repair", action="store_true",
                           help="Repair mode: re-validate and fix all stages")
        parser.add_argument("--status", action="store_true",
                           help="Show detailed pipeline status only")

        args = parser.parse_args()
        pipeline = MalariaPipeline()

        if args.status:
            pipeline.show_pipeline_status()
        elif args.restart:
            pipeline.run_pipeline(force_restart=True)
        elif args.repair:
            pipeline.run_pipeline(repair_mode=True)
        else:  # Default or --continue
            pipeline.run_pipeline()

    else:
        # Interactive Mode
        pipeline = MalariaPipeline()
        pipeline.interactive_menu()


if __name__ == "__main__":
    main()