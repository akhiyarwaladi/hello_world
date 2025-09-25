#!/usr/bin/env python3
"""
Results Manager for Malaria Detection Pipeline
Automatically organizes experiment results into structured folders
"""

import os
import yaml
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class ResultsManager:
    """Manages organized folder structure for experiment results"""

    def __init__(self, config_path: str = "config/results_structure.yaml", pipeline_name: str = None):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.base_dir = Path(self.config.get("results_structure", {}).get("base_dir", "results"))

        # NEW: Support for centralized pipeline results
        self.pipeline_name = pipeline_name
        self.centralized_mode = pipeline_name is not None

        if self.centralized_mode:
            # Use centralized folder inside results directory with shorter name
            self.pipeline_dir = self.base_dir / f"exp_{pipeline_name}"
            self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Original distributed structure
            self.pipeline_dir = None

        # Create directory structure
        self._initialize_directories()

    def _load_config(self) -> Dict:
        """Load results structure configuration"""
        if not self.config_path.exists():
            print(f"[WARNING] Config file {self.config_path} not found, using defaults")
            return self._get_default_config()

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_default_config(self) -> Dict:
        """Return default configuration if config file not found"""
        return {
            "results_structure": {
                "base_dir": "results",
                "directories": {
                    "current_experiments": "current_experiments",
                    "completed_models": "completed_models",
                    "publications": "publications",
                    "archive": "archive",
                    "validation": "validation"
                }
            },
            "defaults": {
                "results_dir": "results/current_experiments",
                "completed_models_dir": "results/completed_models"
            }
        }

    def _initialize_directories(self):
        """Create organized directory structure"""
        # FIXED: Skip folder creation in centralized mode to prevent unwanted folders
        if self.centralized_mode:
            # In centralized mode, only create the main experiment folder
            print(f"[CENTRALIZED] Using clean structure: {self.pipeline_dir}")
            return

        # Only create distributed structure folders when NOT in centralized mode
        directories = self.config.get("results_structure", {}).get("directories", {})

        for dir_type, dir_name in directories.items():
            dir_path = self.base_dir / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"[DIR] Created directory: {dir_path}")

    def get_experiment_path(self, experiment_type: str, model_name: str,
                          experiment_name: str = None) -> Path:
        """Get appropriate path for experiment based on type"""

        # NEW: Use centralized structure if in centralized mode
        if self.centralized_mode:
            return self._get_centralized_path(experiment_type, model_name, experiment_name)

        # ORIGINAL: Distributed structure
        # Determine base directory based on experiment type
        if experiment_type in ["production", "final", "completed"]:
            base = self.base_dir / "completed_models"
        elif experiment_type in ["validation", "test", "pipeline"]:
            base = self.base_dir / "current_experiments" / "validation"
        elif experiment_type in ["training", "model"]:
            base = self.base_dir / "current_experiments" / "training"
        else:
            base = self.base_dir / "current_experiments"

        # Create model-specific subdirectory
        if "detection" in model_name.lower():
            model_path = base / "detection" / model_name
        elif "classification" in model_name.lower():
            model_path = base / "classification" / model_name
        else:
            model_path = base / model_name

        # Add experiment name if provided
        if experiment_name:
            model_path = model_path / experiment_name

        # Don't create folder automatically - let caller decide when needed
        return model_path

    def create_experiment_path(self, experiment_type: str, model_name: str,
                             experiment_name: str = None) -> Path:
        """Get path and CREATE directory for experiment - use when folder is actually needed"""
        path = self.get_experiment_path(experiment_type, model_name, experiment_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_centralized_path(self, experiment_type: str, model_name: str,
                            experiment_name: str = None) -> Path:
        """Get centralized path for pipeline results"""

        # Create organized structure within centralized folder
        if "detection" in model_name.lower():
            model_path = self.pipeline_dir / "detection" / model_name
        elif "classification" in model_name.lower():
            model_path = self.pipeline_dir / "classification" / model_name
        else:
            model_path = self.pipeline_dir / "models" / model_name

        # Add experiment name if provided
        if experiment_name:
            model_path = model_path / experiment_name

        # Don't create folder automatically - let caller decide when needed
        return model_path

    def find_experiment_path(self, experiment_type: str, model_name: str,
                            experiment_name: str = None) -> Path:
        """Find existing experiment path WITHOUT creating directories"""
        if self.centralized_mode:
            return self._find_centralized_path(experiment_type, model_name, experiment_name)

        # ORIGINAL: Distributed structure
        # Determine base directory based on experiment type
        if experiment_type in ["production", "final", "completed"]:
            base = self.base_dir / "completed_models"
        elif experiment_type in ["validation", "test", "pipeline"]:
            base = self.base_dir / "current_experiments" / "validation"
        elif experiment_type in ["training", "model"]:
            base = self.base_dir / "current_experiments" / "training"
        else:
            base = self.base_dir / "current_experiments"

        # Create model-specific subdirectory
        if "detection" in model_name.lower():
            model_path = base / "detection" / model_name
        elif "classification" in model_name.lower():
            model_path = base / "classification" / model_name
        else:
            model_path = base / model_name

        # Add experiment name if provided
        if experiment_name:
            model_path = model_path / experiment_name

        # DO NOT create directories - just return path
        return model_path

    def _find_centralized_path(self, experiment_type: str, model_name: str,
                              experiment_name: str = None) -> Path:
        """Find centralized path WITHOUT creating directories"""

        # Create organized structure within centralized folder
        if "detection" in model_name.lower():
            model_path = self.pipeline_dir / "detection" / model_name
        elif "classification" in model_name.lower():
            model_path = self.pipeline_dir / "classification" / model_name
        else:
            model_path = self.pipeline_dir / "models" / model_name

        # Add experiment name if provided
        if experiment_name:
            model_path = model_path / experiment_name

        # DO NOT create directories - just return path
        return model_path

    def get_publication_path(self, publication_type: str = "journal") -> Path:
        """Get path for publication exports"""
        if self.centralized_mode:
            pub_path = self.pipeline_dir / "publications" / publication_type
        else:
            pub_path = self.base_dir / "publications" / publication_type
        pub_path.mkdir(parents=True, exist_ok=True)
        return pub_path

    def get_crops_path(self, detection_model: str, experiment_name: str) -> Path:
        """Get path for generated crops"""
        if self.centralized_mode:
            crops_path = self.pipeline_dir / "crop_data" / f"crops_from_{detection_model}_{experiment_name}"
        else:
            crops_path = Path(f"data/crops_from_{detection_model}_{experiment_name}")
        crops_path.mkdir(parents=True, exist_ok=True)
        return crops_path

    def get_analysis_path(self, analysis_type: str = "general") -> Path:
        """Get path for analysis results"""
        if self.centralized_mode:
            analysis_path = self.pipeline_dir / "analysis" / analysis_type
        else:
            analysis_path = self.base_dir / "analysis" / analysis_type
        analysis_path.mkdir(parents=True, exist_ok=True)
        return analysis_path

    def promote_to_completed(self, current_path: Path, model_performance: Dict) -> Path:
        """Move experiment to completed models if meets criteria"""
        rules = self.config.get("organization_rules", {})
        auto_promote = rules.get("auto_promote_to_completed", {})

        should_promote = False

        # Check detection model criteria
        if "detection" in str(current_path).lower():
            map50_threshold = auto_promote.get("detection_map50_threshold", 0.85)
            current_map50 = model_performance.get("mAP50", 0.0)
            if current_map50 >= map50_threshold:
                should_promote = True

        # Check classification model criteria
        elif "classification" in str(current_path).lower():
            acc_threshold = auto_promote.get("classification_accuracy_threshold", 0.85)
            current_acc = model_performance.get("accuracy", 0.0)
            if current_acc >= acc_threshold:
                should_promote = True

        if should_promote:
            # Create completed models path
            model_name = current_path.name
            completed_path = self.base_dir / "completed_models"

            if "detection" in str(current_path).lower():
                completed_path = completed_path / "detection" / model_name
            else:
                completed_path = completed_path / "classification" / model_name

            completed_path.mkdir(parents=True, exist_ok=True)

            # Copy results to completed (keep original for now)
            if current_path.exists():
                self._copy_directory(current_path, completed_path)
                print(f"Promoted model to completed: {completed_path}")
                return completed_path

        return current_path

    def _copy_directory(self, src: Path, dst: Path):
        """Copy directory contents"""
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    def archive_old_experiments(self):
        """Archive experiments older than configured threshold"""
        rules = self.config.get("organization_rules", {})
        archive_days = rules.get("auto_archive_after_days", 30)
        cutoff_date = datetime.now() - timedelta(days=archive_days)

        current_exp_dir = self.base_dir / "current_experiments"
        archive_dir = self.base_dir / "archive"

        if not current_exp_dir.exists():
            return

        archived_count = 0
        for exp_path in current_exp_dir.rglob("*"):
            if exp_path.is_dir():
                # Check modification time
                mod_time = datetime.fromtimestamp(exp_path.stat().st_mtime)
                if mod_time < cutoff_date:
                    # Move to archive
                    rel_path = exp_path.relative_to(current_exp_dir)
                    archive_path = archive_dir / rel_path
                    archive_path.parent.mkdir(parents=True, exist_ok=True)

                    shutil.move(str(exp_path), str(archive_path))
                    archived_count += 1

        if archived_count > 0:
            print(f"Archived {archived_count} old experiments")

    def get_results_summary(self) -> Dict:
        """Get summary of current results organization"""
        summary = {
            "current_experiments": [],
            "completed_models": [],
            "publications": [],
            "archive": []
        }

        for category in summary.keys():
            category_path = self.base_dir / category.replace("_", "")
            if category_path.exists():
                for item in category_path.iterdir():
                    if item.is_dir():
                        summary[category].append({
                            "name": item.name,
                            "path": str(item),
                            "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                        })

        return summary

    def cleanup_empty_directories(self):
        """Remove empty directories"""
        for root, dirs, files in os.walk(self.base_dir, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        print(f"[CLEANUP] Removed empty directory: {dir_path}")
                except OSError:
                    # Directory not empty or permission issue
                    pass

# Convenience functions
def get_results_manager(pipeline_name: str = None) -> ResultsManager:
    """Get results manager instance (centralized if pipeline_name provided)"""
    return ResultsManager(pipeline_name=pipeline_name)

def get_experiment_path(experiment_type: str, model_name: str,
                       experiment_name: str = None, pipeline_name: str = None) -> Path:
    """Get experiment path using results manager"""
    manager = get_results_manager(pipeline_name)
    return manager.get_experiment_path(experiment_type, model_name, experiment_name)

def get_publication_path(publication_type: str = "journal", pipeline_name: str = None) -> Path:
    """Get publication path using results manager"""
    manager = get_results_manager(pipeline_name)
    return manager.get_publication_path(publication_type)

def get_crops_path(detection_model: str, experiment_name: str, pipeline_name: str = None) -> Path:
    """Get crops path using results manager"""
    manager = get_results_manager(pipeline_name)
    return manager.get_crops_path(detection_model, experiment_name)

def get_analysis_path(analysis_type: str = "general", pipeline_name: str = None) -> Path:
    """Get analysis path using results manager"""
    manager = get_results_manager(pipeline_name)
    return manager.get_analysis_path(analysis_type)
