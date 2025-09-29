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
            "organization_rules": {
                "auto_promote_to_completed": {
                    "detection_map50_threshold": 0.95,
                    "classification_accuracy_threshold": 0.90
                }
            },
            "paths": {
                "results_dir": "results/current_experiments",
                "completed_models_dir": "results/completed_models"
            }
        }

    def _initialize_directories(self):
        """Create organized directory structure"""
        # SIMPLIFIED: Only create base results directory - no complex structure
        if self.centralized_mode:
            # In centralized mode, ONLY create the main experiment folder when needed
            print(f"[SIMPLE] Minimal structure will be created on demand: {self.pipeline_dir}")
            return

        # For distributed mode, create minimal base structure
        self.base_dir.mkdir(parents=True, exist_ok=True)

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

    def _get_centralized_path(self, experiment_type: str, model_name: str,
                            experiment_name: str = None) -> Path:
        """Get SIMPLIFIED centralized path for experiment (Option A Pipeline)"""

        # SIMPLIFIED: Use flat structure instead of nested folders
        if experiment_name:
            # Use simple naming: exp_name_modeltype
            if "detection" in model_name.lower():
                path = self.pipeline_dir / f"{experiment_name}_detection"
            elif "classification" in model_name.lower():
                path = self.pipeline_dir / f"{experiment_name}_classification"
            else:
                path = self.pipeline_dir / experiment_name
        else:
            # Fallback to simple folder names
            if "detection" in model_name.lower():
                path = self.pipeline_dir / "detection_models"
            elif "classification" in model_name.lower():
                path = self.pipeline_dir / "classification_models"
            else:
                path = self.pipeline_dir / model_name

        # DO NOT create directories - just return path
        return path

    def get_publication_path(self, publication_type: str = "journal") -> Path:
        """Get path for publication exports"""
        if self.centralized_mode:
            pub_path = self.pipeline_dir / "publications" / publication_type
        else:
            pub_path = self.base_dir / "publications" / publication_type
        pub_path.mkdir(parents=True, exist_ok=True)
        return pub_path

    def get_crops_path(self, detection_model: str, experiment_name: str) -> Path:
        """Get SIMPLIFIED path for generated crops (without creating directory)"""
        if self.centralized_mode:
            # SIMPLIFIED: Use simple folder name instead of complex nesting
            crops_path = self.pipeline_dir / f"crops_{experiment_name}"
        else:
            crops_path = Path(f"data/crops_from_{detection_model}_{experiment_name}")
        return crops_path

    def create_crops_path(self, detection_model: str, experiment_name: str) -> Path:
        """Get path for generated crops and CREATE directory"""
        crops_path = self.get_crops_path(detection_model, experiment_name)
        crops_path.mkdir(parents=True, exist_ok=True)
        return crops_path

    def get_analysis_path(self, analysis_type: str = "general") -> Path:
        """Get SIMPLIFIED path for analysis results (without creating directory)"""
        if self.centralized_mode:
            # SIMPLIFIED: Use simple folder name
            analysis_path = self.pipeline_dir / f"analysis_{analysis_type}"
        else:
            analysis_path = self.base_dir / f"analysis_{analysis_type}"
        return analysis_path

    def create_analysis_path(self, analysis_type: str = "general") -> Path:
        """Get path for analysis results and CREATE directory"""
        analysis_path = self.get_analysis_path(analysis_type)
        analysis_path.mkdir(parents=True, exist_ok=True)
        return analysis_path

    def create_experiment_path(self, experiment_type: str, model_name: str, experiment_name: str = None) -> Path:
        """Get experiment path and CREATE directory"""
        experiment_path = self.get_experiment_path(experiment_type, model_name, experiment_name)
        experiment_path.mkdir(parents=True, exist_ok=True)
        return experiment_path

    def promote_to_completed(self, current_path: Path, model_performance: Dict) -> Path:
        """Move experiment to completed models if meets criteria"""
        rules = self.config.get("organization_rules", {})
        auto_promote = rules.get("auto_promote_to_completed", {})

        should_promote = False

        # Check detection model criteria
        if "map50" in model_performance:
            map50_threshold = auto_promote.get("detection_map50_threshold", 0.95)
            if model_performance["map50"] >= map50_threshold:
                should_promote = True

        # Check classification model criteria
        if "accuracy" in model_performance:
            accuracy_threshold = auto_promote.get("classification_accuracy_threshold", 0.90)
            if model_performance["accuracy"] >= accuracy_threshold:
                should_promote = True

        if should_promote:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            completed_path = self.base_dir / "completed_models" / f"{current_path.name}_{timestamp}"
            completed_path.mkdir(parents=True, exist_ok=True)

            # Move files
            if current_path.exists():
                shutil.move(str(current_path), str(completed_path))
                print(f"[PROMOTED] Moved to completed: {completed_path}")
                return completed_path

        return current_path

    def cleanup_old_experiments(self, days_old: int = 7):
        """Remove old experiment folders older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_old)

        for experiment_dir in self.base_dir.glob("**/experiment_*"):
            if experiment_dir.is_dir():
                # Get creation time
                creation_time = datetime.fromtimestamp(experiment_dir.stat().st_ctime)

                if creation_time < cutoff_date:
                    print(f"[CLEANUP] Removing old experiment: {experiment_dir}")
                    shutil.rmtree(experiment_dir)

    def archive_experiment(self, experiment_path: Path, archive_reason: str = "completed"):
        """Archive an experiment to the archive directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.base_dir / "archive" / f"{experiment_path.name}_{timestamp}_{archive_reason}"
        archive_path.mkdir(parents=True, exist_ok=True)

        if experiment_path.exists():
            shutil.move(str(experiment_path), str(archive_path))
            print(f"[ARCHIVED] Moved to archive: {archive_path}")

    def get_experiment_summary(self) -> Dict:
        """Get summary of all experiments"""
        summary = {
            "total_experiments": 0,
            "completed_models": 0,
            "current_experiments": 0,
            "archived_experiments": 0
        }

        # Count experiments in each category
        for category in ["current_experiments", "completed_models", "archive"]:
            category_path = self.base_dir / category
            if category_path.exists():
                experiment_count = len(list(category_path.glob("*")))
                summary[category] = experiment_count
                summary["total_experiments"] += experiment_count

        return summary


class OptionAResultsManager:
    """Advanced Results Manager for Option A Pipeline with Parent Folder Structure"""

    def __init__(self, parent_experiment_name: str = None):
        self.base_dir = Path("results")

        # Create parent experiment folder with timestamp
        if parent_experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parent_experiment_name = f"OPTION_A_{timestamp}"

        self.parent_folder = self.base_dir / parent_experiment_name
        self.experiments_folder = self.parent_folder / "experiments"
        self.consolidated_analysis_folder = self.parent_folder / "consolidated_analysis"

        # Create parent structure
        self.parent_folder.mkdir(parents=True, exist_ok=True)
        self.experiments_folder.mkdir(parents=True, exist_ok=True)
        self.consolidated_analysis_folder.mkdir(parents=True, exist_ok=True)

        print(f"[OPTION_A_MANAGER] Created parent structure: {self.parent_folder}")
        self._create_readme()

    def add_experiment(self, experiment_name: str, dataset: str, models: List[str]) -> Path:
        """Add new experiment to parent folder structure"""
        experiment_path = self.experiments_folder / f"{experiment_name}_{dataset}"
        experiment_path.mkdir(parents=True, exist_ok=True)

        # Create experiment info file
        experiment_info = {
            "experiment_name": experiment_name,
            "dataset": dataset,
            "models": models,
            "created_at": datetime.now().isoformat(),
            "status": "running"
        }

        info_file = experiment_path / "experiment_info.yaml"
        with open(info_file, 'w') as f:
            yaml.dump(experiment_info, f)

        print(f"[OPTION_A_EXPERIMENT] Added: {experiment_path}")
        return experiment_path

    def create_consolidated_analysis(self, analysis_type: str = "multi_dataset") -> Path:
        """Create consolidated analysis folder"""
        analysis_path = self.consolidated_analysis_folder / analysis_type
        analysis_path.mkdir(parents=True, exist_ok=True)
        print(f"[OPTION_A_ANALYSIS] Created: {analysis_path}")
        return analysis_path

    def _create_readme(self):
        """Create README for parent experiment folder"""
        readme_content = f"""# Option A Experiment Results

**Created**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Experiment Type**: Option A - Shared Classification Architecture

## Structure
- `experiments/`: Individual dataset experiments
- `consolidated_analysis/`: Cross-dataset analysis results

## Benefits of Option A
- ~70% storage reduction compared to traditional approach
- ~60% training time reduction through shared classification
- Consistent classification models across all detection methods

Generated by Option A Results Manager
"""
        readme_file = self.parent_folder / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)

    @property
    def parent_experiment_name(self) -> str:
        """Get parent experiment name"""
        return self.parent_folder.name


def get_results_manager(config_path: str = None, pipeline_name: str = None) -> ResultsManager:
    """Factory function to get appropriate results manager"""
    if config_path is None:
        config_path = "config/results_structure.yaml"

    return ResultsManager(config_path=config_path, pipeline_name=pipeline_name)


def create_option_a_manager(parent_experiment_name: str = None) -> OptionAResultsManager:
    """Factory function to create Option A results manager"""
    return OptionAResultsManager(parent_experiment_name=parent_experiment_name)


# Standalone functions for backward compatibility
def get_experiment_path(experiment_type: str, model_name: str, experiment_name: str = None, pipeline_name: str = None) -> Path:
    """Standalone function for getting experiment path"""
    manager = get_results_manager(pipeline_name=pipeline_name)
    return manager.get_experiment_path(experiment_type, model_name, experiment_name)


def get_crops_path(detection_model: str, experiment_name: str, pipeline_name: str = None) -> Path:
    """Standalone function for getting crops path"""
    manager = get_results_manager(pipeline_name=pipeline_name)
    return manager.get_crops_path(detection_model, experiment_name)


def create_crops_path(detection_model: str, experiment_name: str, pipeline_name: str = None) -> Path:
    """Standalone function for creating crops path"""
    manager = get_results_manager(pipeline_name=pipeline_name)
    return manager.create_crops_path(detection_model, experiment_name)


def get_analysis_path(analysis_type: str = "general", pipeline_name: str = None) -> Path:
    """Standalone function for getting analysis path"""
    manager = get_results_manager(pipeline_name=pipeline_name)
    return manager.get_analysis_path(analysis_type)


def create_analysis_path(analysis_type: str = "general", pipeline_name: str = None) -> Path:
    """Standalone function for creating analysis path"""
    manager = get_results_manager(pipeline_name=pipeline_name)
    return manager.create_analysis_path(analysis_type)