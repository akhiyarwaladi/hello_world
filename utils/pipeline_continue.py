#!/usr/bin/env python3
"""
Pipeline Continue/Resume Functionality
Provides smart stage detection and resume capabilities for the multiple models pipeline
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

def check_completed_stages(experiment_dir: str) -> Dict[str, bool]:
    """
    Auto-detect which stages are completed in an experiment

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary with stage completion status
    """
    exp_path = Path(experiment_dir)
    stages = {
        'detection': False,
        'crop': False,
        'classification': False,
        'analysis': False
    }

    if not exp_path.exists():
        return stages

    # Stage 1: Detection - check for best.pt files in detection subfolders
    detection_weights = list(exp_path.glob("detection/*/*/weights/best.pt"))
    stages['detection'] = len(detection_weights) > 0

    # Stage 2: Crops - check for crop directories with data
    crop_dirs = list(exp_path.glob("crop_data/crops_from_*"))
    if crop_dirs:
        # Verify crop directories have train/val/test structure with files
        for crop_dir in crop_dirs:
            crops_path = crop_dir / "crops"
            if crops_path.exists():
                train_dirs = list(crops_path.glob("train/*"))
                if train_dirs and any(list(d.glob("*")) for d in train_dirs if d.is_dir()):
                    stages['crop'] = True
                    break
            # Also check direct structure without 'crops' subfolder
            elif crop_dir.exists():
                train_dirs = list(crop_dir.glob("train/*"))
                if train_dirs and any(list(d.glob("*")) for d in train_dirs if d.is_dir()):
                    stages['crop'] = True
                    break

    # Stage 3: Classification - check for classification model directories
    cls_dirs = list(exp_path.glob("models/*"))
    stages['classification'] = len(cls_dirs) > 0

    # Stage 4: Analysis - check for analysis files
    analysis_files = list(exp_path.glob("analysis/*.json"))
    analysis_dirs = list(exp_path.glob("analysis"))
    stages['analysis'] = len(analysis_files) > 0 or len(analysis_dirs) > 0

    return stages

def determine_next_stage(completed_stages: Dict[str, bool]) -> str:
    """
    Auto-determine next stage to run based on completed stages

    Args:
        completed_stages: Dictionary of stage completion status

    Returns:
        Next stage to execute
    """
    if not completed_stages['detection']:
        return 'detection'
    elif not completed_stages['crop']:
        return 'crop'
    elif not completed_stages['classification']:
        return 'classification'
    else:
        return 'analysis'

def get_available_experiments() -> List[str]:
    """
    Get list of available experiments that can be continued

    Returns:
        List of experiment directory names
    """
    results_path = Path("results")
    if not results_path.exists():
        return []

    experiments = []
    for item in results_path.iterdir():
        if item.is_dir() and item.name.startswith("exp_"):
            experiments.append(item.name)

    experiments.sort(reverse=True)  # Most recent first
    return experiments

def find_detection_models(experiment_dir: str) -> Dict[str, List[Path]]:
    """
    Find all detection models in an experiment

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary mapping model types to list of model paths
    """
    exp_path = Path(experiment_dir)
    models = {}

    detection_path = exp_path / "detection"
    if detection_path.exists():
        for model_dir in detection_path.iterdir():
            if model_dir.is_dir():
                model_type = model_dir.name.replace("_detection", "")
                weights = list(model_dir.glob("*/weights/best.pt"))
                if weights:
                    models[model_type] = weights

    return models

def find_crop_data(experiment_dir: str) -> List[Path]:
    """
    Find all crop data directories in an experiment

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        List of crop data directory paths
    """
    exp_path = Path(experiment_dir)
    crop_data_path = exp_path / "crop_data"

    if not crop_data_path.exists():
        return []

    return list(crop_data_path.glob("crops_from_*"))

def save_experiment_metadata(experiment_dir: str, args, stage_info: Dict = None):
    """
    Save experiment metadata for continue functionality

    Args:
        experiment_dir: Path to experiment directory
        args: Command line arguments
        stage_info: Additional stage information
    """
    exp_path = Path(experiment_dir)
    exp_path.mkdir(parents=True, exist_ok=True)

    # Load existing metadata or create new
    metadata_file = exp_path / 'pipeline_metadata.json'
    metadata = {}
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            metadata = {}

    # Update metadata
    current_time = datetime.now().isoformat()

    if 'created_at' not in metadata:
        metadata['created_at'] = current_time

    metadata.update({
        'last_updated': current_time,
        'original_args': vars(args),
        'completed_stages': check_completed_stages(experiment_dir),
        'pipeline_version': '2.0_continue',
        'stage_info': stage_info or {}
    })

    # Add continue history
    if 'continue_history' not in metadata:
        metadata['continue_history'] = []

    if hasattr(args, 'continue_from') and args.continue_from:
        metadata['continue_history'].append({
            'timestamp': current_time,
            'continued_from_stage': getattr(args, 'start_stage', 'auto'),
            'args_override': {k: v for k, v in vars(args).items()
                           if k not in ['continue_from', 'start_stage']}
        })

    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_experiment_metadata(experiment_dir: str) -> Dict:
    """
    Load existing experiment metadata

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Metadata dictionary
    """
    metadata_file = Path(experiment_dir) / 'pipeline_metadata.json'
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return {}

def merge_parameters(original_args: Dict, new_args) -> Dict:
    """
    Smart parameter merging with conflict detection

    Args:
        original_args: Original experiment arguments
        new_args: New command line arguments

    Returns:
        Merged arguments dictionary
    """
    merged = original_args.copy()
    new_args_dict = vars(new_args)

    # Critical parameters that should warn on change
    critical_params = [
        'epochs_det', 'epochs_cls', 'use_kaggle_dataset',
        'classification_models', 'include', 'exclude_detection'
    ]

    critical_changes = []
    for param in critical_params:
        old_val = original_args.get(param)
        new_val = new_args_dict.get(param)

        if new_val is not None and old_val != new_val:
            critical_changes.append(f"{param}: {old_val} → {new_val}")

    # Show warnings for critical changes
    if critical_changes:
        print("[WARNING] Parameter changes detected:")
        for change in critical_changes:
            print(f"   {change}")
        print()

    # Merge parameters (new overrides old)
    for key, value in new_args_dict.items():
        if value is not None and key not in ['continue_from', 'start_stage']:
            merged[key] = value

    return merged

def validate_experiment_dir(experiment_dir: str) -> bool:
    """
    Validate that experiment directory exists and has valid structure

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        True if valid, False otherwise
    """
    exp_path = Path(experiment_dir)

    if not exp_path.exists():
        print(f"[ERROR] Experiment directory not found: {experiment_dir}")
        return False

    if not exp_path.is_dir():
        print(f"[ERROR] Path is not a directory: {experiment_dir}")
        return False

    # Check if it looks like an experiment directory
    expected_subdirs = ['detection', 'crop_data', 'models']
    has_any_subdir = any((exp_path / subdir).exists() for subdir in expected_subdirs)

    # FIX: Also check for multi-dataset parent structure
    is_multi_dataset_parent = (exp_path / 'experiments').exists() and (exp_path / 'experiments').is_dir()

    if not has_any_subdir and not is_multi_dataset_parent:
        print(f"[WARNING] Directory doesn't appear to be an experiment (no detection/crop_data/models or experiments dirs)")
        print(f"   Contents: {[item.name for item in exp_path.iterdir()]}")

        import sys
        # Skip confirmation if not in interactive terminal
        if not sys.stdin.isatty():
            print(f"[AUTO] Non-interactive mode, skipping validation")
            return True

        confirm = input("Continue anyway? (y/n): ")
        if confirm.lower() != 'y':
            return False

    return True

def select_model_for_continue(experiment_dir: str, model_type: str) -> Optional[Path]:
    """
    Smart model selection for continue functionality

    Args:
        experiment_dir: Path to experiment directory
        model_type: Type of model to select (e.g., 'yolo11')

    Returns:
        Path to selected model or None
    """
    models = find_detection_models(experiment_dir)

    if model_type not in models:
        return None

    model_paths = models[model_type]

    if len(model_paths) == 1:
        return model_paths[0]
    elif len(model_paths) > 1:
        print(f"Multiple {model_type} models found:")
        for i, model_path in enumerate(model_paths):
            print(f"  {i+1}. {model_path}")

        try:
            choice = int(input("Select model (number): ")) - 1
            if 0 <= choice < len(model_paths):
                return model_paths[choice]
        except (ValueError, IndexError):
            pass

        print("Invalid selection, using first model")
        return model_paths[0]

    return None

def print_experiment_status(experiment_dir: str):
    """
    Print detailed status of an experiment

    Args:
        experiment_dir: Path to experiment directory
    """
    print(f"[STATUS] Experiment Status: {experiment_dir}")
    print("=" * 60)

    # Load metadata
    metadata = load_experiment_metadata(experiment_dir)
    if metadata:
        print(f"Created: {metadata.get('created_at', 'Unknown')}")
        print(f"Last Updated: {metadata.get('last_updated', 'Unknown')}")
        print(f"Pipeline Version: {metadata.get('pipeline_version', 'Unknown')}")
        print()

    # Check completed stages
    stages = check_completed_stages(experiment_dir)
    print("Stage Completion:")
    for stage, completed in stages.items():
        status = "[OK]" if completed else "[MISSING]"
        print(f"  {status} {stage.capitalize()}")

    print()

    # Show available models
    models = find_detection_models(experiment_dir)
    if models:
        print("Available Detection Models:")
        for model_type, paths in models.items():
            print(f"  [MODELS] {model_type}: {len(paths)} model(s)")

    # Show crop data
    crop_dirs = find_crop_data(experiment_dir)
    if crop_dirs:
        print(f"Crop Data: {len(crop_dirs)} dataset(s)")

    # Determine next stage
    next_stage = determine_next_stage(stages)
    print(f"[NEXT] Recommended next stage: {next_stage}")
    print()

def list_available_experiments():
    """List all available experiments that can be continued"""
    experiments = get_available_experiments()

    if not experiments:
        print("No experiments found in results/ directory")
        return

    print("[EXPERIMENTS] Available Experiments:")
    print("=" * 50)

    for exp in experiments[:10]:  # Show latest 10
        exp_path = Path("results") / exp
        stages = check_completed_stages(str(exp_path))
        completed_count = sum(stages.values())

        print(f"  {exp}")
        print(f"     Stages completed: {completed_count}/4")
        print(f"     Next: {determine_next_stage(stages)}")
        print()

    if len(experiments) > 10:
        print(f"... and {len(experiments) - 10} more experiments")