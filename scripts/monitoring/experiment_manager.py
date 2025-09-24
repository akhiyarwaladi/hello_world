#!/usr/bin/env python3
"""
Clear Experiment Management System
Eliminates ambiguity in experiment tracking and organization
"""

import os
import json
from pathlib import Path
from datetime import datetime
import shutil

class ExperimentManager:
    def __init__(self, base_dir="results"):
        self.base_dir = Path(base_dir)
        self.current_experiments = self.base_dir / "current_experiments"
        self.completed_experiments = self.base_dir / "completed_experiments"
        self.active_experiments = self.base_dir / "active_experiments"

        # Create directories
        for dir_path in [self.current_experiments, self.completed_experiments, self.active_experiments]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_experiment(self, model_type, experiment_base_name, config=None):
        """Create a new experiment with timestamp and clear naming"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{experiment_base_name}_{timestamp}"

        # Create experiment directory
        exp_path = self.active_experiments / model_type / experiment_name
        exp_path.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = {
            "experiment_name": experiment_name,
            "model_type": model_type,
            "base_name": experiment_base_name,
            "created_at": timestamp,
            "status": "active",
            "config": config or {},
            "weights_saved": False,
            "epochs_completed": 0,
            "target_epochs": config.get("epochs", 50) if config else 50
        }

        # Save metadata
        with open(exp_path / "experiment_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[SUCCESS] Created experiment: {experiment_name}")
        print(f"Location: {exp_path}")

        return exp_path, experiment_name

    def mark_completed(self, experiment_path, success=True):
        """Mark experiment as completed and move to appropriate folder"""
        exp_path = Path(experiment_path)
        metadata_file = exp_path / "experiment_metadata.json"

        if not metadata_file.exists():
            print(f"[ERROR] No metadata found for {exp_path}")
            return False

        # Load metadata
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Update metadata
        metadata["completed_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata["status"] = "completed" if success else "failed"

        # Check if weights exist
        weights_dir = exp_path / "weights"
        if weights_dir.exists():
            best_weights = weights_dir / "best.pt"
            last_weights = weights_dir / "last.pt"
            metadata["weights_saved"] = best_weights.exists() and last_weights.exists()

        # Count epochs
        results_csv = exp_path / "results.csv"
        if results_csv.exists():
            with open(results_csv, "r") as f:
                lines = f.readlines()
                metadata["epochs_completed"] = len(lines) - 1  # Minus header

        # Save updated metadata
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Move to completed folder
        target_dir = self.completed_experiments / metadata["model_type"]
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / exp_path.name
        shutil.move(str(exp_path), str(target_path))

        status_icon = "[OK]" if success else "[FAIL]"
        print(f"{status_icon} Experiment marked as {'completed' if success else 'failed'}")
        print(f"Moved to: {target_path}")

        return True

    def list_active_experiments(self):
        """List all currently active experiments"""
        print("\nACTIVE EXPERIMENTS:")
        print("=" * 50)

        active_found = False
        for model_dir in self.active_experiments.iterdir():
            if model_dir.is_dir():
                for exp_dir in model_dir.iterdir():
                    if exp_dir.is_dir():
                        metadata_file = exp_dir / "experiment_metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)

                            print(f"{metadata['experiment_name']}")
                            print(f"   Model: {metadata['model_type']}")
                            print(f"   Created: {metadata['created_at']}")
                            print(f"   Progress: {metadata.get('epochs_completed', 0)}/{metadata.get('target_epochs', 50)} epochs")

                            # Check if training is still running
                            weights_exist = (exp_dir / "weights").exists()
                            results_exist = (exp_dir / "results.csv").exists()

                            if weights_exist and results_exist:
                                print(f"   Status: Training (weights saved)")
                            elif results_exist:
                                print(f"   Status: [WARNING] Training (no weights yet)")
                            else:
                                print(f"   Status: Starting")

                            print(f"   Path: {exp_dir}")
                            print()
                            active_found = True

        if not active_found:
            print("   No active experiments found")

        print("=" * 50)

    def list_completed_experiments(self, limit=10):
        """List recently completed experiments"""
        print(f"\n[SUCCESS] COMPLETED EXPERIMENTS (last {limit}):")
        print("=" * 50)

        completed_experiments = []
        for model_dir in self.completed_experiments.iterdir():
            if model_dir.is_dir():
                for exp_dir in model_dir.iterdir():
                    if exp_dir.is_dir():
                        metadata_file = exp_dir / "experiment_metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)
                            metadata["path"] = exp_dir
                            completed_experiments.append(metadata)

        # Sort by completion time
        completed_experiments.sort(key=lambda x: x.get("completed_at", ""), reverse=True)

        for i, exp in enumerate(completed_experiments[:limit]):
            status_icon = "[OK]" if exp["status"] == "completed" else "[FAIL]"
            weights_icon = "[SAVE]" if exp.get("weights_saved", False) else "[NONE]"

            print(f"{i+1}. {status_icon} {exp['experiment_name']}")
            print(f"   Model: {exp['model_type']}")
            print(f"   Epochs: {exp.get('epochs_completed', 0)}/{exp.get('target_epochs', 50)}")
            print(f"   Weights: {weights_icon}")
            print(f"   Completed: {exp.get('completed_at', 'Unknown')}")
            print(f"   Path: {exp['path']}")
            print()

        print("=" * 50)

    def cleanup_ambiguous_experiments(self):
        """Clean up old experiment structure and organize properly"""
        print("\nCLEANING UP AMBIGUOUS EXPERIMENTS:")
        print("=" * 50)

        # Find old structure experiments
        old_current = self.base_dir / "current_experiments"
        old_completed = self.base_dir / "completed_models"

        cleanup_count = 0

        # Process old current_experiments
        if old_current.exists():
            for category in ["training", "validation"]:
                category_path = old_current / category
                if category_path.exists():
                    for model_type in category_path.iterdir():
                        if model_type.is_dir():
                            for exp_dir in model_type.iterdir():
                                if exp_dir.is_dir() and not (exp_dir / "experiment_metadata.json").exists():
                                    # This is an old experiment, add metadata
                                    self._add_metadata_to_old_experiment(exp_dir, model_type.name, category)
                                    cleanup_count += 1

        print(f"[SUCCESS] Processed {cleanup_count} old experiments")
        print("=" * 50)

    def _add_metadata_to_old_experiment(self, exp_path, model_type, category):
        """Add metadata to old experiments"""
        # Determine status based on weights existence
        weights_dir = exp_path / "weights"
        has_weights = weights_dir.exists() and any(weights_dir.glob("*.pt"))

        # Count epochs from results.csv
        epochs_completed = 0
        results_csv = exp_path / "results.csv"
        if results_csv.exists():
            with open(results_csv, "r") as f:
                lines = f.readlines()
                epochs_completed = len(lines) - 1

        # Create metadata
        metadata = {
            "experiment_name": exp_path.name,
            "model_type": model_type,
            "base_name": exp_path.name,
            "created_at": "unknown",
            "completed_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status": "completed" if has_weights else "failed",
            "category": category,
            "weights_saved": has_weights,
            "epochs_completed": epochs_completed,
            "target_epochs": 50,
            "migrated_from_old_structure": True
        }

        # Save metadata
        with open(exp_path / "experiment_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Added metadata to: {exp_path.name}")

def main():
    manager = ExperimentManager()

    print("EXPERIMENT MANAGER")
    print("1. List active experiments")
    print("2. List completed experiments")
    print("3. Cleanup old structure")
    print("4. Full status report")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == "1":
        manager.list_active_experiments()
    elif choice == "2":
        manager.list_completed_experiments()
    elif choice == "3":
        manager.cleanup_ambiguous_experiments()
    elif choice == "4":
        manager.list_active_experiments()
        manager.list_completed_experiments()
        print("\n" + "="*50)
        print("SUMMARY: Use this tool to track experiments clearly!")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()