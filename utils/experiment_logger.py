#!/usr/bin/env python3
"""
Integrated Experiment Logger for Malaria Detection Pipeline
Automatically captures and stores all training/testing results with metadata
"""

import json
import time
import psutil
import platform
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class ExperimentLogger:
    def __init__(self, experiment_name: str, experiment_type: str, output_dir: str = "experiment_logs"):
        self.experiment_name = experiment_name
        self.experiment_type = experiment_type  # 'detection', 'classification', 'validation'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.log_file = self.output_dir / f"{experiment_name}_{int(time.time())}.json"
        self.start_time = time.time()

        # Initialize log structure
        self.log_data = {
            "experiment_info": {
                "name": experiment_name,
                "type": experiment_type,
                "start_time": datetime.now().isoformat(),
                "status": "running"
            },
            "system_info": self._get_system_info(),
            "dataset_info": {},
            "model_config": {},
            "training_results": {},
            "validation_results": {},
            "test_results": {},
            "performance_metrics": {},
            "artifacts": [],
            "errors": []
        }

        # Save initial log
        self._save_log()

    def _get_system_info(self) -> Dict[str, Any]:
        """Capture system information"""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_space_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
        }

    def log_dataset_info(self, dataset_path: str, num_classes: int, **kwargs):
        """Log dataset information"""
        self.log_data["dataset_info"] = {
            "dataset_path": dataset_path,
            "num_classes": num_classes,
            "logged_at": datetime.now().isoformat(),
            **kwargs
        }

        # Try to automatically detect dataset size
        self._auto_detect_dataset_size(dataset_path)
        self._save_log()

    def _auto_detect_dataset_size(self, dataset_path: str):
        """Automatically detect and log dataset statistics"""
        dataset_path = Path(dataset_path)

        if dataset_path.is_file() and dataset_path.suffix == '.yaml':
            # YOLO dataset yaml
            self._parse_yolo_dataset(dataset_path)
        elif dataset_path.is_dir():
            # Classification dataset directory
            self._parse_classification_dataset(dataset_path)

    def _parse_yolo_dataset(self, yaml_path: Path):
        """Parse YOLO dataset yaml to get statistics"""
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)

            base_dir = yaml_path.parent

            # Count images in train/val/test
            splits = {}
            for split in ['train', 'val', 'test']:
                split_dir = base_dir / split / 'images'
                if split_dir.exists():
                    splits[f'{split}_images'] = len(list(split_dir.glob('*.jpg'))) + len(list(split_dir.glob('*.png')))

            self.log_data["dataset_info"].update({
                "dataset_type": "detection",
                "classes": config.get('names', []),
                **splits
            })

        except Exception as e:
            self.log_error(f"Failed to parse YOLO dataset: {e}")

    def _parse_classification_dataset(self, dataset_dir: Path):
        """Parse classification dataset directory to get statistics"""
        try:
            class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

            class_stats = {}
            total_images = 0

            for class_dir in class_dirs:
                count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
                class_stats[class_dir.name] = count
                total_images += count

            self.log_data["dataset_info"].update({
                "dataset_type": "classification",
                "total_images": total_images,
                "class_distribution": class_stats,
                "num_classes": len(class_stats)
            })

        except Exception as e:
            self.log_error(f"Failed to parse classification dataset: {e}")

    def log_model_config(self, model_name: str, **config):
        """Log model configuration"""
        self.log_data["model_config"] = {
            "model_name": model_name,
            "config": config,
            "logged_at": datetime.now().isoformat()
        }
        self._save_log()

    def log_training_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log training epoch results"""
        if "epochs" not in self.log_data["training_results"]:
            self.log_data["training_results"]["epochs"] = []

        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }

        self.log_data["training_results"]["epochs"].append(epoch_data)
        self._save_log()

    def log_validation_results(self, metrics: Dict[str, float], confusion_matrix: Optional[List[List]] = None):
        """Log validation results"""
        self.log_data["validation_results"] = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

        if confusion_matrix:
            self.log_data["validation_results"]["confusion_matrix"] = confusion_matrix

        self._save_log()

    def log_test_results(self, test_predictions: List[Dict], test_metrics: Dict[str, float]):
        """Log test results with predictions"""
        self.log_data["test_results"] = {
            "metrics": test_metrics,
            "num_predictions": len(test_predictions),
            "timestamp": datetime.now().isoformat()
        }

        # Save detailed predictions separately
        predictions_file = self.output_dir / f"{self.experiment_name}_predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(test_predictions, f, indent=2)

        self.log_artifact(str(predictions_file), "test_predictions")
        self._save_log()

    def log_final_results(self, final_metrics: Dict[str, float]):
        """Log final experiment results"""
        elapsed_time = time.time() - self.start_time

        self.log_data["performance_metrics"] = {
            "final_metrics": final_metrics,
            "training_time_seconds": elapsed_time,
            "training_time_minutes": elapsed_time / 60,
            "training_time_hours": elapsed_time / 3600,
            "completed_at": datetime.now().isoformat()
        }

        self.log_data["experiment_info"]["status"] = "completed"
        self._save_log()

    def log_artifact(self, file_path: str, artifact_type: str, description: str = ""):
        """Log artifact files (models, plots, results)"""
        artifact = {
            "file_path": file_path,
            "artifact_type": artifact_type,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "file_size_mb": self._get_file_size_mb(file_path)
        }

        self.log_data["artifacts"].append(artifact)
        self._save_log()

    def log_error(self, error_message: str, error_type: str = "general"):
        """Log errors and exceptions"""
        error = {
            "message": error_message,
            "type": error_type,
            "timestamp": datetime.now().isoformat()
        }

        self.log_data["errors"].append(error)
        self._save_log()

    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        try:
            return round(Path(file_path).stat().st_size / (1024*1024), 2)
        except:
            return 0.0

    def _save_log(self):
        """Save log data to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary"""
        return {
            "name": self.experiment_name,
            "type": self.experiment_type,
            "status": self.log_data["experiment_info"]["status"],
            "duration_minutes": (time.time() - self.start_time) / 60,
            "final_metrics": self.log_data.get("performance_metrics", {}).get("final_metrics", {}),
            "artifacts_count": len(self.log_data["artifacts"]),
            "errors_count": len(self.log_data["errors"])
        }

class ResultsCollector:
    """Collects and aggregates results from multiple experiments"""

    def __init__(self, logs_dir: str = "experiment_logs"):
        self.logs_dir = Path(logs_dir)

    def collect_all_results(self) -> Dict[str, Any]:
        """Collect results from all logged experiments"""

        all_results = {
            "collection_time": datetime.now().isoformat(),
            "experiments": [],
            "summary_statistics": {},
            "performance_comparison": {}
        }

        # Find all log files
        log_files = list(self.logs_dir.glob("*.json"))

        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    experiment_data = json.load(f)
                all_results["experiments"].append(experiment_data)
            except Exception as e:
                print(f"Failed to load {log_file}: {e}")

        # Generate summary statistics
        all_results["summary_statistics"] = self._generate_summary_stats(all_results["experiments"])

        # Generate performance comparison
        all_results["performance_comparison"] = self._generate_performance_comparison(all_results["experiments"])

        return all_results

    def _generate_summary_stats(self, experiments: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics across experiments"""

        stats = {
            "total_experiments": len(experiments),
            "experiment_types": {},
            "avg_training_time_hours": 0,
            "total_training_time_hours": 0
        }

        training_times = []

        for exp in experiments:
            exp_type = exp.get("experiment_info", {}).get("type", "unknown")
            stats["experiment_types"][exp_type] = stats["experiment_types"].get(exp_type, 0) + 1

            perf_metrics = exp.get("performance_metrics", {})
            if "training_time_hours" in perf_metrics:
                training_times.append(perf_metrics["training_time_hours"])

        if training_times:
            stats["avg_training_time_hours"] = round(np.mean(training_times), 2)
            stats["total_training_time_hours"] = round(sum(training_times), 2)

        return stats

    def _generate_performance_comparison(self, experiments: List[Dict]) -> Dict[str, Any]:
        """Generate performance comparison across experiments"""

        comparison = {
            "detection_experiments": [],
            "classification_experiments": [],
            "best_performers": {}
        }

        best_map50 = 0
        best_accuracy = 0
        best_map50_exp = None
        best_accuracy_exp = None

        for exp in experiments:
            final_metrics = exp.get("performance_metrics", {}).get("final_metrics", {})
            exp_name = exp.get("experiment_info", {}).get("name", "unknown")

            # Detection experiments
            if "map50" in final_metrics or "mAP@0.5" in str(final_metrics):
                map50 = final_metrics.get("map50", final_metrics.get("mAP@0.5", 0))
                comparison["detection_experiments"].append({
                    "name": exp_name,
                    "map50": map50,
                    "dataset_size": exp.get("dataset_info", {}).get("total_images", 0)
                })

                if map50 > best_map50:
                    best_map50 = map50
                    best_map50_exp = exp_name

            # Classification experiments
            if "accuracy" in final_metrics:
                accuracy = final_metrics["accuracy"]
                comparison["classification_experiments"].append({
                    "name": exp_name,
                    "accuracy": accuracy,
                    "dataset_size": exp.get("dataset_info", {}).get("total_images", 0)
                })

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_accuracy_exp = exp_name

        comparison["best_performers"] = {
            "best_detection": {"name": best_map50_exp, "map50": best_map50},
            "best_classification": {"name": best_accuracy_exp, "accuracy": best_accuracy}
        }

        return comparison

    def export_for_journal(self, output_dir: str = "journal_export"):
        """Export results in journal-ready format"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Collect all results
        results = self.collect_all_results()

        # Save complete results
        with open(output_path / "complete_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Create performance tables
        self._create_performance_tables(results, output_path)

        # Create summary report
        self._create_summary_report(results, output_path)

        print(f"Journal export completed: {output_path}")

    def _create_performance_tables(self, results: Dict, output_path: Path):
        """Create performance tables for journal"""

        # Detection performance table
        detection_data = results["performance_comparison"]["detection_experiments"]
        if detection_data:
            df_detection = pd.DataFrame(detection_data)
            df_detection.to_csv(output_path / "detection_performance_table.csv", index=False)

        # Classification performance table
        classification_data = results["performance_comparison"]["classification_experiments"]
        if classification_data:
            df_classification = pd.DataFrame(classification_data)
            df_classification.to_csv(output_path / "classification_performance_table.csv", index=False)

    def _create_summary_report(self, results: Dict, output_path: Path):
        """Create summary report"""

        summary = results["summary_statistics"]

        report = f"""# Experiment Results Summary

Generated: {results["collection_time"]}

## Overview
- Total Experiments: {summary["total_experiments"]}
- Total Training Time: {summary["total_training_time_hours"]:.2f} hours
- Average Training Time: {summary["avg_training_time_hours"]:.2f} hours

## Experiment Types
"""

        for exp_type, count in summary["experiment_types"].items():
            report += f"- {exp_type.title()}: {count} experiments\n"

        best_performers = results["performance_comparison"]["best_performers"]

        report += f"""
## Best Performers
- Best Detection: {best_performers["best_detection"]["name"]} (mAP@0.5: {best_performers["best_detection"]["map50"]:.3f})
- Best Classification: {best_performers["best_classification"]["name"]} (Accuracy: {best_performers["best_classification"]["accuracy"]:.3f})
"""

        with open(output_path / "experiment_summary.md", 'w') as f:
            f.write(report)

# Usage example functions to integrate with existing scripts
def create_detection_logger(experiment_name: str, dataset_path: str, model_config: Dict) -> ExperimentLogger:
    """Create logger for detection experiments"""
    logger = ExperimentLogger(experiment_name, "detection")
    logger.log_dataset_info(dataset_path, num_classes=1)  # Malaria detection = 1 class
    logger.log_model_config(**model_config)
    return logger

def create_classification_logger(experiment_name: str, dataset_path: str, num_classes: int, model_config: Dict) -> ExperimentLogger:
    """Create logger for classification experiments"""
    logger = ExperimentLogger(experiment_name, "classification")
    logger.log_dataset_info(dataset_path, num_classes=num_classes)
    logger.log_model_config(**model_config)
    return logger