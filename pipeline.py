#!/usr/bin/env python3
"""
Unified Malaria Detection Pipeline
Consolidates all training, validation, and evaluation functionality
"""

import os
import sys
import time
import yaml
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import existing utilities
sys.path.append(str(Path(__file__).parent))
from utils.experiment_logger import ExperimentLogger, ResultsCollector
from utils.results_manager import ResultsManager

class MalariaPipeline:
    """Unified pipeline for malaria detection training and evaluation"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.project_root = Path(".")

        # Load configurations
        self.models_config = self._load_config("models.yaml")
        self.datasets_config = self._load_config("datasets.yaml")

        # Setup organized results management
        self.results_manager = ResultsManager()

        # Setup directories (now organized)
        self.results_dir = Path("results/current_experiments")
        self.logs_dir = Path("results/experiment_logs")

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, filename: str) -> Dict:
        """Load YAML configuration file"""
        config_path = self.config_dir / filename
        if not config_path.exists():
            print(f"⚠️  Warning: Config file {config_path} not found")
            return {}

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.models_config.get("models", {}).keys())

    def list_datasets(self) -> List[str]:
        """List all available processed datasets"""
        return list(self.datasets_config.get("processed_datasets", {}).keys())

    def validate_prerequisites(self) -> Dict[str, bool]:
        """Validate that all prerequisites are met"""
        status = {
            "datasets_exist": True,
            "scripts_exist": True,
            "environment_ready": True
        }

        # Check datasets
        for dataset_name, dataset_config in self.datasets_config.get("processed_datasets", {}).items():
            dataset_path = Path(dataset_config["path"])
            if not dataset_path.exists():
                print(f"❌ Dataset not found: {dataset_path}")
                status["datasets_exist"] = False

        # Check scripts
        for model_name, model_config in self.models_config.get("models", {}).items():
            script = model_config.get("script", "").split()[0]
            if script.startswith("python") and len(script.split()) > 1:
                script_path = Path(script.split()[1])
                if not script_path.exists():
                    print(f"❌ Script not found: {script_path}")
                    status["scripts_exist"] = False

        return status

    def quick_validate(self, models: List[str] = None, timeout: int = 300) -> Dict[str, bool]:
        """Quick validation of models (2 epochs each)"""
        if models is None:
            models = self.list_models()
        elif isinstance(models, str):
            if models == "all":
                models = self.list_models()
            else:
                models = [models]

        print("🧪 QUICK MODEL VALIDATION")
        print("=" * 50)
        print(f"Testing {len(models)} models with 2 epochs each")
        print("=" * 50)

        results = {}

        # Check prerequisites first
        prereq_status = self.validate_prerequisites()
        if not all(prereq_status.values()):
            print("❌ Prerequisites not met. Cannot proceed with validation.")
            return {}

        for model_name in models:
            if model_name not in self.models_config.get("models", {}):
                print(f"⚠️  Unknown model: {model_name}")
                results[model_name] = False
                continue

            print(f"\n🔍 Testing {model_name}...")
            success = self._test_model_quick(model_name, timeout)
            results[model_name] = success

            status_emoji = "✅" if success else "❌"
            print(f"{status_emoji} {model_name}: {'PASS' if success else 'FAIL'}")

        # Summary
        passed = sum(results.values())
        total = len(results)

        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"✅ Passed: {passed}/{total}")
        print(f"❌ Failed: {total - passed}")

        if passed == total:
            print("\n🎉 All models validated successfully!")
        else:
            print(f"\n⚠️  {total - passed} model(s) failed validation")

        return results

    def _test_model_quick(self, model_name: str, timeout: int) -> bool:
        """Test a single model with 2 epochs"""
        model_config = self.models_config["models"][model_name]
        script = model_config["script"]
        validation_args = model_config.get("validation_args", {})

        # Build command
        if script.startswith("python"):
            cmd = script.split()
        else:
            cmd = [script]

        # Add validation arguments
        for key, value in validation_args.items():
            if script.startswith("python"):
                cmd.extend([f"--{key}", str(value)])
            else:
                cmd.append(f"{key}={value}")

        # Add name for validation
        validation_name = f"validation_{model_name}_{int(time.time())}"
        if script.startswith("python"):
            cmd.extend(["--name", validation_name])
        else:
            cmd.append(f"name={validation_name}")

        # Run test
        start_time = time.time()
        try:
            env = os.environ.copy()
            if "NNPACK_DISABLE" in model_config.get("env", {}):
                env["NNPACK_DISABLE"] = "1"

            result = subprocess.run(
                cmd,
                env=env,
                timeout=timeout,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                print(f"  ✅ Success ({duration:.1f}s)")
                return True
            else:
                print(f"  ❌ Failed ({duration:.1f}s)")
                if result.stderr:
                    print(f"  Error: {result.stderr[-200:]}")
                return False

        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout ({timeout}s)")
            return False
        except Exception as e:
            print(f"  💥 Error: {str(e)}")
            return False

    def train(self, model: str, epochs: int = None, batch: int = None,
              device: str = None, name: str = None, background: bool = False) -> bool:
        """Train a specific model"""

        if model not in self.models_config.get("models", {}):
            print(f"❌ Unknown model: {model}")
            return False

        model_config = self.models_config["models"][model]
        script = model_config["script"]
        args = model_config.get("args", {}).copy()

        # Override with provided parameters
        if epochs is not None:
            args["epochs"] = epochs
        if batch is not None:
            args["batch"] = batch
        if device is not None:
            args["device"] = device
        if name is not None:
            args["name"] = name
        else:
            args["name"] = f"{model}_{int(time.time())}"

        # Build command
        if script.startswith("python"):
            cmd = script.split()
            for key, value in args.items():
                cmd.extend([f"--{key}", str(value)])
        else:
            cmd = [script]
            for key, value in args.items():
                cmd.append(f"{key}={value}")

        print(f"🚀 Training {model}")
        print(f"📝 Command: {' '.join(cmd)}")

        if background:
            # Run in background
            env = os.environ.copy()
            process = subprocess.Popen(cmd, env=env, cwd=self.project_root)
            print(f"🔄 Training started in background (PID: {process.pid})")
            return True
        else:
            # Run in foreground
            try:
                result = subprocess.run(cmd, cwd=self.project_root)
                success = result.returncode == 0

                if success:
                    print(f"✅ Training completed successfully")
                else:
                    print(f"❌ Training failed")

                return success

            except KeyboardInterrupt:
                print(f"\n⏹️  Training interrupted by user")
                return False
            except Exception as e:
                print(f"❌ Training error: {str(e)}")
                return False

    def evaluate(self, models: List[str] = None, comprehensive: bool = False) -> Dict[str, Any]:
        """Evaluate trained models"""
        if models is None:
            models = self.list_models()
        elif isinstance(models, str):
            if models == "all":
                models = self.list_models()
            else:
                models = [models]

        print("📊 MODEL EVALUATION")
        print("=" * 50)

        if comprehensive:
            print("Running comprehensive evaluation...")
            # Use existing test_models_comprehensive.py functionality
            return self._comprehensive_evaluation(models)
        else:
            print("Running quick evaluation...")
            return self._quick_evaluation(models)

    def _comprehensive_evaluation(self, models: List[str]) -> Dict[str, Any]:
        """Run comprehensive evaluation using existing ComprehensiveModelTester"""
        try:
            # Import and use existing comprehensive tester
            from test_models_comprehensive import ComprehensiveModelTester

            tester = ComprehensiveModelTester(results_dir=str(self.results_dir / "comprehensive_evaluation"))

            results = {}
            for model_name in models:
                print(f"\n🔍 Comprehensive evaluation of {model_name}...")

                # Find trained model weights
                model_weights = self._find_model_weights(model_name)
                if not model_weights:
                    print(f"⚠️  No trained weights found for {model_name}")
                    continue

                model_config = self.models_config["models"][model_name]

                if model_config["type"] == "detection":
                    # Test detection model
                    dataset_config = self.datasets_config["processed_datasets"]["detection_multispecies"]
                    result = tester.test_detection_model(
                        model_path=model_weights,
                        test_dataset=dataset_config["path"],
                        experiment_name=f"eval_{model_name}"
                    )

                elif model_config["type"] == "classification":
                    # Test classification model
                    dataset_config = self.datasets_config["processed_datasets"]["classification_multispecies"]
                    result = tester.test_classification_model(
                        model_path=model_weights,
                        test_dataset=dataset_config["path"],
                        class_names=dataset_config["class_names"],
                        experiment_name=f"eval_{model_name}"
                    )

                results[model_name] = result

            return results

        except ImportError:
            print("❌ Comprehensive evaluation requires test_models_comprehensive.py")
            return {}

    def _quick_evaluation(self, models: List[str]) -> Dict[str, Any]:
        """Quick evaluation of models"""
        results = {}

        for model_name in models:
            model_weights = self._find_model_weights(model_name)
            if model_weights:
                results[model_name] = {
                    "status": "found_weights",
                    "weights_path": str(model_weights),
                    "file_size_mb": model_weights.stat().st_size / (1024*1024)
                }
            else:
                results[model_name] = {"status": "no_weights"}

        return results

    def _find_model_weights(self, model_name: str) -> Optional[Path]:
        """Find trained model weights"""
        # Common locations for trained weights
        search_paths = [
            self.results_dir / f"{model_name}" / "weights" / "best.pt",
            self.results_dir / "weights" / f"{model_name}_best.pt",
            Path("runs") / "detect" / f"{model_name}" / "weights" / "best.pt",
            Path("runs") / "classify" / f"{model_name}" / "weights" / "best.pt",
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "models_available": len(self.list_models()),
            "datasets_available": len(self.list_datasets()),
            "prerequisites": self.validate_prerequisites(),
            "trained_models": {},
            "running_processes": self._get_running_processes()
        }

        # Check for trained models
        for model_name in self.list_models():
            weights_path = self._find_model_weights(model_name)
            status["trained_models"][model_name] = {
                "trained": weights_path is not None,
                "weights_path": str(weights_path) if weights_path else None
            }

        return status

    def _get_running_processes(self) -> List[Dict[str, Any]]:
        """Get currently running training processes"""
        try:
            # Simple process detection for common training commands
            result = subprocess.run(
                ["pgrep", "-f", "python.*train"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                return [{"pid": pid, "type": "training"} for pid in pids if pid]

        except Exception:
            pass

        return []

    def export_results(self, format_type: str = "journal") -> str:
        """Export results in specified format"""
        if format_type == "journal":
            # Use existing journal export functionality
            try:
                from generate_journal_report import JournalReportGenerator

                generator = JournalReportGenerator()
                generator.generate_complete_report()

                return "journal_publication/"

            except ImportError:
                print("❌ Journal export requires generate_journal_report.py")
                return ""

        return ""

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Unified Malaria Detection Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show pipeline status")

    # List command
    list_parser = subparsers.add_parser("list", help="List available models/datasets")
    list_parser.add_argument("--models", action="store_true", help="List models")
    list_parser.add_argument("--datasets", action="store_true", help="List datasets")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Quick validation of models")
    validate_parser.add_argument("--models", default="all", help="Models to validate (default: all)")
    validate_parser.add_argument("--timeout", type=int, default=300, help="Timeout per model (seconds)")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("model", help="Model to train")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--batch", type=int, help="Batch size")
    train_parser.add_argument("--device", help="Device (cpu/cuda)")
    train_parser.add_argument("--name", help="Experiment name")
    train_parser.add_argument("--background", action="store_true", help="Run in background")

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    evaluate_parser.add_argument("--models", default="all", help="Models to evaluate")
    evaluate_parser.add_argument("--comprehensive", action="store_true", help="Comprehensive evaluation")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export results")
    export_parser.add_argument("--format", default="journal", help="Export format")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize pipeline
    pipeline = MalariaPipeline()

    # Execute commands
    if args.command == "status":
        status = pipeline.status()
        print(json.dumps(status, indent=2))

    elif args.command == "list":
        if args.models or (not args.models and not args.datasets):
            print("📋 Available Models:")
            for model in pipeline.list_models():
                print(f"  - {model}")

        if args.datasets or (not args.models and not args.datasets):
            print("📋 Available Datasets:")
            for dataset in pipeline.list_datasets():
                print(f"  - {dataset}")

    elif args.command == "validate":
        models = args.models.split(",") if args.models != "all" else "all"
        results = pipeline.quick_validate(models, args.timeout)

    elif args.command == "train":
        success = pipeline.train(
            model=args.model,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            name=args.name,
            background=args.background
        )

        if not success:
            sys.exit(1)

    elif args.command == "evaluate":
        models = args.models.split(",") if args.models != "all" else "all"
        results = pipeline.evaluate(models, args.comprehensive)
        print(json.dumps(results, indent=2, default=str))

    elif args.command == "export":
        output_dir = pipeline.export_results(args.format)
        if output_dir:
            print(f"✅ Results exported to: {output_dir}")
        else:
            print("❌ Export failed")

if __name__ == "__main__":
    main()