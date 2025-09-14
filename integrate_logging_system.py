#!/usr/bin/env python3
"""
Integration script to add logging system to existing training scripts
Modifies existing scripts to automatically log all results
"""

import sys
from pathlib import Path
import re

class LoggingIntegrator:
    def __init__(self):
        self.scripts_dir = Path("scripts")
        self.backup_dir = Path("scripts_backup")
        self.backup_dir.mkdir(exist_ok=True)

    def integrate_all_scripts(self):
        """Integrate logging system into all training scripts"""

        print("🔧 INTEGRATING LOGGING SYSTEM INTO EXISTING SCRIPTS")
        print("=" * 60)

        # List of scripts to modify
        scripts_to_modify = [
            "train_yolo_detection.py",
            "train_classification_crops.py"
        ]

        for script_name in scripts_to_modify:
            script_path = self.scripts_dir / script_name
            if script_path.exists():
                print(f"📝 Modifying {script_name}...")
                self._backup_script(script_path)
                self._add_logging_to_script(script_path)
            else:
                print(f"⚠️  Script not found: {script_name}")

        # Modify train_multispecies.py
        main_script = Path("train_multispecies.py")
        if main_script.exists():
            print("📝 Modifying train_multispecies.py...")
            self._backup_script(main_script)
            self._add_logging_to_main_script(main_script)

        print("✅ Logging integration completed!")

    def _backup_script(self, script_path: Path):
        """Create backup of original script"""
        backup_path = self.backup_dir / f"{script_path.stem}_backup.py"
        backup_path.write_text(script_path.read_text())

    def _add_logging_to_script(self, script_path: Path):
        """Add logging imports and initialization to training script"""

        content = script_path.read_text()

        # Add import statement at the top
        if "from utils.experiment_logger import" not in content:
            # Find where to insert import
            import_insertion = content.find("import")
            if import_insertion == -1:
                import_insertion = 0

            # Find end of imports
            lines = content.split('\n')
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_end = i + 1

            # Insert logging import
            lines.insert(import_end, "from utils.experiment_logger import ExperimentLogger")
            content = '\n'.join(lines)

        # Modify main function to add logging
        content = self._modify_main_function_for_logging(content, script_path.stem)

        script_path.write_text(content)

    def _modify_main_function_for_logging(self, content: str, script_name: str) -> str:
        """Modify main function to include logging"""

        # Find main function or training loop
        if "def main(" in content:
            # Add logging initialization at the start of main function
            content = self._add_logging_to_main_function(content, script_name)
        elif "if __name__ ==" in content:
            # Add logging around the main execution
            content = self._add_logging_to_main_execution(content, script_name)

        return content

    def _add_logging_to_main_function(self, content: str, script_name: str) -> str:
        """Add logging to main function"""

        # Find the main function
        main_pattern = r"def main\([^)]*\):\s*\n"
        match = re.search(main_pattern, content)

        if match:
            # Insert logging initialization after function definition
            insert_pos = match.end()

            logging_code = f'''
    # Initialize experiment logger
    experiment_name = f"{script_name}_{{int(time.time())}}"
    logger = ExperimentLogger(experiment_name, "{script_name.split('_')[1]}")

    try:
        # Log dataset and model config (will be populated by script)
        logger.log_dataset_info(args.data if 'args' in locals() else "unknown",
                               num_classes=1 if "detection" in "{script_name}" else 4)

        # Start training with logging
'''

            content = content[:insert_pos] + logging_code + content[insert_pos:]

            # Add logging at the end of main function
            content = self._add_logging_completion(content)

        return content

    def _add_logging_completion(self, content: str) -> str:
        """Add logging completion at the end of training"""

        # Find return statements in main function
        lines = content.split('\n')
        modified_lines = []
        in_main_function = False
        main_indent = 0

        for line in lines:
            if "def main(" in line:
                in_main_function = True
                main_indent = len(line) - len(line.lstrip())

            elif in_main_function and line.strip().startswith("return") and len(line) - len(line.lstrip()) == main_indent + 4:
                # Add logging before return
                modified_lines.append("        # Log final results")
                modified_lines.append("        if 'results' in locals():")
                modified_lines.append("            logger.log_final_results({'training_completed': True})")
                modified_lines.append("        logger.log_artifact(str(save_dir / 'weights' / 'best.pt'), 'model')")

            elif in_main_function and not line.strip() and len([l for l in modified_lines if "def " in l and len(l) - len(l.lstrip()) == main_indent]) > 1:
                in_main_function = False

            modified_lines.append(line)

        return '\n'.join(modified_lines)

    def _add_logging_to_main_script(self, script_path: Path):
        """Add logging to train_multispecies.py"""

        content = script_path.read_text()

        # Add import if not present
        if "from utils.experiment_logger import" not in content:
            # Find the imports section
            lines = content.split('\n')
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_end = i + 1

            lines.insert(import_end, "from utils.experiment_logger import ExperimentLogger")
            content = '\n'.join(lines)

        # Add logging to each training function
        content = self._add_logging_to_training_functions(content)

        script_path.write_text(content)

    def _add_logging_to_training_functions(self, content: str) -> str:
        """Add logging to training functions in train_multispecies.py"""

        # Find training functions and add logging
        function_patterns = [
            "def run_yolov8_detection_training",
            "def run_yolov11_detection_training",
            "def run_rtdetr_detection_training",
            "def run_yolov8_classification_training",
            "def run_yolov11_classification_training"
        ]

        for pattern in function_patterns:
            if pattern in content:
                content = self._add_logging_to_function(content, pattern)

        return content

    def _add_logging_to_function(self, content: str, function_pattern: str) -> str:
        """Add logging to a specific function"""

        # Find function definition
        func_start = content.find(function_pattern)
        if func_start == -1:
            return content

        # Find the end of function definition line
        func_def_end = content.find('\n', func_start)
        if func_def_end == -1:
            return content

        # Find the first line of function body
        body_start = func_def_end + 1

        # Extract function name for logging
        func_name = function_pattern.replace("def ", "").replace("(", "")
        experiment_type = "detection" if "detection" in func_name else "classification"

        # Prepare logging code
        logging_init = f'''
    # Initialize logging
    logger = ExperimentLogger(f"{func_name}_{{int(time.time())}}", "{experiment_type}")

    try:
'''

        # Insert logging initialization
        content = content[:body_start] + logging_init + content[body_start:]

        # Find the end of function and add completion logging
        # This is simplified - in practice, you'd want more sophisticated parsing
        content = self._add_function_completion_logging(content, func_name)

        return content

    def _add_function_completion_logging(self, content: str, func_name: str) -> str:
        """Add completion logging to function"""

        # Find subprocess.run calls and add logging around them
        pattern = r"subprocess\.run\(cmd[^)]*\)"

        def replace_subprocess(match):
            original = match.group(0)
            return f'''
        # Log training start
        logger.log_model_config(model_name=cmd[1] if len(cmd) > 1 else "unknown")

        # Run training
        result = {original}

        # Log completion
        logger.log_final_results({{"subprocess_completed": True, "return_code": result.returncode}})

        return result'''

        content = re.sub(pattern, replace_subprocess, content)

        return content

    def create_example_usage(self):
        """Create example script showing how to use the logging system"""

        example_script = '''#!/usr/bin/env python3
"""
Example: How to use the integrated logging system
"""

import time
from utils.experiment_logger import ExperimentLogger

def example_training_with_logging():
    """Example training function with comprehensive logging"""

    # 1. Initialize logger
    logger = ExperimentLogger("example_experiment", "detection")

    # 2. Log dataset information
    logger.log_dataset_info(
        dataset_path="data/detection_multispecies/dataset.yaml",
        num_classes=1,
        train_images=145,
        val_images=31,
        total_images=176
    )

    # 3. Log model configuration
    logger.log_model_config(
        model_name="yolov8n",
        epochs=30,
        batch_size=8,
        image_size=640,
        device="cpu"
    )

    # 4. Simulate training loop
    for epoch in range(5):
        # Simulate training metrics
        metrics = {
            "loss": 2.5 - epoch * 0.3,
            "map50": 0.1 + epoch * 0.15,
            "precision": 0.2 + epoch * 0.1,
            "recall": 0.15 + epoch * 0.12
        }

        # Log epoch results
        logger.log_training_epoch(epoch + 1, metrics)
        time.sleep(1)  # Simulate training time

    # 5. Log validation results
    val_metrics = {
        "val_map50": 0.8,
        "val_precision": 0.75,
        "val_recall": 0.82
    }
    logger.log_validation_results(val_metrics)

    # 6. Log test results
    test_predictions = [
        {"image": "test1.jpg", "prediction": 0.9, "true_label": 1},
        {"image": "test2.jpg", "prediction": 0.7, "true_label": 1}
    ]
    test_metrics = {"test_accuracy": 0.85, "test_f1": 0.83}
    logger.log_test_results(test_predictions, test_metrics)

    # 7. Log artifacts
    logger.log_artifact("model_weights/best.pt", "model", "Best trained model")
    logger.log_artifact("plots/training_curve.png", "visualization", "Training curve")

    # 8. Log final results
    final_metrics = {
        "final_map50": 0.8,
        "training_time_minutes": 45.2,
        "best_epoch": 23
    }
    logger.log_final_results(final_metrics)

    print(f"✅ Training completed! Log saved to: {logger.log_file}")
    return logger.get_summary()

if __name__ == "__main__":
    summary = example_training_with_logging()
    print("📊 Experiment Summary:")
    print(f"Name: {summary['name']}")
    print(f"Type: {summary['type']}")
    print(f"Status: {summary['status']}")
    print(f"Duration: {summary['duration_minutes']:.1f} minutes")
    print(f"Final Metrics: {summary['final_metrics']}")
'''

        example_path = Path("example_logging_usage.py")
        example_path.write_text(example_script)
        print(f"📝 Example usage script created: {example_path}")

def main():
    integrator = LoggingIntegrator()

    print("This script will modify your existing training scripts to add automatic logging.")
    print("Backups will be created in scripts_backup/ directory.")

    choice = input("\\nProceed with integration? [y/N]: ").strip().lower()

    if choice == 'y':
        integrator.integrate_all_scripts()
        integrator.create_example_usage()

        print("\\n🎉 Integration completed!")
        print("📋 Next steps:")
        print("1. Review modified scripts")
        print("2. Run example_logging_usage.py to test")
        print("3. Use test_models_comprehensive.py for model evaluation")
        print("4. Use generate_journal_report.py to create publication materials")
    else:
        print("Integration cancelled.")

if __name__ == "__main__":
    main()