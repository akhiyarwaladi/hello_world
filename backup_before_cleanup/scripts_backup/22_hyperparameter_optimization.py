#!/usr/bin/env python3
"""
Hyperparameter Optimization for Malaria Detection Models
Implements Bayesian Optimization, Grid Search, and Random Search
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import random

# Optional: Bayesian optimization with scikit-optimize
try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    print("⚠️ scikit-optimize not available. Bayesian optimization disabled.")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.results_manager import ResultsManager

class HyperparameterOptimizer:
    """Comprehensive hyperparameter optimization suite"""

    def __init__(self, base_results_dir: str = "results/hyperparameter_optimization"):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(parents=True, exist_ok=True)
        self.results_manager = ResultsManager()
        self.optimization_history = []

    def define_search_spaces(self) -> Dict[str, Dict]:
        """Define hyperparameter search spaces for different model types"""
        return {
            'yolo_detection': {
                'epochs': [10, 20, 30, 50],
                'batch_size': [2, 4, 8, 16],
                'learning_rate': [0.001, 0.01, 0.1],
                'momentum': [0.9, 0.937, 0.95],
                'weight_decay': [0.0005, 0.001, 0.005],
                'image_size': [320, 416, 640],
                'augment': [True, False],
                'mosaic': [0.0, 0.5, 1.0],
                'mixup': [0.0, 0.1, 0.2]
            },
            'yolo_classification': {
                'epochs': [10, 20, 30, 50],
                'batch_size': [4, 8, 16, 32],
                'learning_rate': [0.001, 0.01, 0.1],
                'momentum': [0.9, 0.937, 0.95],
                'weight_decay': [0.0005, 0.001, 0.005],
                'image_size': [224, 320, 416],
                'dropout': [0.0, 0.1, 0.2, 0.3]
            },
            'pytorch_classification': {
                'epochs': [10, 20, 30],
                'batch_size': [8, 16, 32],
                'learning_rate': [0.0001, 0.001, 0.01],
                'weight_decay': [0.0001, 0.001, 0.01],
                'scheduler_step': [5, 10, 15],
                'scheduler_gamma': [0.1, 0.5, 0.8],
                'dropout': [0.0, 0.1, 0.2, 0.5]
            }
        }

    def bayesian_optimization(self, model_type: str, script_path: str, data_path: str,
                            n_calls: int = 20, device: str = "cpu") -> Dict:
        """Perform Bayesian optimization using Gaussian Process"""
        if not BAYESIAN_OPT_AVAILABLE:
            print("❌ Bayesian optimization requires scikit-optimize")
            return {}

        print(f"🔍 Starting Bayesian optimization for {model_type}")

        search_spaces = self.define_search_spaces()
        if model_type not in search_spaces:
            print(f"❌ Unknown model type: {model_type}")
            return {}

        space = search_spaces[model_type]

        # Convert to skopt format
        dimensions = []
        param_names = []

        for param, values in space.items():
            param_names.append(param)
            if isinstance(values[0], int):
                dimensions.append(Integer(min(values), max(values), name=param))
            elif isinstance(values[0], float):
                dimensions.append(Real(min(values), max(values), name=param))
            elif isinstance(values[0], bool):
                dimensions.append(Categorical([True, False], name=param))
            else:
                dimensions.append(Categorical(values, name=param))

        @use_named_args(dimensions)
        def objective(**params):
            return self._evaluate_hyperparameters(
                model_type, script_path, data_path, params, device
            )

        # Run Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            n_random_starts=5,
            random_state=42
        )

        best_params = dict(zip(param_names, result.x))

        optimization_result = {
            'method': 'bayesian',
            'best_params': best_params,
            'best_score': result.fun,
            'n_evaluations': len(result.func_vals),
            'optimization_history': list(result.func_vals)
        }

        return optimization_result

    def grid_search(self, model_type: str, script_path: str, data_path: str,
                   device: str = "cpu", max_combinations: int = 50) -> Dict:
        """Perform grid search over hyperparameter space"""
        print(f"🎯 Starting grid search for {model_type}")

        search_spaces = self.define_search_spaces()
        if model_type not in search_spaces:
            print(f"❌ Unknown model type: {model_type}")
            return {}

        space = search_spaces[model_type]

        # Generate all parameter combinations
        param_names = list(space.keys())
        param_values = list(space.values())

        all_combinations = list(itertools.product(*param_values))

        # Limit combinations if too many
        if len(all_combinations) > max_combinations:
            print(f"⚠️ Too many combinations ({len(all_combinations)}), sampling {max_combinations}")
            all_combinations = random.sample(all_combinations, max_combinations)

        best_score = float('inf')
        best_params = None
        results_history = []

        print(f"📊 Evaluating {len(all_combinations)} parameter combinations...")

        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))

            print(f"\n🔄 Combination {i+1}/{len(all_combinations)}: {params}")

            score = self._evaluate_hyperparameters(
                model_type, script_path, data_path, params, device
            )

            results_history.append({
                'combination': i+1,
                'params': params.copy(),
                'score': score
            })

            if score < best_score:
                best_score = score
                best_params = params.copy()
                print(f"🎉 New best score: {best_score}")

        optimization_result = {
            'method': 'grid_search',
            'best_params': best_params,
            'best_score': best_score,
            'n_evaluations': len(results_history),
            'results_history': results_history
        }

        return optimization_result

    def random_search(self, model_type: str, script_path: str, data_path: str,
                     device: str = "cpu", n_trials: int = 30) -> Dict:
        """Perform random search over hyperparameter space"""
        print(f"🎲 Starting random search for {model_type}")

        search_spaces = self.define_search_spaces()
        if model_type not in search_spaces:
            print(f"❌ Unknown model type: {model_type}")
            return {}

        space = search_spaces[model_type]

        best_score = float('inf')
        best_params = None
        results_history = []

        print(f"📊 Evaluating {n_trials} random parameter combinations...")

        for trial in range(n_trials):
            # Sample random parameters
            params = {}
            for param, values in space.items():
                if isinstance(values[0], bool):
                    params[param] = random.choice(values)
                elif isinstance(values[0], (int, float)):
                    params[param] = random.uniform(min(values), max(values))
                    if isinstance(values[0], int):
                        params[param] = int(params[param])
                else:
                    params[param] = random.choice(values)

            print(f"\n🔄 Trial {trial+1}/{n_trials}: {params}")

            score = self._evaluate_hyperparameters(
                model_type, script_path, data_path, params, device
            )

            results_history.append({
                'trial': trial+1,
                'params': params.copy(),
                'score': score
            })

            if score < best_score:
                best_score = score
                best_params = params.copy()
                print(f"🎉 New best score: {best_score}")

        optimization_result = {
            'method': 'random_search',
            'best_params': best_params,
            'best_score': best_score,
            'n_evaluations': len(results_history),
            'results_history': results_history
        }

        return optimization_result

    def _evaluate_hyperparameters(self, model_type: str, script_path: str, data_path: str,
                                params: Dict, device: str = "cpu") -> float:
        """Evaluate a set of hyperparameters by training a model"""
        try:
            # Create experiment name
            experiment_name = f"hyperparam_{model_type}_{int(time.time())}"

            # Build command based on model type
            cmd = ["python", script_path, "--data", data_path, "--device", device]

            # Add common parameters
            if 'epochs' in params:
                cmd.extend(["--epochs", str(int(params['epochs']))])
            if 'batch_size' in params:
                cmd.extend(["--batch", str(int(params['batch_size']))])
            if 'learning_rate' in params:
                cmd.extend(["--lr", str(params['learning_rate'])])

            # Add model-specific parameters
            if model_type == 'yolo_detection':
                if 'momentum' in params:
                    cmd.extend(["--momentum", str(params['momentum'])])
                if 'weight_decay' in params:
                    cmd.extend(["--weight-decay", str(params['weight_decay'])])
                if 'image_size' in params:
                    cmd.extend(["--imgsz", str(int(params['image_size']))])
                if 'mosaic' in params:
                    cmd.extend(["--mosaic", str(params['mosaic'])])
                if 'mixup' in params:
                    cmd.extend(["--mixup", str(params['mixup'])])

            elif model_type == 'yolo_classification':
                if 'image_size' in params:
                    cmd.extend(["--imgsz", str(int(params['image_size']))])
                if 'dropout' in params:
                    cmd.extend(["--dropout", str(params['dropout'])])

            elif model_type == 'pytorch_classification':
                if 'weight_decay' in params:
                    cmd.extend(["--weight-decay", str(params['weight_decay'])])
                if 'scheduler_step' in params:
                    cmd.extend(["--scheduler-step", str(int(params['scheduler_step']))])
                if 'scheduler_gamma' in params:
                    cmd.extend(["--scheduler-gamma", str(params['scheduler_gamma'])])

            cmd.extend(["--name", experiment_name])

            # Run training with timeout
            print(f"🏃‍♂️ Running: {' '.join(cmd)}")

            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                env={**os.environ, 'NNPACK_DISABLE': '1'}
            )

            training_time = time.time() - start_time

            if result.returncode == 0:
                # Parse the results to extract validation accuracy
                validation_acc = self._extract_validation_accuracy(result.stdout, experiment_name)

                # Return negative accuracy (for minimization)
                score = -validation_acc if validation_acc is not None else float('inf')

                print(f"✅ Training completed - Validation Accuracy: {validation_acc}% (Score: {score})")

                # Save results
                self.optimization_history.append({
                    'experiment_name': experiment_name,
                    'params': params,
                    'validation_accuracy': validation_acc,
                    'training_time': training_time,
                    'score': score
                })

                return score
            else:
                print(f"❌ Training failed: {result.stderr}")
                return float('inf')

        except subprocess.TimeoutExpired:
            print(f"⏱️ Training timed out")
            return float('inf')
        except Exception as e:
            print(f"💥 Error during evaluation: {e}")
            return float('inf')

    def _extract_validation_accuracy(self, stdout: str, experiment_name: str) -> Optional[float]:
        """Extract validation accuracy from training output"""
        try:
            lines = stdout.split('\n')

            # Look for validation accuracy patterns
            patterns = [
                'Best validation accuracy:',
                'Val Acc:',
                'best_accuracy:',
                'validation_accuracy:'
            ]

            for line in reversed(lines):  # Search from end
                for pattern in patterns:
                    if pattern in line:
                        # Extract number
                        parts = line.split(pattern)
                        if len(parts) > 1:
                            acc_part = parts[1].strip().replace('%', '')
                            try:
                                return float(acc_part.split()[0])
                            except:
                                continue

            # If no pattern found, look for results file
            results_file = self._find_results_file(experiment_name)
            if results_file and results_file.exists():
                with open(results_file, 'r') as f:
                    content = f.read()
                    if 'Val Acc:' in content:
                        for line in content.split('\n'):
                            if 'Val Acc:' in line:
                                acc_str = line.split('Val Acc:')[1].strip().replace('%', '')
                                return float(acc_str.split()[0])

            return None

        except Exception as e:
            print(f"⚠️ Error extracting validation accuracy: {e}")
            return None

    def _find_results_file(self, experiment_name: str) -> Optional[Path]:
        """Find results file for an experiment"""
        search_dirs = [
            Path("results"),
            Path("results/current_experiments"),
            Path("results/detection"),
            Path("results/classification")
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                for results_file in search_dir.rglob(f"*{experiment_name}*/results.txt"):
                    return results_file

        return None

    def save_optimization_results(self, optimization_results: Dict, output_file: Path):
        """Save optimization results to file"""
        # Add optimization history
        optimization_results['optimization_history'] = self.optimization_history

        with open(output_file, 'w') as f:
            json.dump(optimization_results, f, indent=2, default=str)

        print(f"💾 Optimization results saved to: {output_file}")

    def generate_optimization_report(self, optimization_results: Dict, output_file: Path):
        """Generate comprehensive optimization report"""
        with open(output_file, 'w') as f:
            f.write("# Hyperparameter Optimization Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"## Optimization Summary\n\n")
            f.write(f"- Method: {optimization_results['method']}\n")
            f.write(f"- Number of evaluations: {optimization_results['n_evaluations']}\n")
            f.write(f"- Best score: {optimization_results['best_score']:.6f}\n")
            f.write(f"- Best validation accuracy: {-optimization_results['best_score']:.2f}%\n\n")

            f.write(f"## Best Parameters\n\n")
            for param, value in optimization_results['best_params'].items():
                f.write(f"- {param}: {value}\n")

            f.write(f"\n## Optimization History\n\n")
            if self.optimization_history:
                # Sort by score (best first)
                sorted_history = sorted(self.optimization_history, key=lambda x: x['score'])

                f.write("### Top 10 Experiments\n\n")
                for i, exp in enumerate(sorted_history[:10]):
                    f.write(f"#### Rank {i+1}: {exp['experiment_name']}\n")
                    f.write(f"- Validation Accuracy: {exp['validation_accuracy']:.2f}%\n")
                    f.write(f"- Training Time: {exp['training_time']:.1f} seconds\n")
                    f.write(f"- Parameters:\n")
                    for param, value in exp['params'].items():
                        f.write(f"  - {param}: {value}\n")
                    f.write("\n")

        print(f"📝 Optimization report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Malaria Detection")
    parser.add_argument("--method", default="random", choices=["bayesian", "grid", "random"],
                       help="Optimization method")
    parser.add_argument("--model_type", required=True,
                       choices=["yolo_detection", "yolo_classification", "pytorch_classification"],
                       help="Type of model to optimize")
    parser.add_argument("--script_path", required=True, help="Path to training script")
    parser.add_argument("--data_path", required=True, help="Path to dataset")
    parser.add_argument("--device", default="cpu", help="Device to use for training")
    parser.add_argument("--n_trials", type=int, default=20,
                       help="Number of trials for random/bayesian search")
    parser.add_argument("--max_combinations", type=int, default=50,
                       help="Maximum combinations for grid search")
    parser.add_argument("--output_dir", default="hyperparameter_optimization",
                       help="Output directory for results")

    args = parser.parse_args()

    print("=" * 60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    optimizer = HyperparameterOptimizer(args.output_dir)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run optimization
    print(f"🎯 Model Type: {args.model_type}")
    print(f"📊 Method: {args.method}")
    print(f"🎲 Trials: {args.n_trials}")

    start_time = time.time()

    if args.method == "bayesian":
        if not BAYESIAN_OPT_AVAILABLE:
            print("❌ Bayesian optimization requires scikit-optimize")
            print("💡 Install with: pip install scikit-optimize")
            return
        results = optimizer.bayesian_optimization(
            args.model_type, args.script_path, args.data_path,
            n_calls=args.n_trials, device=args.device
        )
    elif args.method == "grid":
        results = optimizer.grid_search(
            args.model_type, args.script_path, args.data_path,
            device=args.device, max_combinations=args.max_combinations
        )
    else:  # random
        results = optimizer.random_search(
            args.model_type, args.script_path, args.data_path,
            device=args.device, n_trials=args.n_trials
        )

    total_time = time.time() - start_time

    # Save results
    results_file = output_dir / f"{args.method}_{args.model_type}_results.json"
    optimizer.save_optimization_results(results, results_file)

    # Generate report
    report_file = output_dir / f"{args.method}_{args.model_type}_report.md"
    optimizer.generate_optimization_report(results, report_file)

    print("\n" + "=" * 60)
    print("🎉 HYPERPARAMETER OPTIMIZATION COMPLETED!")
    print("=" * 60)
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"🎯 Best accuracy: {-results['best_score']:.2f}%")
    print(f"📊 Evaluations: {results['n_evaluations']}")
    print(f"💾 Results: {results_file}")
    print(f"📝 Report: {report_file}")

if __name__ == "__main__":
    main()