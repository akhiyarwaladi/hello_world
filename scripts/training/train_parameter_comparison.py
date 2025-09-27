#!/usr/bin/env python3
"""
Parameter Comparison System for Malaria Classification
Runs comprehensive parameter sweep to find optimal configurations
"""

import os
import sys
import time
import subprocess
import itertools
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_training_experiment(config, data_path, base_save_dir):
    """Run single training experiment with given configuration"""

    # Create experiment name from config
    exp_name = f"{config['model']}_{config['loss']}"
    if config['loss'] == 'focal':
        exp_name += f"_a{config['focal_alpha']}_g{config['focal_gamma']}"
    exp_name += f"_lr{config['lr']}_bs{config['batch']}_ep{config['epochs']}"

    # Create save directory for this experiment
    save_dir = base_save_dir / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"{'='*80}")
    print(f"Configuration: {config}")
    print(f"Save directory: {save_dir}")

    # Build command arguments
    cmd = [
        "python",
        str(Path(__file__).parent / "12_train_pytorch_classification.py"),
        "--data", str(data_path),
        "--model", config['model'],
        "--epochs", str(config['epochs']),
        "--batch", str(config['batch']),
        "--lr", str(config['lr']),
        "--loss", config['loss'],
        "--focal_alpha", str(config['focal_alpha']),
        "--focal_gamma", str(config['focal_gamma']),
        "--image_size", str(config['image_size']),
        "--name", exp_name,
        "--save-dir", str(save_dir)
    ]

    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        success = result.returncode == 0
        output = result.stdout
        error = result.stderr if result.stderr else None
    except subprocess.TimeoutExpired:
        success = False
        output = "Training timed out after 2 hours"
        error = "Timeout"
    except Exception as e:
        success = False
        output = f"Failed to run training: {str(e)}"
        error = str(e)

    end_time = time.time()
    duration = end_time - start_time

    # Parse results if successful
    results = {
        'experiment_name': exp_name,
        'config': config,
        'success': success,
        'duration_minutes': duration / 60,
        'best_val_acc': 0.0,
        'test_acc': 0.0,
        'output': output,
        'error': error
    }

    if success and (save_dir / "results.txt").exists():
        try:
            # Parse results.txt for metrics
            with open(save_dir / "results.txt", 'r') as f:
                content = f.read()

            # Extract metrics using simple string parsing
            for line in content.split('\\n'):
                if line.startswith('Best Val Acc:'):
                    results['best_val_acc'] = float(line.split(':')[1].strip().replace('%', ''))
                elif line.startswith('Test Acc:') and 'N/A' not in line:
                    results['test_acc'] = float(line.split(':')[1].strip().replace('%', ''))

        except Exception as e:
            print(f"Warning: Could not parse results file: {e}")

    print(f"\\n[RESULT] {exp_name}: Success={success}, Val Acc={results['best_val_acc']:.2f}%, Test Acc={results['test_acc']:.2f}%")

    return results

def generate_parameter_configurations():
    """Generate all parameter combinations to test"""

    # Define parameter grids
    models = ['resnet18', 'resnet34', 'efficientnet_b0', 'mobilenet_v2']

    # Parameter configurations for different scenarios
    base_configs = [
        # Standard configurations
        {
            'loss': 'cross_entropy',
            'lr': 0.001,
            'batch': 32,
            'epochs': 25,
            'focal_alpha': 1.0,  # Not used for cross_entropy
            'focal_gamma': 2.0,  # Not used for cross_entropy
            'image_size': 224
        },
        {
            'loss': 'cross_entropy',
            'lr': 0.0005,  # Lower learning rate
            'batch': 16,   # Smaller batch
            'epochs': 30,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'image_size': 224
        },
        # Focal loss configurations
        {
            'loss': 'focal',
            'lr': 0.0005,  # Lower lr for focal loss
            'batch': 32,
            'epochs': 25,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'image_size': 224
        },
        {
            'loss': 'focal',
            'lr': 0.0005,
            'batch': 32,
            'epochs': 25,
            'focal_alpha': 2.0,  # Higher alpha for imbalance
            'focal_gamma': 3.0,  # Higher gamma for hard examples
            'image_size': 224
        },
        # High resolution configurations
        {
            'loss': 'cross_entropy',
            'lr': 0.0005,  # Lower lr for larger images
            'batch': 16,   # Smaller batch for memory
            'epochs': 20,  # Fewer epochs due to slower training
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'image_size': 384
        }
    ]

    # Generate all combinations
    configurations = []
    for model in models:
        for base_config in base_configs:
            config = base_config.copy()
            config['model'] = model
            configurations.append(config)

    return configurations

def analyze_results(results_list, save_path):
    """Analyze and summarize results from all experiments"""

    # Convert to DataFrame for analysis
    df_data = []
    for result in results_list:
        row = result['config'].copy()
        row.update({
            'experiment_name': result['experiment_name'],
            'success': result['success'],
            'duration_minutes': result['duration_minutes'],
            'best_val_acc': result['best_val_acc'],
            'test_acc': result['test_acc']
        })
        df_data.append(row)

    df = pd.DataFrame(df_data)

    # Save detailed results
    df.to_csv(save_path / "parameter_comparison_results.csv", index=False)

    # Generate summary report
    report_path = save_path / "parameter_comparison_summary.md"

    with open(report_path, 'w') as f:
        f.write("# 🔬 Parameter Comparison Results\\n\\n")
        f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")

        # Overall statistics
        successful_runs = df[df['success'] == True]
        f.write(f"## 📊 Overall Statistics\\n\\n")
        f.write(f"- **Total Experiments**: {len(df)}\\n")
        f.write(f"- **Successful Runs**: {len(successful_runs)}\\n")
        f.write(f"- **Success Rate**: {len(successful_runs)/len(df)*100:.1f}%\\n")
        f.write(f"- **Average Duration**: {df['duration_minutes'].mean():.1f} minutes\\n\\n")

        if len(successful_runs) > 0:
            # Best configurations
            f.write("## 🏆 Top Performing Configurations\\n\\n")
            top_configs = successful_runs.nlargest(5, 'best_val_acc')

            f.write("### By Validation Accuracy\\n\\n")
            f.write("| Rank | Model | Loss | LR | Batch | Val Acc | Test Acc | Duration |\\n")
            f.write("|------|-------|------|----|----|---------|----------|----------|\\n")

            for i, (_, row) in enumerate(top_configs.iterrows(), 1):
                f.write(f"| {i} | {row['model']} | {row['loss']} | {row['lr']} | {row['batch']} | {row['best_val_acc']:.2f}% | {row['test_acc']:.2f}% | {row['duration_minutes']:.1f}m |\\n")

            # Analysis by factors
            f.write("\\n## 📈 Performance Analysis\\n\\n")

            # By model
            f.write("### By Model Architecture\\n\\n")
            model_perf = successful_runs.groupby('model')['best_val_acc'].agg(['mean', 'max', 'count']).round(2)
            f.write("| Model | Avg Val Acc | Best Val Acc | Experiments |\\n")
            f.write("|-------|-------------|--------------|-------------|\\n")
            for model, row in model_perf.iterrows():
                f.write(f"| {model} | {row['mean']:.2f}% | {row['max']:.2f}% | {row['count']} |\\n")

            # By loss function
            f.write("\\n### By Loss Function\\n\\n")
            loss_perf = successful_runs.groupby('loss')['best_val_acc'].agg(['mean', 'max', 'count']).round(2)
            f.write("| Loss Function | Avg Val Acc | Best Val Acc | Experiments |\\n")
            f.write("|---------------|-------------|--------------|-------------|\\n")
            for loss, row in loss_perf.iterrows():
                f.write(f"| {loss} | {row['mean']:.2f}% | {row['max']:.2f}% | {row['count']} |\\n")

            # By learning rate
            f.write("\\n### By Learning Rate\\n\\n")
            lr_perf = successful_runs.groupby('lr')['best_val_acc'].agg(['mean', 'max', 'count']).round(2)
            f.write("| Learning Rate | Avg Val Acc | Best Val Acc | Experiments |\\n")
            f.write("|---------------|-------------|--------------|-------------|\\n")
            for lr, row in lr_perf.iterrows():
                f.write(f"| {lr} | {row['mean']:.2f}% | {row['max']:.2f}% | {row['count']} |\\n")

            # Recommendations
            f.write("\\n## 💡 Recommendations\\n\\n")
            best_overall = successful_runs.loc[successful_runs['best_val_acc'].idxmax()]
            f.write(f"### Best Overall Configuration\\n\\n")
            f.write(f"- **Model**: {best_overall['model']}\\n")
            f.write(f"- **Loss Function**: {best_overall['loss']}\\n")
            f.write(f"- **Learning Rate**: {best_overall['lr']}\\n")
            f.write(f"- **Batch Size**: {best_overall['batch']}\\n")
            f.write(f"- **Image Size**: {best_overall['image_size']}\\n")
            f.write(f"- **Validation Accuracy**: {best_overall['best_val_acc']:.2f}%\\n")
            f.write(f"- **Test Accuracy**: {best_overall['test_acc']:.2f}%\\n")

            # Focal loss specific analysis
            focal_runs = successful_runs[successful_runs['loss'] == 'focal']
            if len(focal_runs) > 0:
                f.write("\\n### Focal Loss Parameter Analysis\\n\\n")
                best_focal = focal_runs.loc[focal_runs['best_val_acc'].idxmax()]
                f.write(f"- **Best Focal Alpha**: {best_focal['focal_alpha']}\\n")
                f.write(f"- **Best Focal Gamma**: {best_focal['focal_gamma']}\\n")
                f.write(f"- **Best Focal Val Acc**: {best_focal['best_val_acc']:.2f}%\\n")

    print(f"\\n[ANALYSIS] Results analysis saved to {report_path}")
    return df

def main():
    """Main parameter comparison function"""
    import argparse

    parser = argparse.ArgumentParser(description='Parameter comparison for malaria classification')
    parser.add_argument('--data', required=True, help='Path to classification dataset')
    parser.add_argument('--results-dir', default='results/parameter_comparison',
                       help='Directory to save results')
    parser.add_argument('--max-experiments', type=int, default=None,
                       help='Maximum number of experiments to run (for testing)')

    args = parser.parse_args()

    # Validate data path
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        return

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PARAMETER COMPARISON SYSTEM")
    print("=" * 80)
    print(f"Data path: {data_path}")
    print(f"Results directory: {results_dir}")

    # Generate configurations
    configurations = generate_parameter_configurations()

    if args.max_experiments:
        configurations = configurations[:args.max_experiments]

    print(f"\\nTotal experiments to run: {len(configurations)}")

    # Run experiments
    all_results = []
    start_time = time.time()

    for i, config in enumerate(configurations, 1):
        print(f"\\n[PROGRESS] Experiment {i}/{len(configurations)}")

        try:
            result = run_training_experiment(config, data_path, results_dir)
            all_results.append(result)

            # Save intermediate results
            with open(results_dir / "intermediate_results.json", 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

        except Exception as e:
            print(f"[ERROR] Experiment {i} failed: {e}")
            # Continue with next experiment
            continue

    total_time = time.time() - start_time

    # Analyze results
    print(f"\\n{'='*80}")
    print("PARAMETER COMPARISON COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Completed experiments: {len(all_results)}")

    if len(all_results) > 0:
        # Generate comprehensive analysis
        df = analyze_results(all_results, results_dir)

        # Save final results
        with open(results_dir / "final_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        # Show quick summary
        successful_results = [r for r in all_results if r['success']]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['best_val_acc'])
            print(f"\\n[BEST] Best configuration: {best_result['experiment_name']}")
            print(f"[BEST] Best validation accuracy: {best_result['best_val_acc']:.2f}%")
            print(f"[BEST] Best test accuracy: {best_result['test_acc']:.2f}%")

        print(f"\\n[RESULTS] Detailed analysis saved to: {results_dir}")
    else:
        print("\\n[WARNING] No successful experiments completed")

if __name__ == "__main__":
    main()