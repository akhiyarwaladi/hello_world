"""
Run Baseline Classification Comparison
Simple training with NO over-customization to compare with our customized version
"""

import subprocess
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

def run_baseline_model(model_name, data_path, output_dir, epochs=50):
    """Run baseline training for one model"""
    print(f"\n{'='*80}")
    print(f"Running baseline: {model_name}")
    print(f"{'='*80}\n")

    cmd = [
        'python', 'scripts/training/baseline_classification.py',
        '--data', str(data_path),
        '--model', model_name,
        '--epochs', str(epochs),
        '--batch', '32',
        '--lr', '0.001',
        '--save-dir', str(output_dir / f'baseline_{model_name}')
    ]

    subprocess.run(cmd, check=True)

def main():
    # Configuration
    data_path = Path('results/optA_20251005_142821/experiments/experiment_iml_lifecycle/crops_gt_crops/crops')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'results/baseline_comparison_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        'densenet121',
        'efficientnet_b0',
        'efficientnet_b1',
        'efficientnet_b2',
        'resnet50',
        'resnet101'
    ]

    print(f"\n{'='*80}")
    print("BASELINE CLASSIFICATION COMPARISON")
    print(f"{'='*80}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Models: {', '.join(models)}")
    print(f"\nSettings:")
    print(f"  - Simple focal loss (alpha=1.0, gamma=2.0)")
    print(f"  - Standard early stopping (patience=10, monitor=val_loss)")
    print(f"  - Warmup: 12 epochs (skip chaos phase)")
    print(f"  - NO threshold, minimal customization")
    print(f"  - Adam optimizer (lr=0.001)")
    print(f"  - OneCycleLR scheduler")
    print(f"{'='*80}\n")

    # Run all models
    results = []
    for model_name in models:
        try:
            run_baseline_model(model_name, data_path, output_dir, epochs=50)

            # Load results from model folder
            model_dir = output_dir / f'baseline_{model_name}'
            result_file = model_dir / 'results.json'
            if result_file.exists():
                with open(result_file) as f:
                    model_results = json.load(f)
                    results.append(model_results)
        except Exception as e:
            print(f"Error running {model_name}: {e}")
            continue

    # Create summary
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('test_accuracy', ascending=False)

        print(f"\n{'='*80}")
        print("BASELINE RESULTS SUMMARY")
        print(f"{'='*80}\n")
        print(df.to_string(index=False))

        # Save summary
        df.to_csv(output_dir / 'baseline_summary.csv', index=False)

        print(f"\n{'='*80}")
        print(f"Best Model: {df.iloc[0]['model']} ({df.iloc[0]['test_accuracy']:.2f}%)")
        print(f"{'='*80}")

        # Compare with customized version
        print(f"\n{'='*80}")
        print("COMPARISON WITH CUSTOMIZED VERSION")
        print(f"{'='*80}")
        print(f"\nCustomized version (optA_20251005_135256):")
        print(f"  Best: DenseNet121 Focal = 85.39%")
        print(f"\nBaseline version (this run):")
        print(f"  Best: {df.iloc[0]['model']} = {df.iloc[0]['test_accuracy']:.2f}%")
        print(f"\nDifference: {df.iloc[0]['test_accuracy'] - 85.39:.2f}%")
        print(f"\n{'='*80}")

        print(f"\nResults saved to: {output_dir}/baseline_summary.csv")

        # Create README
        readme_path = output_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write("# Baseline Classification Results\n\n")
            f.write("Simple PyTorch training with NO over-customization.\n\n")
            f.write("## Settings\n\n")
            f.write("- **Loss**: Focal Loss (alpha=1.0, gamma=2.0)\n")
            f.write("- **Early Stopping**: patience=10, monitor=val_loss\n")
            f.write("- **NO warmup**: Save from epoch 1\n")
            f.write("- **NO threshold**: Save best model regardless of accuracy\n")
            f.write("- **Optimizer**: Adam (lr=0.001)\n")
            f.write("- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)\n\n")
            f.write("## Results Summary\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n## Folder Structure\n\n")
            f.write("```\n")
            f.write("baseline_comparison_{timestamp}/\n")
            for model in models:
                f.write(f"├── baseline_{model}/\n")
                f.write(f"│   ├── results.csv          # Training history per epoch\n")
                f.write(f"│   ├── results.json         # Final test metrics\n")
                f.write(f"│   ├── results.txt          # Summary text\n")
                f.write(f"│   ├── best.pt              # Best model (lowest val_loss)\n")
                f.write(f"│   ├── last.pt              # Last epoch model\n")
                f.write(f"│   └── classification_report.txt\n")
            f.write("└── baseline_summary.csv    # All models comparison\n")
            f.write("```\n")

        print(f"README saved to: {readme_path}")
    else:
        print("No results to summarize!")

if __name__ == '__main__':
    main()
