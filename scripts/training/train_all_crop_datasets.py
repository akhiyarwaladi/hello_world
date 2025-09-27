#!/usr/bin/env python3
"""
Automated Training Script for All Crop Datasets
Runs comprehensive training and evaluation on all generated crop datasets.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import the training class directly instead of using subprocess
sys.path.append(str(Path(__file__).parent))
from train_classification_from_crops import CropClassificationTrainer

class MultiDatasetTrainer:
    """Automated trainer for multiple crop datasets"""

    def __init__(self, base_crops_dir="data/ground_truth_crops_224", base_results_dir="results/multi_dataset_training"):
        self.base_crops_dir = Path(base_crops_dir)
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(parents=True, exist_ok=True)

        # Dataset configurations - UPDATED FOR CURRENT STRUCTURE
        self.datasets = {
            'lifecycle': {
                'name': 'IML Lifecycle 224px',
                'description': 'Malaria parasite lifecycle stages (4 classes) - 224px resolution',
                'path': self.base_crops_dir / 'lifecycle' / 'crops',
                'expected_classes': ['gametocyte', 'ring', 'schizont', 'trophozoite'],
                'imbalance_level': 'moderate',
                'samples': {'train': 376, 'val': 102, 'test': 51},
                'resolution': '224x224'
            },
            'species': {
                'name': 'MP-IDB Species 224px',
                'description': 'Malaria parasite species classification (4 classes) - 224px resolution',
                'path': self.base_crops_dir / 'species' / 'crops',
                'expected_classes': ['P_falciparum', 'P_malariae', 'P_ovale', 'P_vivax'],
                'imbalance_level': 'extreme',
                'samples': {'train': 922, 'val': 322, 'test': 192},
                'resolution': '224x224'
            },
            'stages': {
                'name': 'MP-IDB Stages 224px',
                'description': 'Malaria parasite stages from MP-IDB (4 classes) - 224px resolution',
                'path': self.base_crops_dir / 'stages' / 'crops',
                'expected_classes': ['gametocyte', 'ring', 'schizont', 'trophozoite'],
                'imbalance_level': 'extreme',
                'samples': {'train': 999, 'val': 339, 'test': 98},
                'resolution': '224x224'
            }
        }

        # Training configurations to test - IMPROVED VERSION
        self.training_configs = [
            {
                'name': 'ResNet18_Baseline',
                'model': 'resnet18',
                'loss': 'cross_entropy',
                'epochs': 25,  # Increased from 15
                'lr': 0.001,
                'weighted_sampling': False,
                'early_stopping': True,
                'description': 'Baseline ResNet18 with standard cross-entropy'
            },
            {
                'name': 'ResNet18_Weighted_Extended',
                'model': 'resnet18',
                'loss': 'weighted_ce',
                'epochs': 30,  # Increased for better convergence
                'lr': 0.001,
                'weighted_sampling': True,
                'early_stopping': True,
                'description': 'ResNet18 with weighted loss, extended training'
            },
            {
                'name': 'ResNet18_Focal_Optimized',
                'model': 'resnet18',
                'loss': 'focal',
                'epochs': 35,  # More epochs for focal loss convergence
                'lr': 0.0005,  # Lower LR for focal loss
                'weighted_sampling': True,
                'early_stopping': True,
                'description': 'ResNet18 with optimized Focal Loss for extreme imbalance'
            },
            {
                'name': 'ResNet50_Heavy_Augment',
                'model': 'resnet50',
                'loss': 'weighted_ce',
                'epochs': 30,
                'lr': 0.0008,
                'weighted_sampling': True,
                'early_stopping': True,
                'heavy_augment': True,
                'description': 'ResNet50 with heavy augmentation for minority classes'
            },
            {
                'name': 'EfficientNet_Adaptive',
                'model': 'efficientnet_b0',
                'loss': 'focal',
                'epochs': 25,
                'lr': 0.0005,
                'weighted_sampling': True,
                'early_stopping': True,
                'adaptive_lr': True,
                'description': 'EfficientNet with adaptive learning rate'
            },
            {
                'name': 'SimpleCNN_Extended',
                'model': 'simple_cnn',
                'loss': 'weighted_ce',
                'epochs': 40,  # More epochs for simple model
                'lr': 0.002,   # Higher LR for simple model
                'weighted_sampling': True,
                'early_stopping': True,
                'description': 'Simple CNN with extended training'
            }
        ]

        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def verify_dataset(self, dataset_key):
        """Verify dataset exists and has proper structure"""
        config = self.datasets[dataset_key]
        dataset_path = config['path']

        print(f"\n🔍 Verifying {config['name']}:")
        print(f"   Path: {dataset_path}")
        print(f"   Resolution: {config['resolution']}")
        print(f"   Imbalance Level: {config['imbalance_level']}")

        if not dataset_path.exists():
            print(f"❌ Dataset not found: {dataset_path}")
            return False

        # Check splits and compare with expected counts
        expected_samples = config.get('samples', {})
        total_found = 0

        for split in ['train', 'val', 'test']:
            split_path = dataset_path / split
            if not split_path.exists():
                print(f"❌ Split not found: {split_path}")
                return False

            # Count samples
            sample_count = len(list(split_path.rglob("*.jpg")))
            if sample_count == 0:
                print(f"❌ No samples found in: {split_path}")
                return False

            expected = expected_samples.get(split, 'unknown')
            status = "✅" if sample_count == expected else "⚠️"
            print(f"   {status} {split}: {sample_count} samples (expected: {expected})")
            total_found += sample_count

        print(f"   📊 Total samples: {total_found}")

        # Verify classes exist
        train_classes = [d.name for d in (dataset_path / 'train').iterdir() if d.is_dir()]
        expected_classes = config['expected_classes']

        if set(train_classes) == set(expected_classes):
            print(f"   ✅ Classes: {train_classes}")
        else:
            print(f"   ⚠️ Class mismatch:")
            print(f"     Found: {train_classes}")
            print(f"     Expected: {expected_classes}")

        return True

    def run_single_training(self, dataset_key, config_key):
        """Run training for a single dataset-config combination"""
        dataset_config = self.datasets[dataset_key]
        training_config = self.training_configs[config_key]

        experiment_name = f"{dataset_key}_{training_config['name']}"
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING: {experiment_name}")
        print(f"{'='*80}")
        print(f"Dataset: {dataset_config['name']}")
        print(f"Config: {training_config['description']}")

        # Setup paths
        dataset_path = dataset_config['path']
        results_dir = self.base_results_dir / experiment_name

        # Dataset-specific epoch adjustment based on imbalance level
        epochs = training_config['epochs']
        imbalance_level = dataset_config.get('imbalance_level', 'moderate')
        if imbalance_level == 'extreme':
            # Add more epochs for extreme imbalance
            epochs = int(epochs * 1.2)  # 20% more epochs
            print(f"📈 Adjusted epochs for extreme imbalance: {training_config['epochs']} → {epochs}")

        try:
            # Initialize trainer directly (no subprocess!)
            trainer = CropClassificationTrainer(
                train_dir=str(dataset_path / "train"),
                val_dir=str(dataset_path / "val"),
                test_dir=str(dataset_path / "test"),
                results_dir=str(results_dir),
                dataset_name=experiment_name
            )

            # Load data with configuration
            trainer.load_data(
                input_size=224,
                augment=not training_config.get('no_augment', False),
                use_weighted_sampling=training_config.get('weighted_sampling', False)
            )

            # Train model with configuration
            training_results = trainer.train_model(
                model_type=training_config['model'],
                loss_type=training_config['loss'],
                epochs=epochs,
                lr=training_config.get('lr', 0.001),
                pretrained=not training_config.get('no_pretrained', False)
            )

            # Evaluate model
            evaluation_results = trainer.evaluate_model(save_plots=True)

            print("✅ Training completed successfully!")

            # Store results
            self.results[experiment_name] = {
                'dataset': dataset_config,
                'training_config': training_config,
                'training_metrics': training_results,
                'evaluation_results': evaluation_results,
                'status': 'success'
            }

            # Print key metrics
            print(f"📊 Results:")
            print(f"   Accuracy: {evaluation_results['accuracy']:.4f}")
            print(f"   Balanced Accuracy: {evaluation_results['balanced_accuracy']:.4f}")
            print(f"   Macro F1: {evaluation_results['macro_f1']:.4f}")
            print(f"   Best Val Acc: {training_results['best_val_acc']:.4f}")

        except Exception as e:
            print(f"💥 Training error: {e}")
            import traceback
            traceback.print_exc()

            self.results[experiment_name] = {
                'dataset': dataset_config,
                'training_config': training_config,
                'status': 'error',
                'error': str(e)
            }

    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print(f"\n{'='*80}")
        print("📊 GENERATING COMPARISON REPORT")
        print(f"{'='*80}")

        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if v['status'] == 'success'}

        if not successful_results:
            print("❌ No successful training results to analyze")
            return

        # Create summary table
        summary_data = []
        for experiment_name, data in successful_results.items():
            metrics = data['evaluation_results']  # Updated structure
            training_metrics = data['training_metrics']
            summary_data.append({
                'Experiment': experiment_name,
                'Dataset': data['dataset']['name'],
                'Model': data['training_config']['model'],
                'Loss': data['training_config']['loss'],
                'Epochs': data['training_config']['epochs'],
                'LR': data['training_config'].get('lr', 0.001),
                'Accuracy': metrics['accuracy'],
                'Balanced_Accuracy': metrics['balanced_accuracy'],
                'Macro_F1': metrics['macro_f1'],
                'Weighted_F1': metrics['weighted_f1'],
                'Best_Val_Acc': training_metrics['best_val_acc'],
                'Imbalance_Level': data['dataset']['imbalance_level']
            })

        # Convert to DataFrame
        df = pd.DataFrame(summary_data)

        # Save detailed results
        results_summary_file = self.base_results_dir / f"training_summary_{self.timestamp}.json"
        with open(results_summary_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Save CSV summary
        csv_file = self.base_results_dir / f"results_summary_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)

        # Print summary table
        print(f"\n📋 RESULTS SUMMARY:")
        print("=" * 140)
        print(f"{'Experiment':<30} {'Dataset':<20} {'Model':<12} {'Loss':<12} {'Epochs':<8} {'Accuracy':<10} {'Bal_Acc':<10} {'Macro_F1':<10}")
        print("-" * 140)

        for _, row in df.iterrows():
            print(f"{row['Experiment']:<30} {row['Dataset']:<20} {row['Model']:<12} {row['Loss']:<12} "
                  f"{row['Epochs']:<8} {row['Accuracy']:<10.3f} {row['Balanced_Accuracy']:<10.3f} {row['Macro_F1']:<10.3f}")

        # Find best performers
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"Highest Accuracy: {df.loc[df['Accuracy'].idxmax(), 'Experiment']} ({df['Accuracy'].max():.3f})")
        print(f"Highest Balanced Accuracy: {df.loc[df['Balanced_Accuracy'].idxmax(), 'Experiment']} ({df['Balanced_Accuracy'].max():.3f})")
        print(f"Highest Macro F1: {df.loc[df['Macro_F1'].idxmax(), 'Experiment']} ({df['Macro_F1'].max():.3f})")

        # Create visualization
        self.create_comparison_plots(df)

        print(f"\n📁 Files saved:")
        print(f"   Detailed results: {results_summary_file}")
        print(f"   Summary CSV: {csv_file}")

    def create_comparison_plots(self, df):
        """Create comparison visualization plots"""

        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multi-Dataset Training Comparison', fontsize=16, fontweight='bold')

        # 1. Accuracy comparison by dataset
        ax1 = axes[0, 0]
        datasets = df['Dataset'].unique()
        x_pos = np.arange(len(datasets))

        for i, dataset in enumerate(datasets):
            dataset_data = df[df['Dataset'] == dataset]
            ax1.bar(x_pos[i] - 0.2, dataset_data['Accuracy'].mean(), 0.4,
                   label='Accuracy', alpha=0.7, color='skyblue')
            ax1.bar(x_pos[i] + 0.2, dataset_data['Balanced_Accuracy'].mean(), 0.4,
                   label='Balanced Accuracy' if i == 0 else "", alpha=0.7, color='lightcoral')

        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Score')
        ax1.set_title('Average Accuracy by Dataset')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(datasets, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Model comparison
        ax2 = axes[0, 1]
        model_performance = df.groupby('Model')[['Accuracy', 'Balanced_Accuracy', 'Macro_F1']].mean()
        model_performance.plot(kind='bar', ax=ax2, alpha=0.7)
        ax2.set_title('Model Performance Comparison')
        ax2.set_ylabel('Score')
        ax2.legend(['Accuracy', 'Balanced Accuracy', 'Macro F1'])
        ax2.grid(True, alpha=0.3)

        # 3. Loss function comparison
        ax3 = axes[1, 0]
        loss_performance = df.groupby('Loss')[['Accuracy', 'Balanced_Accuracy', 'Macro_F1']].mean()
        loss_performance.plot(kind='bar', ax=ax3, alpha=0.7)
        ax3.set_title('Loss Function Performance')
        ax3.set_ylabel('Score')
        ax3.legend(['Accuracy', 'Balanced Accuracy', 'Macro F1'])
        ax3.grid(True, alpha=0.3)

        # 4. Imbalance level impact
        ax4 = axes[1, 1]
        imbalance_performance = df.groupby('Imbalance_Level')[['Accuracy', 'Balanced_Accuracy']].mean()
        imbalance_performance.plot(kind='bar', ax=ax4, alpha=0.7)
        ax4.set_title('Performance vs Class Imbalance')
        ax4.set_ylabel('Score')
        ax4.legend(['Accuracy', 'Balanced Accuracy'])
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = self.base_results_dir / f"comparison_plots_{self.timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Comparison plots: {plot_file}")

    def run_all_experiments(self, datasets_to_run=None, configs_to_run=None):
        """Run all training experiments"""

        # Default to all datasets and configs if not specified
        if datasets_to_run is None:
            datasets_to_run = list(self.datasets.keys())
        if configs_to_run is None:
            configs_to_run = list(range(len(self.training_configs)))

        print(f"🎯 MULTI-DATASET TRAINING PIPELINE")
        print(f"📅 Timestamp: {self.timestamp}")
        print(f"📊 Datasets to train: {datasets_to_run}")
        print(f"⚙️ Configurations to test: {len(configs_to_run)}")
        print(f"🔢 Total experiments: {len(datasets_to_run)} × {len(configs_to_run)} = {len(datasets_to_run) * len(configs_to_run)}")

        # Verify all datasets first
        print(f"\n📋 VERIFYING DATASETS:")
        for dataset_key in datasets_to_run:
            print(f"\n🔍 Checking {dataset_key}:")
            if not self.verify_dataset(dataset_key):
                print(f"❌ Skipping {dataset_key} due to verification failure")
                datasets_to_run.remove(dataset_key)

        if not datasets_to_run:
            print("❌ No valid datasets found!")
            return

        # Run all experiments
        total_experiments = len(datasets_to_run) * len(configs_to_run)
        current_experiment = 0

        for dataset_key in datasets_to_run:
            for config_idx in configs_to_run:
                current_experiment += 1
                print(f"\n🎯 EXPERIMENT {current_experiment}/{total_experiments}")
                self.run_single_training(dataset_key, config_idx)

        # Generate final report
        self.generate_comparison_report()

        print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
        print(f"📁 Results saved in: {self.base_results_dir}")

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train all crop datasets with multiple configurations')
    parser.add_argument('--datasets', nargs='+', choices=['lifecycle', 'species', 'stages'],
                       default=['lifecycle', 'species', 'stages'],
                       help='Datasets to train (default: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Run only ResNet18 weighted configuration (quick test)')
    parser.add_argument('--base_dir', default='data/ground_truth_crops_224',
                       help='Base directory for crop datasets')

    args = parser.parse_args()

    # Initialize trainer
    trainer = MultiDatasetTrainer(base_crops_dir=args.base_dir)

    # Determine configurations to run
    if args.quick:
        # Only run ResNet18 weighted configuration for quick testing
        configs_to_run = [1]  # ResNet18_Weighted
        print("🚀 QUICK MODE: Running only ResNet18 with weighted loss")
    else:
        # Run all configurations
        configs_to_run = None
        print("🔥 FULL MODE: Running all configurations")

    # Run experiments
    trainer.run_all_experiments(
        datasets_to_run=args.datasets,
        configs_to_run=configs_to_run
    )

if __name__ == "__main__":
    main()