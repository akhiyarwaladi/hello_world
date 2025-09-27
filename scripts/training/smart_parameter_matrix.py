#!/usr/bin/env python3
"""
Smart Parameter Matrix for Pipeline Integration
Automatically selects optimal parameters based on dataset characteristics
"""

import json
from pathlib import Path

class SmartParameterMatrix:
    """Intelligently configure parameters based on dataset characteristics"""

    def __init__(self):
        # Define model categories by computational efficiency
        self.model_categories = {
            "lightweight": ["mobilenet_v3_large", "convnext_tiny"],
            "balanced": ["efficientnet_b1", "densenet121"],
            "heavy": ["efficientnet_b2", "resnet101"]
        }

        # Dataset-specific configurations based on comprehensive analysis
        self.dataset_configs = {
            "iml_lifecycle": {
                "imbalance_ratio": 12.5,
                "classes": 4,
                "difficulty": "moderate",
                "success_rate": "high"
            },
            "mp_idb_species": {
                "imbalance_ratio": 40.5,
                "classes": 4,
                "difficulty": "extreme_imbalance",
                "success_rate": "medium"
            },
            "mp_idb_stages": {
                "imbalance_ratio": 43.2,
                "classes": 4,
                "difficulty": "extreme_critical",
                "success_rate": "low"
            }
        }

    def get_optimal_configurations(self, dataset_name, model_name):
        """Get optimized parameter sets for dataset-model combination"""

        dataset_config = self.dataset_configs.get(dataset_name, {})
        imbalance_ratio = dataset_config.get("imbalance_ratio", 10.0)
        difficulty = dataset_config.get("difficulty", "moderate")

        # Base configuration
        base_config = {
            "model": model_name,
            "image_size": 224,
            "pretrained": True
        }

        # Generate multiple configurations based on dataset difficulty
        configurations = []

        if difficulty == "moderate":  # IML Lifecycle
            # Configuration 1: Standard approach (proven successful)
            config1 = base_config.copy()
            config1.update({
                "loss": "cross_entropy",
                "epochs": 25,
                "batch": 32,
                "lr": 0.001,
                "name_suffix": "standard"
            })
            configurations.append(config1)

            # Configuration 2: Focal loss (for further improvement)
            config2 = base_config.copy()
            config2.update({
                "loss": "focal",
                "focal_alpha": 1.0,
                "focal_gamma": 2.0,
                "epochs": 25,
                "batch": 32,
                "lr": 0.0005,  # Lower LR for focal loss
                "name_suffix": "focal"
            })
            configurations.append(config2)

        elif difficulty == "extreme_imbalance":  # MP-IDB Species
            # Configuration 1: Aggressive focal loss
            config1 = base_config.copy()
            config1.update({
                "loss": "focal",
                "focal_alpha": 2.0,  # Higher alpha for extreme imbalance
                "focal_gamma": 3.0,  # Higher gamma for hard examples
                "epochs": 30,
                "batch": 16,  # Smaller batch for stability
                "lr": 0.0003,  # Very conservative LR
                "name_suffix": "aggressive_focal"
            })
            configurations.append(config1)

            # Configuration 2: Extended training with focal
            config2 = base_config.copy()
            config2.update({
                "loss": "focal",
                "focal_alpha": 1.5,
                "focal_gamma": 2.5,
                "epochs": 40,  # Extended training
                "batch": 24,
                "lr": 0.0005,
                "name_suffix": "extended_focal"
            })
            configurations.append(config2)

        elif difficulty == "extreme_critical":  # MP-IDB Stages
            # Configuration 1: Maximum focal loss with micro batch
            config1 = base_config.copy()
            config1.update({
                "loss": "focal",
                "focal_alpha": 3.0,  # Maximum alpha
                "focal_gamma": 4.0,  # Maximum gamma
                "epochs": 50,  # Extended training
                "batch": 8,   # Micro batch for stability
                "lr": 0.0001, # Very low LR
                "name_suffix": "max_focal"
            })
            configurations.append(config1)

            # Configuration 2: Conservative approach with longer training
            config2 = base_config.copy()
            config2.update({
                "loss": "focal",
                "focal_alpha": 2.5,
                "focal_gamma": 3.5,
                "epochs": 60,  # Very extended training
                "batch": 12,
                "lr": 0.0002,
                "name_suffix": "conservative_extended"
            })
            configurations.append(config2)

        return configurations

    def get_pipeline_configurations(self, dataset_name, selected_models=None):
        """Get complete configuration matrix for pipeline integration"""

        if selected_models is None:
            # Use all models
            all_models = []
            for category in self.model_categories.values():
                all_models.extend(category)
            selected_models = all_models

        pipeline_configs = {}

        for model_name in selected_models:
            model_configs = self.get_optimal_configurations(dataset_name, model_name)

            for i, config in enumerate(model_configs):
                # Create unique experiment name
                exp_key = f"{model_name}_{config['name_suffix']}"

                # Convert to pipeline format
                pipeline_config = {
                    "type": "pytorch",
                    "script": "scripts/training/12_train_pytorch_classification.py",
                    "model": config["model"],
                    "epochs": config["epochs"],
                    "batch": config["batch"],
                    "lr": config["lr"],
                    "loss": config["loss"],
                    "image_size": config["image_size"]
                }

                # Add focal loss parameters if needed
                if config["loss"] == "focal":
                    pipeline_config["focal_alpha"] = config["focal_alpha"]
                    pipeline_config["focal_gamma"] = config["focal_gamma"]

                pipeline_configs[exp_key] = pipeline_config

        return pipeline_configs

    def generate_analysis_report(self, dataset_name):
        """Generate expected results analysis"""

        report = {
            "dataset": dataset_name,
            "configuration_strategy": {},
            "expected_improvements": {},
            "total_experiments": 0
        }

        dataset_config = self.dataset_configs.get(dataset_name, {})
        difficulty = dataset_config.get("difficulty", "moderate")

        all_models = []
        for category in self.model_categories.values():
            all_models.extend(category)

        total_configs = 0
        for model in all_models:
            configs = self.get_optimal_configurations(dataset_name, model)
            total_configs += len(configs)

        report["total_experiments"] = total_configs
        report["configuration_strategy"][difficulty] = {
            "approach": self._get_strategy_description(difficulty),
            "expected_improvement": self._get_expected_improvement(difficulty)
        }

        return report

    def _get_strategy_description(self, difficulty):
        """Get strategy description for difficulty level"""
        strategies = {
            "moderate": "Standard + Focal Loss approaches for proven performance boost",
            "extreme_imbalance": "Aggressive focal loss with extended training for minority class handling",
            "extreme_critical": "Maximum focal loss with micro-batches and extended training for critical imbalance"
        }
        return strategies.get(difficulty, "Adaptive approach")

    def _get_expected_improvement(self, difficulty):
        """Get expected improvement percentage"""
        improvements = {
            "moderate": "5-10% improvement over baseline (proven: 86.27% → 88.24%)",
            "extreme_imbalance": "10-20% balanced accuracy improvement for minority classes",
            "extreme_critical": "20-50% improvement in minority class detection (critical for 31.63% baseline)"
        }
        return improvements.get(difficulty, "Variable improvement expected")

def main():
    """Demonstrate smart parameter matrix"""
    matrix = SmartParameterMatrix()

    # Generate configurations for each dataset
    datasets = ["iml_lifecycle", "mp_idb_species", "mp_idb_stages"]

    for dataset in datasets:
        print(f"\\n{'='*60}")
        print(f"SMART PARAMETER MATRIX: {dataset.upper()}")
        print(f"{'='*60}")

        # Get pipeline configurations
        configs = matrix.get_pipeline_configurations(dataset)

        print(f"Total configurations: {len(configs)}")
        print(f"\\nConfigurations:")

        for exp_name, config in configs.items():
            print(f"\\n{exp_name}:")
            for key, value in config.items():
                print(f"  {key}: {value}")

        # Generate analysis report
        report = matrix.generate_analysis_report(dataset)
        print(f"\\n[ANALYSIS] Expected total experiments: {report['total_experiments']}")

        difficulty = list(report['configuration_strategy'].keys())[0]
        strategy = report['configuration_strategy'][difficulty]
        print(f"[STRATEGY] {strategy['approach']}")
        print(f"[EXPECTED] {strategy['expected_improvement']}")

if __name__ == "__main__":
    main()