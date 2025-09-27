#!/usr/bin/env python3
"""
Journal-Worthy Experimental Design for Malaria Classification
Focus on methodological contributions rather than hyperparameter tuning
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ExperimentalFactor:
    """Represents a factor for systematic experimental design"""
    name: str
    levels: List[str]
    description: str
    journal_relevance: str

class JournalExperimentalDesign:
    """Design systematic experiments for journal publication"""

    def __init__(self):
        # Define journal-worthy experimental factors
        self.experimental_factors = {
            # Factor 1: Loss Function (Novel contribution for extreme imbalance)
            "loss_function": ExperimentalFactor(
                name="Loss Function",
                levels=["cross_entropy", "focal_2_2", "focal_3_3", "focal_adaptive"],
                description="Systematic evaluation of loss functions for extreme class imbalance",
                journal_relevance="Novel contribution: Focal Loss parameter optimization for medical imaging"
            ),

            # Factor 2: Architecture Family (Comprehensive architectural study)
            "architecture_family": ExperimentalFactor(
                name="Architecture Family",
                levels=["cnn_traditional", "cnn_modern", "efficient_nets", "vision_transformer"],
                description="Systematic comparison of architectural paradigms for malaria classification",
                journal_relevance="Comprehensive study: First systematic architectural comparison for malaria parasites"
            ),

            # Factor 3: Transfer Learning Strategy (Medical domain specific)
            "transfer_strategy": ExperimentalFactor(
                name="Transfer Learning Strategy",
                levels=["imagenet_pretrained", "medical_finetuned", "from_scratch"],
                description="Impact of transfer learning on medical domain performance",
                journal_relevance="Domain adaptation: ImageNet vs Medical domain pretraining effectiveness"
            ),

            # Factor 4: Imbalance Handling Method (Systematic methodology)
            "imbalance_method": ExperimentalFactor(
                name="Imbalance Handling",
                levels=["none", "weighted_loss", "weighted_sampling", "combined_approach"],
                description="Systematic evaluation of class imbalance mitigation strategies",
                journal_relevance="Methodological framework: Imbalance handling for medical classification"
            )
        }

        # Define model mapping to architectural families
        self.architecture_mapping = {
            "cnn_traditional": ["resnet18", "resnet34", "densenet121"],
            "cnn_modern": ["convnext_tiny", "resnet101"],
            "efficient_nets": ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"],
            "vision_transformer": ["vit_b_16"]  # Add when available
        }

        # Define focal loss configurations with scientific justification
        self.focal_configurations = {
            "focal_2_2": {"alpha": 2.0, "gamma": 2.0, "justification": "Standard focal loss parameters"},
            "focal_3_3": {"alpha": 3.0, "gamma": 3.0, "justification": "Aggressive focusing for extreme imbalance"},
            "focal_adaptive": {"alpha": "dataset_dependent", "gamma": "dataset_dependent",
                             "justification": "Adaptive parameters based on imbalance ratio"}
        }

    def generate_systematic_experiments(self, dataset_name: str) -> Dict[str, Any]:
        """Generate systematic experimental design for journal publication"""

        # Get dataset characteristics for adaptive parameters
        dataset_info = self._get_dataset_characteristics(dataset_name)

        experiments = {
            "study_design": {
                "title": f"Systematic Evaluation of Deep Learning Approaches for {dataset_name.title()} Malaria Classification",
                "objective": "Comprehensive comparison of loss functions, architectures, and imbalance handling methods",
                "factors": len(self.experimental_factors),
                "total_conditions": self._calculate_total_conditions(dataset_name)
            },
            "experimental_conditions": {},
            "research_contributions": self._define_research_contributions(dataset_name),
            "expected_publications": self._generate_publication_strategy(dataset_name)
        }

        # Generate experimental conditions
        condition_id = 1

        # Focus on the most impactful combinations for journal
        priority_combinations = self._get_priority_combinations(dataset_name)

        for combination in priority_combinations:
            condition_name = f"condition_{condition_id:02d}_{combination['name']}"

            experiments["experimental_conditions"][condition_name] = {
                "condition_id": condition_id,
                "description": combination["description"],
                "scientific_rationale": combination["rationale"],
                "configuration": combination["config"],
                "expected_outcome": combination["expected"],
                "journal_significance": combination["significance"]
            }
            condition_id += 1

        return experiments

    def _get_dataset_characteristics(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset characteristics for adaptive configuration"""
        characteristics = {
            "iml_lifecycle": {
                "imbalance_ratio": 12.5,
                "difficulty": "moderate",
                "baseline_performance": 82.35,
                "priority_challenge": "minority_class_detection"
            },
            "mp_idb_species": {
                "imbalance_ratio": 40.5,
                "difficulty": "extreme",
                "baseline_performance": 95.83,
                "priority_challenge": "extreme_imbalance_bias"
            },
            "mp_idb_stages": {
                "imbalance_ratio": 43.2,
                "difficulty": "critical",
                "baseline_performance": 31.63,
                "priority_challenge": "complete_minority_failure"
            }
        }
        return characteristics.get(dataset_name, {})

    def _get_priority_combinations(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get priority experimental combinations for journal impact"""

        dataset_chars = self._get_dataset_characteristics(dataset_name)
        difficulty = dataset_chars.get("difficulty", "moderate")

        combinations = []

        if difficulty == "moderate":  # IML Lifecycle
            combinations.extend([
                {
                    "name": "baseline_comparison",
                    "description": "Standard CNN with Cross-Entropy vs Focal Loss",
                    "rationale": "Establish baseline and demonstrate focal loss effectiveness",
                    "config": {
                        "models": ["resnet18", "efficientnet_b1"],
                        "losses": ["cross_entropy", "focal_2_2"],
                        "transfer": "imagenet_pretrained",
                        "imbalance": "weighted_loss"
                    },
                    "expected": "5-10% improvement with focal loss",
                    "significance": "Validates focal loss for moderate imbalance"
                },
                {
                    "name": "architecture_comparison",
                    "description": "Architectural paradigm comparison with optimal loss",
                    "rationale": "Systematic evaluation of architectural impact",
                    "config": {
                        "models": ["resnet18", "densenet121", "efficientnet_b1", "convnext_tiny"],
                        "losses": ["focal_2_2"],
                        "transfer": "imagenet_pretrained",
                        "imbalance": "combined_approach"
                    },
                    "expected": "Identify best architecture for medical imaging",
                    "significance": "First comprehensive architectural study for malaria"
                }
            ])

        elif difficulty == "extreme":  # MP-IDB Species
            combinations.extend([
                {
                    "name": "extreme_imbalance_methodology",
                    "description": "Progressive focal loss parameters for extreme imbalance",
                    "rationale": "Develop methodology for extreme imbalance (40:1 ratio)",
                    "config": {
                        "models": ["efficientnet_b1", "resnet18"],
                        "losses": ["cross_entropy", "focal_2_2", "focal_3_3"],
                        "transfer": "imagenet_pretrained",
                        "imbalance": "combined_approach"
                    },
                    "expected": "Significant minority class improvement",
                    "significance": "Novel methodology for extreme medical imbalance"
                },
                {
                    "name": "transfer_learning_medical",
                    "description": "Medical domain transfer learning effectiveness",
                    "rationale": "Evaluate medical vs natural image pretraining",
                    "config": {
                        "models": ["efficientnet_b1"],
                        "losses": ["focal_3_3"],
                        "transfer": ["imagenet_pretrained", "from_scratch"],
                        "imbalance": "combined_approach"
                    },
                    "expected": "Quantify transfer learning benefit in medical domain",
                    "significance": "Domain adaptation insights for medical AI"
                }
            ])

        elif difficulty == "critical":  # MP-IDB Stages
            combinations.extend([
                {
                    "name": "critical_imbalance_rescue",
                    "description": "Maximum intervention for critical imbalance failure",
                    "rationale": "Rescue approach for complete minority class failure",
                    "config": {
                        "models": ["efficientnet_b1", "densenet121"],
                        "losses": ["focal_3_3", "focal_adaptive"],
                        "transfer": "imagenet_pretrained",
                        "imbalance": "combined_approach"
                    },
                    "expected": "Recover minority class detection from 0%",
                    "significance": "Rescue methodology for extreme medical imbalance"
                },
                {
                    "name": "ensemble_methodology",
                    "description": "Multi-model ensemble for critical cases",
                    "rationale": "When single models fail, systematic ensemble approach",
                    "config": {
                        "models": ["resnet18", "efficientnet_b1", "densenet121"],
                        "losses": ["focal_3_3"],
                        "transfer": "imagenet_pretrained",
                        "imbalance": "combined_approach",
                        "ensemble": True
                    },
                    "expected": "Systematic improvement through ensemble",
                    "significance": "Ensemble methodology for medical edge cases"
                }
            ])

        return combinations

    def _define_research_contributions(self, dataset_name: str) -> Dict[str, str]:
        """Define research contributions for journal publication"""
        return {
            "methodological": "Systematic framework for extreme class imbalance in medical imaging",
            "empirical": f"Comprehensive evaluation on {dataset_name} malaria classification dataset",
            "architectural": "First systematic comparison of modern CNN architectures for malaria parasites",
            "loss_function": "Focal loss parameter optimization for medical domain extreme imbalance",
            "practical": "Actionable guidelines for medical AI practitioners dealing with imbalanced datasets"
        }

    def _generate_publication_strategy(self, dataset_name: str) -> Dict[str, Any]:
        """Generate publication strategy based on experimental design"""
        return {
            "primary_paper": {
                "title": f"Deep Learning for Extreme Class Imbalance: A Systematic Study on {dataset_name.title()} Malaria Classification",
                "target_journals": ["IEEE Transactions on Medical Imaging", "Medical Image Analysis", "Pattern Recognition"],
                "key_contributions": [
                    "Focal loss optimization methodology for medical extreme imbalance",
                    "Comprehensive architectural evaluation for malaria classification",
                    "Practical framework for medical AI imbalance handling"
                ]
            },
            "follow_up_papers": [
                {
                    "title": "Transfer Learning Effectiveness in Medical Domain: Natural vs Medical Image Pretraining",
                    "focus": "Transfer learning strategy comparison"
                },
                {
                    "title": "Ensemble Methods for Critical Medical Imbalance: When Single Models Fail",
                    "focus": "Ensemble methodology for extreme cases"
                }
            ]
        }

    def _calculate_total_conditions(self, dataset_name: str) -> int:
        """Calculate total experimental conditions"""
        priority_combinations = self._get_priority_combinations(dataset_name)
        total = 0
        for combination in priority_combinations:
            config = combination["config"]
            models = len(config["models"]) if isinstance(config["models"], list) else 1
            losses = len(config["losses"]) if isinstance(config["losses"], list) else 1
            total += models * losses
        return total

    def generate_pipeline_integration(self, dataset_name: str) -> Dict[str, Any]:
        """Generate pipeline-compatible configuration from experimental design"""

        experiments = self.generate_systematic_experiments(dataset_name)
        pipeline_configs = {}

        for condition_name, condition in experiments["experimental_conditions"].items():
            config = condition["configuration"]

            # Generate pipeline configurations for each model-loss combination
            models = config["models"] if isinstance(config["models"], list) else [config["models"]]
            losses = config["losses"] if isinstance(config["losses"], list) else [config["losses"]]

            for model in models:
                for loss in losses:
                    # Create pipeline-compatible configuration
                    exp_key = f"{model}_{loss}_{condition['condition_id']}"

                    pipeline_config = {
                        "type": "pytorch",
                        "script": "scripts/training/12_train_pytorch_classification.py",
                        "model": model,
                        "loss": loss,
                        "epochs": 25,  # Standard epochs for all experiments
                        "batch": 32,   # Standard batch for all experiments
                        "lr": 0.001 if loss == "cross_entropy" else 0.0005,  # Only essential LR difference
                        "experimental_condition": condition_name,
                        "scientific_rationale": condition["scientific_rationale"]
                    }

                    # Add focal loss parameters if needed
                    if loss.startswith("focal"):
                        focal_config = self.focal_configurations.get(loss, {})
                        if focal_config.get("alpha") != "dataset_dependent":
                            pipeline_config["focal_alpha"] = focal_config["alpha"]
                            pipeline_config["focal_gamma"] = focal_config["gamma"]
                        else:
                            # Adaptive focal parameters based on dataset
                            dataset_chars = self._get_dataset_characteristics(dataset_name)
                            imbalance_ratio = dataset_chars.get("imbalance_ratio", 10.0)
                            pipeline_config["focal_alpha"] = min(3.0, imbalance_ratio / 15.0)
                            pipeline_config["focal_gamma"] = min(4.0, imbalance_ratio / 10.0)

                    pipeline_configs[exp_key] = pipeline_config

        return {
            "experimental_design": experiments,
            "pipeline_configurations": pipeline_configs,
            "total_experiments": len(pipeline_configs)
        }

def main():
    """Demonstrate journal-worthy experimental design"""
    design = JournalExperimentalDesign()

    datasets = ["iml_lifecycle", "mp_idb_species", "mp_idb_stages"]

    for dataset in datasets:
        print(f"\\n{'='*80}")
        print(f"JOURNAL-WORTHY EXPERIMENTAL DESIGN: {dataset.upper()}")
        print(f"{'='*80}")

        # Generate complete experimental design
        result = design.generate_pipeline_integration(dataset)

        experiments = result["experimental_design"]
        pipeline_configs = result["pipeline_configurations"]

        print(f"\\n[STUDY DESIGN]")
        print(f"Title: {experiments['study_design']['title']}")
        print(f"Objective: {experiments['study_design']['objective']}")
        print(f"Total Experimental Conditions: {experiments['study_design']['total_conditions']}")
        print(f"Total Pipeline Configurations: {result['total_experiments']}")

        print(f"\\n[RESEARCH CONTRIBUTIONS]")
        for contrib_type, description in experiments["research_contributions"].items():
            print(f"• {contrib_type.title()}: {description}")

        print(f"\\n[EXPERIMENTAL CONDITIONS]")
        for condition_name, condition in experiments["experimental_conditions"].items():
            print(f"\\n{condition_name}:")
            print(f"  Description: {condition['description']}")
            print(f"  Rationale: {condition['scientific_rationale']}")
            print(f"  Expected: {condition['expected_outcome']}")
            print(f"  Significance: {condition['journal_significance']}")

        print(f"\\n[PUBLICATION STRATEGY]")
        primary = experiments["expected_publications"]["primary_paper"]
        print(f"Primary Paper: {primary['title']}")
        print(f"Target Journals: {', '.join(primary['target_journals'])}")

if __name__ == "__main__":
    main()