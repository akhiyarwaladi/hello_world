#!/usr/bin/env python3
"""
Smart Dataset Detector for Malaria Detection Analysis
Automatically detects which dataset was used for training and uses the same for validation
"""

import os
import yaml
from pathlib import Path
import json

def detect_training_dataset(model_path_or_experiment):
    """
    Detect which dataset was used for training by reading args.yaml or experiment structure

    Args:
        model_path_or_experiment: Path to model or experiment directory

    Returns:
        dict: Dataset information with path and type
    """

    model_path = Path(model_path_or_experiment)

    # Case 1: Direct model path (e.g., results/.../weights/best.pt)
    if model_path.name == "best.pt":
        experiment_dir = model_path.parent.parent
        args_file = experiment_dir / "args.yaml"

    # Case 2: Experiment directory path
    elif "detection" in str(model_path):
        # Find args.yaml in detection subdirectory
        args_files = list(model_path.rglob("args.yaml"))
        if args_files:
            args_file = args_files[0]
            experiment_dir = args_file.parent
        else:
            return None

    # Case 3: Top-level experiment directory
    else:
        detection_dirs = list(model_path.glob("detection/*/"))
        if detection_dirs:
            detection_dir = detection_dirs[0]
            experiment_dirs = list(detection_dir.glob("*/"))
            if experiment_dirs:
                args_file = experiment_dirs[0] / "args.yaml"
                experiment_dir = experiment_dirs[0]
            else:
                return None
        else:
            return None

    # Read args.yaml to get dataset info
    if args_file.exists():
        try:
            with open(args_file, 'r') as f:
                args_data = yaml.safe_load(f)

            training_data_path = args_data.get('data', '')

            # Determine dataset type and correct test path
            if 'kaggle_pipeline_ready' in training_data_path:
                dataset_info = {
                    'type': 'kaggle',
                    'train_data_yaml': training_data_path,
                    'test_data_yaml': training_data_path,  # Use same dataset for consistent evaluation
                    'name': 'Kaggle MP-IDB Pipeline Ready',
                    'description': 'Optimized Kaggle dataset with corrected annotations'
                }
            elif 'integrated/yolo' in training_data_path:
                dataset_info = {
                    'type': 'integrated',
                    'train_data_yaml': training_data_path,
                    'test_data_yaml': training_data_path,  # Use same dataset for consistent evaluation
                    'name': 'Integrated YOLO Dataset',
                    'description': 'Integrated MP-IDB dataset in YOLO format'
                }
            else:
                # Default case - use the same dataset that was used for training
                dataset_info = {
                    'type': 'custom',
                    'train_data_yaml': training_data_path,
                    'test_data_yaml': training_data_path,
                    'name': 'Custom Dataset',
                    'description': f'Custom dataset: {training_data_path}'
                }

            dataset_info['experiment_dir'] = str(experiment_dir)
            dataset_info['args_file'] = str(args_file)

            return dataset_info

        except Exception as e:
            print(f"Error reading args.yaml: {e}")
            return None

    else:
        print(f"❌ args.yaml not found at: {args_file}")
        return None

def get_consistent_dataset_for_analysis(model_path):
    """
    Get the consistent dataset path for analysis based on training dataset

    Args:
        model_path: Path to the model or experiment

    Returns:
        str: Path to the correct data.yaml for analysis
    """

    dataset_info = detect_training_dataset(model_path)

    if dataset_info:
        print(f"✅ Detected training dataset: {dataset_info['name']}")
        print(f"📁 Dataset type: {dataset_info['type']}")
        print(f"🎯 Using consistent dataset: {dataset_info['test_data_yaml']}")
        return dataset_info['test_data_yaml']
    else:
        print("⚠️ Could not detect training dataset, using default")
        return "data/integrated/yolo/data.yaml"

def validate_dataset_consistency(experiment_path):
    """
    Validate that analysis is using the same dataset as training

    Args:
        experiment_path: Path to experiment directory

    Returns:
        dict: Validation report
    """

    dataset_info = detect_training_dataset(experiment_path)

    if not dataset_info:
        return {
            'status': 'error',
            'message': 'Could not detect training dataset'
        }

    # Check if test dataset exists
    test_data_path = Path(dataset_info['test_data_yaml'])
    if not test_data_path.exists():
        return {
            'status': 'error',
            'message': f'Test dataset not found: {test_data_path}',
            'dataset_info': dataset_info
        }

    # Load test dataset yaml to verify structure
    try:
        with open(test_data_path, 'r') as f:
            test_data = yaml.safe_load(f)

        # Check for required keys
        required_keys = ['train', 'val', 'test', 'names']
        missing_keys = [key for key in required_keys if key not in test_data]

        if missing_keys:
            return {
                'status': 'warning',
                'message': f'Dataset yaml missing keys: {missing_keys}',
                'dataset_info': dataset_info
            }

        return {
            'status': 'success',
            'message': 'Dataset consistency validated',
            'dataset_info': dataset_info,
            'test_data': test_data
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error validating dataset: {e}',
            'dataset_info': dataset_info
        }

def create_dataset_consistency_report(experiment_path, output_path=None):
    """
    Create a comprehensive dataset consistency report

    Args:
        experiment_path: Path to experiment directory
        output_path: Where to save the report (optional)

    Returns:
        dict: Complete consistency report
    """

    validation = validate_dataset_consistency(experiment_path)

    report = {
        'timestamp': str(Path(experiment_path).stat().st_mtime),
        'experiment_path': str(experiment_path),
        'validation': validation
    }

    if validation['status'] == 'success':
        dataset_info = validation['dataset_info']
        test_data = validation['test_data']

        report['analysis'] = {
            'training_dataset': dataset_info['train_data_yaml'],
            'test_dataset': dataset_info['test_data_yaml'],
            'consistency': 'PASS' if dataset_info['train_data_yaml'] == dataset_info['test_data_yaml'] else 'FAIL',
            'dataset_type': dataset_info['type'],
            'dataset_name': dataset_info['name'],
            'num_classes': len(test_data.get('names', [])),
            'class_names': test_data.get('names', [])
        }

        # Check if paths exist
        train_path = Path(dataset_info['train_data_yaml']).parent / test_data.get('train', '')
        val_path = Path(dataset_info['train_data_yaml']).parent / test_data.get('val', '')
        test_path = Path(dataset_info['train_data_yaml']).parent / test_data.get('test', '')

        report['analysis']['path_validation'] = {
            'train_exists': train_path.exists() if train_path.name else False,
            'val_exists': val_path.exists() if val_path.name else False,
            'test_exists': test_path.exists() if test_path.name else False,
            'train_path': str(train_path),
            'val_path': str(val_path),
            'test_path': str(test_path)
        }

    # Save report if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📊 Dataset consistency report saved: {output_file}")

    return report

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Smart Dataset Detector for Malaria Analysis")
    parser.add_argument('--experiment', type=str, required=True,
                       help='Path to experiment directory or model')
    parser.add_argument('--report', type=str,
                       help='Output path for consistency report (optional)')
    parser.add_argument('--validate', action='store_true',
                       help='Run full validation and report')

    args = parser.parse_args()

    if args.validate:
        report = create_dataset_consistency_report(args.experiment, args.report)

        print("📊 DATASET CONSISTENCY REPORT")
        print("=" * 50)
        print(f"Experiment: {report['experiment_path']}")
        print(f"Status: {report['validation']['status'].upper()}")
        print(f"Message: {report['validation']['message']}")

        if 'analysis' in report:
            analysis = report['analysis']
            print(f"\nDataset Analysis:")
            print(f"  Type: {analysis['dataset_type']}")
            print(f"  Name: {analysis['dataset_name']}")
            print(f"  Consistency: {analysis['consistency']}")
            print(f"  Classes: {analysis['num_classes']}")
            print(f"  Training Data: {analysis['training_dataset']}")
            print(f"  Test Data: {analysis['test_dataset']}")

            path_val = analysis['path_validation']
            print(f"\nPath Validation:")
            print(f"  Train exists: {path_val['train_exists']}")
            print(f"  Val exists: {path_val['val_exists']}")
            print(f"  Test exists: {path_val['test_exists']}")

    else:
        # Simple detection mode
        dataset_path = get_consistent_dataset_for_analysis(args.experiment)
        print(f"Consistent dataset path: {dataset_path}")

if __name__ == "__main__":
    main()