#!/usr/bin/env python3
"""
Generate Table2 Classification Performance Summary from Latest Experiment
Extracts classification results from experiment folders and creates publication-ready table
"""

import json
import csv
from pathlib import Path
import argparse
import re
from typing import Dict, List


# Model parameter counts (in millions)
MODEL_PARAMS = {
    'densenet121': 8.0,
    'efficientnet_b0': 5.3,
    'efficientnet_b1': 7.8,
    'efficientnet_b2': 9.2,
    'resnet50': 25.6,
    'resnet101': 44.5,
}

# Model display names
MODEL_NAMES = {
    'densenet121': 'DenseNet121',
    'efficientnet_b0': 'EfficientNet-B0',
    'efficientnet_b1': 'EfficientNet-B1',
    'efficientnet_b2': 'EfficientNet-B2',
    'resnet50': 'ResNet50',
    'resnet101': 'ResNet101',
}

# Dataset display names and class info
DATASET_INFO = {
    'iml_lifecycle': {
        'name': 'IML Lifecycle',
        'classes': ['gametocyte', 'ring', 'schizont', 'trophozoite']
    },
    'mp_idb_species': {
        'name': 'MP-IDB Species',
        'classes': ['P_falciparum', 'P_malariae', 'P_ovale', 'P_vivax']
    },
    'mp_idb_stages': {
        'name': 'MP-IDB Stages',
        'classes': ['gametocyte', 'ring', 'schizont', 'trophozoite']
    },
}


def extract_metrics_from_json(json_path: Path) -> Dict:
    """Extract metrics from table9_metrics.json"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        return {
            'accuracy': data.get('test_accuracy', data.get('overall_accuracy', 0.0)) * 100,
            'balanced_accuracy': data.get('overall_balanced_accuracy', 0.0) * 100,
            'per_class_metrics': data.get('per_class_metrics', {})
        }
    except Exception as e:
        print(f"[WARNING] Error reading {json_path}: {e}")
        return None


def extract_metrics_from_txt(txt_path: Path) -> Dict:
    """Extract metrics from results.txt (fallback)"""
    try:
        with open(txt_path, 'r') as f:
            content = f.read()

        # Extract accuracy
        test_acc_match = re.search(r'Test Acc: ([\d.]+)%', content)
        balanced_acc_match = re.search(r'Balanced Acc: ([\d.]+)%', content)

        if not test_acc_match:
            return None

        # Extract per-class F1 scores
        per_class_metrics = {}
        lines = content.split('\n')
        for line in lines:
            # Match classification report lines like "  gametocyte       0.93      0.95      0.94        41"
            match = re.match(r'\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if match:
                class_name = match.group(1)
                precision = float(match.group(2))
                recall = float(match.group(3))
                f1_score = float(match.group(4))
                support = float(match.group(5))

                per_class_metrics[class_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'support': support
                }

        return {
            'accuracy': float(test_acc_match.group(1)),
            'balanced_accuracy': float(balanced_acc_match.group(1)) if balanced_acc_match else 0.0,
            'per_class_metrics': per_class_metrics
        }
    except Exception as e:
        print(f"[WARNING] Error reading {txt_path}: {e}")
        return None


def find_minority_class_best_f1(per_class_metrics: Dict, dataset_key: str) -> tuple:
    """Find the minority class with best F1 score"""
    if not per_class_metrics:
        return "N/A", "N/A"

    # Get minority classes (based on support counts)
    classes_by_support = []
    for class_data in per_class_metrics.values():
        if isinstance(class_data, dict):
            class_name = class_data.get('class_name', '')
            support = class_data.get('support', 0)
            f1 = class_data.get('f1_score', 0)
            classes_by_support.append((class_name, support, f1))

    # Sort by support (ascending) to find minority classes
    classes_by_support.sort(key=lambda x: x[1])

    # Get the minority class with best F1 (typically the smallest 1-2 classes)
    if len(classes_by_support) >= 2:
        # Check the 2 smallest classes and pick the one with better F1
        minority_classes = classes_by_support[:2]
        best_minority = max(minority_classes, key=lambda x: x[2])
        return f"{best_minority[2]:.4f}", best_minority[0]
    elif len(classes_by_support) == 1:
        return f"{classes_by_support[0][2]:.4f}", classes_by_support[0][0]

    return "N/A", "N/A"


def process_experiment_folder(exp_path: Path, dataset_key: str) -> List[Dict]:
    """Process one experiment folder and extract all classification results"""
    results = []

    print(f"\n[DATASET] Processing {DATASET_INFO[dataset_key]['name']}...")
    print(f"[PATH] {exp_path}")

    # Find all classification folders (cls_*_focal)
    cls_folders = sorted(exp_path.glob('cls_*_focal'))

    for cls_folder in cls_folders:
        # Extract model name from folder (e.g., cls_densenet121_focal -> densenet121)
        folder_name = cls_folder.name
        model_key = folder_name.replace('cls_', '').replace('_focal', '')

        if model_key not in MODEL_PARAMS:
            print(f"[SKIP] Unknown model: {model_key}")
            continue

        print(f"  [MODEL] {MODEL_NAMES[model_key]}...", end=' ')

        # Try to extract metrics from JSON first, then TXT
        metrics = None
        json_path = cls_folder / 'table9_metrics.json'
        txt_path = cls_folder / 'results.txt'

        if json_path.exists():
            metrics = extract_metrics_from_json(json_path)
            source = 'JSON'
        elif txt_path.exists():
            metrics = extract_metrics_from_txt(txt_path)
            source = 'TXT'

        if not metrics:
            print(f"[FAILED] No metrics found")
            continue

        # Find minority class with best F1
        best_f1, minority_class = find_minority_class_best_f1(
            metrics['per_class_metrics'],
            dataset_key
        )

        results.append({
            'Dataset': DATASET_INFO[dataset_key]['name'],
            'Model': MODEL_NAMES[model_key],
            'Parameters (M)': MODEL_PARAMS[model_key],
            'Accuracy (%)': round(metrics['accuracy'], 2),
            'Balanced Accuracy (%)': round(metrics['balanced_accuracy'], 2),
            'Best F1-score (Minority Class)': f"{best_f1} ({minority_class})"
        })

        print(f"[OK] Acc={metrics['accuracy']:.2f}%, Bal={metrics['balanced_accuracy']:.2f}% ({source})")

    return results


def generate_table2(experiment_base: Path, output_path: Path):
    """Generate Table2 from experiment results"""

    print(f"\n{'='*80}")
    print(f"GENERATE TABLE 2: Classification Performance Summary")
    print(f"{'='*80}")
    print(f"[EXPERIMENT] {experiment_base}")
    print(f"[OUTPUT] {output_path}")

    all_results = []

    # Process each dataset
    for dataset_key in ['iml_lifecycle', 'mp_idb_species', 'mp_idb_stages']:
        exp_path = experiment_base / f'experiment_{dataset_key}'

        if not exp_path.exists():
            print(f"\n[WARNING] Dataset folder not found: {exp_path}")
            continue

        results = process_experiment_folder(exp_path, dataset_key)
        all_results.extend(results)

    if not all_results:
        print(f"\n[ERROR] No results found!")
        return False

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Dataset', 'Model', 'Parameters (M)', 'Accuracy (%)',
            'Balanced Accuracy (%)', 'Best F1-score (Minority Class)'
        ])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*80}")
    print(f"[SUCCESS] Table 2 generated successfully!")
    print(f"[SAVE] {output_path}")
    print(f"[ROWS] {len(all_results)} model results (from {len(all_results)//6} datasets)")
    print(f"{'='*80}\n")

    # Display summary
    print(f"[SUMMARY] BY DATASET:\n")
    for dataset_key in ['iml_lifecycle', 'mp_idb_species', 'mp_idb_stages']:
        dataset_name = DATASET_INFO[dataset_key]['name']
        dataset_results = [r for r in all_results if r['Dataset'] == dataset_name]

        if dataset_results:
            # Find best model by accuracy
            best_model = max(dataset_results, key=lambda x: x['Accuracy (%)'])
            print(f"  {dataset_name}:")
            print(f"    - Models: {len(dataset_results)}")
            print(f"    - Best: {best_model['Model']} ({best_model['Accuracy (%)']}% accuracy, {best_model['Balanced Accuracy (%)']}% balanced)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate Table2 Classification Performance Summary from experiment results'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Path to experiment base folder (e.g., results/optA_20251007_134458/experiments)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='luaran/tables/Table2_Classification_Performance_Summary.csv',
        help='Output CSV file path (default: luaran/tables/Table2_Classification_Performance_Summary.csv)'
    )

    args = parser.parse_args()

    experiment_base = Path(args.experiment)
    output_path = Path(args.output)

    if not experiment_base.exists():
        print(f"[ERROR] Experiment folder not found: {experiment_base}")
        return 1

    success = generate_table2(experiment_base, output_path)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
