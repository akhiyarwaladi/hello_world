#!/usr/bin/env python3
"""
Script to analyze class distributions across the three datasets:
- species, stages, lifecycle
"""
import os
import glob
from collections import defaultdict, Counter
import yaml

def analyze_dataset(dataset_path, dataset_name):
    """Analyze class distribution for a single dataset"""
    print(f"\n{'='*60}")
    print(f"ANALYZING DATASET: {dataset_name.upper()}")
    print(f"{'='*60}")

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        return None

    # Load data.yaml to get class names
    yaml_path = os.path.join(dataset_path, "data.yaml")
    class_names = ["unknown"]  # Default fallback
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    class_names = data['names']
                    print(f"Class names from data.yaml: {class_names}")
        except Exception as e:
            print(f"Error reading data.yaml: {e}")

    results = {}

    # Analyze each split
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(dataset_path, split, 'labels')

        if not os.path.exists(split_path):
            print(f"Split path does not exist: {split_path}")
            continue

        print(f"\nAnalyzing {split} split...")

        # Count files and annotations per class
        label_files = glob.glob(os.path.join(split_path, "*.txt"))

        if not label_files:
            print(f"No label files found in {split_path}")
            continue

        class_counts = Counter()
        total_annotations = 0
        files_with_annotations = 0

        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    if lines:  # File has annotations
                        files_with_annotations += 1
                        for line in lines:
                            line = line.strip()
                            if line:  # Non-empty line
                                parts = line.split()
                                if parts:
                                    class_id = int(parts[0])
                                    class_counts[class_id] += 1
                                    total_annotations += 1
            except Exception as e:
                print(f"Error reading {label_file}: {e}")

        # Store results
        results[split] = {
            'total_files': len(label_files),
            'files_with_annotations': files_with_annotations,
            'total_annotations': total_annotations,
            'class_counts': dict(class_counts),
            'class_percentages': {cls: (count/total_annotations)*100 if total_annotations > 0 else 0
                                for cls, count in class_counts.items()}
        }

        # Print summary
        print(f"  Total label files: {len(label_files)}")
        print(f"  Files with annotations: {files_with_annotations}")
        print(f"  Total annotations: {total_annotations}")
        print(f"  Class distribution:")

        if total_annotations > 0:
            for class_id in sorted(class_counts.keys()):
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                count = class_counts[class_id]
                percentage = (count/total_annotations)*100
                print(f"    Class {class_id} ({class_name}): {count} annotations ({percentage:.1f}%)")
        else:
            print("    No annotations found!")

    return results

def compare_splits(results, dataset_name):
    """Compare class distributions across splits"""
    print(f"\n{'='*40}")
    print(f"SPLIT COMPARISON FOR {dataset_name.upper()}")
    print(f"{'='*40}")

    all_classes = set()
    for split_data in results.values():
        all_classes.update(split_data['class_counts'].keys())

    if not all_classes:
        print("No classes found in any split!")
        return

    print(f"{'Class':<10} {'Train':<10} {'Test':<10} {'Val':<10} {'Total':<10}")
    print("-" * 55)

    for class_id in sorted(all_classes):
        train_count = results.get('train', {}).get('class_counts', {}).get(class_id, 0)
        test_count = results.get('test', {}).get('class_counts', {}).get(class_id, 0)
        val_count = results.get('val', {}).get('class_counts', {}).get(class_id, 0)
        total_count = train_count + test_count + val_count

        print(f"{class_id:<10} {train_count:<10} {test_count:<10} {val_count:<10} {total_count:<10}")

    # Check for classes missing in test/val
    print(f"\nSPLIT QUALITY ANALYSIS:")
    for class_id in sorted(all_classes):
        train_count = results.get('train', {}).get('class_counts', {}).get(class_id, 0)
        test_count = results.get('test', {}).get('class_counts', {}).get(class_id, 0)
        val_count = results.get('val', {}).get('class_counts', {}).get(class_id, 0)

        issues = []
        if train_count > 0 and test_count == 0:
            issues.append("missing from test")
        if train_count > 0 and val_count == 0:
            issues.append("missing from val")
        if test_count == 0 and val_count == 0:
            issues.append("only in train")

        if issues:
            print(f"  Class {class_id}: {', '.join(issues)}")

def main():
    base_path = r"C:\Users\MyPC PRO\Documents\hello_world\data\processed"
    datasets = {
        'species': os.path.join(base_path, 'species'),
        'stages': os.path.join(base_path, 'stages'),
        'lifecycle': os.path.join(base_path, 'lifecycle')
    }

    all_results = {}

    # Analyze each dataset
    for dataset_name, dataset_path in datasets.items():
        results = analyze_dataset(dataset_path, dataset_name)
        if results:
            all_results[dataset_name] = results
            compare_splits(results, dataset_name)

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()} Dataset Summary:")

        total_train = results.get('train', {}).get('total_annotations', 0)
        total_test = results.get('test', {}).get('total_annotations', 0)
        total_val = results.get('val', {}).get('total_annotations', 0)
        total_all = total_train + total_test + total_val

        print(f"  Total annotations: {total_all}")
        print(f"  Train: {total_train} ({total_train/total_all*100 if total_all > 0 else 0:.1f}%)")
        print(f"  Test: {total_test} ({total_test/total_all*100 if total_all > 0 else 0:.1f}%)")
        print(f"  Val: {total_val} ({total_val/total_all*100 if total_all > 0 else 0:.1f}%)")

        # Count unique classes
        all_classes = set()
        for split_data in results.values():
            all_classes.update(split_data['class_counts'].keys())
        print(f"  Number of classes: {len(all_classes)}")

        # Check for severe imbalance (>90% in one class)
        if total_all > 0:
            for split in ['train', 'test', 'val']:
                if split in results:
                    for class_id, count in results[split]['class_counts'].items():
                        percentage = count / results[split]['total_annotations'] * 100
                        if percentage > 90:
                            print(f"  WARNING: Class {class_id} dominates {split} split ({percentage:.1f}%)")

if __name__ == "__main__":
    main()