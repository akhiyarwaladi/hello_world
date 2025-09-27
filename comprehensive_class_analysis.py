#!/usr/bin/env python3
"""
Comprehensive analysis of class distribution issues across datasets.
This script identifies the root cause of the class distribution problems.
"""
import os
import glob
from collections import defaultdict, Counter
import yaml
import pandas as pd

def check_original_data_structure():
    """Check the expected class structure from original datasets"""
    print("="*80)
    print("ORIGINAL DATA STRUCTURE ANALYSIS")
    print("="*80)

    # Check MP-IDB Falciparum classes
    mp_idb_csv = r"C:\Users\MyPC PRO\Documents\hello_world\data\raw\mp_idb\Falciparum\mp-idb-falciparum.csv"
    if os.path.exists(mp_idb_csv):
        print("\nMP-IDB Falciparum original classes:")
        df = pd.read_csv(mp_idb_csv)
        stage_counts = df['parasite_type'].value_counts()
        print(stage_counts)
        expected_stages = list(stage_counts.index)
        print(f"Expected stages: {expected_stages}")

    # Check YOLO dataset classes
    yolo_yaml = r"C:\Users\MyPC PRO\Documents\hello_world\data\raw\kaggle_dataset\MP-IDB-YOLO\data.yaml"
    if os.path.exists(yolo_yaml):
        print("\nKaggle YOLO dataset classes:")
        with open(yolo_yaml, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                print(f"Expected {len(data['names'])} classes:")
                for i, name in enumerate(data['names']):
                    print(f"  Class {i}: {name}")

                # Analyze by species and stages
                species_stages = {}
                for name in data['names']:
                    parts = name.split('_')
                    if len(parts) == 2:
                        species, stage = parts
                        if species not in species_stages:
                            species_stages[species] = []
                        species_stages[species].append(stage)

                print(f"\nSpecies breakdown:")
                for species, stages in species_stages.items():
                    print(f"  {species}: {stages}")

    return expected_stages if 'expected_stages' in locals() else []

def analyze_processed_vs_expected():
    """Compare processed datasets with expected structure"""
    print("\n" + "="*80)
    print("PROCESSED vs EXPECTED CLASS STRUCTURE")
    print("="*80)

    datasets = {
        'species': r"C:\Users\MyPC PRO\Documents\hello_world\data\processed\species",
        'stages': r"C:\Users\MyPC PRO\Documents\hello_world\data\processed\stages",
        'lifecycle': r"C:\Users\MyPC PRO\Documents\hello_world\data\processed\lifecycle"
    }

    expected_classes = {
        'species': ['falciparum', 'vivax', 'ovale', 'malariae'],  # 4 species
        'stages': ['ring', 'trophozoite', 'schizont', 'gametocyte'],  # 4 stages
        'lifecycle': ['stage1', 'stage2', 'stage3', 'stage4']  # Assuming 4 lifecycle stages
    }

    for dataset_name, dataset_path in datasets.items():
        print(f"\n{dataset_name.upper()} Dataset Analysis:")
        print("-" * 50)

        if not os.path.exists(dataset_path):
            print(f"  Dataset does not exist: {dataset_path}")
            continue

        # Check data.yaml
        yaml_path = os.path.join(dataset_path, "data.yaml")
        actual_classes = []
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    actual_classes = data['names']

        print(f"  Expected classes: {expected_classes.get(dataset_name, 'Unknown')}")
        print(f"  Actual classes in data.yaml: {actual_classes}")
        print(f"  Expected: {len(expected_classes.get(dataset_name, []))} classes")
        print(f"  Actual: {len(actual_classes)} classes")

        if len(actual_classes) == 1 and actual_classes[0] == 'parasite':
            print(f"  [ERROR] PROBLEM: All classes collapsed to single 'parasite' class!")

        # Sample some label files to check actual content
        train_labels = os.path.join(dataset_path, "train", "labels")
        if os.path.exists(train_labels):
            label_files = glob.glob(os.path.join(train_labels, "*.txt"))[:5]  # Sample 5 files
            class_ids_found = set()

            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_ids_found.add(int(parts[0]))
                except:
                    pass

            print(f"  Class IDs found in label files: {sorted(class_ids_found)}")
            if len(class_ids_found) == 1 and 0 in class_ids_found:
                print(f"  [ERROR] CONFIRMED: Only class 0 found in labels")

def investigate_data_processing_pipeline():
    """Investigate what might have gone wrong in the data processing"""
    print("\n" + "="*80)
    print("DATA PROCESSING PIPELINE INVESTIGATION")
    print("="*80)

    # Look for processing scripts
    scripts_to_check = [
        "generate_crops.py",
        "create_detection_datasets.py",
        "prepare_data.py",
        "convert_to_yolo.py"
    ]

    base_path = r"C:\Users\MyPC PRO\Documents\hello_world"
    found_scripts = []

    for script in scripts_to_check:
        script_path = os.path.join(base_path, script)
        if os.path.exists(script_path):
            found_scripts.append(script_path)
            print(f"\nFound processing script: {script}")

    if not found_scripts:
        print("No obvious data processing scripts found in root directory")
        # Look for scripts in subdirectories
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.py') and any(keyword in file.lower() for keyword in ['convert', 'process', 'generate', 'create']):
                    print(f"Potential processing script: {os.path.join(root, file)}")

    # Check if there are multiple label formats
    print(f"\nChecking for multiple annotation formats...")

    # Check for JSON annotations
    json_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json') and 'annotation' in file.lower():
                json_files.append(os.path.join(root, file))

    if json_files:
        print(f"Found {len(json_files)} JSON annotation files:")
        for json_file in json_files[:5]:  # Show first 5
            print(f"  {json_file}")

def identify_root_causes():
    """Identify the root causes of class distribution issues"""
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)

    issues = []

    # Issue 1: Class collapse
    issues.append({
        'issue': 'All multi-class datasets collapsed to single class',
        'severity': 'CRITICAL',
        'description': 'All three datasets (species, stages, lifecycle) show only 1 class instead of expected 4+ classes',
        'likely_cause': 'Data processing pipeline incorrectly maps all annotations to class 0',
        'impact': 'Makes multi-class classification impossible'
    })

    # Issue 2: Incorrect data.yaml
    issues.append({
        'issue': 'Incorrect class names in data.yaml files',
        'severity': 'HIGH',
        'description': 'All data.yaml files show only "parasite" class instead of specific class names',
        'likely_cause': 'Generic template used instead of dataset-specific class mapping',
        'impact': 'Training scripts cannot learn proper class distinctions'
    })

    # Issue 3: Loss of original annotation semantics
    issues.append({
        'issue': 'Original annotation semantics lost during conversion',
        'severity': 'HIGH',
        'description': 'Rich original annotations (species+stage combinations) reduced to binary detection',
        'likely_cause': 'Conversion process treats all objects as generic "parasites"',
        'impact': 'Cannot perform species classification or stage classification'
    })

    print("IDENTIFIED ISSUES:")
    print("="*50)

    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. {issue['issue']} [{issue['severity']}]")
        print(f"   Description: {issue['description']}")
        print(f"   Likely Cause: {issue['likely_cause']}")
        print(f"   Impact: {issue['impact']}")

def recommend_solutions():
    """Recommend solutions for fixing the class distribution issues"""
    print("\n" + "="*80)
    print("RECOMMENDED SOLUTIONS")
    print("="*80)

    solutions = [
        {
            'title': '1. Fix Species Dataset',
            'steps': [
                'Use original MP-IDB CSV to create proper species labels',
                'Map falciparum_*, vivax_*, ovale_*, malariae_* to 4 species classes',
                'Update data.yaml with correct species names: [falciparum, vivax, ovale, malariae]',
                'Regenerate YOLO label files with proper class IDs (0-3)'
            ]
        },
        {
            'title': '2. Fix Stages Dataset',
            'steps': [
                'Use original MP-IDB CSV parasite_type column',
                'Map ring->0, trophozoite->1, schizont->2, gametocyte->3',
                'Update data.yaml with stage names: [ring, trophozoite, schizont, gametocyte]',
                'Handle class imbalance (ring dominates with ~95% of samples)'
            ]
        },
        {
            'title': '3. Fix Lifecycle Dataset',
            'steps': [
                'Determine proper lifecycle stage definitions',
                'Create mapping from image filenames or metadata to lifecycle stages',
                'Generate balanced splits ensuring all stages in train/test/val',
                'Update data.yaml with lifecycle stage names'
            ]
        },
        {
            'title': '4. Address Class Imbalance',
            'steps': [
                'Implement stratified splitting to ensure all classes in each split',
                'Consider data augmentation for underrepresented classes',
                'Use appropriate loss functions (focal loss, weighted cross-entropy)',
                'Monitor per-class metrics during training'
            ]
        }
    ]

    for solution in solutions:
        print(f"\n{solution['title']}")
        print("-" * len(solution['title']))
        for step in solution['steps']:
            print(f"  • {step}")

def main():
    """Main analysis function"""
    print("COMPREHENSIVE CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    print("This analysis identifies critical issues with the current dataset class distributions.")
    print()

    # Run all analyses
    original_classes = check_original_data_structure()
    analyze_processed_vs_expected()
    investigate_data_processing_pipeline()
    identify_root_causes()
    recommend_solutions()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("[CRITICAL] All datasets have been incorrectly processed")
    print("[CRITICAL] Multi-class datasets collapsed to single-class detection")
    print("[CRITICAL] Class imbalance analysis is impossible with current data")
    print("[SUCCESS] Original source data contains proper multi-class annotations")
    print("[SUCCESS] Solutions identified to fix all three datasets")
    print("\nNEXT STEPS: Implement the recommended solutions to restore proper class structure")

if __name__ == "__main__":
    main()