#!/usr/bin/env python3
"""
Analyze Classification Dataset Problem
Investigate why classification accuracy is 100% (suspicious)
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

def analyze_dataset_structure():
    """Analyze classification dataset structure"""

    print("🔍 ANALYZING CLASSIFICATION DATASET PROBLEM")
    print("=" * 60)

    dataset_path = Path("data/classification_crops")

    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return

    # Check each split
    splits = ['train', 'val', 'test']
    total_images = 0
    class_distribution = {}

    for split in splits:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"❌ Split not found: {split_path}")
            continue

        print(f"\n📁 {split.upper()} Split:")

        # Find all directories (classes)
        class_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        print(f"   Classes found: {[d.name for d in class_dirs]}")

        split_images = 0
        for class_dir in class_dirs:
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            class_count = len(image_files)
            split_images += class_count

            if class_dir.name not in class_distribution:
                class_distribution[class_dir.name] = {'train': 0, 'val': 0, 'test': 0}
            class_distribution[class_dir.name][split] = class_count

            print(f"   - {class_dir.name}: {class_count} images")

        print(f"   Total: {split_images} images")
        total_images += split_images

    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Total images: {total_images}")
    print(f"   Total classes: {len(class_distribution)}")

    # Analyze class distribution
    print(f"\n📈 CLASS DISTRIBUTION:")
    df_data = []
    for class_name, counts in class_distribution.items():
        total_class = sum(counts.values())
        print(f"   - {class_name}: {total_class} images")
        print(f"     - Train: {counts['train']} | Val: {counts['val']} | Test: {counts['test']}")

        df_data.extend([
            {'Class': class_name, 'Split': 'train', 'Count': counts['train']},
            {'Class': class_name, 'Split': 'val', 'Count': counts['val']},
            {'Class': class_name, 'Split': 'test', 'Count': counts['test']},
        ])

    # Check for single class problem
    if len(class_distribution) == 1:
        print(f"\n🚨 PROBLEM IDENTIFIED: SINGLE CLASS DATASET!")
        print(f"   - Only 1 class found: {list(class_distribution.keys())[0]}")
        print(f"   - This makes classification trivial (always correct)")
        print(f"   - Explains 100% accuracy!")

        return True, class_distribution

    # Create visualization
    if df_data:
        df = pd.DataFrame(df_data)

        plt.figure(figsize=(12, 8))

        # Class distribution plot
        plt.subplot(2, 2, 1)
        class_totals = df.groupby('Class')['Count'].sum()
        plt.pie(class_totals.values, labels=class_totals.index, autopct='%1.1f%%')
        plt.title('Class Distribution (Overall)')

        # Split distribution plot
        plt.subplot(2, 2, 2)
        sns.barplot(data=df, x='Class', y='Count', hue='Split')
        plt.title('Class Distribution by Split')
        plt.xticks(rotation=45)

        # Training curves analysis if results exist
        results_file = Path("results/classification/production_classification/results.csv")
        if results_file.exists():
            results_df = pd.read_csv(results_file)

            plt.subplot(2, 2, 3)
            plt.plot(results_df['epoch'], results_df['train/loss'], label='Train Loss')
            plt.plot(results_df['epoch'], results_df['val/loss'], label='Val Loss')
            plt.title('Training/Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot(results_df['epoch'], results_df['metrics/accuracy_top1'], label='Top-1 Accuracy')
            plt.title('Classification Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.1)
            plt.legend()

        plt.tight_layout()
        plt.savefig('classification_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n📈 Visualization saved: classification_analysis.png")

    return False, class_distribution

def analyze_training_results():
    """Analyze training results to identify issues"""

    print(f"\n🔍 TRAINING RESULTS ANALYSIS:")

    results_file = Path("results/classification/production_classification/results.csv")
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return

    df = pd.read_csv(results_file)

    print(f"   📊 Training Metrics:")
    print(f"   - Epochs: {len(df)}")
    print(f"   - Final train loss: {df['train/loss'].iloc[-1]}")
    print(f"   - Final val loss: {df['val/loss'].iloc[-1]}")
    print(f"   - Final accuracy: {df['metrics/accuracy_top1'].iloc[-1]}")

    # Check for suspicious patterns
    suspicious = []

    if df['train/loss'].iloc[-1] == 0:
        suspicious.append("Training loss is exactly 0")

    if df['val/loss'].iloc[-1] == 0:
        suspicious.append("Validation loss is exactly 0")

    if (df['metrics/accuracy_top1'] == 1.0).all():
        suspicious.append("Accuracy is 100% for all epochs")

    if df['train/loss'].nunique() <= 2:
        suspicious.append("Training loss has very few unique values")

    if suspicious:
        print(f"\n🚨 SUSPICIOUS PATTERNS FOUND:")
        for pattern in suspicious:
            print(f"   - {pattern}")

    return df

def propose_solutions():
    """Propose solutions for the classification problem"""

    print(f"\n💡 PROPOSED SOLUTIONS:")
    print(f"1. 🔧 CREATE MULTI-CLASS DATASET:")
    print(f"   - Need species-specific classes: falciparum, vivax, malariae, ovale")
    print(f"   - Current dataset only has 'parasite' class")
    print(f"   - Use MP-IDB species information for proper classification")

    print(f"\n2. 🔧 FIX CROPPING SCRIPT:")
    print(f"   - Update crop_detections.py to use species labels")
    print(f"   - Create separate folders for each species")
    print(f"   - Ensure proper train/val/test split per species")

    print(f"\n3. 🔧 ADD NEGATIVE CLASS:")
    print(f"   - Include 'uninfected' or 'background' samples")
    print(f"   - Make it a binary classification (infected vs uninfected)")
    print(f"   - Or multi-class with species + uninfected")

    print(f"\n4. 📊 VALIDATE RESULTS:")
    print(f"   - Create proper confusion matrix")
    print(f"   - Check for data leakage between splits")
    print(f"   - Use stratified sampling for balanced splits")

def main():
    """Main analysis function"""

    # Analyze dataset structure
    is_single_class, class_dist = analyze_dataset_structure()

    # Analyze training results
    results_df = analyze_training_results()

    # Propose solutions
    propose_solutions()

    # Summary
    print(f"\n" + "="*60)
    print(f"📋 SUMMARY:")

    if is_single_class:
        print(f"🚨 ROOT CAUSE: Single class dataset makes classification trivial")
        print(f"✅ SOLUTION: Create proper multi-class dataset with species labels")

    print(f"📈 Next steps: Fix dataset creation and re-train classification")
    print(f"="*60)

if __name__ == "__main__":
    main()