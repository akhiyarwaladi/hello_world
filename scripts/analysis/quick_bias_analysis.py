#!/usr/bin/env python3
"""
Quick Bias Analysis for Malaria Classification
Analyzes class imbalance bias using existing ground truth crops.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import argparse

class MalariaDataset(Dataset):
    """Dataset for malaria crop classification"""

    def __init__(self, samples, class_names, transform=None):
        self.samples = samples  # List of (image_path, label) tuples
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # If image loading fails, create a blank image
            image = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

class SimpleCNN(nn.Module):
    """Simple CNN for malaria classification"""

    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_dataset_samples(data_dir):
    """Load all samples from dataset directory"""
    data_dir = Path(data_dir)
    samples = []
    class_names = []
    class_to_idx = {}

    # Get class directories
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    class_dirs.sort()

    print(f"Found class directories: {[d.name for d in class_dirs]}")

    for i, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        class_names.append(class_name)
        class_to_idx[class_name] = i

        # Find all images in this class
        image_files = list(class_dir.glob("*.jpg"))
        print(f"  {class_name}: {len(image_files)} images")

        for img_path in image_files:
            samples.append((str(img_path), i))

    return samples, class_names, class_to_idx

def analyze_dataset_bias(dataset_name, data_dir, results_dir, epochs=5):
    """Analyze bias in a specific dataset"""
    print(f"\n{'='*60}")
    print(f"BIAS ANALYSIS: {dataset_name}")
    print(f"{'='*60}")

    # Load samples
    samples, class_names, class_to_idx = load_dataset_samples(data_dir)

    if len(samples) == 0:
        print("No samples found in dataset!")
        return None

    # Print class distribution
    labels = [label for _, label in samples]
    class_counts = Counter(labels)
    total_samples = len(samples)

    print(f"\nClass Distribution:")
    print(f"Total samples: {total_samples}")
    for class_name, class_idx in class_to_idx.items():
        count = class_counts[class_idx]
        percentage = (count / total_samples) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

    # Calculate imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"\nImbalance Ratio: {imbalance_ratio:.1f}:1")

    # Split data manually since val/test directories are empty
    train_samples, temp_samples = train_test_split(
        samples, test_size=0.3, random_state=42,
        stratify=[label for _, label in samples]
    )
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.5, random_state=42,
        stratify=[label for _, label in temp_samples]
    )

    print(f"\nData Split:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")

    # Setup transforms
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = MalariaDataset(train_samples, class_names, transform_train)
    val_dataset = MalariaDataset(val_samples, class_names, transform_test)
    test_dataset = MalariaDataset(test_samples, class_names, transform_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Calculate class weights
    train_labels = [label for _, label in train_samples]
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    print(f"\nClass weights: {dict(zip(class_names, class_weights))}")

    # Train models
    results = {}

    for use_weights, model_suffix in [(False, "unweighted"), (True, "weighted")]:
        print(f"\n{'-'*40}")
        print(f"TRAINING {model_suffix.upper()} MODEL")
        print(f"{'-'*40}")

        # Initialize model
        model = SimpleCNN(len(class_names)).to(device)

        # Loss function
        if use_weights:
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        best_val_acc = 0.0
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)

            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

        # Load best model for testing
        model.load_state_dict(best_model_state)

        # Test evaluation
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        test_acc = np.mean(np.array(all_predictions) == np.array(all_labels))
        cm = confusion_matrix(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions,
                                     target_names=class_names, output_dict=True)

        print(f"Test Accuracy: {test_acc:.4f}")

        # Per-class analysis
        print(f"\nPer-class Analysis ({model_suffix}):")
        for i, class_name in enumerate(class_names):
            class_samples_in_test = np.sum(cm[i, :])
            if class_samples_in_test > 0:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                print(f"  {class_name}:")
                print(f"    Test samples: {class_samples_in_test}")
                print(f"    Precision: {precision:.3f}")
                print(f"    Recall: {recall:.3f}")
                print(f"    F1-score: {f1:.3f}")

        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {dataset_name} ({model_suffix})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()

        # Save plot
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(results_dir / f"{dataset_name}_{model_suffix}_confusion_matrix.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Store results
        results[model_suffix] = {
            'accuracy': test_acc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'class_distribution': dict(class_counts),
            'imbalance_ratio': imbalance_ratio
        }

    return results

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Quick bias analysis for malaria datasets')
    parser.add_argument('--dataset', choices=['species', 'stages', 'lifecycle', 'all'],
                       default='all', help='Which dataset to analyze')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')

    args = parser.parse_args()

    # Dataset paths (using only train directories since val/test are empty)
    datasets = {
        'species': {
            'path': 'data/ground_truth_crops_224/species/crops/train',
            'name': 'MP-IDB Species'
        },
        'stages': {
            'path': 'data/ground_truth_crops_224/stages/crops/train',
            'name': 'MP-IDB Stages'
        },
        'lifecycle': {
            'path': 'data/ground_truth_crops_224/lifecycle/crops/train',
            'name': 'IML Lifecycle'
        }
    }

    # Determine datasets to analyze
    if args.dataset == 'all':
        datasets_to_analyze = datasets.keys()
    else:
        datasets_to_analyze = [args.dataset]

    # Run analysis
    all_results = {}

    for dataset_key in datasets_to_analyze:
        config = datasets[dataset_key]
        results = analyze_dataset_bias(
            dataset_name=config['name'],
            data_dir=config['path'],
            results_dir=f'results/bias_analysis/{dataset_key}',
            epochs=args.epochs
        )
        if results:
            all_results[dataset_key] = results

    # Final summary
    print(f"\n{'='*80}")
    print("BIAS ANALYSIS SUMMARY")
    print(f"{'='*80}")

    for dataset_key, results in all_results.items():
        config = datasets[dataset_key]
        print(f"\n{config['name']}:")

        # Show imbalance ratio
        imbalance_ratio = results['unweighted']['imbalance_ratio']
        print(f"  Imbalance Ratio: {imbalance_ratio:.1f}:1")

        # Show class distribution
        class_dist = results['unweighted']['class_distribution']
        total_samples = sum(class_dist.values())
        print(f"  Class Distribution:")
        for class_id, count in class_dist.items():
            percentage = (count / total_samples) * 100
            print(f"    Class {class_id}: {count} ({percentage:.1f}%)")

        # Show model performance comparison
        print(f"  Model Performance:")
        print(f"    {'Strategy':<12} {'Accuracy':<10} {'Macro F1':<10}")
        print(f"    {'-'*32}")

        for strategy in ['unweighted', 'weighted']:
            acc = results[strategy]['accuracy']
            macro_f1 = results[strategy]['classification_report']['macro avg']['f1-score']
            print(f"    {strategy.capitalize():<12} {acc:<10.3f} {macro_f1:<10.3f}")

if __name__ == "__main__":
    main()