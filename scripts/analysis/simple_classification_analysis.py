#!/usr/bin/env python3
"""
Simple Classification Analysis for Ground Truth Crops
Analyzes class imbalance and generates confusion matrices for malaria datasets.
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
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import argparse

class MalariaDataset(Dataset):
    """Simple dataset for malaria crop classification"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Find all images and their labels
        self.samples = []
        self.class_names = []
        self.class_to_idx = {}

        # Get class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        class_dirs.sort()

        for i, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_names.append(class_name)
            self.class_to_idx[class_name] = i

            # Find all images in this class
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((str(img_path), i))

        print(f"Found {len(self.samples)} samples across {len(self.class_names)} classes")
        print(f"Classes: {self.class_names}")

        # Print class distribution
        class_counts = Counter([label for _, label in self.samples])
        print("Class distribution:")
        for class_name, class_idx in self.class_to_idx.items():
            count = class_counts[class_idx]
            percentage = (count / len(self.samples)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

class SimpleCNN(nn.Module):
    """Simple CNN for malaria classification"""

    def __init__(self, num_classes, input_size=128):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Second block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Third block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Fourth block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ClassificationAnalyzer:
    """Analyze classification performance with class imbalance"""

    def __init__(self, dataset_path, results_dir, dataset_name):
        self.dataset_path = Path(dataset_path)
        self.results_dir = Path(results_dir)
        self.dataset_name = dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Data transforms
        self.transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_datasets(self):
        """Load train, val, test datasets"""
        print(f"\n=== Loading {self.dataset_name} Dataset ===")

        # Load datasets
        train_dataset = MalariaDataset(
            self.dataset_path / "train",
            transform=self.transform_train
        )

        val_dataset = MalariaDataset(
            self.dataset_path / "val",
            transform=self.transform_test
        )

        test_dataset = MalariaDataset(
            self.dataset_path / "test",
            transform=self.transform_test
        )

        # Store class information
        self.class_names = train_dataset.class_names
        self.num_classes = len(self.class_names)

        # Calculate class weights for imbalance handling
        train_labels = [label for _, label in train_dataset.samples]
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        self.class_weights = torch.FloatTensor(class_weights).to(self.device)

        print(f"Class weights: {dict(zip(self.class_names, class_weights))}")

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_dataset, val_dataset, test_dataset

    def train_model(self, use_class_weights=True, epochs=20):
        """Train a simple CNN classifier"""
        print(f"\n=== Training Model (Class Weights: {use_class_weights}) ===")

        # Initialize model
        model = SimpleCNN(self.num_classes).to(self.device)

        # Loss function with optional class weights
        if use_class_weights:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            model_name = f"{self.dataset_name}_weighted"
        else:
            criterion = nn.CrossEntropyLoss()
            model_name = f"{self.dataset_name}_unweighted"

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Training loop
        best_val_acc = 0.0
        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = val_correct / val_total
            avg_train_loss = train_loss / len(self.train_loader)

            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)

            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.results_dir / f"{model_name}_best.pth")

            scheduler.step()

        print(f"Best validation accuracy: {best_val_acc:.4f}")

        # Load best model for evaluation
        model.load_state_dict(torch.load(self.results_dir / f"{model_name}_best.pth"))

        return model, model_name, train_losses, val_accuracies

    def evaluate_model(self, model, model_name):
        """Evaluate model and generate confusion matrix"""
        print(f"\n=== Evaluating {model_name} ===")

        model.eval()
        all_predictions = []
        all_labels = []
        class_predictions = defaultdict(list)
        class_labels = defaultdict(list)

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Store per-class predictions for detailed analysis
                for i, (pred, true) in enumerate(zip(predicted.cpu().numpy(), labels.cpu().numpy())):
                    class_predictions[true].append(pred)
                    class_labels[true].append(true)

        # Calculate overall accuracy
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        print(f"Test Accuracy: {accuracy:.4f}")

        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Generate classification report
        report = classification_report(all_labels, all_predictions,
                                     target_names=self.class_names, output_dict=True)

        # Save detailed results
        results = {
            'model_name': model_name,
            'dataset': self.dataset_name,
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'class_names': self.class_names
        }

        # Per-class analysis
        print("\nPer-class Analysis:")
        for i, class_name in enumerate(self.class_names):
            class_true = np.sum(cm[i, :])  # Total true samples for this class
            class_pred_correct = cm[i, i]  # Correctly predicted samples for this class
            class_precision = report[class_name]['precision']
            class_recall = report[class_name]['recall']
            class_f1 = report[class_name]['f1-score']

            print(f"  {class_name}:")
            print(f"    Total samples: {class_true}")
            print(f"    Correct predictions: {class_pred_correct}")
            print(f"    Precision: {class_precision:.3f}")
            print(f"    Recall: {class_recall:.3f}")
            print(f"    F1-score: {class_f1:.3f}")

        return results

    def run_analysis(self):
        """Run complete analysis with and without class weights"""
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION ANALYSIS: {self.dataset_name}")
        print(f"{'='*60}")

        # Load datasets
        train_dataset, val_dataset, test_dataset = self.load_datasets()

        # Train models with different strategies
        results = {}

        # 1. Without class weights (baseline)
        print(f"\n{'-'*40}")
        print("TRAINING WITHOUT CLASS WEIGHTS")
        print(f"{'-'*40}")
        model_unweighted, name_unweighted, _, _ = self.train_model(use_class_weights=False)
        results['unweighted'] = self.evaluate_model(model_unweighted, name_unweighted)

        # 2. With class weights (to handle imbalance)
        print(f"\n{'-'*40}")
        print("TRAINING WITH CLASS WEIGHTS")
        print(f"{'-'*40}")
        model_weighted, name_weighted, _, _ = self.train_model(use_class_weights=True)
        results['weighted'] = self.evaluate_model(model_weighted, name_weighted)

        # Save combined results
        import json
        with open(self.results_dir / f"{self.dataset_name}_analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary comparison
        print(f"\n{'='*60}")
        print(f"SUMMARY COMPARISON: {self.dataset_name}")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12}")
        print("-" * 52)

        for strategy, result in results.items():
            acc = result['accuracy']
            macro_f1 = result['classification_report']['macro avg']['f1-score']
            weighted_f1 = result['classification_report']['weighted avg']['f1-score']
            print(f"{strategy.capitalize():<20} {acc:<10.3f} {macro_f1:<10.3f} {weighted_f1:<12.3f}")

        return results

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Analyze malaria classification datasets')
    parser.add_argument('--dataset', choices=['species', 'stages', 'lifecycle', 'all'],
                       default='all', help='Which dataset to analyze')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--results_dir', default='results/classification_analysis',
                       help='Directory to save results')

    args = parser.parse_args()

    # Dataset configurations
    base_path = Path("data/ground_truth_crops")
    datasets_config = {
        'species': {
            'path': base_path / "species" / "crops",
            'name': 'MP-IDB Species'
        },
        'stages': {
            'path': base_path / "stages" / "crops",
            'name': 'MP-IDB Stages'
        },
        'lifecycle': {
            'path': base_path / "lifecycle" / "crops",
            'name': 'IML Lifecycle'
        }
    }

    # Determine which datasets to analyze
    if args.dataset == 'all':
        datasets_to_analyze = datasets_config.keys()
    else:
        datasets_to_analyze = [args.dataset]

    # Run analysis for each dataset
    all_results = {}

    for dataset_key in datasets_to_analyze:
        if dataset_key not in datasets_config:
            print(f"Unknown dataset: {dataset_key}")
            continue

        config = datasets_config[dataset_key]

        # Create analyzer
        analyzer = ClassificationAnalyzer(
            dataset_path=config['path'],
            results_dir=Path(args.results_dir) / dataset_key,
            dataset_name=config['name']
        )

        # Run analysis
        results = analyzer.run_analysis()
        all_results[dataset_key] = results

    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - ALL DATASETS")
    print(f"{'='*80}")

    for dataset_key, dataset_results in all_results.items():
        print(f"\n{datasets_config[dataset_key]['name']}:")
        print(f"{'  Strategy':<15} {'Accuracy':<10} {'Macro F1':<10} {'Dominant Class F1':<18}")
        print("  " + "-" * 50)

        for strategy, result in dataset_results.items():
            acc = result['accuracy']
            macro_f1 = result['classification_report']['macro avg']['f1-score']

            # Find dominant class (usually first class with highest support)
            class_reports = {k: v for k, v in result['classification_report'].items()
                           if k not in ['accuracy', 'macro avg', 'weighted avg']}
            dominant_class = max(class_reports.keys(),
                               key=lambda x: class_reports[x]['support'])
            dominant_f1 = class_reports[dominant_class]['f1-score']

            print(f"  {strategy.capitalize():<15} {acc:<10.3f} {macro_f1:<10.3f} {dominant_f1:<18.3f}")

if __name__ == "__main__":
    main()