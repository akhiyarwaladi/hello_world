#!/usr/bin/env python3
"""
Comprehensive Classification Testing
Tests multiple strategies to solve resolution + class imbalance issues.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import argparse
import json
from datetime import datetime

class MalariaDataset(Dataset):
    """Enhanced dataset with flexible resolution support"""

    def __init__(self, samples, class_names, transform=None, target_size=224):
        self.samples = samples
        self.class_names = class_names
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(target_size=224, augment_level='medium'):
    """Get transforms based on target size and augmentation level"""

    # Base transforms
    base_transforms = [
        transforms.ToPILImage(),
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # Training transforms with different augmentation levels
    if augment_level == 'none':
        train_transforms = base_transforms
    elif augment_level == 'light':
        train_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif augment_level == 'medium':
        train_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif augment_level == 'heavy':
        train_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

    return transforms.Compose(train_transforms), transforms.Compose(base_transforms)

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_model(model_name, num_classes, pretrained=True):
    """Get model architecture"""

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        try:
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        except:
            # Fallback to ResNet if EfficientNet not available
            print("EfficientNet not available, using ResNet18")
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'simple_cnn':
        # Simple CNN for comparison
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    return model

def load_dataset_samples(data_dir):
    """Load all samples from dataset directory"""
    data_dir = Path(data_dir)
    samples = []
    class_names = []
    class_to_idx = {}

    # Get class directories
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    class_dirs.sort()

    for i, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        class_names.append(class_name)
        class_to_idx[class_name] = i

        # Find all images in this class
        image_files = list(class_dir.glob("*.jpg"))
        for img_path in image_files:
            samples.append((str(img_path), i))

    return samples, class_names, class_to_idx

def get_weighted_sampler(samples):
    """Create weighted sampler for handling class imbalance"""
    labels = [label for _, label in samples]
    class_counts = Counter(labels)

    # Calculate weights (inverse frequency)
    weights = {label: 1.0 / count for label, count in class_counts.items()}
    sample_weights = [weights[label] for label in labels]

    return WeightedRandomSampler(sample_weights, len(sample_weights))

class ComprehensiveExperiment:
    """Run comprehensive experiments with different strategies"""

    def __init__(self, dataset_path, dataset_name, results_dir):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_data(self):
        """Load and split dataset"""
        samples, class_names, class_to_idx = load_dataset_samples(self.dataset_path)

        if len(samples) == 0:
            raise ValueError("No samples found!")

        # Print class distribution
        labels = [label for _, label in samples]
        class_counts = Counter(labels)
        total_samples = len(samples)

        print(f"\nDataset: {self.dataset_name}")
        print(f"Total samples: {total_samples}")
        print("Class distribution:")
        for class_name, class_idx in class_to_idx.items():
            count = class_counts[class_idx]
            percentage = (count / total_samples) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

        # Calculate imbalance ratio
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        print(f"Imbalance Ratio: {imbalance_ratio:.1f}:1")

        # Split data
        train_samples, temp_samples = train_test_split(
            samples, test_size=0.3, random_state=42,
            stratify=[label for _, label in samples]
        )
        val_samples, test_samples = train_test_split(
            temp_samples, test_size=0.5, random_state=42,
            stratify=[label for _, label in temp_samples]
        )

        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.imbalance_ratio = imbalance_ratio

        return train_samples, val_samples, test_samples, class_names

    def run_experiment(self, config):
        """Run single experiment with given configuration"""

        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {config['name']}")
        print(f"{'='*60}")
        print(f"Resolution: {config['resolution']}")
        print(f"Model: {config['model']}")
        print(f"Loss: {config['loss_type']}")
        print(f"Sampling: {config['sampling']}")
        print(f"Augmentation: {config['augment_level']}")

        # Get transforms
        train_transform, test_transform = get_transforms(
            target_size=config['resolution'],
            augment_level=config['augment_level']
        )

        # Create datasets
        train_dataset = MalariaDataset(self.train_samples, self.class_names, train_transform)
        val_dataset = MalariaDataset(self.val_samples, self.class_names, test_transform)
        test_dataset = MalariaDataset(self.test_samples, self.class_names, test_transform)

        # Create data loaders
        if config['sampling'] == 'weighted':
            sampler = get_weighted_sampler(self.train_samples)
            train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Get model
        model = get_model(config['model'], self.num_classes, pretrained=config.get('pretrained', True))
        model = model.to(self.device)

        # Loss function
        if config['loss_type'] == 'focal':
            criterion = FocalLoss(alpha=1, gamma=2)
        elif config['loss_type'] == 'weighted_ce':
            train_labels = [label for _, label in self.train_samples]
            class_weights = compute_class_weight(
                'balanced', classes=np.unique(train_labels), y=train_labels
            )
            class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:  # standard_ce
            criterion = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Training
        epochs = config.get('epochs', 15)
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)

            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            scheduler.step()

        # Load best model for testing
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Test evaluation
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        test_acc = np.mean(np.array(all_predictions) == np.array(all_labels))
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions,
                                     target_names=self.class_names, output_dict=True)

        print(f"\nResults:")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")

        # Save results
        results = {
            'config': config,
            'test_accuracy': test_acc,
            'balanced_accuracy': balanced_acc,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'dataset_info': {
                'name': self.dataset_name,
                'imbalance_ratio': self.imbalance_ratio,
                'num_classes': self.num_classes,
                'class_names': self.class_names
            }
        }

        return results

    def run_all_experiments(self):
        """Run comprehensive set of experiments"""

        # Define experiment configurations
        experiments = [
            # Baseline - current approach
            {
                'name': 'Baseline_128px',
                'resolution': 128,
                'model': 'simple_cnn',
                'loss_type': 'standard_ce',
                'sampling': 'normal',
                'augment_level': 'none',
                'epochs': 10
            },

            # Resolution upgrade
            {
                'name': 'HighRes_224px',
                'resolution': 224,
                'model': 'simple_cnn',
                'loss_type': 'standard_ce',
                'sampling': 'normal',
                'augment_level': 'none',
                'epochs': 10
            },

            # Class imbalance solutions
            {
                'name': 'WeightedSampling_224px',
                'resolution': 224,
                'model': 'simple_cnn',
                'loss_type': 'standard_ce',
                'sampling': 'weighted',
                'augment_level': 'medium',
                'epochs': 10
            },

            {
                'name': 'FocalLoss_224px',
                'resolution': 224,
                'model': 'simple_cnn',
                'loss_type': 'focal',
                'sampling': 'normal',
                'augment_level': 'medium',
                'epochs': 10
            },

            # Pretrained models
            {
                'name': 'ResNet18_224px',
                'resolution': 224,
                'model': 'resnet18',
                'loss_type': 'weighted_ce',
                'sampling': 'weighted',
                'augment_level': 'medium',
                'epochs': 10,
                'pretrained': True
            },

            # Best combination
            {
                'name': 'Best_Combination',
                'resolution': 224,
                'model': 'resnet18',
                'loss_type': 'focal',
                'sampling': 'weighted',
                'augment_level': 'heavy',
                'epochs': 15,
                'pretrained': True
            }
        ]

        all_results = {}

        for config in experiments:
            try:
                results = self.run_experiment(config)
                all_results[config['name']] = results
            except Exception as e:
                print(f"Error in experiment {config['name']}: {e}")
                continue

        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"comprehensive_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Print summary
        self.print_summary(all_results)

        return all_results

    def print_summary(self, all_results):
        """Print comprehensive summary"""

        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE EXPERIMENT SUMMARY: {self.dataset_name}")
        print(f"{'='*80}")

        print(f"Dataset Info:")
        print(f"  • Imbalance Ratio: {self.imbalance_ratio:.1f}:1")
        print(f"  • Number of Classes: {self.num_classes}")
        print(f"  • Train/Val/Test: {len(self.train_samples)}/{len(self.val_samples)}/{len(self.test_samples)}")

        print(f"\nExperiment Results:")
        print(f"{'Experiment':<25} {'Test Acc':<10} {'Balanced Acc':<12} {'Macro F1':<10} {'Weighted F1':<12}")
        print("-" * 75)

        for exp_name, results in all_results.items():
            test_acc = results['test_accuracy']
            balanced_acc = results['balanced_accuracy']
            macro_f1 = results['macro_f1']
            weighted_f1 = results['weighted_f1']

            print(f"{exp_name:<25} {test_acc:<10.3f} {balanced_acc:<12.3f} {macro_f1:<10.3f} {weighted_f1:<12.3f}")

        # Find best performing experiment
        best_exp = max(all_results.items(), key=lambda x: x[1]['balanced_accuracy'])
        print(f"\nBest Performing Experiment: {best_exp[0]}")
        print(f"  • Balanced Accuracy: {best_exp[1]['balanced_accuracy']:.3f}")
        print(f"  • Configuration: {best_exp[1]['config']}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Comprehensive classification testing')
    parser.add_argument('--dataset', choices=['species', 'stages', 'lifecycle', 'all'],
                       default='species', help='Dataset to test')

    args = parser.parse_args()

    datasets = {
        'species': {
            'path': 'data/ground_truth_crops/species/crops/train',
            'name': 'MP-IDB Species'
        },
        'stages': {
            'path': 'data/ground_truth_crops/stages/crops/train',
            'name': 'MP-IDB Stages'
        },
        'lifecycle': {
            'path': 'data/ground_truth_crops/lifecycle/crops/train',
            'name': 'IML Lifecycle'
        }
    }

    if args.dataset == 'all':
        datasets_to_test = datasets.keys()
    else:
        datasets_to_test = [args.dataset]

    for dataset_key in datasets_to_test:
        config = datasets[dataset_key]

        experiment = ComprehensiveExperiment(
            dataset_path=config['path'],
            dataset_name=config['name'],
            results_dir=f'results/comprehensive_test/{dataset_key}'
        )

        # Load data
        experiment.load_data()

        # Run all experiments
        results = experiment.run_all_experiments()

if __name__ == "__main__":
    main()