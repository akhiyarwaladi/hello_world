#!/usr/bin/env python3
"""
Training Classification from Ground Truth Crops
Specialized script for training and evaluating classification models on generated crops.
Focuses on detailed confusion matrix analysis and accuracy metrics.
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
import warnings
warnings.filterwarnings('ignore')

class CropDataset(Dataset):
    """Dataset for loading crops from organized folders"""

    def __init__(self, crop_dir, transform=None, class_filter=None):
        self.crop_dir = Path(crop_dir)
        self.transform = transform
        self.samples = []
        self.class_names = []
        self.class_to_idx = {}

        # Get class directories
        class_dirs = [d for d in self.crop_dir.iterdir() if d.is_dir()]
        class_dirs.sort()

        # Filter classes if specified
        if class_filter:
            class_dirs = [d for d in class_dirs if d.name in class_filter]

        for i, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_names.append(class_name)
            self.class_to_idx[class_name] = i

            # Find all images in this class
            image_files = list(class_dir.glob("*.jpg"))
            for img_path in image_files:
                self.samples.append((str(img_path), i))

        print(f"Dataset loaded: {len(self.samples)} samples, {len(self.class_names)} classes")
        print(f"Classes: {self.class_names}")

        # Print class distribution
        class_counts = Counter([label for _, label in self.samples])
        total_samples = len(self.samples)
        print("Class distribution:")
        for class_name, class_idx in self.class_to_idx.items():
            count = class_counts[class_idx]
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Create dummy image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

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

def get_model(model_type, num_classes, pretrained=True):
    """Get model based on type"""

    if model_type == 'simple_cnn':
        model = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),

            # Classifier
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    elif model_type == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == 'efficientnet_b0':
        try:
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        except:
            print("EfficientNet not available, using ResNet18")
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def get_transforms(input_size=224, augment=True):
    """Get data transforms"""

    # Base transforms
    base_transforms = [
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # Training transforms with augmentation
    if augment:
        train_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        train_transforms = base_transforms

    return transforms.Compose(train_transforms), transforms.Compose(base_transforms)

def create_weighted_sampler(dataset):
    """Create weighted sampler for handling class imbalance"""
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)

    # Calculate weights (inverse frequency)
    weights = {label: 1.0 / count for label, count in class_counts.items()}
    sample_weights = [weights[label] for label in labels]

    return WeightedRandomSampler(sample_weights, len(sample_weights))

def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations with both counts and percentages
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percent[i, j]
            if count > 0:
                row.append(f'{count}\n({percent:.1f}%)')
            else:
                row.append('0\n(0.0%)')
        annotations.append(row)

    # Plot heatmap
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved: {save_path}")

def detailed_classification_report(y_true, y_pred, class_names):
    """Generate detailed classification metrics"""

    # Basic metrics
    accuracy = (np.array(y_true) == np.array(y_pred)).mean()
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class detailed analysis
    print(f"\n{'='*60}")
    print("DETAILED CLASSIFICATION ANALYSIS")
    print(f"{'='*60}")

    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")

    print(f"\nPer-Class Analysis:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 65)

    for i, class_name in enumerate(class_names):
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        support = int(report[class_name]['support'])

        print(f"{class_name:<15} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10}")

    print(f"\nConfusion Matrix:")
    print(f"{'True/Pred':<12}", end="")
    for class_name in class_names:
        print(f"{class_name:<12}", end="")
    print()

    for i, class_name in enumerate(class_names):
        print(f"{class_name:<12}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:<12}", end="")
        print()

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'confusion_matrix': cm,
        'classification_report': report
    }

class CropClassificationTrainer:
    """Main trainer class for crop classification"""

    def __init__(self, train_dir, val_dir, test_dir, results_dir, dataset_name):
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir) if val_dir else None
        self.test_dir = Path(test_dir) if test_dir else None
        self.results_dir = Path(results_dir)
        self.dataset_name = dataset_name

        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_data(self, input_size=224, augment=True, use_weighted_sampling=False):
        """Load and prepare datasets"""

        print(f"\n{'='*60}")
        print(f"LOADING DATA: {self.dataset_name}")
        print(f"{'='*60}")

        # Get transforms
        train_transform, test_transform = get_transforms(input_size, augment)

        # Load training data
        self.train_dataset = CropDataset(self.train_dir, train_transform)

        # Load validation data (or create from train if not available)
        if self.val_dir and self.val_dir.exists():
            self.val_dataset = CropDataset(self.val_dir, test_transform)
        else:
            print("No validation directory found, splitting training data...")
            # Split training data for validation
            train_samples, val_samples = train_test_split(
                self.train_dataset.samples, test_size=0.2, random_state=42,
                stratify=[label for _, label in self.train_dataset.samples]
            )

            # Create new datasets with splits
            train_dataset_full = self.train_dataset
            self.train_dataset = CropDataset.__new__(CropDataset)
            self.train_dataset.crop_dir = train_dataset_full.crop_dir
            self.train_dataset.transform = train_transform
            self.train_dataset.samples = train_samples
            self.train_dataset.class_names = train_dataset_full.class_names
            self.train_dataset.class_to_idx = train_dataset_full.class_to_idx

            self.val_dataset = CropDataset.__new__(CropDataset)
            self.val_dataset.crop_dir = train_dataset_full.crop_dir
            self.val_dataset.transform = test_transform
            self.val_dataset.samples = val_samples
            self.val_dataset.class_names = train_dataset_full.class_names
            self.val_dataset.class_to_idx = train_dataset_full.class_to_idx

        # Load test data (or create from remaining data if not available)
        if self.test_dir and self.test_dir.exists():
            self.test_dataset = CropDataset(self.test_dir, test_transform)
        else:
            print("No test directory found, using validation data for testing...")
            self.test_dataset = self.val_dataset

        # Store class information
        self.class_names = self.train_dataset.class_names
        self.num_classes = len(self.class_names)

        # Create data loaders
        if use_weighted_sampling:
            sampler = create_weighted_sampler(self.train_dataset)
            self.train_loader = DataLoader(self.train_dataset, batch_size=32, sampler=sampler)
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)

        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        print(f"\nData loaded successfully:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
        print(f"  Number of classes: {self.num_classes}")

    def train_model(self, model_type='simple_cnn', loss_type='cross_entropy',
                   epochs=20, lr=0.001, pretrained=True):
        """Train the classification model"""

        print(f"\n{'='*60}")
        print(f"TRAINING MODEL: {model_type}")
        print(f"{'='*60}")
        print(f"Loss function: {loss_type}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Pretrained: {pretrained}")

        # Get model
        model = get_model(model_type, self.num_classes, pretrained)
        model = model.to(self.device)

        # Loss function
        if loss_type == 'focal':
            criterion = FocalLoss(alpha=1, gamma=2)
        elif loss_type == 'weighted_ce':
            # Calculate class weights
            train_labels = [label for _, label in self.train_dataset.samples]
            class_weights = compute_class_weight(
                'balanced', classes=np.unique(train_labels), y=train_labels
            )
            class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            print(f"Class weights: {dict(zip(self.class_names, class_weights))}")
        else:  # cross_entropy
            criterion = nn.CrossEntropyLoss()

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in self.train_loader:
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
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate metrics
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            avg_train_loss = train_loss / len(self.train_loader)

            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)

            print(f"Epoch [{epoch+1:3d}/{epochs}] - "
                  f"Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            scheduler.step()

        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        self.model = model

        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }

    def evaluate_model(self, save_plots=True):
        """Evaluate model on test set with detailed analysis"""

        print(f"\n{'='*60}")
        print("MODEL EVALUATION")
        print(f"{'='*60}")

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                # Get predictions and probabilities
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Detailed analysis
        results = detailed_classification_report(all_labels, all_predictions, self.class_names)

        # Save confusion matrix plot
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cm_path = self.results_dir / f"confusion_matrix_{timestamp}.png"
            plot_confusion_matrix(
                results['confusion_matrix'],
                self.class_names,
                f"Confusion Matrix - {self.dataset_name}",
                cm_path
            )

        # Save detailed results
        results_with_metadata = {
            'dataset_name': self.dataset_name,
            'timestamp': timestamp,
            'class_names': self.class_names,
            'test_samples': len(self.test_dataset),
            'results': results
        }

        results_path = self.results_dir / f"classification_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = results_with_metadata.copy()
            serializable_results['results']['confusion_matrix'] = results['confusion_matrix'].tolist()
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved:")
        print(f"  Confusion matrix: {cm_path}")
        print(f"  Detailed results: {results_path}")

        return results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train classification model on ground truth crops')
    parser.add_argument('--train_dir', required=True, help='Training crops directory')
    parser.add_argument('--val_dir', help='Validation crops directory (optional)')
    parser.add_argument('--test_dir', help='Test crops directory (optional)')
    parser.add_argument('--results_dir', default='results/crop_classification', help='Results output directory')
    parser.add_argument('--dataset_name', default='Malaria_Crops', help='Dataset name for results')

    # Model parameters
    parser.add_argument('--model', choices=['simple_cnn', 'resnet18', 'resnet50', 'efficientnet_b0'],
                       default='resnet18', help='Model architecture')
    parser.add_argument('--loss', choices=['cross_entropy', 'weighted_ce', 'focal'],
                       default='weighted_ce', help='Loss function')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')

    # Training options
    parser.add_argument('--no_pretrained', action='store_true', help='Don\'t use pretrained weights')
    parser.add_argument('--no_augment', action='store_true', help='Don\'t use data augmentation')
    parser.add_argument('--weighted_sampling', action='store_true', help='Use weighted sampling for class imbalance')

    args = parser.parse_args()

    # Initialize trainer
    trainer = CropClassificationTrainer(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        results_dir=args.results_dir,
        dataset_name=args.dataset_name
    )

    # Load data
    trainer.load_data(
        input_size=args.input_size,
        augment=not args.no_augment,
        use_weighted_sampling=args.weighted_sampling
    )

    # Train model
    training_results = trainer.train_model(
        model_type=args.model,
        loss_type=args.loss,
        epochs=args.epochs,
        lr=args.lr,
        pretrained=not args.no_pretrained
    )

    # Evaluate model
    evaluation_results = trainer.evaluate_model(save_plots=True)

    print(f"\n{'='*80}")
    print("TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"Final Results:")
    print(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {evaluation_results['balanced_accuracy']:.4f}")
    print(f"  Macro F1-Score: {evaluation_results['macro_f1']:.4f}")
    print(f"  Weighted F1-Score: {evaluation_results['weighted_f1']:.4f}")
    print(f"\nResults saved in: {args.results_dir}")

if __name__ == "__main__":
    main()