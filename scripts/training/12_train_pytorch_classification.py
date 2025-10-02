#!/usr/bin/env python3
"""
Train PyTorch Classification Models on Cropped Parasites
Supports ResNet, EfficientNet, MobileNet, DenseNet, ViT
For comparison with YOLO classification models
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler  # Mixed precision for RTX 3060
from torchvision import datasets, transforms, models
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils.results_manager import ResultsManager

class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance

    Reference: https://arxiv.org/abs/1708.02002
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Specifies reduction to apply to output
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Calculate p_t
        pt = torch.exp(-ce_loss)

        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_model(model_name, num_classes=4, pretrained=True):
    """Get specified model architecture"""

    if model_name.startswith('resnet'):
        if model_name == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'resnet34':
            model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        elif model_name == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V2' if pretrained else None)
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")

        # Modify final layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name.startswith('efficientnet'):
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'efficientnet_b2':
            model = models.efficientnet_b2(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f"Unknown EfficientNet model: {model_name}")

        # Modify final layer
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name.startswith('mobilenet'):
        if model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        elif model_name == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        else:
            raise ValueError(f"Unknown MobileNet model: {model_name}")

    elif model_name.startswith('densenet'):
        if model_name == 'densenet121':
            model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'densenet161':
            model = models.densenet161(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'densenet169':
            model = models.densenet169(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f"Unknown DenseNet model: {model_name}")

        # Modify final layer
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name.startswith('vgg'):
        if model_name == 'vgg16':
            model = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'vgg19':
            model = models.vgg19(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f"Unknown VGG model: {model_name}")

        # Modify final layer
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name.startswith('vit'):
        if model_name == 'vit_b_16':
            model = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'vit_b_32':
            model = models.vit_b_32(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f"Unknown ViT model: {model_name}")

        # Modify final layer
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif model_name.startswith('swin'):
        if model_name == 'swin_t':
            model = models.swin_t(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'swin_s':
            model = models.swin_s(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'swin_b':
            model = models.swin_b(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f"Unknown Swin model: {model_name}")

        # Modify final layer
        model.head = nn.Linear(model.head.in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

def get_transforms(image_size=224):
    """Get data transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def get_class_weights(dataset, device):
    """Calculate class weights for balanced loss function"""
    # Get all labels from dataset
    labels = []
    for _, label in dataset:
        labels.append(label)

    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )

    # Convert to tensor
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"[CLASS WEIGHTS] {dict(zip(np.unique(labels), class_weights.cpu().numpy()))}")

    return class_weights

def create_weighted_sampler(dataset):
    """Create weighted random sampler for balanced training"""
    # Count samples per class
    labels = []
    for _, label in dataset:
        labels.append(label)

    class_counts = Counter(labels)
    total_samples = len(labels)

    print(f"[CLASS DISTRIBUTION] {dict(class_counts)}")

    # Calculate sample weights (inverse frequency)
    sample_weights = []
    for label in labels:
        weight = total_samples / (len(class_counts) * class_counts[label])
        sample_weights.append(weight)

    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler

def get_enhanced_transforms(image_size=224, minority_classes=None):
    """Get enhanced transforms with aggressive augmentation for minority classes"""

    # Base augmentation for all classes
    base_augment = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # Aggressive augmentation for minority classes
    minority_augment = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose(base_augment)
    minority_transform = transforms.Compose(minority_augment)

    return train_transform, minority_transform, val_transform

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, scheduler=None):
    """Train for one epoch with RTX 3060 Mixed Precision optimization"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None and device.type == 'cuda'

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            # Mixed precision training for RTX 3060 - 2x speedup
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Step OneCycleLR scheduler per batch for maximum optimization
        if scheduler and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch with RTX 3060 optimization"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Use mixed precision for validation too (faster inference)
            if device.type == 'cuda':
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc, all_preds, all_labels

def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Simple GPU setup
    torch.set_num_threads(4)

    parser = argparse.ArgumentParser(description="Train PyTorch Classification Models")
    parser.add_argument("--data", default="data/classification_multispecies",
                       help="Classification dataset root")
    parser.add_argument("--model", default="efficientnet_b0",  # OPTIMAL: Changed from resnet18 to efficientnet_b0
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                               'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                               'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
                               'densenet121', 'densenet161', 'densenet169',
                               'vgg16', 'vgg19',  # VGG models for strong feature extraction
                               'vit_b_16', 'vit_b_32'],  # Keep ViT but optimize CPU usage
                       help="Model architecture (EfficientNet-B0 recommended for medical AI)")
    parser.add_argument("--epochs", type=int, default=25,  # Increased from 10
                       help="Number of epochs (default: 25 for better convergence)")
    parser.add_argument("--batch", type=int, default=32,  # Optimized for 224px images
                       help="Batch size (default: 32 optimized for 224px images)")
    parser.add_argument("--lr", type=float, default=0.0005,  # OPTIMAL: Changed from 0.001 to 0.0005
                       help="Learning rate (default: 0.0005 optimized for focal loss)")
    parser.add_argument("--loss", choices=['cross_entropy', 'focal', 'class_balanced'], default='focal',
                       help="Loss function type (focal/class_balanced recommended for medical AI)")
    parser.add_argument("--focal_alpha", type=float, default=1.0,
                       help="Focal loss alpha parameter")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                       help="Focal loss gamma parameter")
    parser.add_argument("--cb_beta", type=float, default=0.9999,
                       help="Class-Balanced loss beta parameter (default: 0.9999)")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (auto-detect: cuda if available, else cpu)")
    parser.add_argument("--name", default="pytorch_classification",
                       help="Experiment name")
    parser.add_argument("--pretrained", action="store_true", default=True,
                       help="Use pretrained weights")
    parser.add_argument("--save-dir", default=None,
                       help="Override default save directory for centralized results")

    args = parser.parse_args()

    print("=" * 60)
    print("PYTORCH PARASITE CLASSIFICATION TRAINING")
    print("=" * 60)

    # Initialize Results Manager for organized folder structure
    # Use custom save directory if provided, otherwise use results manager
    if args.save_dir:
        experiment_path = Path(args.save_dir)
        print(f"[SAVE] Using custom save directory: {experiment_path}")
    else:
        results_manager = ResultsManager()
        # Get organized experiment path with consistent naming
        model_name = f"pytorch_classification_{args.model}"  # Include model name for consistency
        experiment_path = results_manager.get_experiment_path(
            experiment_type="training",
            model_name=model_name,
            experiment_name=args.name
        )

    experiment_path.mkdir(parents=True, exist_ok=True)

    # Check if data exists
    if not Path(args.data).exists():
        print(f"[ERROR] Dataset directory not found: {args.data}")
        return

    print(f"[DATA] Using dataset: {args.data}")
    print(f"[MODEL] Model: {args.model}")
    print(f"[TRAIN] Epochs: {args.epochs}")
    print(f"[IMG] Image size: {args.image_size}")
    print(f"[BATCH] Batch size: {args.batch}")
    print(f"[DEVICE] Device: {args.device}")
    print(f"[LR] Learning rate: {args.lr}")

    # Setup device
    device = torch.device(args.device)

    # Get transforms
    train_transform, val_transform = get_transforms(args.image_size)

    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=Path(args.data) / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=Path(args.data) / "val",
        transform=val_transform
    )

    # Check if test directory exists and has class folders
    test_path = Path(args.data) / "test"
    test_dataset = None
    test_loader = None

    if test_path.exists():
        try:
            # Check if test directory has any class folders
            class_folders = [d for d in test_path.iterdir() if d.is_dir()]
            if not class_folders:
                # Create empty class folders matching train dataset classes
                print(f"[TEST] Test directory empty, creating class folders...")
                for class_name in train_dataset.classes:
                    (test_path / class_name).mkdir(exist_ok=True)
                print(f"[TEST] Created {len(train_dataset.classes)} empty class folders")

            test_dataset = datasets.ImageFolder(
                root=test_path,
                transform=val_transform
            )
            if len(test_dataset) > 0:
                print(f"[TEST] Found test dataset with {len(test_dataset)} images")
            else:
                print(f"[TEST] Test dataset exists but is empty, skipping test evaluation")
                test_dataset = None
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[TEST] Error loading test dataset: {e}")
            test_dataset = None
    else:
        print(f"[TEST] Creating test directory with class folders...")
        test_path.mkdir(parents=True, exist_ok=True)
        for class_name in train_dataset.classes:
            (test_path / class_name).mkdir(exist_ok=True)
        print(f"[TEST] Created test directory with {len(train_dataset.classes)} class folders")

    # Standard GPU DataLoader setup
    num_workers = 0
    pin_memory = True

    # Create data loaders
    # FIXED: Adjust batch size and drop_last for small datasets
    actual_batch_size = min(args.batch, len(train_dataset))
    use_drop_last = len(train_dataset) >= args.batch * 2  # Only drop last if we have enough data

    print(f"[BATCH] Adjusted batch size: {actual_batch_size} (original: {args.batch})")
    print(f"[BATCH] Drop last: {use_drop_last}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=use_drop_last  # Only drop if we have enough data
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(actual_batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=min(actual_batch_size, len(test_dataset)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    # Get class names
    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"\n[DATASET] Dataset composition:")
    print(f"   Classes: {class_names}")
    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val: {len(val_dataset)} images")
    if test_dataset is not None:
        print(f"   Test: {len(test_dataset)} images")
    else:
        print(f"   Test: 0 images (no test data)")

    # Initialize model
    print(f"\n[LOAD] Loading {args.model} model...")
    model = get_model(args.model, num_classes, args.pretrained)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total parameters: {total_params:,}")
    print(f"[MODEL] Trainable parameters: {trainable_params:,}")

    # Calculate class weights for balanced training
    print(f"\n[CLASS BALANCE] Calculating class weights...")
    class_weights = get_class_weights(train_dataset, device)

    # Create weighted sampler for balanced batches
    print(f"[SAMPLING] Creating weighted random sampler...")
    weighted_sampler = create_weighted_sampler(train_dataset)

    # Recreate train loader with weighted sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        sampler=weighted_sampler,  # Use weighted sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=use_drop_last
    )

    # Setup loss function based on --loss parameter
    print(f"\n[LOSS] Using {args.loss} loss function")
    if args.loss == 'focal':
        print(f"[FOCAL] Alpha: {args.focal_alpha}, Gamma: {args.focal_gamma}")
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        # Note: Focal loss has built-in handling for imbalance, so we don't use class weights
    elif args.loss == 'class_balanced':
        # Import Class-Balanced Loss from advanced_losses
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        from scripts.training.advanced_losses import ClassBalancedLoss

        # Calculate samples per class from training dataset
        from collections import Counter
        all_labels = [label for _, label in train_dataset]
        class_counts = Counter(all_labels)
        samples_per_class = [class_counts[i] for i in range(num_classes)]

        # Get beta parameter (default: 0.9999)
        cb_beta = args.cb_beta if hasattr(args, 'cb_beta') else 0.9999
        criterion = ClassBalancedLoss(samples_per_class, beta=cb_beta, loss_type='ce')

        print(f"[CLASS_BALANCED] Beta: {cb_beta}")
        print(f"[CLASS_BALANCED] Samples per class: {samples_per_class}")
        print(f"[CLASS_BALANCED] Imbalance ratio: {max(samples_per_class)/min(samples_per_class):.2f}:1")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)  # Use class weights for balanced loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  # AdamW for better performance
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr*10, epochs=args.epochs, steps_per_epoch=len(train_loader))  # OneCycle for faster convergence

    # Initialize mixed precision scaler for RTX 3060
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        print(f"[RTX 3060] Mixed precision training enabled - expect 2x speedup!")
    else:
        print(f"[CPU] Standard precision training")

    # Training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Create results.csv file for epoch tracking (like detection models)
    results_csv_path = experiment_path / "results.csv"
    with open(results_csv_path, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    print("\n[TIMER] Starting training...")
    start_time = time.time()

    # Early stopping parameters
    best_val_acc = 0.0
    patience = 10  # Stop if no improvement for 10 epochs
    patience_counter = 0
    print(f"[EARLY STOPPING] Patience: {patience} epochs")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)

        # Train with Mixed Precision and OneCycleLR per-batch scheduling
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, scheduler)

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)

        # Note: OneCycleLR steps per batch inside train_epoch, other schedulers step per epoch
        if not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        # Save history
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Save epoch results to CSV (like detection models)
        with open(results_csv_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_acc:.2f},{val_loss:.6f},{val_acc:.2f}\n")

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model and early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0  # Reset patience counter
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_name': args.model,
                'class_names': class_names
            }, experiment_path / 'best.pt')
            print(f"[SAVE] Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"[EARLY STOPPING] No improvement for {patience_counter}/{patience} epochs")

        # Early stopping check
        if patience_counter >= patience:
            print(f"[EARLY STOPPING] Stopping training due to no improvement for {patience} epochs")
            print(f"[EARLY STOPPING] Best validation accuracy: {best_val_acc:.2f}%")
            break

    # Save final model
    torch.save({
        'epoch': args.epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'model_name': args.model,
        'class_names': class_names
    }, experiment_path / 'last.pt')

    end_time = time.time()
    training_time = end_time - start_time

    print("\n" + "=" * 60)
    print("[DONE] PYTORCH CLASSIFICATION TRAINING COMPLETED!")
    print("=" * 60)
    print(f"[TIMER] Training time: {training_time/60:.1f} minutes")
    print(f"[RESULTS] Results saved to: {experiment_path}")
    print(f"[BEST] Best validation accuracy: {best_val_acc:.2f}%")

    # Test evaluation
    print("\n[TEST] Running test evaluation...")

    # Test evaluation (only if test data exists)
    if test_loader is not None and len(test_dataset) > 0:
        # Load best model for testing
        checkpoint = torch.load(experiment_path / 'best.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc, test_preds, test_labels = validate_epoch(model, test_loader, criterion, device)

        print(f"\n[TEST] Test Results:")
        print(f"   Test Accuracy: {test_acc:.2f}%")
        print(f"   Test Loss: {test_loss:.4f}")

        # MEDICAL AI SAFETY: Calculate balanced accuracy
        balanced_acc = balanced_accuracy_score(test_labels, test_preds) * 100  # Convert to percentage
        print(f"   Balanced Accuracy: {balanced_acc:.2f}% (Medical AI critical metric)")

        # Generate classification report with zero_division handling
        report = classification_report(test_labels, test_preds, target_names=class_names, zero_division=0)
        print(f"\n[REPORT] Classification Report:")
        print(report)

        # Generate structured classification report (JSON) for Table 9 analysis
        report_dict = classification_report(test_labels, test_preds, target_names=class_names,
                                          zero_division=0, output_dict=True)

        # Create structured metrics for Table 9
        table9_metrics = {
            'overall_accuracy': report_dict['accuracy'],
            'overall_balanced_accuracy': balanced_acc / 100,  # Convert back to decimal
            'test_accuracy': test_acc / 100,  # Convert to decimal for consistency
            'per_class_metrics': {}
        }

        # Extract per-class metrics for Table 9
        for class_idx, class_name in enumerate(class_names):
            if class_name in report_dict:
                table9_metrics['per_class_metrics'][f'class_{class_idx}'] = {
                    'class_name': class_name,
                    'class_index': class_idx,
                    'precision': report_dict[class_name]['precision'],
                    'recall': report_dict[class_name]['recall'],
                    'f1_score': report_dict[class_name]['f1-score'],
                    'support': report_dict[class_name]['support']
                }

        print(f"\n[TABLE9] Structured metrics saved for Table 9 analysis")
    else:
        print(f"\n[TEST] No test data available, skipping test evaluation")
        test_acc = 0.0
        test_loss = 0.0
        balanced_acc = 0.0
        report = "No test data available"

        # Create placeholder metrics for no-test case
        table9_metrics = {
            'overall_accuracy': 0.0,
            'overall_balanced_accuracy': 0.0,
            'test_accuracy': 0.0,
            'per_class_metrics': {},
            'note': 'No test data available'
        }

    # Save results
    with open(experiment_path / 'results.txt', 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Best Val Acc: {best_val_acc:.2f}%\n")
        if test_loader is not None and len(test_dataset) > 0:
            f.write(f"Test Acc: {test_acc:.2f}%\n")
            f.write(f"Balanced Acc: {balanced_acc:.2f}%\n")
        else:
            f.write(f"Test Acc: N/A (no test data)\n")
            f.write(f"Balanced Acc: N/A (no test data)\n")
        f.write(f"Training Time: {training_time/60:.1f} min\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Save structured metrics for Table 9 analysis (JSON format)
    with open(experiment_path / 'table9_metrics.json', 'w') as f:
        json.dump(table9_metrics, f, indent=2)
    print(f"[TABLE9] [OK] Structured metrics saved: {experiment_path / 'table9_metrics.json'}")

    # Save confusion matrix (only if test data exists)
    if test_loader is not None and len(test_dataset) > 0:
        save_confusion_matrix(test_labels, test_preds, class_names,
                             experiment_path / 'confusion_matrix.png')

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(experiment_path / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[DONE] {args.model} classification training completed successfully!")
    print(f"[SUMMARY] Results summary:")
    print(f"   - Model: {args.model}")
    print(f"   - Classes: {len(class_names)}")
    print(f"   - Best Val Acc: {best_val_acc:.2f}%")
    print(f"   - Test Acc: {test_acc:.2f}%")
    print(f"   - Parameters: {total_params:,}")

if __name__ == "__main__":
    main()
