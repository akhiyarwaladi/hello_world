#!/usr/bin/env python3
"""
Advanced Loss Functions for Imbalanced Medical Image Classification

Provides additional loss functions beyond Cross-Entropy and Focal Loss:
- Label Smoothing Cross-Entropy
- Class-Balanced Loss
- Weighted Focal Loss
- Combined Losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss

    Prevents model overconfidence by smoothing hard labels.
    Instead of [0, 1, 0, 0], becomes [0.025, 0.925, 0.025, 0.025]

    Args:
        smoothing (float): Smoothing factor (default: 0.1)
        reduction (str): 'mean', 'sum', or 'none'

    Reference:
        "Rethinking the Inception Architecture for Computer Vision" (2016)
        https://arxiv.org/abs/1512.00567
    """

    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (logits) [batch_size, num_classes]
            target: Ground truth labels [batch_size]
        """
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)

        # Negative log likelihood for correct class
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)

        # Uniform distribution smoothing
        smooth_loss = -log_preds.mean(dim=-1)

        if self.reduction == 'mean':
            smooth_loss = smooth_loss.mean()
        elif self.reduction == 'sum':
            smooth_loss = smooth_loss.sum()

        # Combine
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth_loss
        return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss Based on Effective Number of Samples

    Automatically reweights classes based on their frequency.
    More effective than simple inverse frequency weighting.

    Args:
        samples_per_class (list/array): Number of samples per class
        beta (float): Hyperparameter for reweighting (default: 0.9999)
        loss_type (str): 'focal' or 'ce' (default: 'ce')
        gamma (float): Focal loss gamma (only if loss_type='focal')

    Reference:
        "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
        https://arxiv.org/abs/1901.05555
    """

    def __init__(self, samples_per_class, beta=0.9999, loss_type='ce', gamma=1.5):
        super().__init__()
        self.loss_type = loss_type
        self.gamma = gamma

        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)

        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (logits) [batch_size, num_classes]
            target: Ground truth labels [batch_size]
        """
        # FIXED: Ensure weights on same device as predictions (CPU vs CUDA)
        weights = self.weights.to(pred.device)

        if self.loss_type == 'focal':
            # Focal loss with class balancing
            ce_loss = F.cross_entropy(pred, target, weight=weights, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()
        else:
            # Standard cross-entropy with class balancing
            return F.cross_entropy(pred, target, weight=weights)


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with Per-Class Weights

    Combines the focusing mechanism of Focal Loss with explicit class weights.
    Best for extreme class imbalance.

    Args:
        alpha (list/tensor): Per-class weights [num_classes]
        gamma (float): Focusing parameter (default: 1.5, reduced from 2.0 for stability)
        reduction (str): 'mean', 'sum', or 'none'

    Reference:
        "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
        super().__init__()
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (logits) [batch_size, num_classes]
            target: Ground truth labels [batch_size]
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')

        # Calculate pt
        pt = torch.exp(-ce_loss)

        # Focal term
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply per-class weights
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss: Weighted sum of multiple losses

    Useful for combining complementary loss functions.
    Example: CE + Focal, or CE + Label Smoothing

    Args:
        losses (list): List of (loss_fn, weight) tuples
    """

    def __init__(self, losses):
        super().__init__()
        self.losses = losses

    def forward(self, pred, target):
        total_loss = 0
        for loss_fn, weight in self.losses:
            total_loss += weight * loss_fn(pred, target)
        return total_loss


class DiceLoss(nn.Module):
    """
    Dice Loss (Adapted for Classification)

    Originally for segmentation, adapted for multi-class classification.
    Good for medical imaging where some classes overlap.

    Args:
        smooth (float): Smoothing factor to avoid division by zero
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (logits) [batch_size, num_classes]
            target: Ground truth labels [batch_size]
        """
        # Convert to probabilities
        pred = F.softmax(pred, dim=1)

        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).float()

        # Calculate Dice coefficient per class
        intersection = (pred * target_one_hot).sum(dim=0)
        union = pred.sum(dim=0) + target_one_hot.sum(dim=0)

        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)

        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice_coeff.mean()


# Helper function to get loss by name
def get_loss_function(loss_name, **kwargs):
    """
    Factory function to get loss by name

    Args:
        loss_name (str): Name of loss function
        **kwargs: Additional arguments for the loss function

    Returns:
        loss_fn: Instantiated loss function

    Example:
        >>> loss_fn = get_loss_function('label_smoothing', smoothing=0.1)
        >>> loss = loss_fn(predictions, targets)
    """

    if loss_name == 'cross_entropy' or loss_name == 'ce':
        return nn.CrossEntropyLoss()

    elif loss_name == 'label_smoothing' or loss_name == 'ls':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)

    elif loss_name == 'class_balanced' or loss_name == 'cb':
        samples_per_class = kwargs.get('samples_per_class')
        if samples_per_class is None:
            raise ValueError("samples_per_class required for ClassBalancedLoss")
        beta = kwargs.get('beta', 0.9999)
        loss_type = kwargs.get('loss_type', 'ce')
        gamma = kwargs.get('gamma', 1.5)
        return ClassBalancedLoss(samples_per_class, beta, loss_type, gamma)

    elif loss_name == 'focal':
        gamma = kwargs.get('gamma', 1.5)
        alpha = kwargs.get('alpha', 1.0)
        # Import from parent module if needed
        from scripts.training.train_pytorch_classification import FocalLoss
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_name == 'weighted_focal' or loss_name == 'wf':
        alpha = kwargs.get('alpha')
        gamma = kwargs.get('gamma', 1.5)
        return WeightedFocalLoss(alpha=alpha, gamma=gamma)

    elif loss_name == 'dice':
        smooth = kwargs.get('smooth', 1.0)
        return DiceLoss(smooth=smooth)

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


if __name__ == "__main__":
    # Test losses
    print("Testing Advanced Loss Functions...")

    # Dummy data
    batch_size = 8
    num_classes = 4
    pred = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))

    # Test each loss
    print("\n1. Label Smoothing CE:")
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    print(f"   Loss: {ls_loss(pred, target).item():.4f}")

    print("\n2. Class-Balanced Loss:")
    samples_per_class = [100, 50, 30, 20]  # Imbalanced
    cb_loss = ClassBalancedLoss(samples_per_class)
    print(f"   Loss: {cb_loss(pred, target).item():.4f}")

    print("\n3. Weighted Focal Loss:")
    alpha = [1.0, 2.0, 3.0, 4.0]  # Higher weight for rare classes
    wf_loss = WeightedFocalLoss(alpha=alpha, gamma=1.5)
    print(f"   Loss: {wf_loss(pred, target).item():.4f}")

    print("\n4. Dice Loss:")
    dice_loss = DiceLoss()
    print(f"   Loss: {dice_loss(pred, target).item():.4f}")

    print("\n5. Combined Loss (CE + Focal):")
    ce_loss = nn.CrossEntropyLoss()
    focal_loss = WeightedFocalLoss(gamma=1.5)
    combined = CombinedLoss([(ce_loss, 0.5), (focal_loss, 0.5)])
    print(f"   Loss: {combined(pred, target).item():.4f}")

    print("\nAll tests passed! [OK]")
