"""Loss functions for medical image classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss = -alpha * (1-p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        reduction: Reduction method
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for class imbalance.
    
    Args:
        class_weights: Weights for each class
        reduction: Reduction method
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Weighted cross entropy loss
        """
        return F.cross_entropy(
            inputs, 
            targets, 
            weight=self.class_weights,
            reduction=self.reduction
        )


class TverskyLoss(nn.Module):
    """Tversky Loss for medical image segmentation/classification.
    
    Tversky Loss = 1 - Tversky Index
    Tversky Index = TP / (TP + alpha*FN + beta*FP)
    
    Args:
        alpha: Weight for false negatives
        beta: Weight for false positives
        smooth: Smoothing factor
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Tversky loss value
        """
        # Convert to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # Calculate TP, FP, FN
        tp = (probs * targets_one_hot).sum(dim=0)
        fp = (probs * (1 - targets_one_hot)).sum(dim=0)
        fn = ((1 - probs) * targets_one_hot).sum(dim=0)
        
        # Calculate Tversky Index
        tversky_index = tp / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Return Tversky Loss
        return 1 - tversky_index.mean()


class DiceLoss(nn.Module):
    """Dice Loss for medical image classification.
    
    Dice Loss = 1 - Dice Coefficient
    Dice Coefficient = 2*TP / (2*TP + FP + FN)
    
    Args:
        smooth: Smoothing factor
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Dice loss value
        """
        # Convert to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # Calculate intersection and union
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        # Calculate Dice Coefficient
        dice_coeff = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice Loss
        return 1 - dice_coeff.mean()


class CombinedLoss(nn.Module):
    """Combined loss function for medical image classification.
    
    Combines multiple loss functions with weights.
    
    Args:
        losses: List of loss functions
        weights: Weights for each loss function
    """
    
    def __init__(
        self,
        losses: list,
        weights: Optional[list] = None
    ):
        super().__init__()
        
        self.losses = nn.ModuleList(losses)
        
        if weights is None:
            weights = [1.0] * len(losses)
        
        self.weights = weights
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Combined loss value
        """
        total_loss = 0
        
        for loss_fn, weight in zip(self.losses, self.weights):
            loss = loss_fn(inputs, targets)
            total_loss += weight * loss
        
        return total_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Cross Entropy Loss.
    
    Args:
        smoothing: Label smoothing factor
        num_classes: Number of classes
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Label smoothing loss
        """
        log_preds = F.log_softmax(inputs, dim=1)
        
        # Create smoothed labels
        smoothed_labels = torch.zeros_like(log_preds)
        smoothed_labels.fill_(self.smoothing / (self.num_classes - 1))
        smoothed_labels.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # Calculate loss
        loss = -(smoothed_labels * log_preds).sum(dim=1)
        
        return loss.mean()


def create_loss_function(
    loss_name: str,
    num_classes: int = 2,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """Create a loss function instance.
    
    Args:
        loss_name: Name of the loss function
        num_classes: Number of classes
        class_weights: Class weights for imbalanced datasets
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function instance
    """
    loss_name = loss_name.lower()
    
    if loss_name == "cross_entropy":
        if class_weights is not None:
            return WeightedCrossEntropyLoss(class_weights=class_weights)
        else:
            return nn.CrossEntropyLoss()
    
    elif loss_name == "focal":
        return FocalLoss(**kwargs)
    
    elif loss_name == "tversky":
        return TverskyLoss(**kwargs)
    
    elif loss_name == "dice":
        return DiceLoss(**kwargs)
    
    elif loss_name == "label_smoothing":
        return LabelSmoothingLoss(num_classes=num_classes, **kwargs)
    
    elif loss_name == "combined":
        # Create combined loss with multiple components
        losses = []
        weights = kwargs.get("weights", [1.0, 0.5])
        
        # Add Cross Entropy
        if class_weights is not None:
            losses.append(WeightedCrossEntropyLoss(class_weights=class_weights))
        else:
            losses.append(nn.CrossEntropyLoss())
        
        # Add Focal Loss
        losses.append(FocalLoss())
        
        return CombinedLoss(losses, weights)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def calculate_class_weights(dataset) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets.
    
    Args:
        dataset: Dataset with labels
        
    Returns:
        Class weights tensor
    """
    # Count class frequencies
    class_counts = {}
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Calculate weights
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = []
    for i in range(num_classes):
        if i in class_counts:
            weight = total_samples / (num_classes * class_counts[i])
            weights.append(weight)
        else:
            weights.append(1.0)
    
    return torch.tensor(weights, dtype=torch.float32)
