"""Advanced model architectures for medical image classification."""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from transformers import ViTModel, ViTConfig


class MedicalCNN(nn.Module):
    """Baseline CNN model for medical image classification.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        features = self.backbone(x)
        return self.classifier(features)


class EfficientNetMedical(nn.Module):
    """EfficientNet model for medical image classification.
    
    Args:
        model_name: EfficientNet variant (e.g., 'efficientnet_b0')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classifier
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        features = self.backbone(x)
        return self.classifier(features)


class VisionTransformerMedical(nn.Module):
    """Vision Transformer for medical image classification.
    
    Args:
        model_name: ViT variant (e.g., 'vit_base_patch16_224')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Load pretrained ViT
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classifier
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        features = self.backbone(x)
        return self.classifier(features)


class ConvNeXtMedical(nn.Module):
    """ConvNeXt model for medical image classification.
    
    Args:
        model_name: ConvNeXt variant (e.g., 'convnext_tiny')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        model_name: str = "convnext_tiny",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Load pretrained ConvNeXt
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classifier
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        features = self.backbone(x)
        return self.classifier(features)


class EnsembleModel(nn.Module):
    """Ensemble model combining multiple architectures.
    
    Args:
        models: List of models to ensemble
        weights: Optional weights for each model
    """
    
    def __init__(
        self,
        models: list,
        weights: Optional[list] = None
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0] * len(models)
        
        # Normalize weights
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble output logits
        """
        outputs = []
        
        for model, weight in zip(self.models, self.weights):
            output = model(x)
            outputs.append(output * weight)
        
        return torch.stack(outputs).sum(dim=0)


class UncertaintyModel(nn.Module):
    """Model with uncertainty estimation using Monte Carlo Dropout.
    
    Args:
        base_model: Base model architecture
        dropout_rate: Dropout rate for uncertainty estimation
        num_samples: Number of Monte Carlo samples
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        dropout_rate: float = 0.5,
        num_samples: int = 10
    ):
        super().__init__()
        
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
        
        # Enable dropout in eval mode for uncertainty estimation
        self._enable_dropout()
        
    def _enable_dropout(self):
        """Enable dropout layers for uncertainty estimation."""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional uncertainty estimation.
        
        Args:
            x: Input tensor
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Output logits or (logits, uncertainty)
        """
        if not return_uncertainty:
            return self.base_model(x)
        
        # Monte Carlo Dropout for uncertainty estimation
        outputs = []
        
        for _ in range(self.num_samples):
            output = self.base_model(x)
            outputs.append(F.softmax(output, dim=1))
        
        # Stack outputs
        outputs = torch.stack(outputs)  # [num_samples, batch_size, num_classes]
        
        # Calculate mean and variance
        mean_output = outputs.mean(dim=0)
        variance = outputs.var(dim=0)
        
        # Calculate uncertainty as entropy
        uncertainty = -(mean_output * torch.log(mean_output + 1e-8)).sum(dim=1)
        
        return mean_output, uncertainty


def create_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    uncertainty: bool = False
) -> nn.Module:
    """Create a model instance.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        uncertainty: Whether to add uncertainty estimation
        
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == "resnet18":
        model = MedicalCNN(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif model_name.startswith("efficientnet"):
        model = EfficientNetMedical(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_name.startswith("vit"):
        model = VisionTransformerMedical(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_name.startswith("convnext"):
        model = ConvNeXtMedical(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if uncertainty:
        model = UncertaintyModel(model)
    
    return model
