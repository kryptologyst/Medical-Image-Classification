"""Explainability and uncertainty estimation for medical image classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Generates visual explanations for CNN predictions by computing
    gradients of the target class with respect to feature maps.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        """Initialize GradCAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of target layer for GradCAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        if self.target_layer is None:
            # Default to last convolutional layer
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    self.target_layer = name
        
        # Register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                break
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Generate GradCAM heatmap.
        
        Args:
            input_tensor: Input image tensor
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            GradCAM heatmap
        """
        # Forward pass
        self.model.eval()
        input_tensor.requires_grad_()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Normalize CAM
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def visualize(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        alpha: float = 0.4,
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Visualize GradCAM overlay.
        
        Args:
            input_tensor: Input image tensor
            class_idx: Target class index
            alpha: Overlay transparency
            save_path: Path to save visualization
            
        Returns:
            Tuple of (original_image, overlay_image)
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Convert input to numpy
        img = input_tensor[0].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        
        # Denormalize if needed (assuming ImageNet normalization)
        if img.min() < 0:
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        
        img = np.clip(img, 0, 1)
        
        # Resize CAM to match image
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = cv2.addWeighted(
            np.uint8(255 * img), 
            alpha, 
            heatmap, 
            1 - alpha, 
            0
        )
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('GradCAM Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return img, overlay
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()


class ScoreCAM:
    """Score-weighted Class Activation Mapping (Score-CAM).
    
    Alternative to GradCAM that uses forward pass scores
    instead of gradients for generating explanations.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        """Initialize ScoreCAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of target layer
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook = None
        
        # Register hook
        self._register_hook()
    
    def _register_hook(self):
        """Register forward hook."""
        def forward_hook(module, input, output):
            self.activations = output
        
        # Find target layer
        if self.target_layer is None:
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    self.target_layer = name
        
        # Register hook
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hook = module.register_forward_hook(forward_hook)
                break
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Generate ScoreCAM heatmap.
        
        Args:
            input_tensor: Input image tensor
            class_idx: Target class index
            
        Returns:
            ScoreCAM heatmap
        """
        self.model.eval()
        
        # Forward pass to get activations
        _ = self.model(input_tensor)
        activations = self.activations[0]  # [C, H, W]
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(input_tensor)
            if class_idx is None:
                class_idx = baseline_output.argmax(dim=1).item()
            baseline_score = baseline_output[0, class_idx].item()
        
        # Generate CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        
        for i in range(activations.shape[0]):
            # Create masked input
            masked_input = input_tensor.clone()
            
            # Upsample activation to input size
            activation = activations[i:i+1]  # [1, H, W]
            activation_upsampled = F.interpolate(
                activation.unsqueeze(0),
                size=input_tensor.shape[2:],
                mode='bilinear',
                align_corners=False
            )[0, 0]  # [H, W]
            
            # Normalize activation
            activation_norm = (activation_upsampled - activation_upsampled.min()) / \
                             (activation_upsampled.max() - activation_upsampled.min())
            
            # Apply mask
            masked_input[0] = masked_input[0] * activation_norm
            
            # Get score for masked input
            with torch.no_grad():
                masked_output = self.model(masked_input)
                masked_score = masked_output[0, class_idx].item()
            
            # Weight by score increase
            weight = max(0, masked_score - baseline_score)
            cam += weight * activations[i]
        
        # Normalize CAM
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def cleanup(self):
        """Remove hook."""
        if self.hook is not None:
            self.hook.remove()


class UncertaintyEstimator:
    """Uncertainty estimation using Monte Carlo Dropout."""
    
    def __init__(self, model: nn.Module, num_samples: int = 10):
        """Initialize uncertainty estimator.
        
        Args:
            model: PyTorch model
            num_samples: Number of Monte Carlo samples
        """
        self.model = model
        self.num_samples = num_samples
        
        # Enable dropout in eval mode
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout layers for uncertainty estimation."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def estimate_uncertainty(
        self,
        input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate prediction uncertainty.
        
        Args:
            input_tensor: Input tensor
            
        Returns:
            Tuple of (mean_predictions, uncertainty)
        """
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(input_tensor)
                predictions.append(F.softmax(output, dim=1))
        
        # Stack predictions
        predictions = torch.stack(predictions)  # [num_samples, batch_size, num_classes]
        
        # Calculate mean and variance
        mean_pred = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        # Calculate uncertainty as entropy
        uncertainty = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=1)
        
        return mean_pred, uncertainty
    
    def get_confidence_intervals(
        self,
        input_tensor: torch.Tensor,
        confidence_level: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get confidence intervals for predictions.
        
        Args:
            input_tensor: Input tensor
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(input_tensor)
                predictions.append(output)
        
        predictions = torch.stack(predictions)
        
        # Calculate percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = torch.quantile(predictions, lower_percentile / 100, dim=0)
        upper_bound = torch.quantile(predictions, upper_percentile / 100, dim=0)
        
        return lower_bound, upper_bound


class AttentionVisualizer:
    """Visualize attention maps for transformer-based models."""
    
    def __init__(self, model: nn.Module):
        """Initialize attention visualizer.
        
        Args:
            model: Transformer model
        """
        self.model = model
        self.attention_weights = None
        self.hook = None
        
        # Register hook for attention weights
        self._register_hook()
    
    def _register_hook(self):
        """Register hook to capture attention weights."""
        def attention_hook(module, input, output):
            if hasattr(module, 'attention_weights'):
                self.attention_weights = module.attention_weights
        
        # Find attention modules
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                self.hook = module.register_forward_hook(attention_hook)
                break
    
    def visualize_attention(
        self,
        input_tensor: torch.Tensor,
        patch_size: int = 16,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """Visualize attention maps.
        
        Args:
            input_tensor: Input tensor
            patch_size: Size of patches
            save_path: Path to save visualization
            
        Returns:
            Attention visualization
        """
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if self.attention_weights is None:
            print("No attention weights found")
            return None
        
        # Process attention weights
        attention = self.attention_weights[0]  # [num_heads, num_patches, num_patches]
        
        # Average across heads
        attention = attention.mean(dim=0)  # [num_patches, num_patches]
        
        # Get attention for CLS token (first token)
        cls_attention = attention[0, 1:]  # [num_patches-1]
        
        # Reshape to spatial dimensions
        num_patches = int(np.sqrt(len(cls_attention)))
        attention_map = cls_attention.reshape(num_patches, num_patches)
        
        # Resize to input image size
        img_size = input_tensor.shape[-1]
        attention_map = cv2.resize(
            attention_map.numpy(),
            (img_size, img_size)
        )
        
        # Visualize
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        img = input_tensor[0].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        if img.min() < 0:
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(attention_map, cmap='jet')
        plt.title('Attention Map')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return attention_map
    
    def cleanup(self):
        """Remove hook."""
        if self.hook is not None:
            self.hook.remove()
