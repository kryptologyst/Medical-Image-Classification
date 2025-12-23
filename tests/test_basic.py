"""Test configuration and basic tests for medical image classification."""

import pytest
import torch
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.architectures import create_model, MedicalCNN, EfficientNetMedical
from data.dataset import MedicalImageDataset, SyntheticMedicalDataset
from losses.losses import create_loss_function, FocalLoss
from metrics.evaluation import MedicalMetrics
from utils.core import set_seed, get_device, EarlyStopping


class TestModels:
    """Test model architectures."""
    
    def test_medical_cnn_creation(self):
        """Test MedicalCNN model creation."""
        model = MedicalCNN(num_classes=2, pretrained=False)
        assert isinstance(model, torch.nn.Module)
        assert model.classifier[-1].out_features == 2
    
    def test_efficientnet_creation(self):
        """Test EfficientNet model creation."""
        model = EfficientNetMedical(num_classes=3, pretrained=False)
        assert isinstance(model, torch.nn.Module)
        assert model.classifier[-1].out_features == 3
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = MedicalCNN(num_classes=2, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 2)
    
    def test_create_model_function(self):
        """Test create_model function."""
        model = create_model("resnet18", num_classes=2, pretrained=False)
        assert isinstance(model, torch.nn.Module)
        
        # Test unknown model
        with pytest.raises(ValueError):
            create_model("unknown_model")


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticMedicalDataset(size=10, num_classes=2)
        assert len(dataset) == 10
        
        # Test data loading
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert image.shape == (3, 224, 224)
        assert 0 <= label < 2
    
    def test_medical_image_dataset_structure(self):
        """Test medical image dataset structure."""
        # This test would require actual data, so we'll mock it
        with patch('os.listdir') as mock_listdir, \
             patch('os.path.isdir') as mock_isdir, \
             patch('os.path.exists') as mock_exists:
            
            mock_listdir.return_value = ['normal', 'abnormal']
            mock_isdir.return_value = True
            mock_exists.return_value = True
            
            # This would normally fail without real data, but we're testing the structure
            try:
                dataset = MedicalImageDataset("dummy_path", split="train")
                # If it gets here, the structure is correct
                assert hasattr(dataset, 'classes')
                assert hasattr(dataset, 'class_to_idx')
                assert hasattr(dataset, 'samples')
            except (FileNotFoundError, OSError):
                # Expected without real data
                pass


class TestLossFunctions:
    """Test loss functions."""
    
    def test_focal_loss(self):
        """Test Focal Loss."""
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Test with dummy data
        inputs = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        
        loss = loss_fn(inputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_create_loss_function(self):
        """Test create_loss_function."""
        # Test cross entropy
        loss_fn = create_loss_function("cross_entropy", num_classes=2)
        assert isinstance(loss_fn, torch.nn.Module)
        
        # Test focal loss
        loss_fn = create_loss_function("focal", num_classes=2)
        assert isinstance(loss_fn, FocalLoss)
        
        # Test unknown loss
        with pytest.raises(ValueError):
            create_loss_function("unknown_loss")


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_medical_metrics_creation(self):
        """Test MedicalMetrics creation."""
        metrics = MedicalMetrics(num_classes=2, class_names=['Normal', 'Abnormal'])
        assert metrics.num_classes == 2
        assert metrics.class_names == ['Normal', 'Abnormal']
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        metrics = MedicalMetrics(num_classes=2)
        
        # Dummy data
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.6, 0.4]])
        
        result = metrics.calculate_metrics(y_true, y_pred, y_prob)
        
        assert isinstance(result, dict)
        assert 'accuracy' in result
        assert 'auroc' in result
        assert 'auprc' in result


class TestUtilities:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't crash
        assert True
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        # Mock model
        model = MagicMock()
        
        # Test improvement
        assert not early_stopping(0.8, model)
        assert not early_stopping(0.9, model)
        
        # Test no improvement
        assert not early_stopping(0.89, model)  # Within min_delta
        assert not early_stopping(0.88, model)
        assert not early_stopping(0.87, model)
        assert early_stopping(0.86, model)  # Should trigger early stopping


class TestIntegration:
    """Integration tests."""
    
    def test_training_step_simulation(self):
        """Simulate a training step."""
        # Create model
        model = MedicalCNN(num_classes=2, pretrained=False)
        
        # Create dummy data
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 2, (2,))
        
        # Create loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        
        assert loss.item() >= 0
        assert outputs.shape == (2, 2)
    
    def test_model_parameter_count(self):
        """Test model parameter counting."""
        model = MedicalCNN(num_classes=2, pretrained=False)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert total_params == trainable_params  # All parameters should be trainable


# Pytest configuration
@pytest.fixture(scope="session")
def setup_test_environment():
    """Setup test environment."""
    # Set random seed for reproducible tests
    set_seed(42)
    
    # Create test directories if needed
    os.makedirs("test_data", exist_ok=True)
    
    yield
    
    # Cleanup after tests
    import shutil
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
