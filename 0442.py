#!/usr/bin/env python3
"""
Medical Image Classification - Entry Point

This is the main entry point for the medical image classification project.
For the modernized version, please use the scripts in the scripts/ directory.

RESEARCH DEMO ONLY - NOT FOR CLINICAL USE
"""

import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.core import setup_logging, get_device
from models.architectures import create_model
from data.dataset import SyntheticMedicalDataset


def main():
    """Main entry point for the medical image classification demo."""
    
    # Setup logging
    logger = setup_logging('INFO')
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Medical Image Classification Demo')
    parser.add_argument('--mode', choices=['demo', 'train', 'evaluate'], default='demo',
                       help='Mode to run the application')
    parser.add_argument('--model', default='efficientnet_b0', 
                       help='Model architecture to use')
    parser.add_argument('--config', default='configs/default.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Display disclaimer
    print("=" * 80)
    print("MEDICAL IMAGE CLASSIFICATION - RESEARCH DEMO")
    print("=" * 80)
    print("⚠️  IMPORTANT DISCLAIMER:")
    print("   This software is for RESEARCH AND EDUCATIONAL PURPOSES ONLY")
    print("   NOT approved for clinical diagnosis or treatment decisions")
    print("   NOT a medical device")
    print("   NOT validated for clinical use")
    print("   NOT intended to replace professional medical judgment")
    print("   Always consult qualified healthcare professionals for medical decisions")
    print("=" * 80)
    
    if args.mode == 'demo':
        run_demo(args)
    elif args.mode == 'train':
        run_training(args)
    elif args.mode == 'evaluate':
        run_evaluation(args)


def run_demo(args):
    """Run a simple demo of the medical image classification."""
    logger = setup_logging('INFO')
    logger.info("Starting medical image classification demo...")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info(f"Creating model: {args.model}")
    model = create_model(
        model_name=args.model,
        num_classes=2,
        pretrained=True,
        dropout=0.3
    )
    model = model.to(device)
    model.eval()
    
    # Create synthetic dataset
    logger.info("Creating synthetic medical dataset...")
    dataset = SyntheticMedicalDataset(size=20, num_classes=2)
    
    # Make predictions
    logger.info("Making predictions on sample images...")
    correct = 0
    total = 0
    
    import torch
    import torch.nn.functional as F
    
    with torch.no_grad():
        for i in range(min(10, len(dataset))):
            image, true_label = dataset[i]
            image_tensor = image.unsqueeze(0).to(device)
            
            # Forward pass
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            if predicted_class == true_label:
                correct += 1
            total += 1
            
            print(f"Sample {i+1}: True={true_label}, Pred={predicted_class}, Conf={confidence:.3f}")
    
    accuracy = correct / total
    print(f"\nDemo Accuracy: {accuracy:.2%}")
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    logger.info("Demo completed successfully!")


def run_training(args):
    """Run training using the modern training script."""
    print("To run training, use: python scripts/train.py --config configs/default.yaml")
    print("Available configs:")
    print("  - configs/default.yaml")
    print("  - configs/efficientnet.yaml") 
    print("  - configs/vit.yaml")


def run_evaluation(args):
    """Run evaluation using the modern evaluation script."""
    print("To run evaluation, use: python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth")


if __name__ == '__main__':
    main()