"""Evaluation script for medical image classification."""

import argparse
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.architectures import create_model
from src.data.dataset import create_data_loaders
from src.metrics.evaluation import MedicalMetrics, ClinicalReport
from src.utils.explainability import GradCAM, ScoreCAM, UncertaintyEstimator
from src.utils.core import load_config, get_device, setup_logging


class Evaluator:
    """Evaluator class for medical image classification."""
    
    def __init__(self, config: Dict, checkpoint_path: str):
        """Initialize evaluator.
        
        Args:
            config: Model configuration
            checkpoint_path: Path to model checkpoint
        """
        self.config = config
        self.device = get_device()
        self.logger = setup_logging('INFO')
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Create data loaders
        self.data_loaders = self._create_data_loaders()
        
        # Initialize metrics
        self.metrics = MedicalMetrics(
            num_classes=config['model']['num_classes'],
            class_names=config.get('class_names', ['Normal', 'Abnormal'])
        )
        
        # Initialize explainability tools
        self.gradcam = GradCAM(self.model)
        self.scorecam = ScoreCAM(self.model)
        self.uncertainty_estimator = UncertaintyEstimator(self.model)
        
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint."""
        # Create model
        model_config = self.config['model']
        model = create_model(
            model_name=model_config['name'],
            num_classes=model_config['num_classes'],
            pretrained=model_config.get('pretrained', True),
            dropout=model_config.get('dropout', 0.3),
            uncertainty=model_config.get('uncertainty', False)
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Loaded model from {checkpoint_path}")
        return model
    
    def _create_data_loaders(self) -> Dict[str, torch.utils.data.DataLoader]:
        """Create data loaders."""
        data_config = self.config['data']
        return create_data_loaders(
            data_dir=data_config['data_dir'],
            batch_size=data_config['batch_size'],
            image_size=tuple(data_config['image_size']),
            num_workers=data_config.get('num_workers', 4),
            augmentation=False,  # No augmentation for evaluation
            synthetic=data_config.get('synthetic', False)
        )
    
    def evaluate(self, split: str = 'test') -> Dict[str, float]:
        """Evaluate model on specified split.
        
        Args:
            split: Data split to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Evaluating on {split} split...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_uncertainties = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.data_loaders[split], desc=f'Evaluating {split}'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                # Uncertainty estimation if available
                if hasattr(self.model, 'forward') and 'uncertainty' in str(type(self.model)):
                    try:
                        _, uncertainties = self.model(images, return_uncertainty=True)
                        all_uncertainties.extend(uncertainties.cpu().numpy())
                    except:
                        pass
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(
            all_labels, all_preds, all_probs
        )
        
        # Add uncertainty metrics if available
        if all_uncertainties:
            metrics['mean_uncertainty'] = np.mean(all_uncertainties)
            metrics['std_uncertainty'] = np.std(all_uncertainties)
        
        return metrics
    
    def generate_explanations(
        self,
        num_samples: int = 5,
        save_dir: str = "assets/explanations"
    ):
        """Generate explanations for sample images.
        
        Args:
            num_samples: Number of samples to explain
            save_dir: Directory to save explanations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info(f"Generating explanations for {num_samples} samples...")
        
        # Get sample images
        sample_images = []
        sample_labels = []
        
        for images, labels in self.data_loaders['test']:
            sample_images.extend(images[:2])  # Take 2 from each batch
            sample_labels.extend(labels[:2])
            if len(sample_images) >= num_samples:
                break
        
        sample_images = sample_images[:num_samples]
        sample_labels = sample_labels[:num_samples]
        
        # Generate explanations
        for i, (image, label) in enumerate(zip(sample_images, sample_labels)):
            image_tensor = image.unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                pred_prob = torch.softmax(output, dim=1)
                pred_class = torch.argmax(output, dim=1).item()
                confidence = pred_prob[0, pred_class].item()
            
            # Generate GradCAM
            try:
                gradcam_path = os.path.join(save_dir, f'sample_{i}_gradcam.png')
                self.gradcam.visualize(
                    image_tensor,
                    class_idx=pred_class,
                    save_path=gradcam_path
                )
            except Exception as e:
                self.logger.warning(f"GradCAM failed for sample {i}: {e}")
            
            # Generate ScoreCAM
            try:
                scorecam_path = os.path.join(save_dir, f'sample_{i}_scorecam.png')
                scorecam = self.scorecam.generate_cam(image_tensor, pred_class)
                
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                img = image.permute(1, 2, 0).numpy()
                if img.min() < 0:
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                plt.imshow(img)
                plt.title(f'True: {label}, Pred: {pred_class}, Conf: {confidence:.3f}')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(scorecam, cmap='jet')
                plt.title('ScoreCAM')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(scorecam_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                self.logger.warning(f"ScoreCAM failed for sample {i}: {e}")
    
    def generate_uncertainty_analysis(
        self,
        num_samples: int = 100,
        save_dir: str = "assets/uncertainty"
    ):
        """Generate uncertainty analysis.
        
        Args:
            num_samples: Number of samples to analyze
            save_dir: Directory to save analysis
        """
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info(f"Generating uncertainty analysis for {num_samples} samples...")
        
        uncertainties = []
        confidences = []
        correct_predictions = []
        
        sample_count = 0
        
        with torch.no_grad():
            for images, labels in self.data_loaders['test']:
                if sample_count >= num_samples:
                    break
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions and uncertainties
                mean_preds, uncertainties_batch = self.uncertainty_estimator.estimate_uncertainty(images)
                pred_classes = torch.argmax(mean_preds, dim=1)
                confidences_batch = torch.max(mean_preds, dim=1)[0]
                
                uncertainties.extend(uncertainties_batch.cpu().numpy())
                confidences.extend(confidences_batch.cpu().numpy())
                correct_predictions.extend((pred_classes == labels).cpu().numpy())
                
                sample_count += len(images)
        
        # Create uncertainty plots
        plt.figure(figsize=(15, 5))
        
        # Uncertainty distribution
        plt.subplot(1, 3, 1)
        plt.hist(uncertainties, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Uncertainty (Entropy)')
        plt.ylabel('Frequency')
        plt.title('Uncertainty Distribution')
        plt.grid(True, alpha=0.3)
        
        # Confidence vs Uncertainty
        plt.subplot(1, 3, 2)
        plt.scatter(confidences, uncertainties, alpha=0.6)
        plt.xlabel('Confidence')
        plt.ylabel('Uncertainty')
        plt.title('Confidence vs Uncertainty')
        plt.grid(True, alpha=0.3)
        
        # Accuracy vs Uncertainty
        plt.subplot(1, 3, 3)
        uncertainty_bins = np.linspace(min(uncertainties), max(uncertainties), 10)
        bin_accuracies = []
        bin_centers = []
        
        for i in range(len(uncertainty_bins) - 1):
            mask = (np.array(uncertainties) >= uncertainty_bins[i]) & \
                   (np.array(uncertainties) < uncertainty_bins[i + 1])
            if np.sum(mask) > 0:
                bin_accuracies.append(np.mean(np.array(correct_predictions)[mask]))
                bin_centers.append((uncertainty_bins[i] + uncertainty_bins[i + 1]) / 2)
        
        plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=6)
        plt.xlabel('Uncertainty (Entropy)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Uncertainty')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print uncertainty statistics
        print(f"\nUncertainty Analysis:")
        print(f"Mean Uncertainty: {np.mean(uncertainties):.4f}")
        print(f"Std Uncertainty: {np.std(uncertainties):.4f}")
        print(f"Mean Confidence: {np.mean(confidences):.4f}")
        print(f"Correlation (Confidence vs Uncertainty): {np.corrcoef(confidences, uncertainties)[0, 1]:.4f}")
    
    def generate_comprehensive_report(
        self,
        save_dir: str = "assets/reports"
    ):
        """Generate comprehensive evaluation report.
        
        Args:
            save_dir: Directory to save reports
        """
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info("Generating comprehensive evaluation report...")
        
        # Evaluate on all splits
        splits = ['train', 'val', 'test']
        all_metrics = {}
        
        for split in splits:
            if split in self.data_loaders:
                metrics = self.evaluate(split)
                all_metrics[split] = metrics
                
                # Generate plots for test split
                if split == 'test':
                    self._generate_evaluation_plots(metrics, save_dir)
        
        # Generate clinical report
        test_metrics = all_metrics.get('test', {})
        if test_metrics:
            # Get test predictions for report
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for images, labels in tqdm(self.data_loaders['test'], desc='Generating report'):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            # Generate clinical report
            report = ClinicalReport(self.metrics)
            clinical_report = report.generate_report(
                all_labels, all_preds, all_probs,
                model_name=self.config['model']['name']
            )
            
            # Save report
            report_path = os.path.join(save_dir, 'clinical_report.txt')
            with open(report_path, 'w') as f:
                f.write(clinical_report)
            
            print(clinical_report)
    
    def _generate_evaluation_plots(self, metrics: Dict[str, float], save_dir: str):
        """Generate evaluation plots."""
        # This would generate ROC curves, PR curves, calibration plots, etc.
        # Implementation depends on the specific metrics available
        pass


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate medical image classification model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--explanations', action='store_true', help='Generate explanations')
    parser.add_argument('--uncertainty', action='store_true', help='Generate uncertainty analysis')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create evaluator
    evaluator = Evaluator(config, args.checkpoint)
    
    # Run evaluations
    if args.report:
        evaluator.generate_comprehensive_report()
    else:
        # Basic evaluation
        metrics = evaluator.evaluate('test')
        print("\nTest Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
    
    # Generate explanations if requested
    if args.explanations:
        evaluator.generate_explanations()
    
    # Generate uncertainty analysis if requested
    if args.uncertainty:
        evaluator.generate_uncertainty_analysis()
    
    # Cleanup
    evaluator.gradcam.cleanup()
    evaluator.scorecam.cleanup()


if __name__ == '__main__':
    main()
