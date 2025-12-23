"""Training script for medical image classification."""

import argparse
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import yaml

from src.models.architectures import create_model
from src.data.dataset import create_data_loaders
from src.losses.losses import create_loss_function, calculate_class_weights
from src.metrics.evaluation import MedicalMetrics, ClinicalReport
from src.utils.core import (
    setup_logging, set_seed, get_device, load_config, save_config,
    create_directories, EarlyStopping, format_time
)


class Trainer:
    """Trainer class for medical image classification."""
    
    def __init__(self, config: Dict):
        """Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = get_device()
        self.logger = setup_logging(config.get('log_level', 'INFO'))
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Create directories
        self.checkpoint_dir = config['checkpoint_dir']
        self.log_dir = config['log_dir']
        create_directories([self.checkpoint_dir, self.log_dir])
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize data loaders
        self.data_loaders = self._create_data_loaders()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize metrics
        self.metrics = MedicalMetrics(
            num_classes=config['model']['num_classes'],
            class_names=config.get('class_names', ['Normal', 'Abnormal'])
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            min_delta=config.get('early_stopping_min_delta', 0.001)
        )
        
        # Initialize logging
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_metrics = []
        
    def _create_model(self) -> nn.Module:
        """Create model instance."""
        model_config = self.config['model']
        return create_model(
            model_name=model_config['name'],
            num_classes=model_config['num_classes'],
            pretrained=model_config.get('pretrained', True),
            dropout=model_config.get('dropout', 0.3),
            uncertainty=model_config.get('uncertainty', False)
        )
    
    def _create_data_loaders(self) -> Dict[str, DataLoader]:
        """Create data loaders."""
        data_config = self.config['data']
        return create_data_loaders(
            data_dir=data_config['data_dir'],
            batch_size=data_config['batch_size'],
            image_size=tuple(data_config['image_size']),
            num_workers=data_config.get('num_workers', 4),
            augmentation=data_config.get('augmentation', True),
            synthetic=data_config.get('synthetic', False)
        )
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function."""
        loss_config = self.config['loss']
        
        # Calculate class weights if needed
        class_weights = None
        if loss_config.get('use_class_weights', False):
            class_weights = calculate_class_weights(self.data_loaders['train'].dataset)
            class_weights = class_weights.to(self.device)
        
        return create_loss_function(
            loss_name=loss_config['name'],
            num_classes=self.config['model']['num_classes'],
            class_weights=class_weights,
            **loss_config.get('params', {})
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        opt_config = self.config['optimizer']
        
        if opt_config['name'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        elif opt_config['name'] == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        elif opt_config['name'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['name']}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_config = self.config.get('scheduler')
        
        if scheduler_config is None:
            return None
        
        if scheduler_config['name'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_config['name'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', self.config['training']['epochs'])
            )
        elif scheduler_config['name'] == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5)
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.data_loaders['train'], desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                self.writer.add_scalar(
                    'Train/Loss_Batch',
                    loss.item(),
                    self.current_epoch * len(self.data_loaders['train']) + batch_idx
                )
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.data_loaders['val'], desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(
            all_labels, all_preds, all_probs
        )
        
        self.val_metrics.append(metrics)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Model: {self.config['model']['name']}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('auroc', val_metrics.get('accuracy', 0)))
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            self.logger.info(f"Epoch {epoch}: Val AUROC = {val_metrics.get('auroc', 0):.4f}")
            self.logger.info(f"Epoch {epoch}: Val Accuracy = {val_metrics.get('accuracy', 0):.4f}")
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })
            
            # Save checkpoint
            current_metric = val_metrics.get('auroc', val_metrics.get('accuracy', 0))
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
            
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.early_stopping(current_metric, self.model):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {format_time(total_time)}")
        self.logger.info(f"Best metric: {self.best_metric:.4f}")
        
        # Generate final report
        self.generate_final_report()
        
        # Close writer
        self.writer.close()
    
    def generate_final_report(self):
        """Generate final training report."""
        # Load best model
        best_checkpoint = os.path.join(self.checkpoint_dir, 'best.pth')
        if os.path.exists(best_checkpoint):
            self.load_checkpoint(best_checkpoint)
        
        # Evaluate on test set
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.data_loaders['test'], desc='Test Evaluation'):
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
        report_path = os.path.join(self.checkpoint_dir, 'clinical_report.txt')
        with open(report_path, 'w') as f:
            f.write(clinical_report)
        
        self.logger.info(f"Clinical report saved to {report_path}")
        print(clinical_report)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train medical image classification model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project="medical-image-classification",
            config=config,
            name=f"{config['model']['name']}_{config['data']['image_size'][0]}x{config['data']['image_size'][1]}"
        )
        config['use_wandb'] = True
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
