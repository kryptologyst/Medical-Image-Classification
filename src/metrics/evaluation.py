"""Comprehensive evaluation metrics for medical image classification."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


class MedicalMetrics:
    """Comprehensive metrics for medical image classification.
    
    Provides clinical metrics including AUROC, AUPRC, sensitivity,
    specificity, calibration metrics, and more.
    """
    
    def __init__(self, num_classes: int = 2, class_names: Optional[List[str]] = None):
        """Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
    def calculate_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        if y_prob is not None:
            y_prob = self._to_numpy(y_prob)
        
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._calculate_classification_metrics(y_true, y_pred))
        
        # Probability-based metrics
        if y_prob is not None:
            metrics.update(self._calculate_probability_metrics(y_true, y_prob))
            
            # Calibration metrics
            metrics.update(self._calculate_calibration_metrics(y_true, y_prob))
        
        return metrics
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data
    
    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        metrics = {}
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if self.num_classes == 2:
            # Binary classification metrics
            tn, fp, fn, tp = cm.ravel()
            
            metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics["recall"] = metrics["sensitivity"]
            metrics["f1_score"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            
            # Additional clinical metrics
            metrics["ppv"] = metrics["precision"]  # Positive Predictive Value
            metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
            metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
            metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
            
        else:
            # Multi-class metrics
            metrics["accuracy"] = np.mean(y_true == y_pred)
            
            # Per-class metrics
            for i in range(self.num_classes):
                class_mask = (y_true == i)
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(y_pred[class_mask] == i)
                    metrics[f"accuracy_class_{i}"] = class_acc
        
        return metrics
    
    def _calculate_probability_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate probability-based metrics."""
        metrics = {}
        
        if self.num_classes == 2:
            # Binary classification
            if y_prob.ndim > 1:
                y_prob = y_prob[:, 1]  # Take positive class probability
            
            # AUROC
            try:
                metrics["auroc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics["auroc"] = 0.0
            
            # AUPRC
            try:
                metrics["auprc"] = average_precision_score(y_true, y_prob)
            except ValueError:
                metrics["auprc"] = 0.0
                
        else:
            # Multi-class metrics
            try:
                metrics["auroc_macro"] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics["auroc_weighted"] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except ValueError:
                metrics["auroc_macro"] = 0.0
                metrics["auroc_weighted"] = 0.0
        
        return metrics
    
    def _calculate_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate calibration metrics."""
        metrics = {}
        
        if self.num_classes == 2:
            if y_prob.ndim > 1:
                y_prob = y_prob[:, 1]
            
            # Brier Score
            try:
                metrics["brier_score"] = brier_score_loss(y_true, y_prob)
            except ValueError:
                metrics["brier_score"] = 0.0
            
            # Log Loss
            try:
                metrics["log_loss"] = log_loss(y_true, y_prob)
            except ValueError:
                metrics["log_loss"] = 0.0
            
            # Expected Calibration Error (ECE)
            try:
                metrics["ece"] = self._calculate_ece(y_true, y_prob)
            except ValueError:
                metrics["ece"] = 0.0
        
        return metrics
    
    def _calculate_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def plot_confusion_matrix(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix."""
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_prob: Union[np.ndarray, torch.Tensor],
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curve."""
        y_true = self._to_numpy(y_true)
        y_prob = self._to_numpy(y_prob)
        
        if self.num_classes == 2:
            if y_prob.ndim > 1:
                y_prob = y_prob[:, 1]
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_precision_recall_curve(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_prob: Union[np.ndarray, torch.Tensor],
        save_path: Optional[str] = None
    ) -> None:
        """Plot Precision-Recall curve."""
        y_true = self._to_numpy(y_true)
        y_prob = self._to_numpy(y_prob)
        
        if self.num_classes == 2:
            if y_prob.ndim > 1:
                y_prob = y_prob[:, 1]
            
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            auc = average_precision_score(y_true, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'PR Curve (AUC = {auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_calibration_curve(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_prob: Union[np.ndarray, torch.Tensor],
        save_path: Optional[str] = None
    ) -> None:
        """Plot calibration curve."""
        y_true = self._to_numpy(y_true)
        y_prob = self._to_numpy(y_prob)
        
        if self.num_classes == 2:
            if y_prob.ndim > 1:
                y_prob = y_prob[:, 1]
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            
            plt.figure(figsize=(8, 6))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Curve')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()


class ClinicalReport:
    """Generate clinical evaluation report."""
    
    def __init__(self, metrics: MedicalMetrics):
        """Initialize clinical report generator.
        
        Args:
            metrics: MedicalMetrics instance
        """
        self.metrics = metrics
    
    def generate_report(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
        model_name: str = "Model"
    ) -> str:
        """Generate comprehensive clinical report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            Formatted clinical report
        """
        # Calculate metrics
        metrics_dict = self.metrics.calculate_metrics(y_true, y_pred, y_prob)
        
        # Generate report
        report = f"""
CLINICAL EVALUATION REPORT
==========================

Model: {model_name}
Evaluation Date: {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CLASSIFICATION PERFORMANCE
--------------------------
"""
        
        if self.metrics.num_classes == 2:
            report += f"""
Accuracy:           {metrics_dict.get('accuracy', 0):.3f}
Sensitivity (TPR):  {metrics_dict.get('sensitivity', 0):.3f}
Specificity (TNR):  {metrics_dict.get('specificity', 0):.3f}
Precision (PPV):    {metrics_dict.get('precision', 0):.3f}
Negative Pred Value: {metrics_dict.get('npv', 0):.3f}
F1-Score:           {metrics_dict.get('f1_score', 0):.3f}
False Positive Rate: {metrics_dict.get('fpr', 0):.3f}
False Negative Rate: {metrics_dict.get('fnr', 0):.3f}
"""
        
        if y_prob is not None:
            report += f"""
PROBABILITY-BASED METRICS
-------------------------
AUROC:              {metrics_dict.get('auroc', 0):.3f}
AUPRC:              {metrics_dict.get('auprc', 0):.3f}

CALIBRATION METRICS
-------------------
Brier Score:        {metrics_dict.get('brier_score', 0):.3f}
Log Loss:           {metrics_dict.get('log_loss', 0):.3f}
Expected Cal Error: {metrics_dict.get('ece', 0):.3f}
"""
        
        report += f"""
CLINICAL INTERPRETATION
------------------------
"""
        
        # Add clinical interpretation
        if self.metrics.num_classes == 2:
            sensitivity = metrics_dict.get('sensitivity', 0)
            specificity = metrics_dict.get('specificity', 0)
            
            if sensitivity >= 0.9:
                report += "• High sensitivity: Good at detecting positive cases\n"
            elif sensitivity >= 0.7:
                report += "• Moderate sensitivity: Reasonable detection rate\n"
            else:
                report += "• Low sensitivity: May miss positive cases\n"
            
            if specificity >= 0.9:
                report += "• High specificity: Good at ruling out negative cases\n"
            elif specificity >= 0.7:
                report += "• Moderate specificity: Reasonable exclusion rate\n"
            else:
                report += "• Low specificity: May have false positives\n"
        
        report += f"""
DISCLAIMER
----------
This evaluation is for research purposes only. 
Not intended for clinical diagnosis or treatment decisions.
Always consult qualified healthcare professionals.
"""
        
        return report
