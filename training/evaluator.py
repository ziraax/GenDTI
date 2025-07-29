"""
DTI Model Evaluator with comprehensive evaluation metrics and analysis.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dti_model import DTIModel


class DTIEvaluator:
    """Comprehensive evaluator for DTI models."""
    
    def __init__(self, model: DTIModel, device: torch.device):
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, dataloader: DataLoader, criterion: Optional[nn.Module] = None) -> Dict[str, float]:
        """
        Evaluate model on given dataloader.
        
        Args:
            dataloader: DataLoader for evaluation
            criterion: Loss function (optional)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for drug_emb, prot_emb, labels in tqdm(dataloader, desc="Evaluating", leave=False):
                drug_emb = drug_emb.to(self.device)
                prot_emb = prot_emb.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(drug_emb, prot_emb)
                
                # Calculate loss if criterion provided
                if criterion:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * labels.size(0)
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                labels_np = labels.cpu().numpy()
                
                all_probabilities.extend(probabilities)
                all_predictions.extend(predictions)
                all_labels.extend(labels_np)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_probabilities, all_labels)
        
        # Add average loss if criterion was provided
        if criterion:
            metrics['loss'] = total_loss / len(dataloader.dataset)
        
        return metrics
    
    def _calculate_metrics(self, predictions: np.ndarray, probabilities: np.ndarray, 
                          labels: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['precision'] = precision_score(labels, predictions, average='binary', zero_division=0)
        metrics['recall'] = recall_score(labels, predictions, average='binary', zero_division=0)
        metrics['f1'] = f1_score(labels, predictions, average='binary', zero_division=0)
        metrics['specificity'] = self._calculate_specificity(labels, predictions)
        
        # Probability-based metrics
        try:
            metrics['roc_auc'] = roc_auc_score(labels, probabilities)
        except ValueError:
            metrics['roc_auc'] = 0.0
            
        try:
            metrics['pr_auc'] = average_precision_score(labels, probabilities)
        except ValueError:
            metrics['pr_auc'] = 0.0
        
        # Matthews correlation coefficient
        metrics['mcc'] = matthews_corrcoef(labels, predictions)
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = self._calculate_balanced_accuracy(labels, predictions)
        
        return metrics
    
    def _calculate_specificity(self, labels: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_balanced_accuracy(self, labels: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate balanced accuracy."""
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return (sensitivity + specificity) / 2
    
    def detailed_evaluation(self, dataloader: DataLoader, save_dir: str = None) -> Dict:
        """
        Perform detailed evaluation with visualizations and analysis.
        
        Args:
            dataloader: DataLoader for evaluation
            save_dir: Directory to save results and plots
            
        Returns:
            Dictionary containing detailed evaluation results
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for drug_emb, prot_emb, labels in tqdm(dataloader, desc="Detailed Evaluation", leave=False):
                drug_emb = drug_emb.to(self.device)
                prot_emb = prot_emb.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(drug_emb, prot_emb)
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                labels_np = labels.cpu().numpy()
                
                all_probabilities.extend(probabilities)
                all_predictions.extend(predictions)
                all_labels.extend(labels_np)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_probabilities, all_labels)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Generate classification report
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        # Prepare results
        results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': all_predictions.tolist(),
            'probabilities': all_probabilities.tolist(),
            'labels': all_labels.tolist()
        }
        
        # Generate and save plots if save_dir provided
        if save_dir:
            self._generate_evaluation_plots(
                all_labels, all_predictions, all_probabilities, cm, save_dir
            )
            
            # Save results to JSON
            results_path = f"{save_dir}/evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Detailed evaluation results saved to {save_dir}")
        
        return results
    
    def _generate_evaluation_plots(self, labels: np.ndarray, predictions: np.ndarray,
                                  probabilities: np.ndarray, cm: np.ndarray, save_dir: str):
        """Generate evaluation plots."""
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Binding', 'Binding'],
                   yticklabels=['Non-Binding', 'Binding'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = roc_auc_score(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        pr_auc = average_precision_score(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Probability Distribution
        plt.figure(figsize=(10, 6))
        binding_probs = probabilities[labels == 1]
        non_binding_probs = probabilities[labels == 0]
        
        plt.hist(non_binding_probs, bins=50, alpha=0.7, label='Non-Binding', density=True)
        plt.hist(binding_probs, bins=50, alpha=0.7, label='Binding', density=True)
        plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Distribution of Predicted Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/probability_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def threshold_analysis(self, dataloader: DataLoader, thresholds: List[float] = None) -> pd.DataFrame:
        """
        Analyze model performance across different decision thresholds.
        
        Args:
            dataloader: DataLoader for evaluation
            thresholds: List of thresholds to evaluate
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        self.model.eval()
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for drug_emb, prot_emb, labels in tqdm(dataloader, desc="Threshold Analysis", leave=False):
                drug_emb = drug_emb.to(self.device)
                prot_emb = prot_emb.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(drug_emb, prot_emb)
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_probabilities.extend(probabilities)
                all_labels.extend(labels_np)
        
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Evaluate each threshold
        results = []
        for threshold in thresholds:
            predictions = (all_probabilities > threshold).astype(int)
            metrics = self._calculate_metrics(predictions, all_probabilities, all_labels)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def print_evaluation_summary(self, metrics: Dict[str, float]):
        """Print a formatted evaluation summary."""
        print("\n" + "="*50)
        print("           EVALUATION SUMMARY")
        print("="*50)
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Precision:         {metrics['precision']:.4f}")
        print(f"Recall:            {metrics['recall']:.4f}")
        print(f"F1-Score:          {metrics['f1']:.4f}")
        print(f"Specificity:       {metrics['specificity']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"ROC AUC:           {metrics['roc_auc']:.4f}")
        print(f"PR AUC:            {metrics['pr_auc']:.4f}")
        print(f"MCC:               {metrics['mcc']:.4f}")
        if 'loss' in metrics:
            print(f"Loss:              {metrics['loss']:.4f}")
        print("="*50)
