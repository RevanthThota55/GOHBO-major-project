"""
Evaluation module for medical image classification models.

Provides comprehensive evaluation metrics and visualizations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import json


class Evaluator:
    """
    Evaluator class for comprehensive model evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: List[str],
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model to evaluate
            test_loader: Test data loader
            class_names: List of class names
            config: Configuration dictionary
            device: Device to use for evaluation
        """
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Results storage
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        self.misclassified = []

        # Output directory
        self.output_dir = Path(config['evaluation']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """
        Perform complete evaluation of the model.

        Returns:
            Dictionary containing all evaluation metrics
        """
        print("\nEvaluating model on test set...")

        # Collect predictions
        self._collect_predictions()

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Generate visualizations if enabled
        if self.config['evaluation']['visualization']['plot_confusion_matrix']:
            self._plot_confusion_matrix()

        if self.config['evaluation']['visualization']['plot_roc_curves']:
            self._plot_roc_curves()

        if self.config['evaluation']['visualization']['plot_class_distribution']:
            self._plot_class_distribution()

        if self.config['evaluation']['visualization']['save_misclassified']:
            self._save_misclassified_samples()

        # Save results
        self._save_results(metrics)

        return metrics

    def _collect_predictions(self):
        """Collect all predictions from the test set."""
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        self.misclassified = []

        pbar = tqdm(self.test_loader, desc="Collecting predictions")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Get model outputs
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Store results
            self.predictions.extend(preds.cpu().numpy())
            self.true_labels.extend(labels.cpu().numpy())
            self.probabilities.extend(probs.cpu().numpy())

            # Track misclassified samples
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    self.misclassified.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'true_label': labels[i].item(),
                        'pred_label': preds[i].item(),
                        'confidence': probs[i, preds[i]].item(),
                        'true_class': self.class_names[labels[i].item()],
                        'pred_class': self.class_names[preds[i].item()]
                    })

        self.predictions = np.array(self.predictions)
        self.true_labels = np.array(self.true_labels)
        self.probabilities = np.array(self.probabilities)

    def _calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive evaluation metrics.

        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(self.true_labels, self.predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels, self.predictions, average='weighted'
        )

        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
            self.true_labels, self.predictions, average=None
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(self.true_labels, self.predictions)

        # ROC AUC (for binary or one-vs-rest for multiclass)
        if len(self.class_names) == 2:
            # Binary classification
            roc_auc = roc_auc_score(self.true_labels, self.probabilities[:, 1])
        else:
            # Multiclass - one-vs-rest
            roc_auc = roc_auc_score(
                self.true_labels,
                self.probabilities,
                multi_class='ovr',
                average='weighted'
            )

        # Classification report
        class_report = classification_report(
            self.true_labels,
            self.predictions,
            target_names=self.class_names,
            output_dict=True
        )

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_metrics': {
                class_name: {
                    'precision': per_class_precision[i],
                    'recall': per_class_recall[i],
                    'f1_score': per_class_f1[i],
                    'support': int(per_class_support[i])
                }
                for i, class_name in enumerate(self.class_names)
            },
            'classification_report': class_report,
            'num_misclassified': len(self.misclassified),
            'total_samples': len(self.true_labels)
        }

        # Print summary
        self._print_summary(metrics)

        return metrics

    def _print_summary(self, metrics: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        print(f"\nPer-Class Metrics:")
        for class_name in self.class_names:
            class_metrics = metrics['per_class_metrics'][class_name]
            print(f"\n  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall:    {class_metrics['recall']:.4f}")
            print(f"    F1-Score:  {class_metrics['f1_score']:.4f}")
            print(f"    Support:   {class_metrics['support']}")

        print(f"\nMisclassified: {metrics['num_misclassified']}/{metrics['total_samples']}")
        print("=" * 60)

    def _plot_confusion_matrix(self):
        """Plot and save confusion matrix."""
        conf_matrix = confusion_matrix(self.true_labels, self.predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        # Save figure
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Also save normalized confusion matrix
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix_norm,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        plt.savefig(self.output_dir / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Confusion matrices saved to {self.output_dir}")

    def _plot_roc_curves(self):
        """Plot and save ROC curves."""
        if len(self.class_names) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(self.true_labels, self.probabilities[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(self.output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()

        else:
            # Multiclass - plot one-vs-rest ROC curves
            plt.figure(figsize=(10, 8))

            for i, class_name in enumerate(self.class_names):
                # Create binary labels for this class
                binary_labels = (self.true_labels == i).astype(int)
                class_probs = self.probabilities[:, i]

                fpr, tpr, _ = roc_curve(binary_labels, class_probs)
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves (One-vs-Rest)')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(self.output_dir / 'roc_curves_multiclass.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"[OK] ROC curves saved to {self.output_dir}")

    def _plot_class_distribution(self):
        """Plot class distribution in predictions vs true labels."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # True label distribution
        true_counts = np.bincount(self.true_labels)
        axes[0].bar(self.class_names, true_counts, color='blue', alpha=0.7)
        axes[0].set_title('True Label Distribution')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)

        # Predicted label distribution
        pred_counts = np.bincount(self.predictions)
        axes[1].bar(self.class_names, pred_counts, color='green', alpha=0.7)
        axes[1].set_title('Predicted Label Distribution')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Class distribution plot saved to {self.output_dir}")

    def _save_misclassified_samples(self):
        """Save information about misclassified samples."""
        num_to_save = min(
            len(self.misclassified),
            self.config['evaluation']['visualization']['num_misclassified_to_save']
        )

        # Sort by confidence (ascending) to get the worst misclassifications
        sorted_misclassified = sorted(
            self.misclassified,
            key=lambda x: x['confidence']
        )[:num_to_save]

        # Save as JSON
        misclassified_path = self.output_dir / 'misclassified_samples.json'
        with open(misclassified_path, 'w') as f:
            json.dump(sorted_misclassified, f, indent=2)

        print(f"[OK] Misclassified samples saved to {misclassified_path}")

    def _save_results(self, metrics: Dict):
        """
        Save evaluation results to files.

        Args:
            metrics: Dictionary of evaluation metrics
        """
        # Save metrics as JSON
        metrics_path = self.output_dir / 'evaluation_metrics.json'

        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                metrics_serializable[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                metrics_serializable[key] = int(value)
            else:
                metrics_serializable[key] = value

        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

        # Save classification report as text
        report_path = self.output_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(classification_report(
                self.true_labels,
                self.predictions,
                target_names=self.class_names
            ))

        print(f"[OK] Evaluation results saved to {self.output_dir}")