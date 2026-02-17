"""
Metrics utilities for medical image classification.

Provides functions for calculating and analyzing various performance metrics.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score
)
from typing import Dict, List, Tuple, Optional, Any
import json


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        class_names: List of class names (optional)

    Returns:
        Dictionary containing various metrics
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None)

    if class_names:
        metrics['per_class'] = {
            class_names[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            }
            for i in range(len(class_names))
        }

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # Cohen's Kappa (inter-rater agreement)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    # Matthews Correlation Coefficient
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)

    # ROC AUC if probabilities are provided
    if y_prob is not None:
        num_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 2

        if num_classes == 2:
            # Binary classification
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob)
        else:
            # Multiclass
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = None

    return metrics


def calculate_specificity_sensitivity(
    confusion_mat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate specificity and sensitivity from confusion matrix.

    Args:
        confusion_mat: Confusion matrix

    Returns:
        Tuple of (specificity, sensitivity) arrays
    """
    # Sensitivity (True Positive Rate) = TP / (TP + FN)
    sensitivity = np.diag(confusion_mat) / np.sum(confusion_mat, axis=1)

    # Specificity = TN / (TN + FP)
    specificity = []
    for i in range(len(confusion_mat)):
        # True negatives: sum of all elements except row i and column i
        tn = np.sum(confusion_mat) - np.sum(confusion_mat[i, :]) - \
             np.sum(confusion_mat[:, i]) + confusion_mat[i, i]
        # False positives: sum of column i except diagonal
        fp = np.sum(confusion_mat[:, i]) - confusion_mat[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    return np.array(specificity), sensitivity


def calculate_confidence_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Calculate confidence-related metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities

    Returns:
        Dictionary of confidence metrics
    """
    metrics = {}

    # Get maximum probability for each prediction
    max_probs = np.max(y_prob, axis=1)

    # Average confidence
    metrics['mean_confidence'] = np.mean(max_probs)
    metrics['std_confidence'] = np.std(max_probs)

    # Confidence for correct vs incorrect predictions
    correct_mask = y_true == y_pred
    metrics['mean_confidence_correct'] = np.mean(max_probs[correct_mask]) if np.any(correct_mask) else 0
    metrics['mean_confidence_incorrect'] = np.mean(max_probs[~correct_mask]) if np.any(~correct_mask) else 0

    # Confidence gap
    metrics['confidence_gap'] = metrics['mean_confidence_correct'] - metrics['mean_confidence_incorrect']

    # Calibration error (simplified)
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0  # Expected Calibration Error
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct_mask[in_bin])
            avg_confidence_in_bin = np.mean(max_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    metrics['expected_calibration_error'] = ece

    return metrics


def calculate_medical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class: int = 1
) -> Dict[str, float]:
    """
    Calculate medical-specific metrics for binary classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        positive_class: Label for positive class (disease)

    Returns:
        Dictionary of medical metrics
    """
    metrics = {}

    # Convert to binary if needed
    if len(np.unique(y_true)) > 2:
        y_true_binary = (y_true == positive_class).astype(int)
        y_pred_binary = (y_pred == positive_class).astype(int)
    else:
        y_true_binary = y_true
        y_pred_binary = y_pred

    # Calculate confusion matrix elements
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))

    # Sensitivity (Recall, True Positive Rate)
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Specificity (True Negative Rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Positive Predictive Value (Precision)
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Negative Predictive Value
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Diagnostic Odds Ratio
    if fp > 0 and fn > 0:
        metrics['diagnostic_odds_ratio'] = (tp * tn) / (fp * fn)
    else:
        metrics['diagnostic_odds_ratio'] = float('inf') if fp == 0 and fn == 0 else 0

    # Youden's J statistic
    metrics['youdens_j'] = metrics['sensitivity'] + metrics['specificity'] - 1

    # Number Needed to Diagnose (NND)
    if metrics['youdens_j'] > 0:
        metrics['nnd'] = 1 / metrics['youdens_j']
    else:
        metrics['nnd'] = float('inf')

    return metrics


def calculate_multiclass_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str]
) -> Dict[str, float]:
    """
    Calculate AUC for each class in multiclass classification.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: List of class names

    Returns:
        Dictionary of AUC scores per class
    """
    auc_scores = {}

    for i, class_name in enumerate(class_names):
        # Create binary labels for this class
        y_true_binary = (y_true == i).astype(int)
        y_score = y_prob[:, i]

        try:
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            auc_score = auc(fpr, tpr)
            auc_scores[class_name] = auc_score
        except:
            auc_scores[class_name] = None

    # Calculate macro and weighted average
    valid_scores = [v for v in auc_scores.values() if v is not None]
    if valid_scores:
        auc_scores['macro_avg'] = np.mean(valid_scores)

        # Weighted average (by support)
        weights = np.bincount(y_true, minlength=len(class_names))
        weighted_scores = [auc_scores[class_names[i]] * weights[i]
                          for i in range(len(class_names))
                          if auc_scores[class_names[i]] is not None]
        total_weight = sum(weights[i] for i in range(len(class_names))
                          if auc_scores[class_names[i]] is not None)
        auc_scores['weighted_avg'] = sum(weighted_scores) / total_weight if total_weight > 0 else 0

    return auc_scores


def save_metrics_report(
    metrics: Dict[str, Any],
    save_path: str,
    format: str = 'json'
):
    """
    Save metrics report to file.

    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the report
        format: Format to save ('json' or 'txt')
    """
    if format == 'json':
        # Convert numpy types to Python types for JSON serialization
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

        with open(save_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

    elif format == 'txt':
        with open(save_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("METRICS REPORT\n")
            f.write("=" * 60 + "\n\n")

            for key, value in metrics.items():
                if isinstance(value, dict):
                    f.write(f"\n{key.upper()}:\n")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, float):
                            f.write(f"  {sub_key}: {sub_value:.4f}\n")
                        else:
                            f.write(f"  {sub_key}: {sub_value}\n")
                elif isinstance(value, (list, np.ndarray)):
                    f.write(f"\n{key.upper()}:\n")
                    f.write(f"  {value}\n")
                elif isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

    print(f"✓ Metrics report saved to {save_path}")