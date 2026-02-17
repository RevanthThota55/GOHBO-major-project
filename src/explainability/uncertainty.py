"""
Monte Carlo Dropout for Uncertainty Quantification in Medical Image Classification.

Uses dropout at inference time to estimate model uncertainty by performing
multiple stochastic forward passes and analyzing the variance in predictions.

Reference:
    Gal & Ghahramani. "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


class MCDropoutPredictor:
    """
    Monte Carlo Dropout predictor for uncertainty estimation.

    Performs multiple forward passes with dropout enabled to estimate
    predictive uncertainty. Higher variance indicates higher uncertainty.

    Attributes:
        model: Neural network with dropout layers
        num_passes: Number of stochastic forward passes
        device: Device to run computations on

    Example:
        >>> model = MedicalResNet18(num_classes=4, enable_mc_dropout=True)
        >>> mc_predictor = MCDropoutPredictor(model, num_passes=20)
        >>> mean_pred, uncertainty = mc_predictor.predict_with_uncertainty(image)
    """

    def __init__(
        self,
        model: nn.Module,
        num_passes: int = 20,
        device: str = 'cuda'
    ):
        """
        Initialize MC Dropout predictor.

        Args:
            model: Model with dropout layers (should support mc_dropout mode)
            num_passes: Number of forward passes for uncertainty estimation
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model
        self.num_passes = num_passes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Enable MC dropout mode if model supports it
        if hasattr(self.model, 'set_mc_dropout_mode'):
            self.model.set_mc_dropout_mode(True)
        else:
            print("Warning: Model does not have set_mc_dropout_mode method. "
                  "Uncertainty estimates may not be accurate.")

    def predict_with_uncertainty(
        self,
        input_image: torch.Tensor,
        return_all_predictions: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict with uncertainty estimation using MC Dropout.

        Args:
            input_image: Input tensor of shape (1, C, H, W) or (C, H, W)
            return_all_predictions: If True, return all individual predictions

        Returns:
            Tuple of (mean_prediction, uncertainty_dict)
            - mean_prediction: Average prediction across all passes
            - uncertainty_dict: Dictionary containing:
                - 'std': Standard deviation of predictions
                - 'entropy': Predictive entropy
                - 'confidence': Confidence score (1 - entropy/log(num_classes))
                - 'all_predictions': All predictions (if requested)

        Example:
            >>> image = torch.randn(1, 3, 224, 224)
            >>> mean_pred, uncertainty = mc_predictor.predict_with_uncertainty(image)
            >>> print(f"Confidence: {uncertainty['confidence']:.2%}")
        """
        # Ensure input is 4D
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)

        input_image = input_image.to(self.device)

        # Store predictions from all passes
        all_predictions = []

        # Perform multiple forward passes
        for _ in range(self.num_passes):
            with torch.no_grad():
                output = self.model(input_image)
                probabilities = F.softmax(output, dim=1)
                all_predictions.append(probabilities.cpu().numpy())

        # Convert to numpy array (num_passes, 1, num_classes)
        all_predictions = np.array(all_predictions).squeeze(1)  # (num_passes, num_classes)

        # Calculate mean prediction
        mean_prediction = np.mean(all_predictions, axis=0)  # (num_classes,)

        # Calculate standard deviation
        std_prediction = np.std(all_predictions, axis=0)  # (num_classes,)

        # Calculate predictive entropy
        entropy = -np.sum(mean_prediction * np.log(mean_prediction + 1e-10))

        # Calculate confidence (normalized entropy)
        num_classes = mean_prediction.shape[0]
        max_entropy = np.log(num_classes)
        confidence = 1.0 - (entropy / max_entropy)

        # Prepare uncertainty dictionary
        uncertainty_dict = {
            'std': std_prediction,
            'mean_std': float(np.mean(std_prediction)),
            'max_std': float(np.max(std_prediction)),
            'entropy': float(entropy),
            'confidence': float(confidence),
            'predicted_class': int(np.argmax(mean_prediction)),
            'predicted_probability': float(mean_prediction[np.argmax(mean_prediction)])
        }

        if return_all_predictions:
            uncertainty_dict['all_predictions'] = all_predictions

        return mean_prediction, uncertainty_dict

    def predict_batch_with_uncertainty(
        self,
        images: torch.Tensor,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Predict with uncertainty for a batch of images.

        Args:
            images: Batch of images (B, C, H, W)
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (mean_predictions, uncertainty_dicts)
        """
        batch_size = images.size(0)
        mean_predictions = []
        uncertainty_dicts = []

        iterator = tqdm(range(batch_size), desc="MC Dropout") if show_progress else range(batch_size)

        for i in iterator:
            mean_pred, uncertainty = self.predict_with_uncertainty(images[i:i+1])
            mean_predictions.append(mean_pred)
            uncertainty_dicts.append(uncertainty)

        return np.array(mean_predictions), uncertainty_dicts

    def calculate_mutual_information(
        self,
        all_predictions: np.ndarray
    ) -> float:
        """
        Calculate mutual information between predictions and model parameters.

        This measures the information gain from knowing the model parameters,
        which is a measure of model uncertainty.

        Args:
            all_predictions: Array of shape (num_passes, num_classes)

        Returns:
            Mutual information score
        """
        # Average prediction
        mean_pred = np.mean(all_predictions, axis=0)

        # Entropy of average prediction
        entropy_mean = -np.sum(mean_pred * np.log(mean_pred + 1e-10))

        # Average entropy
        individual_entropies = -np.sum(all_predictions * np.log(all_predictions + 1e-10), axis=1)
        mean_entropy = np.mean(individual_entropies)

        # Mutual information
        mutual_info = entropy_mean - mean_entropy

        return float(mutual_info)


class UncertaintyAnalyzer:
    """
    Analyzer for model uncertainty across a dataset.

    Provides tools to analyze uncertainty patterns, identify uncertain cases,
    and generate comprehensive uncertainty reports.

    Example:
        >>> analyzer = UncertaintyAnalyzer(mc_predictor, class_names)
        >>> results = analyzer.analyze_dataset(test_loader)
        >>> analyzer.plot_uncertainty_distribution(results)
    """

    def __init__(
        self,
        mc_predictor: MCDropoutPredictor,
        class_names: List[str],
        confidence_threshold: float = 0.85
    ):
        """
        Initialize uncertainty analyzer.

        Args:
            mc_predictor: MC Dropout predictor
            class_names: List of class names
            confidence_threshold: Threshold below which predictions are flagged as uncertain
        """
        self.mc_predictor = mc_predictor
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold

    def analyze_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze uncertainty across an entire dataset.

        Args:
            dataloader: DataLoader for the dataset
            max_samples: Maximum number of samples to analyze (None = all)

        Returns:
            Dictionary containing:
                - 'predictions': List of predictions
                - 'uncertainties': List of uncertainty dictionaries
                - 'true_labels': List of ground truth labels
                - 'uncertain_indices': Indices of uncertain predictions
                - 'statistics': Summary statistics

        Example:
            >>> results = analyzer.analyze_dataset(test_loader, max_samples=100)
            >>> print(f"Uncertain cases: {len(results['uncertain_indices'])}")
        """
        all_predictions = []
        all_uncertainties = []
        all_true_labels = []
        uncertain_indices = []

        num_processed = 0

        for images, labels in tqdm(dataloader, desc="Analyzing uncertainty"):
            if max_samples and num_processed >= max_samples:
                break

            # Process batch
            mean_preds, uncertainties = self.mc_predictor.predict_batch_with_uncertainty(
                images, show_progress=False
            )

            # Store results
            all_predictions.extend(mean_preds)
            all_uncertainties.extend(uncertainties)
            all_true_labels.extend(labels.numpy())

            # Flag uncertain predictions
            for i, uncertainty in enumerate(uncertainties):
                if uncertainty['confidence'] < self.confidence_threshold:
                    uncertain_indices.append(num_processed + i)

            num_processed += images.size(0)

        # Calculate statistics
        confidences = [u['confidence'] for u in all_uncertainties]
        entropies = [u['entropy'] for u in all_uncertainties]
        mean_stds = [u['mean_std'] for u in all_uncertainties]

        # Check correctness
        predicted_classes = [u['predicted_class'] for u in all_uncertainties]
        correct = [pred == true for pred, true in zip(predicted_classes, all_true_labels)]

        statistics = {
            'total_samples': num_processed,
            'num_uncertain': len(uncertain_indices),
            'uncertainty_rate': len(uncertain_indices) / num_processed,
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'mean_entropy': float(np.mean(entropies)),
            'mean_std': float(np.mean(mean_stds)),
            'accuracy': sum(correct) / len(correct),
            'confidence_threshold': self.confidence_threshold,
            'confidence_when_correct': float(np.mean([confidences[i] for i in range(len(correct)) if correct[i]])),
            'confidence_when_incorrect': float(np.mean([confidences[i] for i in range(len(correct)) if not correct[i]])) if sum([not c for c in correct]) > 0 else 0.0
        }

        results = {
            'predictions': all_predictions,
            'uncertainties': all_uncertainties,
            'true_labels': all_true_labels,
            'uncertain_indices': uncertain_indices,
            'statistics': statistics
        }

        return results

    def plot_uncertainty_distribution(
        self,
        results: Dict[str, Any],
        save_path: Optional[Path] = None
    ):
        """
        Plot distribution of uncertainty metrics.

        Args:
            results: Results from analyze_dataset()
            save_path: Optional path to save the plot
        """
        confidences = [u['confidence'] for u in results['uncertainties']]
        entropies = [u['entropy'] for u in results['uncertainties']]

        # Check correctness
        predicted_classes = [u['predicted_class'] for u in results['uncertainties']]
        correct = [pred == true for pred, true in zip(predicted_classes, results['true_labels'])]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Confidence distribution
        axes[0, 0].hist(confidences, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(self.confidence_threshold, color='red', linestyle='--',
                          label=f'Threshold: {self.confidence_threshold}')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Confidence Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Entropy distribution
        axes[0, 1].hist(entropies, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('Predictive Entropy')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Predictive Entropy Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Confidence vs Correctness
        correct_conf = [confidences[i] for i in range(len(correct)) if correct[i]]
        incorrect_conf = [confidences[i] for i in range(len(correct)) if not correct[i]]

        axes[1, 0].hist([correct_conf, incorrect_conf], bins=20, alpha=0.7,
                       color=['green', 'red'], label=['Correct', 'Incorrect'],
                       edgecolor='black')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Confidence by Correctness')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Per-class uncertainty
        class_uncertainties = {name: [] for name in self.class_names}
        for i, true_label in enumerate(results['true_labels']):
            class_name = self.class_names[true_label]
            class_uncertainties[class_name].append(1 - confidences[i])  # Use uncertainty instead of confidence

        box_data = [class_uncertainties[name] for name in self.class_names]
        axes[1, 1].boxplot(box_data, labels=self.class_names)
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Uncertainty (1 - Confidence)')
        axes[1, 1].set_title('Uncertainty by Class')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Uncertainty Analysis (Threshold: {self.confidence_threshold})', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Uncertainty distribution plot saved to {save_path}")

        plt.show()
        plt.close()

    def generate_uncertainty_report(
        self,
        results: Dict[str, Any],
        save_path: Optional[Path] = None
    ) -> str:
        """
        Generate a comprehensive uncertainty report.

        Args:
            results: Results from analyze_dataset()
            save_path: Optional path to save the report

        Returns:
            Report as a string
        """
        stats = results['statistics']

        report = f"""
================================================================================
UNCERTAINTY QUANTIFICATION REPORT
================================================================================

Dataset Statistics:
-------------------
Total Samples Analyzed: {stats['total_samples']}
Uncertain Predictions: {stats['num_uncertain']} ({stats['uncertainty_rate']:.1%})
Confidence Threshold: {stats['confidence_threshold']:.2f}

Overall Performance:
--------------------
Accuracy: {stats['accuracy']:.2%}
Mean Confidence: {stats['mean_confidence']:.4f} ± {stats['std_confidence']:.4f}
Mean Entropy: {stats['mean_entropy']:.4f}
Mean Std Dev: {stats['mean_std']:.4f}

Confidence Analysis:
--------------------
Confidence (Correct Predictions): {stats['confidence_when_correct']:.4f}
Confidence (Incorrect Predictions): {stats['confidence_when_incorrect']:.4f}
Confidence Gap: {stats['confidence_when_correct'] - stats['confidence_when_incorrect']:.4f}

Interpretation:
---------------
"""

        if stats['uncertainty_rate'] < 0.1:
            report += "✓ Low uncertainty rate - model is generally confident\n"
        elif stats['uncertainty_rate'] < 0.3:
            report += "⚠ Moderate uncertainty rate - review uncertain cases\n"
        else:
            report += "✗ High uncertainty rate - consider model retraining or more data\n"

        if stats['confidence_when_correct'] - stats['confidence_when_incorrect'] > 0.1:
            report += "✓ Model is well-calibrated - higher confidence for correct predictions\n"
        else:
            report += "⚠ Model calibration may need improvement\n"

        report += """
Recommendations:
----------------
"""
        if stats['uncertainty_rate'] > 0.2:
            report += f"- Review {stats['num_uncertain']} uncertain cases manually\n"
            report += "- Consider ensemble methods or additional training data\n"

        if stats['confidence_when_correct'] - stats['confidence_when_incorrect'] < 0.05:
            report += "- Apply confidence calibration techniques (temperature scaling)\n"

        report += """
================================================================================
"""

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"✓ Uncertainty report saved to {save_path}")

        return report

    def save_uncertain_cases(
        self,
        results: Dict[str, Any],
        save_path: Path
    ):
        """
        Save information about uncertain cases to JSON file.

        Args:
            results: Results from analyze_dataset()
            save_path: Path to save the JSON file
        """
        uncertain_cases = []

        for idx in results['uncertain_indices']:
            uncertainty = results['uncertainties'][idx]
            true_label = results['true_labels'][idx]

            case = {
                'index': int(idx),
                'predicted_class': self.class_names[uncertainty['predicted_class']],
                'predicted_class_idx': int(uncertainty['predicted_class']),
                'true_class': self.class_names[true_label],
                'true_class_idx': int(true_label),
                'confidence': float(uncertainty['confidence']),
                'entropy': float(uncertainty['entropy']),
                'mean_std': float(uncertainty['mean_std']),
                'is_correct': uncertainty['predicted_class'] == true_label
            }

            uncertain_cases.append(case)

        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(uncertain_cases, f, indent=2)

        print(f"✓ {len(uncertain_cases)} uncertain cases saved to {save_path}")