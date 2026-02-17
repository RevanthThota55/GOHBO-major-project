"""
Training module for medical image classification with ResNet-18.

Handles the training loop, validation, checkpointing, and TensorBoard logging.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Tuple, Any
import time
import json
from sklearn.metrics import precision_recall_fscore_support


class Trainer:
    """
    Trainer class for medical image classification models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Initialize TensorBoard writer if enabled
        self.writer = None
        if config['logging']['tensorboard']['enabled']:
            log_dir = Path(config['logging']['tensorboard']['log_dir']) / \
                     f"{config['experiment']['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir)

        # Training history
        # D-7: Track accuracy, precision, and recall during training
        # Precision: Of all images predicted as a tumor, how many actually were?
        # Recall: Of all actual tumor images, how many did we find?
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_precision': [],  # Added for D-7
            'train_recall': [],     # Added for D-7
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],    # Added for D-7
            'val_recall': [],       # Added for D-7
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Checkpoint path
        self.checkpoint_dir = Path(config['paths']['models']) / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> optim.Optimizer:
        """
        Create optimizer based on configuration.

        Returns:
            Optimizer instance
        """
        opt_config = self.config['training']['optimizer']
        opt_type = opt_config['type'].lower()

        # Get learning rate (may be optimized by GOHBO)
        learning_rate = self.config.get('optimized_learning_rate',
                                       self.config['training'].get('learning_rate', 1e-3))

        if opt_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=opt_config['betas'],
                eps=opt_config['eps'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                betas=opt_config['betas'],
                eps=opt_config['eps'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler based on configuration.

        Returns:
            Scheduler instance or None
        """
        if 'scheduler' not in self.config['training']:
            return None

        sched_config = self.config['training']['scheduler']
        sched_type = sched_config['type'].lower()

        if sched_type == 'cosineannealinglr':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max'],
                eta_min=sched_config['eta_min']
            )
        elif sched_type == 'steplr':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_type == 'reducelronplateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('factor', 0.1),
                patience=sched_config.get('patience', 10)
            )
        else:
            return None

    def train_epoch(self, epoch: int) -> Tuple[float, float, float, float]:
        """
        Train for one epoch.

        What changed (D-7):
        -------------------
        Now also calculates PRECISION and RECALL, not just accuracy!
        - Precision: How many of our "positive" predictions were correct
        - Recall: How many of the actual positives did we find

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average loss, accuracy, precision, recall)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Store all predictions and labels for precision/recall calculation
        all_predictions = []
        all_labels = []

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')

        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping if enabled
            if 'gradient_clip_value' in self.config['training']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_value']
                )

            # Optimizer step
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and labels for precision/recall
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

            # TensorBoard logging
            if self.writer and batch_idx % self.config['logging']['tensorboard']['log_interval'] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss_Batch', loss.item(), global_step)
                self.writer.add_scalar('Train/Acc_Batch', current_acc, global_step)

        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        # Calculate precision and recall (D-7)
        # precision_recall_fscore_support returns: (precision, recall, f1-score, support)
        # We use 'weighted' average which accounts for class imbalance
        precision, recall, _, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='weighted',  # Weighted by number of samples per class
            zero_division=0  # Return 0 if there are no predictions for a class
        )

        # Convert to percentages for easier reading
        epoch_precision = precision * 100.0
        epoch_recall = recall * 100.0

        return epoch_loss, epoch_acc, epoch_precision, epoch_recall

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float, float, float]:
        """
        Validate the model.

        What changed (D-7):
        -------------------
        Now also calculates PRECISION and RECALL during validation!

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average loss, accuracy, precision, recall)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # Store all predictions and labels for precision/recall calculation
        all_predictions = []
        all_labels = []

        # Progress bar
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')

        for images, labels in pbar:
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            current_loss = running_loss / (len(pbar) + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        # Calculate precision and recall (D-7)
        precision, recall, _, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='weighted',
            zero_division=0
        )

        epoch_precision = precision * 100.0
        epoch_recall = recall * 100.0

        return epoch_loss, epoch_acc, epoch_precision, epoch_recall

    def train(self, num_epochs: int, learning_rate: Optional[float] = None) -> Dict:
        """
        Complete training loop.

        Args:
            num_epochs: Number of epochs to train
            learning_rate: Optional learning rate override (from GOHBO)

        Returns:
            Training history dictionary
        """
        # Override learning rate if provided
        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
            print(f"Using GOHBO-optimized learning rate: {learning_rate:.6f}")

        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc, train_precision, train_recall = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_precision, val_recall = self.validate(epoch)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history (D-7: now includes precision and recall)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_precision'].append(train_precision)
            self.history['train_recall'].append(train_recall)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.history['learning_rates'].append(current_lr)

            # Print epoch summary (D-7: now shows precision and recall)
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                  f"Precision: {train_precision:.2f}%, Recall: {train_recall:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
                  f"Precision: {val_precision:.2f}%, Recall: {val_recall:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")

            # TensorBoard logging (D-7: now logs precision and recall too)
            if self.writer:
                self.writer.add_scalars('Loss', {
                    'train': train_loss,
                    'val': val_loss
                }, epoch)
                self.writer.add_scalars('Accuracy', {
                    'train': train_acc,
                    'val': val_acc
                }, epoch)
                self.writer.add_scalars('Precision', {
                    'train': train_precision,
                    'val': val_precision
                }, epoch)
                self.writer.add_scalars('Recall', {
                    'train': train_recall,
                    'val': val_recall
                }, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Check for improvement
            improved = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                improved = True

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                improved = True

            # Save best model checkpoint if improved
            if improved:
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)
                self.epochs_without_improvement = 0
                print(f"  ✓ Model improved! Saved best checkpoint.")
            else:
                self.epochs_without_improvement += 1

            # D-5: Save periodic checkpoint every 5 epochs (regardless of improvement)
            # This gives us recovery points even if the model doesn't improve
            checkpoint_interval = self.config['training']['checkpoint'].get('save_interval', 5)
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=False)
                print(f"  📦 Periodic checkpoint saved (every {checkpoint_interval} epochs)")

            # D-3: Early stopping if we hit 95% accuracy target
            target_accuracy = self.config['gohbo'].get('target_accuracy', 0.95) * 100
            if val_acc >= target_accuracy:
                print(f"\n{'='*70}")
                print(f"🎉 TARGET ACHIEVED! Validation accuracy {val_acc:.2f}% >= {target_accuracy:.2f}%")
                print(f"Stopping training early at epoch {epoch+1} (no need to continue!)")
                print(f"{'='*70}")
                break

            # D-6: Early stopping if no improvement for patience epochs
            early_stop_config = self.config['training'].get('early_stopping', {})
            if early_stop_config.get('enabled', False):
                patience = early_stop_config['patience']
                if self.epochs_without_improvement >= patience:
                    print(f"\n{'='*70}")
                    print(f"⏸️  Early stopping triggered: No improvement for {patience} epochs")
                    print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                    print(f"Stopped at epoch {epoch+1}/{num_epochs}")
                    print(f"{'='*70}")
                    break

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()

        # Save final model
        self.save_final_model()

        return self.history

    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool = True):
        """
        Save model checkpoint.

        What changed (D-5):
        -------------------
        Now saves TWO types of checkpoints:
        1. Best model checkpoint (when model improves)
        2. Periodic checkpoint every 5 epochs (for recovery)

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            val_acc: Validation accuracy
            is_best: If True, saves as best model; if False, saves as periodic checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history,
            'config': self.config
        }

        if is_best:
            # Save as best model
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
        else:
            # Save as periodic checkpoint (D-5)
            # These are saved with the epoch number so we can go back to any point
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)

    def save_final_model(self):
        """
        Save the final trained model.

        What changed:
        -------------
        Now also saves training history to Phase 1 output directory
        for use in evaluation and reporting.
        """
        final_model = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        final_path = self.checkpoint_dir / 'final_model.pth'
        torch.save(final_model, final_path)

        # Save training history to checkpoints directory
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Also save to Phase1 output directory for easy access by evaluation scripts
        try:
            from pathlib import Path
            phase1_dir = Path(self.config['paths']['results']) / 'phase1'
            phase1_dir.mkdir(parents=True, exist_ok=True)
            phase1_history_path = phase1_dir / 'training_history.json'
            with open(phase1_history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"✓ Training history also saved to {phase1_history_path}")
        except Exception as e:
            # If Phase1 save fails, it's not critical
            pass

        print(f"\n✓ Final model saved to {final_path}")
        print(f"✓ Training history saved to {history_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.history = checkpoint.get('history', self.history)
        self.best_val_acc = checkpoint.get('val_acc', 0.0)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))

        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch'] + 1}")
        print(f"  Val Acc: {checkpoint['val_acc']:.2f}%")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")