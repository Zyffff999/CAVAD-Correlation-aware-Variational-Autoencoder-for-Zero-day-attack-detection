import torch
import numpy as np
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import json
import logging

class EarlyStopping:
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-6,
                 mode: str = 'min',
                 restore_best_weights: bool = True,
                 verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_score = -np.inf
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

    def __call__(self, current_score: float, model: torch.nn.Module, epoch: int = None) -> bool:
        """
        Check if training should stop.

        Args:
            current_score: Current metric value
            model: Model instance
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch if epoch is not None else 0
            self.counter = 0

            if self.restore_best_weights:
                # Deep copy the model state
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if self.verbose:
                print(f"EarlyStopping: Improvement found. Best score: {self.best_score:.6f}")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter} epochs")

            if self.counter >= self.patience:
                self.stopped_epoch = epoch if epoch is not None else 0

                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose:
                        print(f"EarlyStopping: Restoring best weights from epoch {self.best_epoch}")
                    model.load_state_dict(self.best_weights)

                return True

        return False

    def reset(self):
        """Reset the early stopping state."""
        self.best_score = np.inf if self.mode == 'min' else -np.inf
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0


class ModelCheckpoint:
    """
    Callback to save model checkpoints during training.
    """

    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_weights_only: bool = False,
                 verbose: bool = True):
        """
        Args:
            filepath: Path template for saving checkpoints (can include {epoch} placeholder)
            monitor: Metric to monitor
            mode: 'min' or 'max' - whether lower or higher is better
            save_best_only: Save only when monitored metric improves
            save_weights_only: Save only model weights (not full checkpoint)
            verbose: Whether to print messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    def __call__(self,
                 epoch: int,
                 model: torch.nn.Module,
                 metrics: Dict[str, float],
                 optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Save checkpoint if conditions are met.

        Args:
            epoch: Current epoch
            model: Model instance
            metrics: Dictionary of current metrics
            optimizer: Optimizer instance (optional)
        """
        current = metrics.get(self.monitor)
        if current is None:
            logging.warning(f"ModelCheckpoint: Metric '{self.monitor}' not found in metrics")
            return

        save_checkpoint = False

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                if self.verbose:
                    print(f"ModelCheckpoint: {self.monitor} improved from {self.best:.6f} to {current:.6f}")
                self.best = current
                save_checkpoint = True
        else:
            save_checkpoint = True

        if save_checkpoint:
            filepath = self.filepath.format(epoch=epoch)

            if self.save_weights_only:
                torch.save(model.state_dict(), filepath)
            else:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics,
                    f'best_{self.monitor}': self.best
                }

                if optimizer is not None:
                    checkpoint['optimizer_state_dict'] = optimizer.state_dict()

                torch.save(checkpoint, filepath)

            if self.verbose:
                print(f"ModelCheckpoint: Saved checkpoint to {filepath}")


class LearningRateMonitor:
    """
    Callback to monitor and log learning rate during training.
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Args:
            log_file: Optional file to log learning rates
        """
        self.log_file = log_file
        self.history = []

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, optimizer: torch.optim.Optimizer, epoch: int):
        """
        Record current learning rate.

        Args:
            optimizer: Optimizer instance
            epoch: Current epoch
        """
        lrs = [group['lr'] for group in optimizer.param_groups]

        self.history.append({
            'epoch': epoch,
            'learning_rates': lrs
        })

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {epoch}: {lrs}\n")


class GradientMonitor:
    """
    Callback to monitor gradient statistics during training.
    """

    def __init__(self,
                 log_interval: int = 10,
                 verbose: bool = True):
        """
        Args:
            log_interval: Log gradient stats every N batches
            verbose: Whether to print statistics
        """
        self.log_interval = log_interval
        self.verbose = verbose
        self.batch_count = 0
        self.gradient_stats = []

    def __call__(self, model: torch.nn.Module):
        """
        Compute and log gradient statistics.

        Args:
            model: Model instance
        """
        self.batch_count += 1

        if self.batch_count % self.log_interval == 0:
            total_norm = 0
            param_count = 0
            min_grad = float('inf')
            max_grad = float('-inf')

            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    param_count += 1

                    grad_min = p.grad.data.min().item()
                    grad_max = p.grad.data.max().item()

                    min_grad = min(min_grad, grad_min)
                    max_grad = max(max_grad, grad_max)

            total_norm = total_norm ** 0.5

            stats = {
                'batch': self.batch_count,
                'total_norm': total_norm,
                'avg_norm': total_norm / param_count if param_count > 0 else 0,
                'min_grad': min_grad,
                'max_grad': max_grad
            }

            self.gradient_stats.append(stats)

            if self.verbose:
                print(f"GradientMonitor: Batch {self.batch_count} - "
                      f"Total norm: {total_norm:.4f}, "
                      f"Min: {min_grad:.2e}, Max: {max_grad:.2e}")


class MetricTracker:
    """
    Generic callback to track and save metrics during training.
    """

    def __init__(self,
                 save_path: Optional[str] = None,
                 metrics_to_track: Optional[list] = None):
        """
        Args:
            save_path: Path to save metrics history
            metrics_to_track: List of metric names to track (None = all)
        """
        self.save_path = save_path
        self.metrics_to_track = metrics_to_track
        self.history = []

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, epoch: int, metrics: Dict[str, float]):
        """
        Record metrics for current epoch.

        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        entry = {'epoch': epoch}

        if self.metrics_to_track:
            entry.update({k: v for k, v in metrics.items() if k in self.metrics_to_track})
        else:
            entry.update(metrics)

        self.history.append(entry)

        if self.save_path:
            with open(self.save_path, 'w') as f:
                json.dump(self.history, f, indent=2)

    def get_metric_history(self, metric_name: str) -> list:
        """Get history for a specific metric."""
        return [entry.get(metric_name) for entry in self.history if metric_name in entry]


class ProgressBar:
    """
    Simple progress bar callback for training.
    """

    def __init__(self, total_epochs: int, width: int = 50):
        """
        Args:
            total_epochs: Total number of epochs
            width: Width of progress bar
        """
        self.total_epochs = total_epochs
        self.width = width

    def __call__(self, epoch: int, metrics: Optional[Dict[str, float]] = None):
        """
        Display progress bar.

        Args:
            epoch: Current epoch (0-indexed)
            metrics: Optional metrics to display
        """
        progress = (epoch + 1) / self.total_epochs
        filled = int(self.width * progress)
        bar = '█' * filled + '░' * (self.width - filled)

        metrics_str = ""
        if metrics:
            metrics_str = " - " + " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        print(f"\rEpoch {epoch + 1}/{self.total_epochs} [{bar}] {progress * 100:.1f}%{metrics_str}",
              end='', flush=True)

        if epoch + 1 == self.total_epochs:
            print()  # New line at the end


class WarmupScheduler:
    """
    Learning rate warmup scheduler callback.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int,
                 initial_lr: float = 1e-6,
                 target_lr: float = None,
                 warmup_method: str = 'linear'):
        """
        Args:
            optimizer: Optimizer instance
            warmup_epochs: Number of warmup epochs
            initial_lr: Starting learning rate
            target_lr: Target learning rate after warmup
            warmup_method: Warmup method ('linear', 'exponential')
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr or optimizer.param_groups[0]['lr']
        self.warmup_method = warmup_method

        # Store original learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def __call__(self, epoch: int):
        """
        Adjust learning rate based on warmup schedule.

        Args:
            epoch: Current epoch
        """
        if epoch >= self.warmup_epochs:
            return

        if self.warmup_method == 'linear':
            progress = (epoch + 1) / self.warmup_epochs
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * progress
        elif self.warmup_method == 'exponential':
            progress = (epoch + 1) / self.warmup_epochs
            lr = self.initial_lr * (self.target_lr / self.initial_lr) ** progress
        else:
            raise ValueError(f"Unknown warmup method: {self.warmup_method}")

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class CallbackList:
    """
    Container for managing multiple callbacks.
    """

    def __init__(self, callbacks: list = None):
        """
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []

    def on_epoch_begin(self, epoch: int, **kwargs):
        """Call all callbacks' on_epoch_begin methods."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_begin'):
                callback.on_epoch_begin(epoch, **kwargs)

    def on_epoch_end(self, epoch: int, **kwargs):
        """Call all callbacks' on_epoch_end methods."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, **kwargs)

    def on_batch_begin(self, batch: int, **kwargs):
        """Call all callbacks' on_batch_begin methods."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_begin'):
                callback.on_batch_begin(batch, **kwargs)

    def on_batch_end(self, batch: int, **kwargs):
        """Call all callbacks' on_batch_end methods."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_end'):
                callback.on_batch_end(batch, **kwargs)

    def on_train_begin(self, **kwargs):
        """Call all callbacks' on_train_begin methods."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(**kwargs)

    def on_train_end(self, **kwargs):
        """Call all callbacks' on_train_end methods."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(**kwargs)