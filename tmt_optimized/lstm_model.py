"""
Optimized LSTM Model - MPS-accelerated deep learning for Apple Silicon

Features:
1. Automatic MPS/CUDA/CPU device selection
2. Mixed precision training support
3. Gradient checkpointing for memory efficiency
4. Adaptive batch sizing
5. Built-in performance monitoring
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass
import time
import warnings

from .mps_accelerator import MPSAccelerator, get_device, MPSGradScaler
from .memory_manager import AdaptiveMemoryManager, MemoryConfig, estimate_model_memory


@dataclass
class ModelConfig:
    """LSTM model configuration"""
    input_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    use_layer_norm: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    use_mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.0


class OptimizedLSTM(nn.Module):
    """
    Optimized LSTM for Apple Silicon with MPS acceleration.

    Features:
    - Layer normalization for stable training
    - Dropout for regularization
    - Optimized for MPS backend
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the LSTM model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
        )

        # Output dimension
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)

        # Layer normalization
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_output_size)
        else:
            self.layer_norm = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Output layer
        self.fc = nn.Linear(lstm_output_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights for better convergence."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'fc.weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, features)

        Returns:
            Predictions (batch, 1)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Take last time step
        last_output = lstm_out[:, -1, :]

        # Layer norm and dropout
        normalized = self.layer_norm(last_output)
        dropped = self.dropout(normalized)

        # Output
        return self.fc(dropped)

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Generate predictions.

        Args:
            x: Input tensor

        Returns:
            Numpy array of predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x).cpu().numpy().flatten()


class MPSTrainer:
    """
    Trainer optimized for Apple Silicon MPS.

    Features:
    - Automatic device selection
    - Memory-adaptive batch sizing
    - Mixed precision support
    - Gradient accumulation
    - Early stopping
    - Training metrics tracking
    """

    def __init__(
        self,
        model: OptimizedLSTM,
        training_config: TrainingConfig,
        accelerator: Optional[MPSAccelerator] = None,
        memory_manager: Optional[AdaptiveMemoryManager] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: LSTM model to train
            training_config: Training configuration
            accelerator: MPS accelerator instance
            memory_manager: Memory manager instance
        """
        self.model = model
        self.config = training_config

        # Setup accelerator
        self.accelerator = accelerator or MPSAccelerator()
        self.device = self.accelerator.device

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup memory manager
        self.memory_manager = memory_manager or AdaptiveMemoryManager()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Setup loss function
        self.criterion = nn.MSELoss()

        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )

        # Setup gradient scaler for mixed precision
        if self.config.use_mixed_precision:
            self.scaler = MPSGradScaler(enabled=True)
        else:
            self.scaler = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_ic': [],
            'val_sharpe': [],
            'learning_rate': [],
            'batch_size': [],
            'epoch_time': [],
        }

        # Early stopping state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def _to_device(self, *tensors) -> Tuple[torch.Tensor, ...]:
        """Move tensors to device."""
        return tuple(
            self.accelerator.to_device(t) if isinstance(t, torch.Tensor) else t
            for t in tensors
        )

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from scipy.stats import spearmanr

        # Information coefficient
        ic = spearmanr(predictions, actuals)[0]
        if np.isnan(ic):
            ic = 0.0

        # Strategy returns
        strategy_returns = np.sign(predictions) * actuals

        # Sharpe ratio
        if np.std(strategy_returns) > 1e-10:
            sharpe = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
        else:
            sharpe = 0.0

        return {'ic': ic, 'sharpe': sharpe}

    def train_epoch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> float:
        """
        Train for one epoch.

        Args:
            X_train: Training features
            y_train: Training targets
            batch_size: Batch size (uses adaptive if None)

        Returns:
            Average training loss
        """
        self.model.train()

        # Get batch size
        if batch_size is None:
            batch_size = self.memory_manager.get_dynamic_batch_size(
                sample_shape=X_train.shape[1:],
                model_memory_gb=0.5,
            )

        n_samples = len(X_train)
        indices = np.random.permutation(n_samples)

        total_loss = 0.0
        n_batches = 0

        # Gradient accumulation counter
        accumulated_steps = 0

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            # Prepare batch
            X_batch = torch.from_numpy(X_train[batch_indices]).float()
            y_batch = torch.from_numpy(y_train[batch_indices]).float().view(-1, 1)

            X_batch, y_batch = self._to_device(X_batch, y_batch)

            # Forward pass
            if self.config.use_mixed_precision and self.scaler:
                with torch.autocast(device_type=str(self.device.type), dtype=torch.float16):
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    loss = loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

            accumulated_steps += 1

            # Update weights
            if accumulated_steps >= self.config.gradient_accumulation_steps:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                accumulated_steps = 0

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    @torch.no_grad()
    def evaluate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 64,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on validation set.

        Args:
            X_val: Validation features
            y_val: Validation targets
            batch_size: Batch size

        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.eval()

        n_samples = len(X_val)
        total_loss = 0.0
        n_batches = 0
        all_predictions = []

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)

            X_batch = torch.from_numpy(X_val[start_idx:end_idx]).float()
            y_batch = torch.from_numpy(y_val[start_idx:end_idx]).float().view(-1, 1)

            X_batch, y_batch = self._to_device(X_batch, y_batch)

            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            total_loss += loss.item()
            n_batches += 1

            all_predictions.extend(outputs.cpu().numpy().flatten())

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        predictions = np.array(all_predictions)
        metrics = self._compute_metrics(predictions, y_val[:len(predictions)])

        return avg_loss, metrics

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, List]:
        """
        Full training loop.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            verbose: Print progress

        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print("TRAINING OPTIMIZED LSTM")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(X_train)}")
        if X_val is not None:
            print(f"Validation samples: {len(X_val)}")
        print(f"{'='*60}\n")

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            # Get adaptive batch size
            batch_size = self.memory_manager.get_dynamic_batch_size(
                sample_shape=X_train.shape[1:],
                model_memory_gb=0.5,
            )

            # Train epoch
            train_loss = self.train_epoch(X_train, y_train, batch_size)

            # Validate
            if X_val is not None and y_val is not None:
                val_loss, val_metrics = self.evaluate(X_val, y_val)
            else:
                val_loss = train_loss
                val_metrics = {'ic': 0.0, 'sharpe': 0.0}

            epoch_time = time.time() - epoch_start

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_ic'].append(val_metrics['ic'])
            self.history['val_sharpe'].append(val_metrics['sharpe'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['batch_size'].append(batch_size)
            self.history['epoch_time'].append(epoch_time)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1

            # Print progress
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"IC: {val_metrics['ic']:.4f} | "
                      f"Sharpe: {val_metrics['sharpe']:.2f} | "
                      f"Batch: {batch_size} | "
                      f"Time: {epoch_time:.2f}s")

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            # Clear memory periodically
            if (epoch + 1) % 10 == 0:
                self.memory_manager.clear_memory()

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nRestored best model (val_loss: {self.best_val_loss:.6f})")

        return self.history

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Input features
            batch_size: Batch size

        Returns:
            Predictions array
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for start_idx in range(0, len(X), batch_size):
                end_idx = min(start_idx + batch_size, len(X))
                X_batch = torch.from_numpy(X[start_idx:end_idx]).float()
                X_batch = self.accelerator.to_device(X_batch)

                outputs = self.model(X_batch)
                predictions.extend(outputs.cpu().numpy().flatten())

        return np.array(predictions)

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']


def create_model_for_device(
    input_size: int,
    available_memory_gb: float,
    device: Optional[torch.device] = None,
) -> Tuple[OptimizedLSTM, TrainingConfig]:
    """
    Create model configuration optimized for available memory.

    Args:
        input_size: Number of input features
        available_memory_gb: Available RAM in GB
        device: Target device

    Returns:
        Tuple of (model, training_config)
    """
    from .mps_accelerator import optimize_for_memory

    # Get recommended settings
    settings = optimize_for_memory(available_memory_gb)

    # Create model config
    model_config = ModelConfig(
        input_size=input_size,
        hidden_size=settings['hidden_size'],
        num_layers=settings['num_layers'],
        dropout=0.3,
    )

    # Create training config
    training_config = TrainingConfig(
        batch_size=settings['batch_size'],
        gradient_accumulation_steps=settings['gradient_accumulation'],
    )

    # Create model
    model = OptimizedLSTM(model_config)

    if device is not None:
        model = model.to(device)

    return model, training_config
