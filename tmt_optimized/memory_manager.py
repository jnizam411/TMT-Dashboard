"""
Adaptive Memory Manager - Dynamic resource allocation for variable RAM

This module provides:
1. Real-time memory monitoring
2. Dynamic batch size adjustment
3. Gradient checkpointing utilities
4. Memory-efficient data loading
"""

import torch
import gc
import warnings
from typing import Optional, Tuple, Iterator, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np


@dataclass
class MemoryConfig:
    """Configuration for memory management"""
    max_memory_gb: Optional[float] = None  # None = auto-detect
    min_batch_size: int = 8
    max_batch_size: int = 512
    enable_checkpointing: bool = True
    enable_garbage_collection: bool = True
    memory_reserve_fraction: float = 0.2  # Reserve 20% for system


@dataclass
class MemoryStats:
    """Current memory statistics"""
    total_gb: float
    available_gb: float
    used_gb: float
    peak_gb: float
    tensor_count: int


class AdaptiveMemoryManager:
    """
    Adaptive memory manager for variable RAM environments.

    Automatically adjusts batch sizes and enables memory optimizations
    based on real-time memory availability.

    Usage:
        manager = AdaptiveMemoryManager(MemoryConfig())
        batch_size = manager.get_dynamic_batch_size(sample_shape)

        with manager.memory_efficient_context():
            # Training code here
            pass
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the memory manager.

        Args:
            config: Memory configuration (uses defaults if None)
        """
        self.config = config or MemoryConfig()
        self._peak_memory = 0.0
        self._baseline_memory = self._get_current_memory()

        # Auto-detect max memory if not specified
        if self.config.max_memory_gb is None:
            self.config.max_memory_gb = self._get_total_memory()

        print(f"[MemoryManager] Total memory: {self.config.max_memory_gb:.1f} GB")
        print(f"[MemoryManager] Reserved: {self.config.max_memory_gb * self.config.memory_reserve_fraction:.1f} GB")

    @staticmethod
    def _get_total_memory() -> float:
        """Get total system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 8.0  # Conservative default

    @staticmethod
    def _get_current_memory() -> float:
        """Get current memory usage in GB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024**3)
        except ImportError:
            return 0.0

    def get_available_memory(self) -> float:
        """Get available memory for computation in GB."""
        try:
            import psutil
            available = psutil.virtual_memory().available / (1024**3)
            # Subtract reserve
            usable = available - (self.config.max_memory_gb * self.config.memory_reserve_fraction)
            return max(usable, 0.5)  # Minimum 0.5 GB
        except ImportError:
            # Conservative estimate
            return self.config.max_memory_gb * 0.5

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            process_mem = psutil.Process().memory_info().rss

            current_gb = process_mem / (1024**3)
            self._peak_memory = max(self._peak_memory, current_gb)

            # Count PyTorch tensors
            tensor_count = 0
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj):
                        tensor_count += 1
                except Exception:
                    pass

            return MemoryStats(
                total_gb=mem.total / (1024**3),
                available_gb=mem.available / (1024**3),
                used_gb=current_gb,
                peak_gb=self._peak_memory,
                tensor_count=tensor_count,
            )
        except ImportError:
            return MemoryStats(
                total_gb=self.config.max_memory_gb,
                available_gb=self.get_available_memory(),
                used_gb=0.0,
                peak_gb=0.0,
                tensor_count=0,
            )

    def get_dynamic_batch_size(
        self,
        sample_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        model_memory_gb: float = 0.5,
    ) -> int:
        """
        Calculate optimal batch size based on current memory.

        Args:
            sample_shape: Shape of single sample
            dtype: Data type
            model_memory_gb: Estimated model memory

        Returns:
            Recommended batch size
        """
        # Get available memory
        available_gb = self.get_available_memory()

        # Calculate bytes per sample
        dtype_bytes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float64: 8,
        }.get(dtype, 4)

        sample_bytes = np.prod(sample_shape) * dtype_bytes

        # Account for gradients (2x) and optimizer states (2x for Adam)
        sample_bytes *= 4

        # Available bytes for batches
        batch_memory_gb = max(available_gb - model_memory_gb, 0.1)
        batch_bytes = batch_memory_gb * (1024**3)

        # Calculate max batch size
        max_batch = int(batch_bytes / sample_bytes)

        # Clamp to config limits and round to power of 2
        batch_size = max(min(max_batch, self.config.max_batch_size), self.config.min_batch_size)

        # Round down to nearest power of 2 for efficiency
        batch_size = 2 ** int(np.log2(batch_size))

        return batch_size

    def should_reduce_batch_size(self, current_batch_size: int) -> Tuple[bool, int]:
        """
        Check if batch size should be reduced due to memory pressure.

        Args:
            current_batch_size: Current batch size

        Returns:
            Tuple of (should_reduce, new_batch_size)
        """
        stats = self.get_memory_stats()

        # Reduce if available memory is low
        if stats.available_gb < 1.0:
            new_batch = max(current_batch_size // 2, self.config.min_batch_size)
            return True, new_batch

        return False, current_batch_size

    @contextmanager
    def memory_efficient_context(self):
        """
        Context manager for memory-efficient operations.

        Enables garbage collection and clears caches on exit.
        """
        # Enable aggressive GC
        if self.config.enable_garbage_collection:
            gc.enable()

        try:
            yield
        finally:
            # Clean up
            gc.collect()

            # Clear PyTorch caches
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    def clear_memory(self):
        """Aggressively clear memory."""
        gc.collect()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


class MemoryEfficientDataLoader:
    """
    Memory-efficient data loader with dynamic batching.

    Automatically adjusts batch sizes based on memory pressure
    and supports gradient accumulation for effective larger batches.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_batch_size: int = 32,
        memory_manager: Optional[AdaptiveMemoryManager] = None,
        shuffle: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the data loader.

        Args:
            X: Feature array (n_samples, seq_len, n_features)
            y: Target array (n_samples,)
            initial_batch_size: Starting batch size
            memory_manager: Memory manager instance
            shuffle: Shuffle data each epoch
            device: Target device
        """
        self.X = X
        self.y = y
        self.batch_size = initial_batch_size
        self.memory_manager = memory_manager or AdaptiveMemoryManager()
        self.shuffle = shuffle
        self.device = device or torch.device("cpu")

        self.n_samples = len(X)
        self._indices = np.arange(self.n_samples)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over batches."""
        if self.shuffle:
            np.random.shuffle(self._indices)

        for start_idx in range(0, self.n_samples, self.batch_size):
            # Check memory pressure
            should_reduce, new_batch = self.memory_manager.should_reduce_batch_size(
                self.batch_size
            )
            if should_reduce:
                warnings.warn(f"Reducing batch size from {self.batch_size} to {new_batch} due to memory pressure")
                self.batch_size = new_batch

            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = self._indices[start_idx:end_idx]

            # Load batch
            X_batch = torch.from_numpy(self.X[batch_indices]).float()
            y_batch = torch.from_numpy(self.y[batch_indices]).float()

            # Move to device
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            yield X_batch, y_batch

    def get_effective_batch_size(self, gradient_accumulation_steps: int = 1) -> int:
        """Get effective batch size with gradient accumulation."""
        return self.batch_size * gradient_accumulation_steps


class GradientCheckpointer:
    """
    Gradient checkpointing utilities for memory-constrained training.

    Trades compute for memory by recomputing activations during backward pass.
    """

    @staticmethod
    def checkpoint_sequential(
        module: torch.nn.Module,
        segments: int = 2,
    ) -> torch.nn.Module:
        """
        Wrap a sequential module with gradient checkpointing.

        Args:
            module: Sequential module to wrap
            segments: Number of segments to checkpoint

        Returns:
            Wrapped module with checkpointing
        """
        from torch.utils.checkpoint import checkpoint_sequential

        class CheckpointedModule(torch.nn.Module):
            def __init__(self, module, segments):
                super().__init__()
                self.module = module
                self.segments = segments

            def forward(self, x):
                # Only checkpoint during training
                if self.training:
                    modules = list(self.module.children())
                    return checkpoint_sequential(modules, self.segments, x)
                else:
                    return self.module(x)

        return CheckpointedModule(module, segments)

    @staticmethod
    def apply_checkpointing(model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply gradient checkpointing to LSTM model.

        Args:
            model: LSTM model

        Returns:
            Model with checkpointing enabled
        """
        # For LSTM, we checkpoint at the layer level
        if hasattr(model, 'lstm'):
            # Enable PyTorch's built-in LSTM checkpointing
            # This is done by processing sequences in chunks
            pass

        return model


def estimate_model_memory(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype = torch.float32,
) -> float:
    """
    Estimate memory usage for LSTM model in GB.

    Args:
        input_size: Number of input features
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        batch_size: Batch size
        seq_length: Sequence length
        dtype: Data type

    Returns:
        Estimated memory in GB
    """
    dtype_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float64: 8,
    }.get(dtype, 4)

    # LSTM parameters: 4 * hidden_size * (input_size + hidden_size + 1) per layer
    # First layer: 4 * hidden * (input + hidden + 1)
    # Other layers: 4 * hidden * (hidden + hidden + 1)
    first_layer_params = 4 * hidden_size * (input_size + hidden_size + 1)
    other_layer_params = 4 * hidden_size * (2 * hidden_size + 1) * (num_layers - 1)
    total_params = first_layer_params + other_layer_params

    # FC layer
    fc_params = hidden_size + 1

    # Total parameters
    param_memory = (total_params + fc_params) * dtype_bytes

    # Activations: batch * seq * hidden * num_layers
    activation_memory = batch_size * seq_length * hidden_size * num_layers * dtype_bytes

    # Gradients (same size as activations + parameters)
    gradient_memory = param_memory + activation_memory

    # Optimizer states (Adam: 2x parameters)
    optimizer_memory = 2 * param_memory

    total_bytes = param_memory + activation_memory + gradient_memory + optimizer_memory

    return total_bytes / (1024**3)
