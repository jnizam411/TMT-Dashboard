"""
MPS Accelerator - Metal Performance Shaders integration for Apple Silicon

This module provides:
1. Automatic device detection (MPS/CPU)
2. Memory-aware tensor allocation
3. Mixed precision support for M-series
4. Optimized data transfer utilities
"""

import torch
import platform
import subprocess
import warnings
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class DeviceInfo:
    """Information about the compute device"""
    device_type: str  # 'mps', 'cuda', 'cpu'
    device_name: str
    total_memory_gb: float
    available_memory_gb: float
    unified_memory: bool  # True for Apple Silicon
    supports_float16: bool


class MPSAccelerator:
    """
    Metal Performance Shaders accelerator for Apple Silicon.

    Provides automatic device selection, memory management,
    and optimized tensor operations for M-series chips.

    Usage:
        accelerator = MPSAccelerator()
        device = accelerator.device
        tensor = accelerator.to_device(my_tensor)
    """

    def __init__(
        self,
        prefer_mps: bool = True,
        memory_fraction: float = 0.8,
        enable_mixed_precision: bool = True,
    ):
        """
        Initialize the MPS accelerator.

        Args:
            prefer_mps: Use MPS if available (default True)
            memory_fraction: Fraction of available memory to use (0-1)
            enable_mixed_precision: Use float16 where beneficial
        """
        self.prefer_mps = prefer_mps
        self.memory_fraction = memory_fraction
        self.enable_mixed_precision = enable_mixed_precision

        self._device = None
        self._device_info = None
        self._setup_device()

    def _setup_device(self):
        """Detect and configure the optimal device."""
        if self.prefer_mps and torch.backends.mps.is_available():
            self._device = torch.device("mps")
            self._device_info = self._get_mps_info()
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._device_info = self._get_cuda_info()
        else:
            self._device = torch.device("cpu")
            self._device_info = self._get_cpu_info()

        print(f"[TMT] Using device: {self._device_info.device_name}")
        print(f"[TMT] Total memory: {self._device_info.total_memory_gb:.1f} GB")
        print(f"[TMT] Available memory: {self._device_info.available_memory_gb:.1f} GB")

    def _get_mps_info(self) -> DeviceInfo:
        """Get Apple Silicon device information."""
        # Get system memory info (unified memory)
        total_gb, available_gb = self._get_system_memory()

        # Detect chip name
        chip_name = self._get_apple_chip_name()

        return DeviceInfo(
            device_type="mps",
            device_name=f"Apple {chip_name} (MPS)",
            total_memory_gb=total_gb,
            available_memory_gb=available_gb * self.memory_fraction,
            unified_memory=True,
            supports_float16=True,
        )

    def _get_cuda_info(self) -> DeviceInfo:
        """Get NVIDIA GPU information."""
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        available_gb = (props.total_memory - torch.cuda.memory_allocated()) / (1024**3)

        return DeviceInfo(
            device_type="cuda",
            device_name=props.name,
            total_memory_gb=total_gb,
            available_memory_gb=available_gb * self.memory_fraction,
            unified_memory=False,
            supports_float16=True,
        )

    def _get_cpu_info(self) -> DeviceInfo:
        """Get CPU information."""
        total_gb, available_gb = self._get_system_memory()

        return DeviceInfo(
            device_type="cpu",
            device_name=platform.processor() or "CPU",
            total_memory_gb=total_gb,
            available_memory_gb=available_gb * self.memory_fraction,
            unified_memory=True,
            supports_float16=False,
        )

    @staticmethod
    def _get_system_memory() -> Tuple[float, float]:
        """Get system memory in GB."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.total / (1024**3), mem.available / (1024**3)
        except ImportError:
            # Fallback for systems without psutil
            return 8.0, 4.0  # Conservative defaults

    @staticmethod
    def _get_apple_chip_name() -> str:
        """Detect Apple Silicon chip name."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            brand = result.stdout.strip()
            if "Apple" in brand:
                return brand.replace("Apple ", "")
            return "Silicon"
        except Exception:
            return "Silicon"

    @property
    def device(self) -> torch.device:
        """Get the configured device."""
        return self._device

    @property
    def device_info(self) -> DeviceInfo:
        """Get device information."""
        return self._device_info

    @property
    def is_mps(self) -> bool:
        """Check if using MPS."""
        return self._device_info.device_type == "mps"

    @property
    def is_unified_memory(self) -> bool:
        """Check if using unified memory architecture."""
        return self._device_info.unified_memory

    def to_device(
        self,
        tensor: torch.Tensor,
        non_blocking: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Move tensor to device with optional dtype conversion.

        Args:
            tensor: Input tensor
            non_blocking: Use async transfer (faster for unified memory)
            dtype: Target dtype (None keeps original)

        Returns:
            Tensor on device
        """
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)

        # For unified memory, non_blocking is essentially free
        if self.is_unified_memory:
            non_blocking = True

        return tensor.to(self._device, non_blocking=non_blocking)

    def empty_cache(self):
        """Clear device memory cache."""
        if self._device_info.device_type == "mps":
            torch.mps.empty_cache()
        elif self._device_info.device_type == "cuda":
            torch.cuda.empty_cache()

    def synchronize(self):
        """Synchronize device operations."""
        if self._device_info.device_type == "mps":
            torch.mps.synchronize()
        elif self._device_info.device_type == "cuda":
            torch.cuda.synchronize()

    def get_optimal_batch_size(
        self,
        sample_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        model_memory_gb: float = 0.5,
    ) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            sample_shape: Shape of single sample (seq_len, features)
            dtype: Data type
            model_memory_gb: Estimated model memory usage

        Returns:
            Recommended batch size
        """
        # Calculate bytes per sample
        dtype_size = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float64: 8,
        }.get(dtype, 4)

        bytes_per_sample = 1
        for dim in sample_shape:
            bytes_per_sample *= dim
        bytes_per_sample *= dtype_size

        # Account for gradients (2x) and optimizer states (2x)
        bytes_per_sample *= 4

        # Available memory for batches
        available_bytes = (self._device_info.available_memory_gb - model_memory_gb) * (1024**3)

        # Calculate max batch size
        max_batch = int(available_bytes / bytes_per_sample)

        # Round down to power of 2
        if max_batch <= 0:
            return 16

        batch_size = 2 ** int(max_batch.bit_length() - 1)
        return min(max(batch_size, 16), 512)

    def get_optimal_dtype(self) -> torch.dtype:
        """
        Get optimal dtype for this device.

        Returns:
            Recommended dtype (float32 or float16)
        """
        if self.enable_mixed_precision and self._device_info.supports_float16:
            # For MPS, float32 is often faster than float16 for small models
            # Use float16 only for large batch inference
            return torch.float32
        return torch.float32


# Global accelerator instance
_global_accelerator: Optional[MPSAccelerator] = None


def get_device(
    prefer_mps: bool = True,
    memory_fraction: float = 0.8,
) -> torch.device:
    """
    Get the optimal compute device.

    Convenience function that returns a configured device.

    Args:
        prefer_mps: Prefer MPS over CUDA if available
        memory_fraction: Fraction of memory to use

    Returns:
        torch.device configured for optimal performance
    """
    global _global_accelerator

    if _global_accelerator is None:
        _global_accelerator = MPSAccelerator(
            prefer_mps=prefer_mps,
            memory_fraction=memory_fraction,
        )

    return _global_accelerator.device


def get_accelerator() -> MPSAccelerator:
    """Get the global accelerator instance."""
    global _global_accelerator

    if _global_accelerator is None:
        _global_accelerator = MPSAccelerator()

    return _global_accelerator


def optimize_for_memory(available_gb: float) -> dict:
    """
    Get recommended settings for given memory constraints.

    Args:
        available_gb: Available RAM in GB

    Returns:
        Dictionary of recommended settings
    """
    if available_gb >= 32:
        # High memory: maximize performance
        return {
            "batch_size": 128,
            "sequence_length": 60,
            "hidden_size": 256,
            "num_layers": 4,
            "dtype": torch.float32,
            "gradient_accumulation": 1,
            "use_checkpointing": False,
        }
    elif available_gb >= 16:
        # Medium memory: balanced
        return {
            "batch_size": 64,
            "sequence_length": 40,
            "hidden_size": 128,
            "num_layers": 3,
            "dtype": torch.float32,
            "gradient_accumulation": 2,
            "use_checkpointing": False,
        }
    elif available_gb >= 8:
        # Low memory: optimize for efficiency
        return {
            "batch_size": 32,
            "sequence_length": 30,
            "hidden_size": 64,
            "num_layers": 2,
            "dtype": torch.float32,
            "gradient_accumulation": 4,
            "use_checkpointing": True,
        }
    else:
        # Very low memory: minimal footprint
        return {
            "batch_size": 16,
            "sequence_length": 20,
            "hidden_size": 32,
            "num_layers": 1,
            "dtype": torch.float32,
            "gradient_accumulation": 8,
            "use_checkpointing": True,
        }


class MPSGradScaler:
    """
    Gradient scaler compatible with MPS.

    MPS doesn't support torch.cuda.amp.GradScaler, so this
    provides a compatible implementation for mixed precision.
    """

    def __init__(self, enabled: bool = True, init_scale: float = 65536.0):
        self.enabled = enabled
        self._scale = init_scale
        self._growth_factor = 2.0
        self._backoff_factor = 0.5
        self._growth_interval = 2000
        self._step_count = 0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        if not self.enabled:
            return loss
        return loss * self._scale

    def unscale_(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients."""
        if not self.enabled:
            return

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.div_(self._scale)

    def step(self, optimizer: torch.optim.Optimizer):
        """Optimizer step with gradient checking."""
        self.unscale_(optimizer)

        # Check for inf/nan gradients
        has_inf = False
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        has_inf = True
                        break
            if has_inf:
                break

        if not has_inf:
            optimizer.step()
            self._step_count += 1

            # Grow scale periodically
            if self._step_count % self._growth_interval == 0:
                self._scale *= self._growth_factor
        else:
            # Reduce scale on overflow
            self._scale *= self._backoff_factor

    def update(self):
        """Update scaler state (no-op for MPS)."""
        pass

    def get_scale(self) -> float:
        """Get current scale factor."""
        return self._scale


def create_mps_autocast():
    """
    Create autocast context for MPS.

    Returns a context manager for mixed precision on MPS.
    """
    if torch.backends.mps.is_available():
        # MPS autocast is available in recent PyTorch versions
        try:
            return torch.autocast(device_type="mps", dtype=torch.float16)
        except Exception:
            # Fallback to no-op context
            from contextlib import nullcontext
            return nullcontext()
    else:
        from contextlib import nullcontext
        return nullcontext()
