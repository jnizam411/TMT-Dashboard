"""
TMT Optimized - High-Performance Quantitative Finance Library

Optimized for Apple M-series hardware with:
- Metal Performance Shaders (MPS) GPU acceleration
- Rust-based compute kernels with SIMD
- Adaptive memory management
- Variable RAM/GPU usage
"""

from .mps_accelerator import MPSAccelerator, get_device, optimize_for_memory
from .feature_engine import OptimizedFeatureEngine
from .memory_manager import AdaptiveMemoryManager, MemoryConfig
from .lstm_model import OptimizedLSTM, MPSTrainer

__version__ = "0.1.0"
__all__ = [
    "MPSAccelerator",
    "get_device",
    "optimize_for_memory",
    "OptimizedFeatureEngine",
    "AdaptiveMemoryManager",
    "MemoryConfig",
    "OptimizedLSTM",
    "MPSTrainer",
]
