"""
Device management for Apple Silicon M4 Max optimization.
"""
import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages compute device selection and optimization for M4 Max."""

    _instance: Optional["DeviceManager"] = None

    def __new__(cls) -> "DeviceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._device = self._select_device()
        self._dtype = self._select_dtype()

        logger.info(f"Device initialized: {self._device}")
        logger.info(f"Default dtype: {self._dtype}")

    def _select_device(self) -> torch.device:
        """Select the best available device."""
        if torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                logger.info("MPS (Metal Performance Shaders) available - using Apple Silicon GPU")
                return torch.device("mps")
            else:
                logger.warning("MPS available but not built, falling back to CPU")

        if torch.cuda.is_available():
            logger.info("CUDA available - using NVIDIA GPU")
            return torch.device("cuda")

        logger.info("Using CPU")
        return torch.device("cpu")

    def _select_dtype(self) -> torch.dtype:
        """Select optimal dtype for the device."""
        if self._device.type == "mps":
            # M4 Max supports float16 efficiently
            return torch.float32  # Use float32 for stability, float16 for speed
        elif self._device.type == "cuda":
            return torch.float32
        return torch.float32

    @property
    def device(self) -> torch.device:
        """Get the selected device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Get the selected dtype."""
        return self._dtype

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the selected device."""
        return tensor.to(self._device, dtype=self._dtype)

    def empty_cache(self):
        """Clear device memory cache."""
        if self._device.type == "mps":
            torch.mps.empty_cache()
        elif self._device.type == "cuda":
            torch.cuda.empty_cache()

    def synchronize(self):
        """Synchronize device operations."""
        if self._device.type == "mps":
            torch.mps.synchronize()
        elif self._device.type == "cuda":
            torch.cuda.synchronize()

    def get_memory_info(self) -> dict:
        """Get device memory information."""
        info = {"device": str(self._device)}

        if self._device.type == "mps":
            # MPS doesn't have detailed memory API yet
            info["allocated"] = torch.mps.driver_allocated_memory()
            info["type"] = "Apple Silicon"
        elif self._device.type == "cuda":
            info["allocated"] = torch.cuda.memory_allocated()
            info["cached"] = torch.cuda.memory_reserved()
            info["max_allocated"] = torch.cuda.max_memory_allocated()
            info["type"] = "NVIDIA CUDA"
        else:
            info["type"] = "CPU"

        return info

    def optimize_for_inference(self):
        """Apply inference optimizations."""
        torch.set_grad_enabled(False)
        if self._device.type == "mps":
            # MPS-specific optimizations
            pass

    def optimize_for_training(self):
        """Apply training optimizations."""
        torch.set_grad_enabled(True)
        if self._device.type == "mps":
            # Enable gradient checkpointing for memory efficiency
            pass


# Global device manager instance
device_manager = DeviceManager()


def get_device() -> torch.device:
    """Get the global device."""
    return device_manager.device


def get_dtype() -> torch.dtype:
    """Get the global dtype."""
    return device_manager.dtype


def to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to the global device."""
    return device_manager.to_device(tensor)
