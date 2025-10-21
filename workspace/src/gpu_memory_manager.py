"""
GPU Memory Manager

This module provides utilities for managing GPU memory usage, preventing
out-of-memory crashes, and implementing retry logic for GPU operations.
"""

import logging
import os
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def setup_gpu_memory_limits(
    memory_fraction: float = 0.4,
    max_split_size_mb: int = 128,
) -> None:
    """
    Configure GPU memory limits to prevent overflow and system crashes.

    This function should be called at the start of each worker process
    BEFORE importing PyTorch or Docling.

    Args:
        memory_fraction: Fraction of GPU memory each process can use (0.0-1.0).
                        Default 0.4 means each process can use up to 40% of GPU.
                        With 2 workers, this allows both to run without overflow.
        max_split_size_mb: Maximum size in MB for memory allocations.
                          Smaller values reduce fragmentation but may be slower.
    """
    # Set PyTorch memory allocator configuration
    # This prevents memory fragmentation and sets hard limits
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
        f'max_split_size_mb:{max_split_size_mb},'
        'expandable_segments:True'
    )

    # Set memory growth for TensorFlow (if used by any dependencies)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    logger.info(
        f"GPU memory limits configured: "
        f"{memory_fraction*100:.0f}% per process, "
        f"{max_split_size_mb}MB max split size"
    )


def set_process_gpu_memory_fraction(fraction: float = 0.4) -> bool:
    """
    Set the GPU memory fraction for the current process.

    Must be called AFTER importing PyTorch but BEFORE using any models.

    Args:
        fraction: Fraction of GPU memory this process can use (0.0-1.0)

    Returns:
        True if successfully set, False if CUDA not available
    """
    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("CUDA not available, skipping GPU memory fraction setting")
            return False

        # Set memory fraction for all visible devices
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(fraction, device=device_id)

        logger.info(f"Set GPU memory fraction to {fraction*100:.0f}% for {torch.cuda.device_count()} device(s)")
        return True

    except ImportError:
        logger.warning("PyTorch not available, cannot set GPU memory fraction")
        return False
    except Exception as e:
        logger.warning(f"Failed to set GPU memory fraction: {e}")
        return False


def clear_gpu_cache() -> None:
    """
    Clear GPU memory cache to free up memory.

    Call this between processing batches or after errors to ensure
    memory is released.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU cache cleared")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to clear GPU cache: {e}")


def get_gpu_memory_info() -> Optional[Tuple[int, int, float]]:
    """
    Get current GPU memory usage information.

    Returns:
        Tuple of (allocated_mb, total_mb, usage_percent) or None if unavailable
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        # Get memory for device 0 (primary GPU)
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)  # Convert to MB
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        usage_percent = (allocated / total) * 100

        return int(allocated), int(total), usage_percent

    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return None


def check_gpu_memory_available(required_mb: int = 2000) -> bool:
    """
    Check if sufficient GPU memory is available.

    Args:
        required_mb: Minimum required free memory in MB

    Returns:
        True if sufficient memory available, False otherwise
    """
    info = get_gpu_memory_info()
    if info is None:
        # If we can't check, assume it's available
        return True

    allocated_mb, total_mb, usage_percent = info
    free_mb = total_mb - allocated_mb

    is_available = free_mb >= required_mb

    if not is_available:
        logger.warning(
            f"Insufficient GPU memory: {free_mb}MB free, {required_mb}MB required "
            f"(total: {total_mb}MB, allocated: {allocated_mb}MB, {usage_percent:.1f}% used)"
        )

    return is_available


def wait_for_gpu_memory(
    required_mb: int = 2000,
    timeout_seconds: int = 300,
    check_interval: int = 5
) -> bool:
    """
    Wait until sufficient GPU memory becomes available.

    Args:
        required_mb: Minimum required free memory in MB
        timeout_seconds: Maximum time to wait in seconds
        check_interval: Time between checks in seconds

    Returns:
        True if memory became available, False if timeout
    """
    start_time = time.time()

    while True:
        if check_gpu_memory_available(required_mb):
            return True

        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            logger.error(
                f"Timeout waiting for GPU memory after {elapsed:.0f}s "
                f"(required: {required_mb}MB)"
            )
            return False

        logger.info(
            f"Waiting for GPU memory... ({elapsed:.0f}s/{timeout_seconds}s, "
            f"required: {required_mb}MB)"
        )

        # Clear cache in case it helps
        clear_gpu_cache()

        time.sleep(check_interval)


def is_oom_error(exception: Exception) -> bool:
    """
    Check if an exception is a GPU out-of-memory error.

    Args:
        exception: The exception to check

    Returns:
        True if it's an OOM error, False otherwise
    """
    error_msg = str(exception).lower()

    oom_signatures = [
        "cuda out of memory",
        "cuda error: out of memory",
        "cublas error: an illegal memory access was encountered",
        "out of memory on device",
        "hip out of memory",
        "resource exhausted",
        "failed to allocate",
        "cudnn error",
    ]

    return any(sig in error_msg for sig in oom_signatures)


def log_gpu_memory_stats() -> None:
    """Log current GPU memory statistics."""
    info = get_gpu_memory_info()
    if info is None:
        logger.info("GPU memory stats: Not available (CPU mode or no CUDA)")
        return

    allocated_mb, total_mb, usage_percent = info
    free_mb = total_mb - allocated_mb

    logger.info(
        f"GPU memory stats: "
        f"{allocated_mb}MB used / {total_mb}MB total "
        f"({usage_percent:.1f}% used, {free_mb}MB free)"
    )
