"""Device utilities for cross-platform support (CUDA, MPS, CPU)."""

import torch


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_device_type(device) -> str:
    """Return a device type string suitable for ``torch.autocast``.

    Returns the device type directly for CUDA and MPS (both support
    ``torch.autocast``). Falls back to ``"cpu"`` for other devices.
    """
    device_type = device.type if isinstance(device, torch.device) else device
    if device_type in ("cuda", "mps"):
        return device_type
    return "cpu"


def get_autocast_dtype(device) -> torch.dtype:
    """Return a suitable autocast dtype for the given device.

    CUDA supports bfloat16; MPS requires float16; CPU uses bfloat16.
    """
    device_type = device.type if isinstance(device, torch.device) else device
    if device_type == "mps":
        return torch.float16
    return torch.bfloat16
