import torch

def get_device(device_number: int) -> torch.device:
    """Return the configured CUDA device when available, otherwise CPU."""
    if int(device_number) >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{int(device_number)}")
    return torch.device("cpu")
