import os
import torch
import numpy as np
from braindecode.util import set_random_seeds
from config import DEFAULT_SEED

RAND_SEED = DEFAULT_SEED

def set_seeds(seed_num):
    globals.RAND_SEED = seed_num
    """Ensure full reproducibility for PyTorch, NumPy, and random operations."""
    cuda = torch.cuda.is_available()
    set_random_seeds(seed_num, cuda)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)  # For multi-GPU setups
    np.random.seed(seed_num)

    torch.backends.cudnn.deterministic = True  # Enforce deterministic CNN ops
    torch.backends.cudnn.benchmark = False  # Prevent dynamic optimizations
    torch.set_float32_matmul_precision('high')  # Ensures FP32 consistency
    torch.set_default_dtype(torch.float32)  # Standardize float type

    torch.use_deterministic_algorithms(True)  # Ensures determinism in all PyTorch ops
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Ensures deterministic CUBLAS ops (important for CUDA 10.2+)

def get_seed():
    return RAND_SEED