import os
import torch
import numpy as np
from braindecode.util import set_random_seeds
from skorch.callbacks import EarlyStopping

RAND_SEED = 42

def set_seeds(seed_num):
    global RAND_SEED
    RAND_SEED = seed_num
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

# Centralized EarlyStopping configuration
# These parameters are designed to prevent underfitting while still preventing overfitting
# Note: A higher patience value is still cheaper computationally than a full re-train run.
EARLY_STOPPING_PATIENCE = 20  # Increased from 10 to allow more training
EARLY_STOPPING_THRESHOLD = 1e-5  # Relaxed from 1e-4 to be less strict
EARLY_STOPPING_MONITOR = 'valid_loss'  # Monitor validation loss instead of score
EARLY_STOPPING_LOAD_BEST = True  # Load best model weights

def get_early_stopping_callback():
    """
    Get a standardized EarlyStopping callback for all models.
    
    Returns:
        EarlyStopping: Configured EarlyStopping callback
    """
    return EarlyStopping(
        monitor=EARLY_STOPPING_MONITOR,
        threshold=EARLY_STOPPING_THRESHOLD,
        patience=EARLY_STOPPING_PATIENCE,
        load_best=EARLY_STOPPING_LOAD_BEST
    )

# Default max_epochs - increased to allow more training time
DEFAULT_MAX_EPOCHS = 200  # Increased from 100

def get_max_epochs_for_dataset(dataset: str, eval_mode: str = None) -> int:
    """
    Get dataset-specific max_epochs setting.
    
    Args:
        dataset: Dataset name (e.g., "Lee2019_SSVEP", "BI2015a", "BNCI2014_001")
        eval_mode: Evaluation mode (e.g., "CrossSubject", "CrossSession", "WithinSession")
    
    Returns:
        int: Max epochs for the dataset and evaluation mode.
             - CrossSubject: Always 20 (for performance reasons)
             - Lee2019_SSVEP and BI2015a: 100
             - Otherwise: 200
    """
    # CrossSubject evaluation is much slower, so use fewer epochs
    if eval_mode == "CrossSubject":
        return 20
    
    if dataset in ["Lee2019_SSVEP", "BI2015a"]:
        return 100
    return DEFAULT_MAX_EPOCHS

# Underfitting detection threshold (default for backward compatibility)
UNDERFITTING_THRESHOLD = 0.70  # Increased from 0.65 to catch more underfitting cases

def get_underfitting_threshold_for_dataset(dataset: str) -> float:
    """
    Get dataset-specific underfitting threshold.
    
    The threshold should be adjusted based on the number of classes:
    - For 4-class problems (e.g., Lee2019_SSVEP): ~0.35 (random chance = 0.25)
    - For 2-class problems (e.g., BI2015a, BNCI2014_001): ~0.70 (random chance = 0.50)
    
    Args:
        dataset: Dataset name (e.g., "Lee2019_SSVEP", "BI2015a", "BNCI2014_001")
    
    Returns:
        float: Underfitting threshold for the dataset
    """
    if dataset == "Lee2019_SSVEP":
        return 0.35  # 4 classes, random chance = 0.25
    # Default for 2-class problems
    return UNDERFITTING_THRESHOLD  # 0.70 for 2-class problems

# EEGClassifier verbose level
# 0 = silent, 1 = epoch progress bar, 2 = epoch progress + batch info
EEGCLASSIFIER_VERBOSE = 2  # Set to 0 to suppress epoch-level output