import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import moabb_braindecode_compat  # noqa: F401 — before braindecode (via models.*)

from moabb.datasets import BNCI2014_001, Lee2019_SSVEP, BI2015a
from moabb.paradigms import MotorImagery, SSVEP, P300
from models.eegnet import create_eegnet_classifier
from models.reegnet import create_reegnet_classifier
from models.cnnncp import (
    create_cnnncpv2_classifier,
    create_cnnncp_classifier,
    create_cnnncp_residual_skip_classifier,
)
from models.branched_ncp import create_cnnncp_branched_bins_classifier
from models.cnnncp import create_cnnncfc_v2_classifier, create_cnnncfc_compact_classifier
from models.cnnncp import create_cnnsmallworld_classifier, create_cnnwiredcfc_classifier
from models.cnnncp import create_cfc_only_classifier, create_ncp_only_classifier
from models.cnn_wiredcfc_min import create_cnnwiredcfc_min_classifier
from models.diva_ncp import create_diva_ncp_classifier
from models.branched_diva_ncp import create_branched_diva_ncp_classifier
from models.branched_lstm import create_branched_lstm_classifier
from models.branched_wiredcfc import create_branched_wiredcfc_classifier
from models.hydra import create_hydra_v2_classifier
from models.diva_full import create_diva_full_classifier
from models.sppncp import create_sppncp_classifier
from models.ctnet import create_ctnet_classifier

# Import the integration functions for CNNWiredCfC models
from architecture_refinement.integrate_with_evaluation import create_model_factory_from_architecture

# Centralized experiment configuration

# Global verbose flag for controlling print output
VERBOSE = False

def set_verbose(verbose):
    """Set the global verbose flag."""
    global VERBOSE
    VERBOSE = verbose

def verbose_print(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if VERBOSE:
        print(*args, **kwargs)

# Dataset and paradigm defaults
default_subjects = None # or list(range(1, 10)) for debugging
DEFAULT_DATASET = BNCI2014_001
DEFAULT_PARADIGM = MotorImagery(
    events=["left_hand", "right_hand"],
    fmin=8,
    fmax=35,
    tmin=0.0,
    tmax=None,
    baseline=None,
    resample=None,
    n_classes=2
)

# SSVEP paradigm for Lee2019_SSVEP dataset (event keys must match moabb.datasets.Lee2019_SSVEP)
_LEE2019_SSVEP_EVENTS = ["12.0", "8.57", "6.67", "5.45"]
DEFAULT_SSVEP_PARADIGM = SSVEP(
    n_classes=4,
    events=_LEE2019_SSVEP_EVENTS,
    tmin=0.0,
    tmax=4.0,
    baseline=None,
    resample=None
)
DEFAULT_SEED = 42

# Module-level registry for CNNWiredCfC architectures
_wiredcfc_architecture_registry = {}

# Module-level registry for BranchedWiredCfC architectures
_branched_wiredcfc_architecture_registry = {}

# Module-level registry for HYDRAv2 architectures
_hydra_v2_architecture_registry = {}

# Base model registry without CNNWiredCfC architectures
def get_base_model_registry():
    """Get the base model registry with standard models."""
    return {
        "eegnet": create_eegnet_classifier,
        "ctnet": create_ctnet_classifier,
        "reegnet": create_reegnet_classifier,
        "cnn_ncp": create_cnnncp_classifier,
        "cnn_ncp_residual_skip": create_cnnncp_residual_skip_classifier,
        "cnn_ncp_v2": create_cnnncpv2_classifier,
        "cnn_ncp_branch": create_cnnncp_branched_bins_classifier,
        "cnncfc_v2": create_cnnncfc_v2_classifier,
        "diva_ncp": create_diva_ncp_classifier,
        "branched_diva_ncp": create_branched_diva_ncp_classifier,
        "branched_lstm": create_branched_lstm_classifier,
        "branched_wiredcfc": create_branched_wiredcfc_classifier,
        "hydra_v2": create_hydra_v2_classifier,
        "diva_full": create_diva_full_classifier,
        "cnncfc_compact": create_cnnncfc_compact_classifier,
        "cnn_smallworld": create_cnnsmallworld_classifier,
        "cnn_wiredcfc_min": create_cnnwiredcfc_min_classifier,
        "cfc_only": create_cfc_only_classifier,
        "ncp_only": create_ncp_only_classifier,
    }

# Dynamic CNNWiredCfC architecture registry
def get_wiredcfc_architecture_registry():
    """Get the registry of CNNWiredCfC models with optimized architectures."""
    return _wiredcfc_architecture_registry

def add_wiredcfc_architecture(architecture_name, architecture_file_path):
    """
    Add a new CNNWiredCfC architecture to the registry.
    
    Args:
        architecture_name: Name for the architecture (e.g., "wiredcfc_arch1")
        architecture_file_path: Path to the architecture JSON file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create model factory for this architecture
        model_factory = create_model_factory_from_architecture(architecture_file_path)
        
        # Add to the architecture registry
        _wiredcfc_architecture_registry[architecture_name] = {
            'factory': model_factory,
            'file_path': architecture_file_path
        }
        
        # verbose_print(f"Successfully added {architecture_name} from {architecture_file_path}")
        return True
        
    except Exception as e:
        verbose_print(f"Failed to add {architecture_name}: {e}")
        return False

def remove_wiredcfc_architecture(architecture_name):
    """
    Remove a CNNWiredCfC architecture from the registry.
    
    Args:
        architecture_name: Name of the architecture to remove
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if architecture_name in _wiredcfc_architecture_registry:
            del _wiredcfc_architecture_registry[architecture_name]
            verbose_print(f"Successfully removed {architecture_name}")
            return True
        else:
            verbose_print(f"WARNING: Architecture {architecture_name} not found in registry")
            return False
        
    except Exception as e:
        verbose_print(f"Failed to remove {architecture_name}: {e}")
        return False

def list_wiredcfc_architectures():
    """List all registered CNNWiredCfC architectures."""
    if not _wiredcfc_architecture_registry:
        verbose_print("No CNNWiredCfC architectures registered")
        return []
    
    verbose_print("Registered CNNWiredCfC architectures:")
    for name, info in _wiredcfc_architecture_registry.items():
        verbose_print(f" {name}: {info['file_path']}")
    
    return list(_wiredcfc_architecture_registry.keys())

# Module-level dict for runtime-registered models (e.g., from test scripts)
_runtime_model_registry = {}

def get_model_registry():
    """Get the complete model registry including dynamic CNNWiredCfC and BranchedWiredCfC architectures."""
    # Start with base models
    registry = get_base_model_registry()
    # verbose_print("Retrieved model registry: ")
    # verbose_print(registry.keys())
    
    # Add dynamic CNNWiredCfC architectures
    for name, info in _wiredcfc_architecture_registry.items():
        registry[name] = info['factory']
    
    # Add dynamic BranchedWiredCfC architectures
    for name, info in _branched_wiredcfc_architecture_registry.items():
        # Create a factory function that uses the registered wiring
        # Use closure to capture wiring correctly
        def make_branched_factory(wiring_ref):
            def create_branched_wiredcfc_with_wiring(**kwargs):
                return create_branched_wiredcfc_classifier(wiring=wiring_ref, **kwargs)
            return create_branched_wiredcfc_with_wiring
        registry[name] = make_branched_factory(info['wiring'])
    
    # Add dynamic HYDRAv2 architectures
    for name, info in _hydra_v2_architecture_registry.items():
        # Create a factory function that uses the registered wiring
        # Use closure to capture wiring correctly
        def make_hydra_factory(wiring_ref):
            def create_hydra_v2_with_wiring(**kwargs):
                return create_hydra_v2_classifier(wiring=wiring_ref, **kwargs)
            return create_hydra_v2_with_wiring
        registry[name] = make_hydra_factory(info['wiring'])
    
    # Add runtime-registered models (from test scripts, registration files, etc.)
    registry.update(_runtime_model_registry)
    
    # Also check MODEL_REGISTRY for any additions made directly
    # (in case registration files modify it)
    if hasattr(sys.modules[__name__], 'MODEL_REGISTRY'):
        for name, factory in MODEL_REGISTRY.items():
            if name not in registry:
                registry[name] = factory
    
    return registry

def load_architectures_from_directory(architectures_dir="outputs/architectures", prefix="wiredcfc_arch"):
    """
    Automatically load all architecture files from a directory.
    
    Args:
        architectures_dir: Directory containing architecture JSON files
        prefix: Prefix for architecture names
    
    Returns:
        List of successfully loaded architecture names
    """
    import os
    from pathlib import Path
    
    arch_path = Path(architectures_dir)
    if not arch_path.exists():
        verbose_print(f"Architectures directory not found: {architectures_dir}")
        return []
    
    # Find all JSON files
    json_files = list(arch_path.glob("*.json"))
    if not json_files:
        verbose_print(f"No JSON files found in {architectures_dir}")
        return []
    
    # verbose_print(f"Found {len(json_files)} architecture files in {architectures_dir}")
    
    loaded_architectures = []
    
    for i, json_file in enumerate(json_files):
        # Generate architecture name
        architecture_name = f"{prefix}{i+1}"
        verbose_print(f"Adding architecture {architecture_name} from {json_file}")
        # Add to registry
        if add_wiredcfc_architecture(architecture_name, str(json_file)):
            loaded_architectures.append(architecture_name)
    
    verbose_print(f"Successfully loaded {len(loaded_architectures)} architectures")
    return loaded_architectures

def clear_wiredcfc_architectures():
    """Clear all registered CNNWiredCfC architectures."""
    _wiredcfc_architecture_registry.clear()
    verbose_print("Cleared all CNNWiredCfC architectures")

# BranchedWiredCfC architecture management functions
def add_branched_wiredcfc_architecture(architecture_name, wiring):
    """
    Add a new BranchedWiredCfC architecture to the registry.
    
    Args:
        architecture_name: Name for the architecture (e.g., "branched_wiredcfc_arch1")
        wiring: ArbitraryWiring instance or path to architecture file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # If wiring is a string, treat it as a file path and load the architecture
        if isinstance(wiring, str):
            from architecture_refinement.arbitrary_wiring import load_architecture_from_file
            wiring = load_architecture_from_file(wiring)
        
        # Add to the architecture registry
        _branched_wiredcfc_architecture_registry[architecture_name] = {
            'wiring': wiring,
            'file_path': getattr(wiring, 'file_path', None)
        }
        
        # verbose_print(f"Successfully added {architecture_name}")
        return True
        
    except Exception as e:
        verbose_print(f"Failed to add {architecture_name}: {e}")
        return False

def remove_branched_wiredcfc_architecture(architecture_name):
    """
    Remove a BranchedWiredCfC architecture from the registry.
    
    Args:
        architecture_name: Name of the architecture to remove
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if architecture_name in _branched_wiredcfc_architecture_registry:
            del _branched_wiredcfc_architecture_registry[architecture_name]
            verbose_print(f"Successfully removed {architecture_name}")
            return True
        else:
            verbose_print(f"WARNING: Architecture {architecture_name} not found in registry")
            return False
        
    except Exception as e:
        verbose_print(f"Failed to remove {architecture_name}: {e}")
        return False

def list_branched_wiredcfc_architectures():
    """List all registered BranchedWiredCfC architectures."""
    if not _branched_wiredcfc_architecture_registry:
        verbose_print("No BranchedWiredCfC architectures registered")
        return []
    
    verbose_print("Registered BranchedWiredCfC architectures:")
    for name, info in _branched_wiredcfc_architecture_registry.items():
        file_path = info.get('file_path', 'Direct wiring object')
        verbose_print(f" {name}: {file_path}")
    
    return list(_branched_wiredcfc_architecture_registry.keys())

def get_branched_wiredcfc_architecture_registry():
    """Get the registry of BranchedWiredCfC models with optimized architectures."""
    return _branched_wiredcfc_architecture_registry

def clear_branched_wiredcfc_architectures():
    """Clear all registered BranchedWiredCfC architectures."""
    _branched_wiredcfc_architecture_registry.clear()
    verbose_print("Cleared all BranchedWiredCfC architectures")

def load_branched_wiredcfc_architectures_from_directory(architectures_dir="outputs/architectures", prefix="branched_wiredcfc_arch"):
    """
    Automatically load all BranchedWiredCfC architecture files from a directory.
    
    Args:
        architectures_dir: Directory containing architecture JSON files
        prefix: Prefix for architecture names
    
    Returns:
        List of successfully loaded architecture names
    """
    import os
    from pathlib import Path
    
    arch_path = Path(architectures_dir)
    if not arch_path.exists():
        verbose_print(f"Architectures directory not found: {architectures_dir}")
        return []
    
    # Find all JSON files
    json_files = list(arch_path.glob("*.json"))
    if not json_files:
        verbose_print(f"No JSON files found in {architectures_dir}")
        return []
    
    # verbose_print(f"Found {len(json_files)} architecture files in {architectures_dir}")
    
    loaded_architectures = []
    
    for i, json_file in enumerate(json_files):
        # Generate architecture name
        architecture_name = f"{prefix}{i+1}"
        
        # Add to registry
        if add_branched_wiredcfc_architecture(architecture_name, str(json_file)):
            loaded_architectures.append(architecture_name)
    
    # verbose_print(f"Successfully loaded {len(loaded_architectures)} BranchedWiredCfC architectures")
    return loaded_architectures

# HYDRAv2 architecture management functions
def add_hydra_v2_architecture(architecture_name, wiring):
    """
    Add a new HYDRAv2 architecture to the registry.
    
    Args:
        architecture_name: Name for the architecture (e.g., "hydra_v2_arch1")
        wiring: ArbitraryWiring instance or path to architecture file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # If wiring is a string, treat it as a file path and load the architecture
        if isinstance(wiring, str):
            from architecture_refinement.arbitrary_wiring import load_architecture_from_file
            wiring = load_architecture_from_file(wiring)
        
        # Add to the architecture registry
        _hydra_v2_architecture_registry[architecture_name] = {
            'wiring': wiring,
            'file_path': getattr(wiring, 'file_path', None)
        }
        
        # verbose_print(f"Successfully added {architecture_name}")
        return True
        
    except Exception as e:
        verbose_print(f"Failed to add {architecture_name}: {e}")
        return False

def remove_hydra_v2_architecture(architecture_name):
    """
    Remove a HYDRAv2 architecture from the registry.
    
    Args:
        architecture_name: Name of the architecture to remove
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if architecture_name in _hydra_v2_architecture_registry:
            del _hydra_v2_architecture_registry[architecture_name]
            verbose_print(f"Successfully removed {architecture_name}")
            return True
        else:
            verbose_print(f"WARNING: Architecture {architecture_name} not found in registry")
            return False
        
    except Exception as e:
        verbose_print(f"Failed to remove {architecture_name}: {e}")
        return False

def get_hydra_v2_architecture_registry():
    """Get the registry of HYDRAv2 models with optimized architectures."""
    return _hydra_v2_architecture_registry

def load_hydra_v2_architecture_4():
    """
    Load and register HYDRAv2 with architecture 4 (same as branched_wiredcfc_arch4).
    
    Returns:
        True if successful, False otherwise
    """
    import os
    from pathlib import Path
    
    # Try multiple possible paths for architecture 4
    possible_paths = [
        "outputs/architectures/best_architecture_4_trial_178.json",
        "architecture_refinement/outputs/architectures/best_architecture_4_trial_178.json",
    ]
    
    architecture_path = None
    for path in possible_paths:
        full_path = Path(path)
        if full_path.exists():
            architecture_path = str(full_path)
            break
    
    if architecture_path is None:
        verbose_print(f"WARNING: Could not find Architecture 4 file. Tried: {possible_paths}")
        return False
    
    # Register as hydra_v2_arch4 (default HYDRAv2 with architecture 4)
    return add_hydra_v2_architecture("hydra_v2_arch4", architecture_path)

def load_hydra_v2_architectures_from_directory(architectures_dir="outputs/architectures", prefix="hydra_v2_arch"):
    """
    Automatically load all HYDRAv2 architecture files from a directory.
    
    Args:
        architectures_dir: Directory containing architecture JSON files
        prefix: Prefix for architecture names
    
    Returns:
        List of successfully loaded architecture names
    """
    import os
    from pathlib import Path
    
    arch_path = Path(architectures_dir)
    if not arch_path.exists():
        verbose_print(f"Architectures directory not found: {architectures_dir}")
        return []
    
    # Find all best_architecture_*.json files (sorted to get 1-10 in order)
    json_files = sorted(arch_path.glob("best_architecture_*.json"))
    if not json_files:
        verbose_print(f"No architecture JSON files found in {architectures_dir}")
        return []
    
    # verbose_print(f"Found {len(json_files)} architecture files in {architectures_dir}")
    
    loaded_architectures = []
    
    for i, json_file in enumerate(json_files):
        # Generate architecture name (i+1 for 1-based indexing)
        architecture_name = f"{prefix}{i+1}"
        
        # Add to registry
        if add_hydra_v2_architecture(architecture_name, str(json_file)):
            loaded_architectures.append(architecture_name)
    
    # verbose_print(f"Successfully loaded {len(loaded_architectures)} HYDRAv2 architectures")
    return loaded_architectures

# Initialize with some default architectures if they exist
def initialize_default_architectures():
    """Initialize the registry with default architectures if they exist."""
    default_architectures = [
        ("wiredcfc_arch1", "outputs/architectures/best_architecture_1_trial_1.json"),
        ("wiredcfc_arch2", "outputs/architectures/best_architecture_2_trial_6.json"),
        ("wiredcfc_arch3", "outputs/architectures/best_architecture_3_trial_7.json"),
        ("wiredcfc_arch4", "outputs/architectures/best_architecture_4_trial_4.json"),
        ("wiredcfc_arch5", "outputs/architectures/best_architecture_5_trial_2.json"),
    ]
    
    loaded_count = 0
    for name, file_path in default_architectures:
        if os.path.exists(file_path):
            if add_wiredcfc_architecture(name, file_path):
                loaded_count += 1
    
    if loaded_count > 0:
        verbose_print(f"Initialized {loaded_count} default architectures")
    else:
        verbose_print("WARNING: No default architectures found, use load_architectures_from_directory() to load them")

# Try to initialize default architectures
try:
    import os
    # initialize_default_architectures()
    load_architectures_from_directory()
    load_branched_wiredcfc_architectures_from_directory()
    # Load all HYDRAv2 architectures (1-10) from directory
    load_hydra_v2_architectures_from_directory()
except Exception as e:
    verbose_print(f"WARNING: Could not initialize default architectures: {e}")

def classification_num_classes(dataset: str) -> int:
    """Output class count for metrics and EEG classifiers (unified runner)."""
    if dataset == "Lee2019_SSVEP":
        return 4
    if dataset == "Chang2025":
        return 3
    return 2


def get_dataset_sampling_rate(dataset="BNCI2014_001"):
    """
    Get the appropriate sampling rate (Hz) for a given dataset.
    
    Args:
        dataset: Dataset name ("BNCI2014_001", "Lee2019_MI", "Lee2019_SSVEP", "BI2015a")
    
    Returns:
        Sampling rate in Hz (float)
    """
    dataset_rates = {
        "BNCI2014_001": 250.0,      # MOABB provides this dataset at 250 Hz
        "Lee2019_MI": 1000.0,       # OpenBMI MI split (same native rate as Lee2019_SSVEP)
        "Lee2019_SSVEP": 1000.0,    # Native sampling rate is 1000 Hz
        "BI2015a": 512.0,           # Typical ERP datasets are 250 Hz, verify with actual data
        "Shin2017A": 200.0,
        "Chang2025": 1000.0,
        "Yang2025": 1000.0,
    }
    return dataset_rates.get(dataset, 250.0)  # Default to 250 Hz if unknown

def get_paradigm(resample=None, dataset="BNCI2014_001"):
    """
    Get the appropriate paradigm based on dataset.
    
    Args:
        resample: Target sampling rate in Hz. If None, uses dataset-specific default.
        dataset: Dataset name
    
    Returns:
        Configured paradigm instance
    """
    # If resample not specified, use dataset-specific default
    if resample is None:
        resample = get_dataset_sampling_rate(dataset)
    
    if dataset == "Lee2019_SSVEP":
        return SSVEP(
            n_classes=4,
            events=_LEE2019_SSVEP_EVENTS,
            tmin=0.0,
            tmax=4.0,
            baseline=None,
            resample=resample
        )
    elif dataset == "BI2015a":
        return P300(
            fmin=1,
            fmax=24,
            tmin=0.0,
            tmax=1.0,
            baseline=None,  # Use None instead of (None, 0) to avoid the TypeError
            resample=resample
        )
    elif dataset == "Lee2019_MI":
        # Lee2019_MI raw trials are 4 s of MI after cue (dataset interval [0, 4] s). MOABB and
        # Lee et al. (2019) note that *online* decoding in the paper used [1.0, 3.5] s within
        # that window (stable sustained imagery, fewer onset/offset transients). Using the full
        # [0, 4] s window here produced chronically poor valid_acc / underfitting with EEGNet on
        # CrossSession (100 trials per split)—aligning the crop with the paper fixes separability.
        #
        # Bandpass: 8–30 Hz (μ/β MI band; common in MI literature vs default 8–32 in MOABB MotorImagery).
        return MotorImagery(
            events=["left_hand", "right_hand"],
            fmin=8,
            fmax=30,
            tmin=0.5,
            tmax=3.5,
            baseline=None,
            resample=resample,
            n_classes=2,
        )
    elif dataset == "Shin2017A":
        # Feasibility smoke: full 10 s task window per MOABB (tmin/tmax relative to MI event).
        return MotorImagery(
            events=["left_hand", "right_hand"],
            fmin=8,
            fmax=35,
            tmin=0.0,
            tmax=10.0,
            baseline=None,
            resample=resample,
            n_classes=2,
        )
    elif dataset == "Yang2025":
        # Feasibility smoke: 4 s MI segment (exclude cue/rest); 2-class paradigm.
        return MotorImagery(
            events=["left_hand", "right_hand"],
            fmin=8,
            fmax=35,
            tmin=0.0,
            tmax=4.0,
            baseline=None,
            resample=resample,
            n_classes=2,
        )
    elif dataset == "Chang2025":
        # Feasibility smoke: 4 s task interval; MOABB MI paradigm is 3-class.
        return MotorImagery(
            events=["left_hand", "right_hand", "both_hands"],
            fmin=8,
            fmax=35,
            tmin=0.0,
            tmax=4.0,
            baseline=None,
            resample=resample,
            n_classes=3,
        )
    else:  # Default to MotorImagery for BNCI2014_001
        return MotorImagery(
            events=["left_hand", "right_hand"],
            fmin=8, fmax=35,
            tmin=0.0, tmax=None,
            baseline=None,
            resample=resample,
            n_classes=2
        )

MODEL_REGISTRY = get_model_registry()