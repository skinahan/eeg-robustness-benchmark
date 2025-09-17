from moabb.datasets import BNCI2014_001, Lee2019_SSVEP
from moabb.paradigms import MotorImagery, SSVEP
from models.eegnet import create_eegnet_classifier
from models.reegnet import create_reegnet_classifier
from models.cnnncp import create_cnnncp_classifier, create_cnnncfc_v2_classifier, create_cnnncfc_compact_classifier, create_cnnsmallworld_classifier, create_cnnwiredcfc_classifier
from models.sppncp import create_sppncp_classifier

# Import the integration functions for CNNWiredCfC models
from architecture_refinement.integrate_with_evaluation import create_model_factory_from_architecture

# Centralized experiment configuration

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

# SSVEP paradigm for Lee2019_SSVEP dataset
DEFAULT_SSVEP_PARADIGM = SSVEP(
    n_classes=4,
    tmin=0.0,
    tmax=4.0,
    baseline=None,
    resample=None
)
DEFAULT_SEED = 42

# Module-level registry for CNNWiredCfC architectures
_wiredcfc_architecture_registry = {}

# Base model registry without CNNWiredCfC architectures
def get_base_model_registry():
    """Get the base model registry with standard models."""
    return {
        "eegnet": create_eegnet_classifier,
        "reegnet": create_reegnet_classifier,
        "cnn_ncp": create_cnnncp_classifier,
        "cnncfc_v2": create_cnnncfc_v2_classifier,
        "cnncfc_compact": create_cnnncfc_compact_classifier,
        "cnn_smallworld": create_cnnsmallworld_classifier
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
        
        print(f"Successfully added {architecture_name} from {architecture_file_path}")
        return True
        
    except Exception as e:
        print(f"Failed to add {architecture_name}: {e}")
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
            print(f"Successfully removed {architecture_name}")
            return True
        else:
            print(f"WARNING: Architecture {architecture_name} not found in registry")
            return False
        
    except Exception as e:
        print(f"Failed to remove {architecture_name}: {e}")
        return False

def list_wiredcfc_architectures():
    """List all registered CNNWiredCfC architectures."""
    if not _wiredcfc_architecture_registry:
        print("No CNNWiredCfC architectures registered")
        return []
    
    print("Registered CNNWiredCfC architectures:")
    for name, info in _wiredcfc_architecture_registry.items():
        print(f" {name}: {info['file_path']}")
    
    return list(_wiredcfc_architecture_registry.keys())

def get_model_registry():
    """Get the complete model registry including dynamic CNNWiredCfC architectures."""
    # Start with base models
    registry = get_base_model_registry()
    
    # Add dynamic CNNWiredCfC architectures
    for name, info in _wiredcfc_architecture_registry.items():
        registry[name] = info['factory']
    
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
        print(f"Architectures directory not found: {architectures_dir}")
        return []
    
    # Find all JSON files
    json_files = list(arch_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {architectures_dir}")
        return []
    
    print(f"Found {len(json_files)} architecture files in {architectures_dir}")
    
    loaded_architectures = []
    
    for i, json_file in enumerate(json_files):
        # Generate architecture name
        architecture_name = f"{prefix}{i+1}"
        
        # Add to registry
        if add_wiredcfc_architecture(architecture_name, str(json_file)):
            loaded_architectures.append(architecture_name)
    
    print(f"Successfully loaded {len(loaded_architectures)} architectures")
    return loaded_architectures

def clear_wiredcfc_architectures():
    """Clear all registered CNNWiredCfC architectures."""
    _wiredcfc_architecture_registry.clear()
    print("Cleared all CNNWiredCfC architectures")

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
        print(f"Initialized {loaded_count} default architectures")
    else:
        print("WARNING: No default architectures found, use load_architectures_from_directory() to load them")

# Try to initialize default architectures
try:
 import os
 # initialize_default_architectures()
 load_architectures_from_directory()
except Exception as e:
 print(f"WARNING: Could not initialize default architectures: {e}")

def get_paradigm(resample=None, dataset="BNCI2014_001"):
    """Get the appropriate paradigm based on dataset."""
    if dataset == "Lee2019_SSVEP":
        return SSVEP(
            n_classes=4,
            tmin=0.0,
            tmax=4.0,
            baseline=None,
            resample=resample
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