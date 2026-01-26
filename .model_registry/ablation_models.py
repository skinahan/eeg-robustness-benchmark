"""
Registration file for all ablation study model variants.

This file registers all ablation model variants so they can be used with
the unified experiment runner and experiment automation system.
"""
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from config import MODEL_REGISTRY, get_model_registry, _runtime_model_registry, add_branched_wiredcfc_architecture
from architecture_refinement.arbitrary_wiring import load_architecture_from_file
from ablations.ablation_models import (
    create_branched_wiredcfc_no_carry_gate_classifier,
    create_branched_wiredcfc_no_branching_classifier,
    create_branched_lstm_classifier,
    create_branched_wiredcfc_no_snr_gate_classifier,
    create_branched_wiredcfc_no_carry_gate_no_branching_classifier,
    create_branched_wiredcfc_no_carry_gate_no_snr_gate_classifier,
    create_branched_wiredcfc_no_branching_no_snr_gate_classifier,
    create_branched_wiredcfc_no_carry_gate_no_branching_no_snr_gate_classifier,
    create_branched_lstm_no_carry_gate_classifier,
    create_branched_lstm_no_branching_classifier,
    create_branched_lstm_no_snr_gate_classifier,
    create_branched_lstm_no_carry_gate_no_branching_classifier,
    create_branched_lstm_no_carry_gate_no_snr_gate_classifier,
    create_branched_lstm_no_branching_no_snr_gate_classifier,
    create_branched_lstm_no_carry_gate_no_branching_no_snr_gate_classifier,
)

# Try to find architecture file in common locations
possible_paths = [
    script_dir / "outputs" / "architectures" / "best_architecture_4_trial_178.json",
    script_dir / "architecture_refinement" / "outputs" / "architectures" / "best_architecture_4_trial_178.json",
]

architecture_path = None
for path in possible_paths:
    if path.exists():
        architecture_path = str(path)
        break

if architecture_path is None:
    raise FileNotFoundError(
        f"Could not find architecture file. Tried: {[str(p) for p in possible_paths]}"
    )

# Load wiring for CFC-based models
wiring = load_architecture_from_file(architecture_path)
if wiring is None:
    raise RuntimeError("Failed to load wiring from architecture file")

# Register baseline model (if not already registered)
if "branched_wiredcfc_arch4" not in get_model_registry():
    add_branched_wiredcfc_architecture("branched_wiredcfc_arch4", wiring)

# Define all ablation model configurations
# Format: (model_name, factory_function, requires_wiring)
ablation_configs = [
    # Single ablations (CfC-based)
    ("branched_wiredcfc_arch4_no_carry_gate", create_branched_wiredcfc_no_carry_gate_classifier, True),
    ("branched_wiredcfc_arch4_no_branching", create_branched_wiredcfc_no_branching_classifier, True),
    ("branched_wiredcfc_arch4_no_snr_gate", create_branched_wiredcfc_no_snr_gate_classifier, True),
    
    # LSTM baseline
    ("branched_lstm_arch4_equivalent", create_branched_lstm_classifier, False),
    
    # Combination ablations (CfC-based)
    ("branched_wiredcfc_arch4_no_carry_gate_no_branching", create_branched_wiredcfc_no_carry_gate_no_branching_classifier, True),
    ("branched_wiredcfc_arch4_no_carry_gate_no_snr_gate", create_branched_wiredcfc_no_carry_gate_no_snr_gate_classifier, True),
    ("branched_wiredcfc_arch4_no_branching_no_snr_gate", create_branched_wiredcfc_no_branching_no_snr_gate_classifier, True),
    ("branched_wiredcfc_arch4_no_carry_gate_no_branching_no_snr_gate", create_branched_wiredcfc_no_carry_gate_no_branching_no_snr_gate_classifier, True),
    
    # Combination ablations (LSTM-based)
    ("branched_lstm_arch4_no_carry_gate", create_branched_lstm_no_carry_gate_classifier, False),
    ("branched_lstm_arch4_no_branching", create_branched_lstm_no_branching_classifier, False),
    ("branched_lstm_arch4_no_snr_gate", create_branched_lstm_no_snr_gate_classifier, False),
    ("branched_lstm_arch4_no_carry_gate_no_branching", create_branched_lstm_no_carry_gate_no_branching_classifier, False),
    ("branched_lstm_arch4_no_carry_gate_no_snr_gate", create_branched_lstm_no_carry_gate_no_snr_gate_classifier, False),
    ("branched_lstm_arch4_no_branching_no_snr_gate", create_branched_lstm_no_branching_no_snr_gate_classifier, False),
    ("branched_lstm_arch4_no_carry_gate_no_branching_no_snr_gate", create_branched_lstm_no_carry_gate_no_branching_no_snr_gate_classifier, False),
]

# Register all ablation models
for model_name, factory_func, requires_wiring in ablation_configs:
    try:
        if requires_wiring:
            # Create factory with wiring closure (use lambda with default arg to capture wiring correctly)
            def make_factory_with_wiring(wiring_ref, factory):
                def factory_wrapped(n_chans, n_times, n_outputs, **kwargs):
                    return factory(n_chans, n_times, n_outputs, wiring_ref, **kwargs)
                return factory_wrapped
            factory_wrapped = make_factory_with_wiring(wiring, factory_func)
        else:
            # LSTM models don't need wiring
            factory_wrapped = factory_func
        
        # Register in runtime registry (used by get_model_registry())
        _runtime_model_registry[model_name] = factory_wrapped
        
        # Also register in MODEL_REGISTRY for backward compatibility
        MODEL_REGISTRY[model_name] = factory_wrapped
        
    except Exception as e:
        raise RuntimeError(f"Failed to register {model_name}: {e}")
