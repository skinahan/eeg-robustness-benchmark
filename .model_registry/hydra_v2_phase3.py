"""
Auto-generated registration file for hydra_v2_phase3
"""
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from config import MODEL_REGISTRY, get_model_registry, _runtime_model_registry
from models.hydra import create_hydra_v2_classifier
from architecture_refinement.arbitrary_wiring import load_architecture_from_file

# Load wiring
wiring = load_architecture_from_file(r"E:\Research\Dissertation\full_backup_7_16_2025\moabb_experiments\outputs\architectures\best_architecture_4_trial_178.json")

# Feature configuration
feature_config = {'attn_dropout': 0.1,
 'num_attn_queries': 4,
 'use_adaptive_residual': True,
 'use_cross_bin_context': True,
 'use_erp_head': True,
 'use_global_skip': True,
 'use_ssvep_head': True}

# Create factory
def factory(n_chans, n_times, n_outputs, **kwargs):
    merged_kwargs = {**kwargs, **feature_config}
    return create_hydra_v2_classifier(
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        wiring=wiring,
        **merged_kwargs
    )

# Register in runtime registry (used by get_model_registry())
# This is the persistent registry that get_model_registry() checks
_runtime_model_registry["hydra_v2_phase3"] = factory

# Also register in MODEL_REGISTRY for backward compatibility
MODEL_REGISTRY["hydra_v2_phase3"] = factory
