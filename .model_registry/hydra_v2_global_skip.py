"""
Auto-generated registration file for hydra_v2_global_skip
"""
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from config import MODEL_REGISTRY, get_model_registry, _runtime_model_registry
from models.hydra import create_hydra_v2_classifier
from architecture_refinement.arbitrary_wiring import load_architecture_from_file
from project_paths import resolve_architecture_json_path

# Load wiring
wiring = load_architecture_from_file(str(resolve_architecture_json_path()))

# Feature configuration
feature_config = {'attn_dropout': 0.0,
 'num_attn_queries': 1,
 'use_adaptive_residual': False,
 'use_cross_bin_context': False,
 'use_erp_head': False,
 'use_global_skip': True,
 'use_ssvep_head': False}

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
_runtime_model_registry["hydra_v2_global_skip"] = factory

# Also register in MODEL_REGISTRY for backward compatibility
MODEL_REGISTRY["hydra_v2_global_skip"] = factory
