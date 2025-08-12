"""
Configuration settings for the Architecture Refinement project.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

@dataclass
class GraphGenerationConfig:
    """Configuration for graph generation parameters."""
    # Network size parameters
    min_units: int = 16
    max_units: int = 64
    min_output_size: int = 4
    max_output_size: int = 16
    
    # Modular structure parameters
    min_modules: int = 1
    max_modules: int = 8
    min_module_size: int = 2
    
    # Small-world parameters
    min_rewiring_prob: float = 0.1
    max_rewiring_prob: float = 0.5
    
    # Connection density parameters
    min_connection_density: float = 0.1
    max_connection_density: float = 0.9

    min_k_degree: int = 2
    max_k_degree: int = 6
    min_p_rewiring: float = 0.0
    max_p_rewiring: float = 1.0
    
    # Generation strategy
    num_candidates: int = 1000
    seed: Optional[int] = 42

@dataclass
class TopologyMetricsConfig:
    """Configuration for topology analysis parameters."""
    # Entropy calculation
    entropy_bins: int = 20
    entropy_normalize: bool = True
    
    # Ricci curvature parameters
    ricci_curvature_method: str = "ollivier"  # "ollivier" or "forman"
    ricci_curvature_samples: int = 100
    
    # Algebraic connectivity
    laplacian_normalize: bool = True
    
    # Clustering and path length
    clustering_method: str = "average"  # "average", "global", "local"
    
    # Efficiency metrics
    efficiency_normalize: bool = True

@dataclass
class OptimizationConfig:
    n_trials: int = 100
    timeout: int = 3600
    n_jobs: int = 1
    n_pareto_solutions: int = 10
    
    # Objective weights for the 7-objective optimization
    entropy_weight: float = 0.15
    curvature_weight: float = 0.15
    connectivity_weight: float = 0.15
    efficiency_weight: float = 0.15
    modularity_weight: float = 0.15
    redundancy_weight: float = 0.15
    interpretability_weight: float = 0.10
    
    # Parameter ranges for optimization
    min_units: int = 16
    max_units: int = 64
    min_modules: int = 2
    max_modules: int = 8
    min_rewiring_prob: float = 0.1
    max_rewiring_prob: float = 0.5
    min_connection_density: float = 0.3
    max_connection_density: float = 0.8
    min_module_connectivity: float = 0.4
    max_module_connectivity: float = 0.9
    min_inter_module_connectivity: float = 0.1
    max_inter_module_connectivity: float = 0.6
    
    # Watts-Strogatz specific parameters
    min_k_degree: int = 2
    max_k_degree: int = 6
    min_p_rewiring: float = 0.0
    max_p_rewiring: float = 1.0

@dataclass
class ArchitectureConfig:
    """Configuration for WiredCfC architecture conversion."""
    # Model parameters
    input_size: int = 64
    hidden_size: int = 64
    output_size: int = 8
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    
    # CfC specific parameters
    cfc_tau_fast: float = 0.6
    cfc_tau_slow: float = 0.6
    cfc_adaptive_tau: bool = True

@dataclass
class LoggingConfig:
    """Configuration for logging and visualization."""
    # Logging levels
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Output directories
    output_dir: str = "outputs"
    plots_dir: str = "plots"
    models_dir: str = "models"
    logs_dir: str = "logs"
    
    # Visualization settings
    plot_style: str = "seaborn-v0_8"
    figure_dpi: int = 300
    save_format: str = "png"
    
    # Progress tracking
    use_tqdm: bool = True
    use_rich: bool = True

@dataclass
class ExperimentConfig:
    """Configuration for experiment management."""
    # Experiment identification
    experiment_name: str = "architecture_refinement"
    experiment_id: Optional[str] = None
    
    # Data management
    cache_results: bool = True
    save_intermediate: bool = True
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    # Performance
    use_gpu: bool = False
    num_workers: int = 4

class Config:
    """Main configuration class that aggregates all sub-configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.graph_generation = GraphGenerationConfig()
        self.topology_metrics = TopologyMetricsConfig()
        self.optimization = OptimizationConfig()
        self.architecture = ArchitectureConfig()
        self.logging = LoggingConfig()
        self.experiment = ExperimentConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from a JSON or YAML file."""
        # Implementation for loading from file
        pass
    
    def save_to_file(self, config_path: str):
        """Save configuration to a JSON or YAML file."""
        # Implementation for saving to file
        pass
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Get all output directory paths."""
        base_dir = Path(self.logging.output_dir)
        return {
            "base": base_dir,
            "plots": base_dir / self.logging.plots_dir,
            "models": base_dir / self.logging.models_dir,
            "logs": base_dir / self.logging.logs_dir
        }
    
    def create_output_directories(self):
        """Create all necessary output directories."""
        paths = self.get_output_paths()
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

# Default configuration instance
default_config = Config()
