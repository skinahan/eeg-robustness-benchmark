"""
Architecture Refinement for Robustness-Aware CfC Networks

This package implements a comprehensive system for optimizing CfC network architectures
using graph-theoretic metrics and multi-objective optimization. The system generates
candidate wiring graphs, evaluates them using topological metrics, and optimizes
them using Optuna for downstream training.

Key Components:
- Graph generation using modular small-world strategies
- Topological metric computation (entropy, curvature, connectivity)
- Multi-objective optimization with Optuna
- WiredCfC architecture conversion
- Comprehensive logging and visualization
"""

__version__ = "0.1.0"
__author__ = "EEG Noise Robustness Team"

from .graph_generator import ModularSmallWorldGraphGenerator
from .topology_analyzer import TopologyAnalyzer
from .optimizer import MultiObjectiveOptimizer
from .utils import setup_logging, create_experiment_logger

__all__ = [
    "ModularSmallWorldGraphGenerator",
    "TopologyAnalyzer",
    "MultiObjectiveOptimizer",
    "WiredCfCConverter",
    "setup_logging",
    "create_experiment_logger",
]


def __getattr__(name: str):
    """Lazy import: WiredCfCConverter pulls in Torch; keep package import light."""
    if name == "WiredCfCConverter":
        from .architecture_converter import WiredCfCConverter

        return WiredCfCConverter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
