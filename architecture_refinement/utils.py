"""
Utility functions for the Architecture Refinement project.
"""

import logging
import os
import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table
from rich.panel import Panel
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True
) -> logging.Logger:
    """
    Set up logging with optional rich formatting and file output.
    
    Args:
        level: Logging level
        log_file: Optional file path for logging
        use_rich: Whether to use rich formatting
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("architecture_refinement")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if use_rich:
        console_handler = RichHandler(rich_tracebacks=True)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
    
    return logger

def create_experiment_logger(
    experiment_name: str,
    output_dir: str = "outputs",
    use_rich: bool = True
) -> logging.Logger:
    """
    Create a logger specifically for an experiment with file output.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Base output directory
        use_rich: Whether to use rich formatting
        
    Returns:
        Configured experiment logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_path = Path(output_dir) / "logs" / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    return setup_logging(
        level="INFO",
        log_file=str(log_path),
        use_rich=use_rich
    )

def save_results(
    results: Dict[str, Any],
    filename: str,
    output_dir: str = "outputs",
    format: str = "json"
) -> str:
    """
    Save results to file in specified format.
    
    Args:
        results: Results dictionary to save
        filename: Name of the file (without extension)
        output_dir: Output directory
        format: File format ('json', 'pickle', 'csv')
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        file_path = output_path / f"{filename}.json"
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == "pickle":
        file_path = output_path / f"{filename}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
    elif format == "csv":
        file_path = output_path / f"{filename}.csv"
        if isinstance(results, dict):
            # Convert dict to DataFrame if possible
            df = pd.DataFrame([results])
        else:
            df = pd.DataFrame(results)
        df.to_csv(file_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return str(file_path)

def load_results(
    filepath: str,
    format: str = "auto"
) -> Any:
    """
    Load results from file.
    
    Args:
        filepath: Path to the file
        format: File format ('auto', 'json', 'pickle', 'csv')
        
    Returns:
        Loaded data
    """
    file_path = Path(filepath)
    
    if format == "auto":
        format = file_path.suffix[1:]
    
    if format == "json":
        with open(file_path, 'r') as f:
            return json.load(f)
    elif format == "pickle":
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif format == "csv":
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

def create_visualization_style():
    """Set up matplotlib and seaborn visualization style."""
    # Use default matplotlib style for better readability
    plt.style.use('default')
    # Only set basic DPI for high-quality output
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

def plot_optimization_history(
    study: Any,
    output_path: str,
    title: str = "Optimization History"
) -> None:
    """
    Plot optimization history from Optuna study.
    
    Args:
        study: Optuna study object
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title)
    
    # Optimization history
    optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0, 0])
    
    # Parameter importance
    optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[0, 1])
    
    # Parameter relationships
    optuna.visualization.matplotlib.plot_parallel_coordinate(study, ax=axes[1, 0])
    
    # Parameter contour
    optuna.visualization.matplotlib.plot_contour(study, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pareto_front(
    pareto_solutions: List[Dict[str, float]],
    output_path: str,
    title: str = "Pareto Front"
) -> None:
    """
    Plot Pareto front from multi-objective optimization.
    
    Args:
        pareto_solutions: List of Pareto optimal solutions
        output_path: Path to save the plot
        title: Plot title
    """
    if len(pareto_solutions) < 2:
        print("Need at least 2 solutions to plot Pareto front")
        return
    
    # Extract objective values
    objectives = list(pareto_solutions[0].keys())
    if len(objectives) < 2:
        print("Need at least 2 objectives to plot Pareto front")
        return
    
    # Create subplots for different objective combinations
    n_objectives = len(objectives)
    n_plots = n_objectives * (n_objectives - 1) // 2
    
    if n_plots <= 4:
        cols = 2
        rows = (n_plots + 1) // 2
    else:
        cols = 3
        rows = (n_plots + 2) // 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    for i in range(n_objectives):
        for j in range(i + 1, n_objectives):
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                
                # Extract values for this pair of objectives
                x_vals = [sol[objectives[i]] for sol in pareto_solutions]
                y_vals = [sol[objectives[j]] for sol in pareto_solutions]
                
                ax.scatter(x_vals, y_vals, alpha=0.7, s=50)
                ax.set_xlabel(objectives[i])
                ax.set_ylabel(objectives[j])
                ax.set_title(f"{objectives[i]} vs {objectives[j]}")
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_rich_table(
    data: List[Dict[str, Any]],
    title: str = "Results Table"
) -> Table:
    """
    Create a rich table from data.
    
    Args:
        data: List of dictionaries containing the data
        title: Table title
        
    Returns:
        Rich table object
    """
    if not data:
        return Table(title=title, show_header=True, header_style="bold magenta")
    
    # Get column names from first row
    columns = list(data[0].keys())
    
    # Create table
    table = Table(title=title, show_header=True, header_style="bold magenta")
    
    # Add columns
    for col in columns:
        table.add_column(col, style="cyan")
    
    # Add rows
    for row in data:
        table.add_row(*[str(row.get(col, "")) for col in columns])
    
    return table

def print_experiment_summary(
    config: Any,
    results: List[Any], # Changed from Dict[str, Any] to List[Any]
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Print a summary of the experiment configuration and results.
    
    Args:
        config: Configuration object
        results: Results dictionary
        logger: Optional logger for output
    """
    console = Console()
    
    # Configuration summary
    config_panel = Panel(
        f"Experiment: {config.experiment.experiment_name}\n"
        f"Trials: {config.optimization.n_trials}\n"
        f"Timeout: {config.optimization.timeout}s\n"
        f"Seed: {config.experiment.random_seed}",
        title="Configuration",
        border_style="blue"
    )
    
    # Results summary
    if results:
        # Handle OptimizationResult objects
        if hasattr(results[0], 'objectives') and hasattr(results[0], 'parameters'):
            # Extract best values from OptimizationResult objects
            best_entropy = max([r.objectives.get('entropy', 0.0) for r in results if hasattr(r, 'objectives')], default=0.0)
            best_curvature = max([r.objectives.get('curvature', 0.0) for r in results if hasattr(r, 'objectives')], default=0.0)
            best_connectivity = max([r.objectives.get('connectivity', 0.0) for r in results if hasattr(r, 'objectives')], default=0.0)
            
            results_panel = Panel(
                f"Solutions found: {len(results)}\n"
                f"Best entropy: {best_entropy:.4f}\n"
                f"Best curvature: {best_curvature:.4f}\n"
                f"Best connectivity: {best_connectivity:.4f}",
                title="Results Summary",
                border_style="green"
            )
        else:
            # Fallback for other result types
            results_panel = Panel(
                f"Results count: {len(results)}\n"
                f"Result type: {type(results[0]) if results else 'None'}",
                title="Results Summary",
                border_style="green"
            )
    else:
        results_panel = Panel("No results available", title="Results Summary", border_style="red")
    
    console.print(config_panel)
    console.print(results_panel)
    
    if logger:
        logger.info("Experiment summary printed to console")

def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to set deterministic behavior
    """
    np.random.seed(seed)
    random.seed(seed)
    
    if deterministic:
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # For PyTorch
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of numerical values.
    
    Args:
        data: List of numerical values
        
    Returns:
        Dictionary containing statistics
    """
    if not data:
        return {}
    
    data_array = np.array(data)
    
    return {
        "mean": float(np.mean(data_array)),
        "std": float(np.std(data_array)),
        "min": float(np.min(data_array)),
        "max": float(np.max(data_array)),
        "median": float(np.median(data_array)),
        "q25": float(np.percentile(data_array, 25)),
        "q75": float(np.percentile(data_array, 75))
    }

def validate_config(config: Any) -> List[str]:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Graph generation validation
    if config.graph_generation.min_units >= config.graph_generation.max_units:
        errors.append("min_units must be less than max_units")
    
    if config.graph_generation.min_output_size >= config.graph_generation.max_output_size:
        errors.append("min_output_size must be less than max_output_size")
    
    if config.graph_generation.min_modules >= config.graph_generation.max_modules:
        errors.append("min_modules must be less than max_modules")
    
    # Optimization validation
    if config.optimization.n_trials <= 0:
        errors.append("n_trials must be positive")
    
    if config.optimization.timeout <= 0:
        errors.append("timeout must be positive")
    
    # Weights validation
    total_weight = (config.optimization.entropy_weight + 
                   config.optimization.curvature_weight + 
                   config.optimization.connectivity_weight + 
                   config.optimization.efficiency_weight)
    
    if abs(total_weight - 1.0) > 1e-6:
        errors.append(f"Optimization weights must sum to 1.0, got {total_weight}")
    
    return errors
