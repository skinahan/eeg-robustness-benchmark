"""
Multi-objective optimization module using Optuna.
"""

import optuna
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass
import json
from pathlib import Path
from .config import OptimizationConfig
from .topology_analyzer import TopologyAnalyzer
from .graph_generator import ModularSmallWorldGraphGenerator, GraphParameters

@dataclass
class OptimizationResult:
    """Result of a single optimization trial."""
    trial_number: int
    parameters: Dict[str, Any]
    objectives: Dict[str, float]
    robustness_score: float
    graph: Optional[nx.Graph] = None
    metrics: Optional[Dict[str, float]] = None

class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for graph architectures using Optuna.
    
    This class implements optimization strategies that balance multiple
    topological objectives to find robust network architectures.
    """
    
    def __init__(
        self, 
        config: OptimizationConfig,
        graph_generator: ModularSmallWorldGraphGenerator,
        topology_analyzer: TopologyAnalyzer,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the multi-objective optimizer.
        
        Args:
            config: Optimization configuration
            graph_generator: Graph generator instance
            topology_analyzer: Topology analyzer instance
            logger: Optional logger for output
        """
        self.config = config
        self.graph_generator = graph_generator
        self.topology_analyzer = topology_analyzer
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize Optuna study
        self.study = None
        self.results: List[OptimizationResult] = []
        
        # Create output directory
        self.output_dir = Path("outputs/optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def optimize(
        self, 
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        study_name: Optional[str] = None
    ) -> List[OptimizationResult]:
        """
        Run the multi-objective optimization.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Optimization timeout in seconds
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary containing optimization results
        """
        n_trials = n_trials or self.config.n_trials
        timeout = timeout or self.config.timeout
        
        if self.logger:
            self.logger.info(f"Starting optimization with {n_trials} trials, timeout: {timeout}s")
            self.logger.info("Using Watts-Strogatz flex graph generation for expanded search space")
            self.logger.info("Objectives: maximize entropy and Ollivier-Ricci curvature")
            self.logger.info("Optimizing over: units, k_degree, p_rewiring, clustering coefficient, path length, and other graph parameters")
            self.logger.info(f"Using consistent seeds for reproducibility (base seed: 42)")
        
        # Create study with 2 objectives: entropy, curvature
        if not study_name:
            study_name = f"architecture_optimization_{np.random.randint(10000, 99999)}"
            
        self.study = optuna.create_study(
            study_name=study_name,
            storage=None,
            sampler=optuna.samplers.TPESampler(seed=42),
            directions=["maximize"] * 2  # Maximize entropy and curvature
        )
        
        # Run optimization
        self.study.optimize(
            func=self._objective_function,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        if self.logger:
            self.logger.info(f"Optimization completed. Best trials: {self.study.best_trials}")
            self.logger.info(f"All trials used consistent seeds (trial number as seed) for reproducibility")
            
        return self._process_results()

    def _objective_function(self, trial: optuna.Trial) -> Tuple[float, float]:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Tuple of objective values (entropy, curvature)
        """
        try:
            # Suggest parameters for this trial
            params = self._suggest_parameters(trial)
            
            # Generate Watts-Strogatz flex graph with suggested parameters
            graph = self.graph_generator._create_watts_strogatz_flex_graph(
                params.units, 
                params.k_degree,  # Use Optuna-suggested k_degree
                params.p_rewiring,  # Use Optuna-suggested p_rewiring
                params.target_clustering,  # Use Optuna-suggested clustering
                params.target_path_length,  # Use Optuna-suggested path length
                params.seed  # Pass the seed for reproducibility
            )
            
            # Analyze graph topology
            metrics = self.topology_analyzer.analyze_graph(graph)
            
            # Extract objective values using canonical TE/ORC definitions.
            # - TE is already normalized to [0,1] by definition.
            # - ORC is signed; we maximize it directly (no abs).
            entropy = float(np.clip(metrics.get('te', 0.0), 0.0, 1.0))
            curvature = float(metrics.get('orc', metrics.get('avg_ricci_curvature', 0.0)))
            
            # Store results
            self.results.append(OptimizationResult(
                trial_number=trial.number,
                parameters=self._extract_trial_parameters(trial),
                objectives={
                    'entropy': entropy,
                    'curvature': curvature,
                },
                robustness_score=self.topology_analyzer.compute_robustness_score(metrics),
                graph=graph,
                metrics=metrics
            ))
            
            if self.logger and trial.number % 10 == 0:
                self.logger.info(f"Trial {trial.number}: entropy={entropy:.3f}, curvature={curvature:.3f}, "
                               f"k={params.k_degree}, p={params.p_rewiring:.3f}, "
                               f"clustering={params.target_clustering:.3f}, path_length={params.target_path_length:.3f}, "
                               f"seed={params.seed}")
            
            return entropy, curvature
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in trial {trial.number}: {e}")
            # Return worst possible values for failed trials
            return 0.0, 0.0

    def _suggest_parameters(self, trial: optuna.Trial) -> GraphParameters:
        """Suggest parameters for a trial using Optuna's parameter suggestion."""
        # Network size parameters - expand the range for better search
        units = trial.suggest_int(
            'units', 
            self.graph_generator.config.min_units, 
            self.graph_generator.config.max_units
        )
        
        output_size = trial.suggest_int(
            'output_size', 
            max(2, units // 10), 
            min(units // 3, 20)
        )
        
        # n_modules = trial.suggest_int(
        #     'n_modules', 
        #     2, 
        #     min(12, units // 6)
        # )
        
        # Watts-Strogatz specific parameters
        rewiring_prob = trial.suggest_float(
            'rewiring_prob', 
            0.05,  # Very regular
            0.8    # Very random
        )
        
        # connection_density = trial.suggest_float(
        #     'connection_density', 
        #     0.10,  # Sparse
        #     0.95   # Dense
        # )
        
        # Additional parameters for flex graphs
        # module_connectivity = trial.suggest_float('module_connectivity', 0.2, 0.95)
        # inter_module_connectivity = trial.suggest_float('inter_module_connectivity', 0.05, 0.8)
        
        # Watts-Strogatz flex specific parameters
        target_clustering = trial.suggest_float('target_clustering', 0.01, 1.0)
        target_path_length = trial.suggest_float('target_path_length', 1.0, 4.5)
        
        # Core Watts-Strogatz parameters
        k_degree = trial.suggest_int('k_degree', self.config.min_k_degree, self.config.max_k_degree)
        p_rewiring = trial.suggest_float('p_rewiring', self.config.min_p_rewiring, self.config.max_p_rewiring)
        
        return GraphParameters(
            units=units,
            output_size=output_size,
            # n_modules=n_modules,
            rewiring_prob=rewiring_prob,
            # connection_density=connection_density,
            # module_connectivity=module_connectivity,
            # inter_module_connectivity=inter_module_connectivity,
            target_clustering=target_clustering,
            target_path_length=target_path_length,
            k_degree=k_degree,
            p_rewiring=p_rewiring,
            seed=trial.number  # Use trial number as seed for reproducibility
        )

    def _extract_trial_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Extract all parameters from a trial."""
        return {
            'units': trial.params.get('units'),
            'output_size': trial.params.get('output_size'),
            'n_modules': trial.params.get('n_modules'),
            'rewiring_prob': trial.params.get('rewiring_prob'),
            'connection_density': trial.params.get('connection_density'),
            # 'module_connectivity': trial.params.get('module_connectivity'),
            # 'inter_module_connectivity': trial.params.get('inter_module_connectivity'),
            'target_clustering': trial.params.get('target_clustering'),
            'target_path_length': trial.params.get('target_path_length'),
            'k_degree': trial.params.get('k_degree'),
            'p_rewiring': trial.params.get('p_rewiring'),
            'seed': trial.number  # Include the seed for reproducibility
        }

    def _process_results(self) -> List[OptimizationResult]:
        """Process and organize optimization results."""
        if not self.results:
            return []
            
        if self.logger:
            self.logger.info(f"Processing {len(self.results)} optimization results")
            
        # Find Pareto optimal solutions
        pareto_solutions = self._find_pareto_optimal_solutions()
        
        if self.logger:
            self.logger.info(f"Found {len(pareto_solutions)} Pareto optimal solutions")
            
        return self.results

    def _find_pareto_optimal_solutions(self) -> List[Dict[str, Any]]:
        """Find Pareto optimal solutions from the results."""
        if not self.results:
            return []
            
        pareto_solutions = []
        
        for result in self.results:
            is_pareto = True
            
            for other_result in self.results:
                if result.trial_number == other_result.trial_number:
                    continue
                    
                # Check if other_result dominates this result
                if self._dominates(other_result.objectives, result.objectives):
                    is_pareto = False
                    break
                    
            if is_pareto:
                pareto_solutions.append({
                    'trial_number': result.trial_number,
                    'objectives': result.objectives,
                    'robustness_score': result.robustness_score,
                    'parameters': result.parameters
                })
                
        return pareto_solutions

    def _dominates(self, obj1: Dict[str, float], obj2: Dict[str, float]) -> bool:
        """Check if obj1 dominates obj2 (all objectives are at least as good, at least one is better)."""
        at_least_as_good = all(obj1[key] >= obj2[key] for key in obj1.keys())
        at_least_one_better = any(obj1[key] > obj2[key] for key in obj1.keys())
        return at_least_as_good and at_least_one_better

    def _compute_optimization_statistics(self) -> Dict[str, Any]:
        """Compute statistics across all optimization trials."""
        if not self.results:
            return {}
        
        # Extract successful trials
        successful_results = [r for r in self.results if r.robustness_score > 0]
        
        if not successful_results:
            return {}
        
        # Compute statistics for each objective
        objectives = ['entropy', 'curvature']
        statistics = {}
        
        for obj in objectives:
            values = [r.objectives.get(obj, 0.0) for r in successful_results]
            statistics[f'{obj}_stats'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        # Robustness score statistics
        robustness_scores = [r.robustness_score for r in successful_results]
        statistics['robustness_stats'] = {
            'mean': float(np.mean(robustness_scores)),
            'std': float(np.std(robustness_scores)),
            'min': float(np.min(robustness_scores)),
            'max': float(np.max(robustness_scores)),
            'median': float(np.median(robustness_scores))
        }
        
        # Parameter statistics
        # parameters = ['units', 'output_size', 'n_modules', 'rewiring_prob', 'connection_density', 'module_connectivity', 'inter_module_connectivity', 'target_clustering', 'target_path_length', 'k_degree', 'p_rewiring']
        # parameters = ['units', 'output_size', 'rewiring_prob', 'connection_density', 'target_clustering', 'target_path_length', 'k_degree', 'p_rewiring']
        parameters = ['units', 'output_size', 'rewiring_prob', 'target_clustering', 'target_path_length', 'k_degree', 'p_rewiring']
        for param in parameters:
            values = [r.parameters.get(param, 0) for r in successful_results]
            if values:
                statistics[f'{param}_stats'] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        return statistics

    def save_results(self, filename: str) -> None:
        """
        Save optimization results to file.
        
        Args:
            filename: Base filename for saving
            
        Returns:
            Path to saved file
        """
        if not self.results:
            if self.logger:
                self.logger.warning("No results to save")
            return
        
        # Prepare data for saving
        save_data = {
            'config': {
                'n_trials': len(self.results),
                'optimization_params': {
                    'min_units': self.config.min_units,
                    'max_units': self.config.max_units,
                    'rewiring_prob_range': [0.05, 0.8],
                    'connection_density_range': [0.15, 0.95],
                    # 'module_connectivity_range': [0.2, 0.95],
                    # 'inter_module_connectivity_range': [0.05, 0.8],
                    'target_clustering_range': [0.1, 0.9],
                    'target_path_length_range': [2.0, 8.0],
                    'k_degree_range': [self.config.min_k_degree, self.config.max_k_degree],
                    'p_rewiring_range': [self.config.min_p_rewiring, self.config.max_p_rewiring]
                }
            },
            'statistics': self._compute_optimization_statistics(),
            'pareto_solutions': self._find_pareto_optimal_solutions(),
            'all_results': [
                {
                    'trial_number': r.trial_number,
                    'parameters': r.parameters,
                    'objectives': r.objectives,
                    'robustness_score': r.robustness_score
                }
                for r in self.results
            ]
        }
        
        filepath = self.output_dir / f"{filename}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
                
            if self.logger:
                self.logger.info(f"Results saved to {filepath}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving results: {e}")

    def load_results(self, filename: str) -> bool:
        """
        Load optimization results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Loaded results dictionary
        """
        filepath = self.output_dir / f"{filename}.json"
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct results (graphs won't be loaded)
            self.results = []
            for result_data in data.get('all_results', []):
                result = OptimizationResult(
                    trial_number=result_data['trial_number'],
                    parameters=result_data['parameters'],
                    objectives=result_data['objectives'],
                    robustness_score=result_data['robustness_score']
                )
                self.results.append(result)
                
            if self.logger:
                self.logger.info(f"Loaded {len(self.results)} results from {filepath}")
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading results: {e}")
            return False

    def get_best_solutions(self, n_solutions: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the best solutions based on robustness score.
        
        Args:
            n_solutions: Number of top solutions to return
            
        Returns:
            List of best solutions
        """
        if not self.results:
            return []
            
        n_solutions = n_solutions or self.config.n_pareto_solutions
        
        # Sort by robustness score
        sorted_results = sorted(self.results, key=lambda x: x.robustness_score, reverse=True)
        
        return [
            {
                'trial_number': r.trial_number,
                'objectives': r.objectives,
                'robustness_score': r.robustness_score,
                'parameters': r.parameters
            }
            for r in sorted_results[:n_solutions]
        ]

    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot optimization history and results.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.study:
            if self.logger:
                self.logger.warning("No study available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Create subplots for each objective
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle('Optimization History for All Objectives')
            
            objective_names = ['entropy', 'curvature']
            
            for i, objective in enumerate(objective_names):
                ax = axes[i]
                
                # Extract values for this objective
                values = [r.objectives.get(objective, 0.0) for r in self.results]
                trials = list(range(len(values)))
                
                ax.plot(trials, values, 'b-', alpha=0.7)
                ax.set_title(f'{objective.title()} Objective')
                ax.set_xlabel('Trial')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if self.logger:
                    self.logger.info(f"Optimization history plot saved to {save_path}")
                    
            plt.show()
            
        except ImportError:
            if self.logger:
                self.logger.warning("matplotlib not available for plotting")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error plotting optimization history: {e}")

    def plot_pareto_front(self, save_path: Optional[str] = None) -> None:
        """
        Plot Pareto front from optimization results.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.results:
            if self.logger:
                self.logger.warning("No results available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Create subplots for different objective pairs
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle('Pareto Fronts for Objective Pairs')
            
            # Define objective pairs to plot
            objective_pairs = [
                ('entropy', 'curvature'),
            ]
            
            for i, (obj1, obj2) in enumerate(objective_pairs):
                ax = axes[i]
                
                # Extract values for this pair
                x_values = [r.objectives.get(obj1, 0.0) for r in self.results]
                y_values = [r.objectives.get(obj2, 0.0) for r in self.results]
                robustness_scores = [r.robustness_score for r in self.results]
                
                # Color by robustness score
                scatter = ax.scatter(x_values, y_values, c=robustness_scores, cmap='viridis', alpha=0.7)
                ax.set_xlabel(obj1.title())
                ax.set_ylabel(obj2.title())
                ax.set_title(f'{obj1.title()} vs {obj2.title()}')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax, label='Robustness Score')
                
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if self.logger:
                    self.logger.info(f"Pareto front plot saved to {save_path}")
                    
            plt.show()
            
        except ImportError:
            if self.logger:
                self.logger.warning("matplotlib not available for plotting")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error plotting Pareto front: {e}")

    def export_best_graphs(self, output_dir: str) -> None:
        """
        Export the best graphs to files.
        
        Args:
            output_dir: Directory to save the graphs
            
        Returns:
            List of saved graph file paths
        """
        if not self.results:
            if self.logger:
                self.logger.warning("No results to export")
            return
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get best solutions
        best_solutions = self.get_best_solutions()
        
        for i, solution in enumerate(best_solutions):
            trial_number = solution['trial_number']
            
            # Find the corresponding result with graph
            result = next((r for r in self.results if r.trial_number == trial_number), None)
            if result and result.graph:
                # Save graph as GraphML
                graph_file = output_path / f"best_graph_{i+1}_trial_{trial_number}.graphml"
                nx.write_graphml(result.graph, graph_file)
                
                # Save parameters
                param_file = output_path / f"best_graph_{i+1}_trial_{trial_number}_params.json"
                with open(param_file, 'w') as f:
                    json.dump(solution, f, indent=2)
                    
        if self.logger:
            self.logger.info(f"Exported {len(best_solutions)} best graphs to {output_path}")
