#!/usr/bin/env python3
"""
Demo script for the Architecture Refinement system.
This script demonstrates the complete pipeline from graph generation to optimization.
"""

import sys
from pathlib import Path
import numpy as np
import logging
from typing import List, Dict, Any
import random # Added for random seed setting
import networkx as nx

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from architecture_refinement.config import Config
from architecture_refinement.utils import (
    setup_logging, create_experiment_logger, set_random_seed, 
    create_visualization_style, print_experiment_summary
)
from architecture_refinement.graph_generator import ModularSmallWorldGraphGenerator
from architecture_refinement.topology_analyzer import TopologyAnalyzer
from architecture_refinement.optimizer import MultiObjectiveOptimizer
from architecture_refinement.architecture_converter import WiredCfCConverter

def visualize_best_graph(best_solutions: List[Dict[str, Any]], plots_dir: Path, logger: logging.Logger) -> None:
    """
    Visualize the best graph result using NetworkX.
    
    Args:
        best_solutions: List of best solutions from optimization
        plots_dir: Directory to save the visualization
        logger: Logger for output
    """
    if not best_solutions:
        logger.warning("No best solutions to visualize")
        return
    
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Get the first (best) solution
        best_solution = best_solutions[0]
        trial_number = best_solution['trial_number']
        
        logger.info(f"Visualizing best graph from trial {trial_number}")
        logger.info(f"Best solution parameters: {best_solution['parameters']}")
        logger.info(f"Best solution objectives: {best_solution['objectives']}")
        logger.info(f"Best solution robustness score: {best_solution['robustness_score']:.4f}")
        
        # Debug: Log parameter types to identify formatting issues
        logger.info("Parameter types for debugging:")
        for key, value in best_solution['parameters'].items():
            logger.info(f"  {key}: {value} (type: {type(value)})")
        
        # Test basic matplotlib functionality first
        try:
            test_fig, test_ax = plt.subplots(1, 1, figsize=(5, 3))
            test_ax.text(0.5, 0.5, "Matplotlib test successful", ha='center', va='center', transform=test_ax.transAxes)
            test_ax.set_title("Test Plot")
            plt.close(test_fig)
            logger.info("Basic matplotlib functionality test passed")
        except Exception as test_error:
            logger.error(f"Basic matplotlib test failed: {test_error}")
            raise
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Subplot 1: Parameter values bar chart
        ax1 = plt.subplot(2, 2, 1)
        params = best_solution['parameters']
        
        # Filter out None values and convert to numeric
        valid_params = {}
        for key, value in params.items():
            if value is not None:
                try:
                    if isinstance(value, int):
                        valid_params[key] = int(value)
                    elif isinstance(value, float):
                        valid_params[key] = float(value)
                    else:
                        valid_params[key] = float(str(value))                
                except (ValueError, TypeError):
                    # Skip non-numeric parameters
                    logger.debug(f"Skipping non-numeric parameter {key}: {value} (type: {type(value)})")
                    continue
        
        logger.info(f"Extracted {len(valid_params)} valid numeric parameters from {len(params)} total parameters")
        
        if not valid_params:
            ax1.text(0.5, 0.5, 'No valid numeric parameters found', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Best Graph Parameters (No Valid Data)')
        else:
            param_names = list(valid_params.keys())
            param_values = list(valid_params.values())
            
            bars = ax1.bar(range(len(param_names)), param_values, alpha=0.7, color='skyblue')
            ax1.set_xlabel('Parameters')
            ax1.set_ylabel('Values')
            ax1.set_title(f'Best Graph Parameters (Trial {trial_number})')
            ax1.set_xticks(range(len(param_names)))
            ax1.set_xticklabels(param_names, rotation=45, ha='right')
            
            # Add value labels on bars (only for valid numeric values)
            for bar, value in zip(bars, param_values):
                if isinstance(value, (int, float)) and not np.isnan(value):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.3f}', ha='center', va='bottom')
        
        # Subplot 2: Create a sample Watts-Strogatz graph based on best parameters
        ax2 = plt.subplot(2, 2, 2)
        try:
            # Use the EXACT same parameters and seed that generated the best solution
            n = valid_params.get('units', 50)
            k = valid_params.get('k_degree', 4)
            p = valid_params.get('p_rewiring', 0.1)
            
            # Get the exact seed that was used for this solution
            solution_seed = best_solution['parameters'].get('seed')
            if solution_seed is None:
                # Fallback: use trial number as seed if not available
                solution_seed = trial_number
            
            logger.info(f"Visualizing best solution graph using seed: {solution_seed} for exact reproducibility")
            
            # Ensure all parameters are valid numbers
            if not all(isinstance(x, (int, float)) for x in [n, k, p]):
                raise ValueError("Invalid parameter types")
            
            # Ensure k is valid for the graph size
            k = max(2, min(int(k), n // 2))
            
            # Generate the EXACT same graph using the same seed
            # This ensures we see the actual graph that was optimized
            G = nx.watts_strogatz_graph(int(n), k, float(p), seed=int(solution_seed))
            
            # Calculate some graph metrics
            clustering = nx.average_clustering(G)
            path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
            density = nx.density(G)
            
            # Layout the graph - use different layouts for different graph sizes
            if n <= 30:
                pos = nx.spring_layout(G, seed=int(solution_seed), k=1.5)
            elif n <= 100:
                pos = nx.circular_layout(G)
            else:
                pos = nx.kamada_kawai_layout(G)
            
            # Draw the graph with enhanced styling
            node_colors = ['lightblue'] * len(G.nodes())
            edge_colors = ['gray'] * len(G.edges())
            
            # Highlight some key nodes (e.g., high degree nodes)
            degrees = dict(G.degree())
            max_degree = max(degrees.values()) if degrees else 0
            for i, node in enumerate(G.nodes()):
                if degrees.get(node, 0) >= max_degree * 0.8:  # Highlight high-degree nodes
                    node_colors[i] = 'red'
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, ax=ax2, 
                                 node_color=node_colors,
                                 node_size=[max(20, degrees.get(node, 1) * 3) for node in G.nodes()],
                                 alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, ax=ax2,
                                 edge_color=edge_colors,
                                 alpha=0.6,
                                 width=0.5)
            
            # Add some labels for small graphs
            if n <= 20:
                nx.draw_networkx_labels(G, pos, ax=ax2, font_size=8)
            
            ax2.set_title(f'Best Solution Graph (Seed: {solution_seed})\nk={k}, p={p:.3f}\nClustering: {clustering:.3f}\nPath: {path_length:.2f}\nDensity: {density:.3f}')
            ax2.axis('off')
            
        except Exception as e:
            ax2.text(0.5, 0.5, f'Graph generation failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Best Solution Graph (Generation Failed)')
            ax2.axis('off')
        
        # Subplot 3: Objective values
        ax3 = plt.subplot(2, 2, 3)
        objectives = best_solution['objectives']
        obj_names = list(objectives.keys())
        obj_values = list(objectives.values())
        
        bars = ax3.bar(obj_names, obj_values, alpha=0.7, color=['green', 'orange'])
        ax3.set_ylabel('Objective Values')
        ax3.set_title('Best Solution Objectives')
        ax3.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, obj_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Subplot 4: Robustness score and summary
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        # Create a text summary with safe formatting
        try:
            summary_text = f"""Best Solution Summary
        
Trial Number: {trial_number}
Robustness Score: {best_solution['robustness_score']:.4f}
Solution Seed: {best_solution['parameters'].get('seed', 'N/A')}

Key Parameters:
• Units: {valid_params.get('units', 'N/A')}
• k_degree: {valid_params.get('k_degree', 'N/A')}
• p_rewiring: {valid_params.get('p_rewiring', 'N/A') if valid_params.get('p_rewiring') is not None else 'N/A'}
• Target Clustering: {valid_params.get('target_clustering', 'N/A') if valid_params.get('target_clustering') is not None else 'N/A'}
• Target Path Length: {valid_params.get('target_path_length', 'N/A') if valid_params.get('target_path_length') is not None else 'N/A'}

Objectives:
• Entropy: {objectives.get('entropy', 0.0):.4f}
• Curvature: {objectives.get('curvature', 0.0):.4f}
"""
        except Exception as e:
            logger.error(f"Error creating summary text: {e}")
            summary_text = f"""Best Solution Summary
        
Trial Number: {trial_number}
Robustness Score: {best_solution['robustness_score']:.4f}
Error creating detailed summary: {str(e)}
"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the comprehensive visualization
        viz_path = plots_dir / "best_graph_comprehensive.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comprehensive best graph visualization saved to {viz_path}")
        
        # Also create a summary text file
        summary_path = plots_dir / "best_graph_summary.txt"
        try:
            with open(summary_path, 'w') as f:
                f.write(f"Best Graph Summary (Trial {trial_number})\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Robustness Score: {best_solution['robustness_score']:.4f}\n")
                f.write(f"Entropy Objective: {objectives.get('entropy', 0.0):.4f}\n")
                f.write(f"Curvature Objective: {objectives.get('curvature', 0.0):.4f}\n\n")
                f.write("Parameters:\n")
                for param, value in params.items():
                    f.write(f"  {param}: {value}\n")
            
            logger.info(f"Best graph summary saved to {summary_path}")
        except Exception as e:
            logger.error(f"Error saving summary file: {e}")
        
        # Show the plot
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for graph visualization")
    except Exception as e:
        logger.error(f"Error visualizing best graph: {e}")
        # Log the full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Fallback: Create a simple text-based visualization
        try:
            logger.info("Creating fallback text-based visualization...")
            
            # Create a simple summary file
            summary_path = plots_dir / "best_graph_simple_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("BEST GRAPH VISUALIZATION (Fallback Mode)\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Trial Number: {trial_number}\n")
                f.write(f"Robustness Score: {best_solution['robustness_score']}\n")
                f.write(f"Entropy: {objectives.get('entropy', 'N/A')}\n")
                f.write(f"Curvature: {objectives.get('curvature', 'N/A')}\n\n")
                f.write("Parameters:\n")
                for key, value in best_solution['parameters'].items():
                    f.write(f"  {key}: {value}\n")
            
            logger.info(f"Fallback summary saved to {summary_path}")
            
            # Try to create a very simple plot
            try:
                simple_fig, simple_ax = plt.subplots(1, 1, figsize=(8, 6))
                simple_ax.text(0.5, 0.8, f"Best Solution (Trial {trial_number})", 
                             ha='center', va='center', fontsize=16, transform=simple_ax.transAxes)
                simple_ax.text(0.5, 0.6, f"Robustness Score: {best_solution['robustness_score']}", 
                             ha='center', va='center', fontsize=12, transform=simple_ax.transAxes)
                simple_ax.text(0.5, 0.4, f"Entropy: {objectives.get('entropy', 'N/A')}", 
                             ha='center', va='center', fontsize=12, transform=simple_ax.transAxes)
                simple_ax.text(0.5, 0.2, f"Curvature: {objectives.get('curvature', 'N/A')}", 
                             ha='center', va='center', fontsize=12, transform=simple_ax.transAxes)
                simple_ax.set_xlim(0, 1)
                simple_ax.set_ylim(0, 1)
                simple_ax.axis('off')
                
                simple_viz_path = plots_dir / "best_graph_simple.png"
                plt.savefig(simple_viz_path, dpi=300, bbox_inches='tight')
                logger.info(f"Simple fallback visualization saved to {simple_viz_path}")
                plt.show()
                
            except Exception as plot_error:
                logger.error(f"Simple plot creation failed: {plot_error}")
            
        except Exception as fallback_error:
            logger.error(f"Even fallback visualization failed: {fallback_error}")

def main():
    """Main demo function showcasing the complete architecture refinement pipeline."""
    print("=" * 80)
    print("Architecture Refinement for Robustness-Aware CfC Networks")
    print("Simplified Demo: 2-Objective Optimization")
    print("=" * 80)
    print("Objectives: Entropy, Ollivier-Ricci Curvature")
    print("Graph Generation: Watts-Strogatz Flex with Varying Clustering & Path Lengths")
    print("=" * 80)

    # Initialize configuration
    config = Config()
    config.create_output_directories()
    
    # Setup logging
    logger = create_experiment_logger(
        experiment_name=config.experiment.experiment_name,
        output_dir=config.logging.output_dir
    )
    
    # Set random seed for reproducibility
    set_random_seed(config.experiment.random_seed, config.experiment.deterministic)
    
    # Setup visualization style
    create_visualization_style()
    
    logger.info("Starting extended architecture refinement demo with 7 objectives")

    # Step 1: Generate candidate graphs using Watts-Strogatz flex approach
    logger.info("Step 1: Generating candidate graphs using Watts-Strogatz flex approach")
    graph_generator = ModularSmallWorldGraphGenerator(config.graph_generation, logger=logger)
    
    # Ensure graph generator uses the same seed as the experiment
    if hasattr(graph_generator.config, 'seed'):
        graph_generator.config.seed = config.experiment.random_seed
        logger.info(f"Graph generator seed set to: {config.experiment.random_seed}")
    else:
        # Set the seed directly if the config doesn't have it
        np.random.seed(config.experiment.random_seed)
        random.seed(config.experiment.random_seed)
        logger.info(f"Random seeds set to: {config.experiment.random_seed}")
    
    # Use demo-specific configuration
    demo_config = config.graph_generation
    demo_config.num_candidates = 5  # Increased for better search space coverage
    
    # Use the new flex graph generation method
    candidate_graphs = graph_generator.generate_watts_strogatz_flex_graphs(num_candidates=demo_config.num_candidates)
    
    logger.info(f"Generated {len(candidate_graphs)} flex candidate graphs")
    
    # Step 2: Analyze graph topology (focusing on entropy and curvature)
    logger.info("Step 2: Analyzing graph topology for entropy and curvature metrics")
    topology_analyzer = TopologyAnalyzer(config, logger=logger)
    
    # Analyze all graphs
    graphs = [graph for graph, params in candidate_graphs]
    topology_metrics = topology_analyzer.analyze_graph_batch(graphs)
    
    # Log some example metrics
    if topology_metrics:
        # Find the first valid metrics (not empty)
        example_metrics = None
        for metrics in topology_metrics:
            if metrics and isinstance(metrics, dict) and len(metrics) > 0:
                example_metrics = metrics
                break
        
        if example_metrics:
            logger.info(f"Example metrics from first valid graph:")
            # Focus on the two key metrics
            entropy = example_metrics.get('degree_entropy', 0.0)
            if isinstance(entropy, (int, float)):
                logger.info(f"  - Degree Entropy: {entropy:.4f}")
            else:
                logger.info(f"  - Degree Entropy: {entropy}")
                
            curvature = example_metrics.get('avg_ricci_curvature', 0.0)
            if isinstance(curvature, (int, float)):
                logger.info(f"  - Ollivier-Ricci Curvature: {curvature:.4f}")
            else:
                logger.info(f"  - Ollivier-Ricci Curvature: {curvature}")
                
            logger.info(f"  - Number of Nodes: {example_metrics.get('num_nodes', 'N/A')}")
            logger.info(f"  - Number of Edges: {example_metrics.get('num_edges', 'N/A')}")
        else:
            logger.warning("No valid metrics found in any graph")
    else:
        logger.warning("No topology metrics generated")

    # Step 3: Run simplified 2-objective optimization (entropy, curvature)
    logger.info("Step 3: Running 2-objective optimization (entropy, curvature)")
    optimizer = MultiObjectiveOptimizer(config.optimization, graph_generator, topology_analyzer, logger=logger)
    
    # Use demo-specific optimization configuration
    demo_opt_config = config.optimization
    demo_opt_config.n_trials = 500  # Increased for better optimization
    optimization_results = optimizer.optimize(
        n_trials=demo_opt_config.n_trials,
        timeout=1200,  # 20 minutes for demo
        n_jobs=1
    )
    
    logger.info(f"Optimization completed with {len(optimization_results)} trials")

    # Step 4: Convert top architectures to WiredCfC format
    logger.info("Step 4: Converting top architectures to WiredCfC format")
    
    # Get the best solutions
    best_solutions = optimizer.get_best_solutions(n_solutions=5)
    
    if not best_solutions:
        logger.warning("No best solutions found for architecture conversion")
        return
    
    logger.info(f"Found {len(best_solutions)} best solutions for architecture conversion")
    
    # Convert to WiredCfC architectures
    converter = WiredCfCConverter(config.architecture, logger=logger)
    
    architectures = []
    successful_conversions = 0
    failed_conversions = 0
    
    for i, solution in enumerate(best_solutions):
        logger.info(f"Processing solution {i+1}/{len(best_solutions)} (trial {solution['trial_number']})")
        
        try:
            # Get the parameters for this solution
            solution_params = solution['parameters']
            trial_number = solution['trial_number']
            
            logger.info(f"Converting architecture {i+1} from trial {trial_number}")
            logger.info(f"Parameters: {solution_params}")
            
            # Extract key parameters for graph regeneration
            units = solution_params.get('units', 64)
            output_size = solution_params.get('output_size', 8)
            target_clustering = solution_params.get('target_clustering', 0.5)
            target_path_length = solution_params.get('target_path_length', 3.0)
            k_degree = solution_params.get('k_degree', 4)
            p_rewiring = solution_params.get('p_rewiring', 0.1)
            seed = trial_number  # Use trial number as seed for reproducibility
            
            # Validate parameters before graph regeneration
            if not all(isinstance(x, (int, float)) for x in [units, output_size, target_clustering, target_path_length, k_degree, p_rewiring]):
                logger.warning(f"Invalid parameter types for trial {trial_number}: {solution_params}")
                continue
            
            if units < 16 or units > 128:
                logger.warning(f"Invalid units value for trial {trial_number}: {units} (should be 16-128)")
                continue
            
            if output_size < 2 or output_size > 20:
                logger.warning(f"Invalid output_size for trial {trial_number}: {output_size} (should be 2-20)")
                continue
            
            if k_degree < 2 or k_degree > units // 2:
                logger.warning(f"Invalid k_degree for trial {trial_number}: {k_degree} (should be 2-{units//2})")
                continue
            
            if p_rewiring < 0.0 or p_rewiring > 1.0:
                logger.warning(f"Invalid p_rewiring for trial {trial_number}: {p_rewiring} (should be 0.0-1.0)")
                continue
            
            if target_clustering < 0.0 or target_clustering > 1.0:
                logger.warning(f"Invalid target_clustering for trial {trial_number}: {target_clustering} (should be 0.0-1.0)")
                continue
            
            if target_path_length < 1.0 or target_path_length > units:
                logger.warning(f"Invalid target_path_length for trial {trial_number}: {target_path_length} (should be 1.0-{units})")
                continue
            
            logger.info(f"Regenerating graph with: units={units}, output_size={output_size}, "
                      f"k={k_degree}, p={p_rewiring:.3f}, clustering={target_clustering:.3f}, "
                      f"path_length={target_path_length:.3f}, seed={seed}")
            
            # Regenerate the exact same graph using the same parameters
            try:
                graph = graph_generator._create_watts_strogatz_flex_graph(
                    units, k_degree, p_rewiring, target_clustering, target_path_length, seed
                )
                
                if graph is None or graph.number_of_nodes() == 0:
                    logger.warning(f"Failed to regenerate graph for trial {trial_number}")
                    continue
                
                # Validate the regenerated graph
                if not isinstance(graph, nx.Graph):
                    logger.warning(f"Invalid graph type for trial {trial_number}: {type(graph)}")
                    continue
                
                if graph.number_of_nodes() != units:
                    logger.warning(f"Graph node count mismatch for trial {trial_number}: expected {units}, got {graph.number_of_nodes()}")
                    continue
                
                logger.info(f"Successfully regenerated graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
                
            except Exception as graph_error:
                logger.error(f"Error regenerating graph for trial {trial_number}: {graph_error}")
                import traceback
                logger.error(f"Graph regeneration traceback: {traceback.format_exc()}")
                continue
            
            # Convert the regenerated graph to WiredCfC architecture
            architecture = converter.convert_graph_to_wiredcfc(
                graph, 
                input_size=8,  # Use fixed input size
                output_size=output_size  # Use output_size from params
            )
            
            # Validate the architecture
            validation_errors = converter.validate_architecture(architecture)
            if validation_errors:
                logger.warning(f"Architecture {i+1} validation errors: {validation_errors}")
            else:
                logger.info(f"Architecture {i+1} validation passed")
            
            # Get architecture summary
            summary = converter.get_architecture_summary(architecture)
            logger.info(f"Architecture {i+1} summary: {summary}")
            
            # Save the architecture
            filename = f"best_architecture_{i+1}_trial_{trial_number}"
            converter.save_architecture(architecture, filename)
            
            architectures.append(architecture)
            successful_conversions += 1
            
        except Exception as e:
            logger.error(f"Error converting architecture {i+1}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            failed_conversions += 1
            continue
    
    logger.info(f"Successfully converted {successful_conversions} architectures to WiredCfC format")
    logger.info(f"Failed to convert {failed_conversions} architectures.")
    
    if successful_conversions > 0:
        logger.info("Architecture conversion summary:")
        for i, arch in enumerate(architectures):
            logger.info(f"  Architecture {i+1}: {arch.input_size} -> {arch.hidden_size} -> {arch.output_size} "
                      f"(total neurons: {arch.input_size + arch.hidden_size + arch.output_size})")
    else:
        logger.warning("No architectures were successfully converted!")
        logger.warning("This may indicate an issue with the graph generation or conversion process.")

    # Step 5: Generate visualizations and save results
    logger.info("Step 5: Generating visualizations and saving results")
    
    # Save optimization results
    optimizer.save_results("extended_demo_optimization_results")
    
    # Create plots directory
    plots_dir = Path(config.logging.output_dir) / config.logging.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate optimization plots
    optimizer.plot_optimization_history(save_path=str(plots_dir / "extended_optimization_history.png"))
    optimizer.plot_pareto_front(save_path=str(plots_dir / "extended_pareto_front.png"))
    
    # Export best graphs
    optimizer.export_best_graphs(str(Path(config.logging.output_dir) / "best_graphs"))

    # Step 6: Generate comprehensive analysis plots
    logger.info("Step 6: Generating comprehensive analysis plots")
    create_summary_plots(topology_metrics, optimization_results, plots_dir, logger)
    
    # Print experiment summary
    print_experiment_summary(config, optimization_results, logger)
    
    # Visualize the best graph result
    visualize_best_graph(best_solutions, plots_dir, logger)
    
    logger.info("Simplified 2-objective demo completed successfully!")
    print("\n" + "=" * 80)
    print("Demo completed! Check the outputs directory for results and visualizations.")
    print("Focus: Entropy and Ollivier-Ricci Curvature optimization using Watts-Strogatz flex graphs")
    print("=" * 80)

def create_summary_plots(topology_metrics, optimization_results, plots_dir, logger):
    """Create summary plots for the simplified 2-objective demo."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        logger.info("Creating summary plots...")
        
        # Plot 1: Distribution of key metrics across all graphs
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Distribution of Key Topology Metrics')
        
        # Extract key metrics
        entropy_scores = [m.get('degree_entropy', 0.0) for m in topology_metrics if m]
        curvature_scores = [m.get('avg_ricci_curvature', 0.0) for m in topology_metrics if m]
        
        # Plot distributions
        axes[0].hist(entropy_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('Degree Entropy Distribution')
        axes[0].set_xlabel('Entropy Score')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(curvature_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1].set_title('Ollivier-Ricci Curvature Distribution')
        axes[1].set_xlabel('Curvature Score')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "key_metrics_distributions.png", dpi=300, bbox_inches='tight')
        logger.info("Key metrics distribution plot saved")
        
        # Plot 2: Entropy vs Curvature scatter plot
        if optimization_results:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Extract objective values
            entropy_values = []
            curvature_values = []
            robustness_scores = []
            
            for result in optimization_results:
                if hasattr(result, 'objectives') and result.objectives:
                    entropy = result.objectives.get('entropy', 0.0)
                    curvature = result.objectives.get('curvature', 0.0)
                    if hasattr(result, 'robustness_score'):
                        robustness = result.robustness_score
                    else:
                        robustness = 0.0
                    
                    entropy_values.append(entropy)
                    curvature_values.append(curvature)
                    robustness_scores.append(robustness)
            
            if entropy_values and curvature_values:
                # Create scatter plot colored by robustness score
                scatter = ax.scatter(entropy_values, curvature_values, c=robustness_scores, 
                                   cmap='viridis', alpha=0.7, s=50)
                ax.set_xlabel('Degree Entropy')
                ax.set_ylabel('Ollivier-Ricci Curvature')
                ax.set_title('Entropy vs Curvature (colored by robustness)')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax, label='Robustness Score')
                plt.tight_layout()
                plt.savefig(plots_dir / "entropy_vs_curvature.png", dpi=300, bbox_inches='tight')
                logger.info("Entropy vs curvature plot saved")
        
        # Plot 3: Parameter relationships with objectives
        if optimization_results:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle('Parameter Relationships with Objectives')
            
            # Define parameter-objective pairs to explore
            param_obj_pairs = [
                ('units', 'entropy'),
                ('k_degree', 'curvature'),
                ('p_rewiring', 'entropy'),
                ('target_clustering', 'curvature'),
                ('target_path_length', 'entropy')
            ]
            
            for i, (param, obj) in enumerate(param_obj_pairs):
                ax = axes[i]
                
                # Extract values
                param_values = []
                obj_values = []
                
                for result in optimization_results:
                    if hasattr(result, 'parameters') and result.parameters:
                        param_val = result.parameters.get(param, 0)
                        if hasattr(result, 'objectives') and result.objectives:
                            obj_val = result.objectives.get(obj, 0.0)
                            param_values.append(param_val)
                            obj_values.append(obj_val)
                
                if param_values and obj_values:
                    ax.scatter(param_values, obj_values, alpha=0.6, c='purple')
                    ax.set_xlabel(param.replace('_', ' ').title())
                    ax.set_ylabel(obj.title())
                    ax.set_title(f'{param.replace("_", " ").title()} vs {obj.title()}')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "parameter_objective_relationships.png", dpi=300, bbox_inches='tight')
            logger.info("Parameter-objective relationships plot saved")
        
        plt.close('all')
        
    except ImportError:
        logger.warning("matplotlib/seaborn not available for plotting")
    except Exception as e:
        logger.error(f"Error creating summary plots: {e}")

def run_quick_demo():
    """Run a minimal demo for quick testing."""
    print("Running quick demo...")
    
    # Minimal configuration
    config = Config()
    config.create_output_directories()
    
    # Setup basic logging
    logger = create_experiment_logger("quick_demo", config.logging.output_dir)
    
    # Set random seed
    set_random_seed(42, deterministic=True)
    
    logger.info("Quick demo: Testing Watts-Strogatz flex graph generation and 2-objective optimization")
    
    # Test flex graph generation
    graph_generator = ModularSmallWorldGraphGenerator(config.graph_generation, logger=logger)
    
    # Ensure graph generator uses the same seed as the experiment
    if hasattr(graph_generator.config, 'seed'):
        graph_generator.config.seed = 42  # Use the same seed as set_random_seed(42)
        logger.info(f"Graph generator seed set to: 42")
    else:
        # Set the seed directly if the config doesn't have it
        np.random.seed(42)
        random.seed(42)
        logger.info(f"Random seeds set to: 42")
    
    candidate_graphs = graph_generator.generate_watts_strogatz_flex_graphs(num_candidates=10)
    logger.info(f"Generated {len(candidate_graphs)} flex test graphs")
    
    # Test topology analysis
    topology_analyzer = TopologyAnalyzer(config, logger=logger)
    graphs = [graph for graph, params in candidate_graphs]
    topology_metrics = topology_analyzer.analyze_graph_batch(graphs)
    logger.info(f"Analyzed {len(topology_metrics)} graphs")
    
    # Test optimization (minimal)
    optimizer = MultiObjectiveOptimizer(config.optimization, graph_generator, topology_analyzer, logger=logger)
    optimization_results = optimizer.optimize(n_trials=10, timeout=120, n_jobs=1)
    logger.info(f"Completed {len(optimization_results)} optimization trials")
    
    # Show some results
    if optimization_results:
        best_result = max(optimization_results, key=lambda x: x.robustness_score)
        logger.info(f"Best result - Entropy: {best_result.objectives.get('entropy', 0.0):.4f}, "
                   f"Curvature: {best_result.objectives.get('curvature', 0.0):.4f}")
    
    # Print experiment summary
    print_experiment_summary(config, optimization_results, logger)
    
    # Visualize the best graph result
    best_solutions = optimizer.get_best_solutions(n_solutions=5)
    visualize_best_graph(best_solutions, Path(config.logging.output_dir) / config.logging.plots_dir, logger)
    
    logger.info("Quick demo completed successfully!")
    return optimization_results

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_demo()
    else:
        main()
