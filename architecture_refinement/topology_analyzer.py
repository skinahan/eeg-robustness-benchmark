"""
Topology analysis module for computing graph-theoretic metrics.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
import warnings
from collections import defaultdict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class TopologyAnalyzer:
    """
    Analyzes topological properties of graphs using various metrics.
    
    This class implements computation of:
    - Topological entropy
    - Ollivier-Ricci curvature
    - Algebraic connectivity
    - Clustering coefficients
    - Path length metrics
    - Efficiency measures
    """
    
    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        """
        Initialize the topology analyzer.
        
        Args:
            config: Configuration object containing topology analysis parameters
            logger: Optional logger for output
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_graph(self, graph: nx.Graph) -> Dict[str, float]:
        """Analyze a graph and compute all topological metrics."""
        try:
            metrics = {}
            metrics.update(self._compute_basic_properties(graph))
            metrics.update(self._compute_entropy_metrics(graph))
            metrics.update(self._compute_curvature_metrics(graph))
            metrics.update(self._compute_connectivity_metrics(graph))
            metrics.update(self._compute_efficiency_metrics(graph))
            metrics.update(self._compute_clustering_path_metrics(graph))
            metrics.update(self._compute_spectral_metrics(graph))
            
            # Add new metrics for the extended objectives
            # metrics.update(self._compute_modularity_metrics(graph))
            # metrics.update(self._compute_redundant_pathway_metrics(graph))
            # metrics.update(self._compute_interpretability_metrics(graph))
            
            return metrics
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in analyze_graph: {e}")
            # Return minimal metrics on error
            return {
                'num_nodes': graph.number_of_nodes() if hasattr(graph, 'number_of_nodes') else 0,
                'num_edges': graph.number_of_edges() if hasattr(graph, 'number_of_edges') else 0,
                'error': str(e)
            }
    
    def _compute_basic_properties(self, graph: nx.Graph) -> Dict[str, float]:
        """Compute basic graph properties."""
        return {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'avg_degree': np.mean([d for n, d in graph.degree()])
        }

    def _compute_entropy_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Compute entropy-based metrics for diversity assessment."""
        try:
            # Canonical TE (Waqas et al., 2022): exact entropy of empirical degree distribution
            n = int(graph.number_of_nodes())
            if n <= 1:
                degree_entropy_raw = 0.0
                te = 0.0
                te_norm_const = 1.0
            else:
                degrees = np.fromiter((d for _, d in graph.degree()), dtype=int)
                _, counts = np.unique(degrees, return_counts=True)
                p = counts.astype(float) / float(n)
                degree_entropy_raw = float(-np.sum(p * np.log(p)))
                te_norm_const = float(np.log(n))
                te = float(degree_entropy_raw / te_norm_const) if te_norm_const > 0.0 else 0.0
                te = float(np.clip(te, 0.0, 1.0))

            # Keep legacy key name but make it the exact (raw) entropy (nats).
            degree_entropy = float(degree_entropy_raw)
                
            # Weight entropy (if weights exist)
            weights = [graph[u][v].get('weight', 1.0) for u, v in graph.edges()]
            if len(set(weights)) > 1:
                hist_result = np.histogram(weights, bins=min(20, len(set(weights))))
                if len(hist_result) >= 2 and hist_result[0].size > 0:
                    weight_entropy = stats.entropy(hist_result[0] + 1e-10)
                else:
                    weight_entropy = 0.0
            else:
                weight_entropy = 0.0
                
            # Path length entropy
            try:
                path_lengths = []
                for source in list(graph.nodes())[:min(10, graph.number_of_nodes())]:
                    for target in list(graph.nodes())[:min(10, graph.number_of_nodes())]:
                        if source != target:
                            try:
                                path_lengths.append(nx.shortest_path_length(graph, source, target))
                            except nx.NetworkXNoPath:
                                continue
                if path_lengths and len(set(path_lengths)) > 1:
                    hist_result = np.histogram(path_lengths, bins=min(20, len(set(path_lengths))))
                    if len(hist_result) >= 2 and hist_result[0].size > 0:
                        path_entropy = stats.entropy(hist_result[0] + 1e-10)
                    else:
                        path_entropy = 0.0
                else:
                    path_entropy = 0.0
            except:
                path_entropy = 0.0
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error computing entropy metrics: {e}")
            degree_entropy = 0.0
            degree_entropy_raw = 0.0
            te = 0.0
            te_norm_const = 1.0
            weight_entropy = 0.0
            path_entropy = 0.0
            
        return {
            'degree_entropy': float(degree_entropy),
            'degree_entropy_raw': float(degree_entropy_raw),
            'te': float(te),
            'te_norm_const': float(te_norm_const),
            'weight_entropy': float(weight_entropy),
            'path_entropy': float(path_entropy)
        }

    def _compute_curvature_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Compute curvature metrics for robustness assessment."""
        try:
            # Canonical ORC: signed mean Ollivier–Ricci curvature via optimal transport
            # Fall back to the historical heuristic only if dependencies are unavailable.
            orc_alpha = 0.5
            cfg_alpha = getattr(self.config, "orc_alpha", None)
            if cfg_alpha is None:
                topo_cfg = getattr(self.config, "topology", None)
                cfg_alpha = getattr(topo_cfg, "orc_alpha", None) if topo_cfg is not None else None
            if cfg_alpha is not None:
                orc_alpha = float(cfg_alpha)

            orc_max_edges = getattr(self.config, "orc_max_edges", None)
            if orc_max_edges is None:
                topo_cfg = getattr(self.config, "topology", None)
                orc_max_edges = getattr(topo_cfg, "orc_max_edges", None) if topo_cfg is not None else None
            if orc_max_edges is None:
                orc_max_edges = 200  # safe default for general analysis

            try:
                from .metrics_te_orc import ollivier_ricci_mean

                avg_ricci_curvature, orc_debug = ollivier_ricci_mean(
                    graph,
                    alpha=float(orc_alpha),
                    max_edges=int(orc_max_edges) if orc_max_edges is not None else None,
                    return_edge_curvatures=False,
                )
                ricci_curvature_std = float(orc_debug.get("orc_std", 0.0))
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Falling back to heuristic Ricci curvature (exact ORC unavailable): {e}")
                # Historical heuristic (non-canonical)
                curvatures = []
                edges = list(graph.edges())[: min(100, graph.number_of_edges())]
                for u, v in edges:
                    deg_u = graph.degree(u)
                    deg_v = graph.degree(v)
                    common_neighbors = len(list(nx.common_neighbors(graph, u, v)))
                    curvature = 1.0 - (deg_u + deg_v - 2 * common_neighbors) / (deg_u + deg_v)
                    curvatures.append(curvature)
                avg_ricci_curvature = float(np.mean(curvatures)) if curvatures else 0.0
                ricci_curvature_std = float(np.std(curvatures)) if curvatures else 0.0
            
            # Forman-Ricci curvature
            forman_curvatures = []
            edges = list(graph.edges())[:min(100, graph.number_of_edges())]
            for u, v in edges:
                deg_u = graph.degree(u)
                deg_v = graph.degree(v)
                forman_curvature = 4 - deg_u - deg_v
                forman_curvatures.append(forman_curvature)
            
            avg_forman_curvature = np.mean(forman_curvatures) if forman_curvatures else 0.0
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error computing curvature metrics: {e}")
            avg_ricci_curvature = 0.0
            ricci_curvature_std = 0.0
            avg_forman_curvature = 0.0
            
        return {
            'avg_ricci_curvature': float(avg_ricci_curvature),
            'ricci_curvature_std': float(ricci_curvature_std),
            'orc': float(avg_ricci_curvature),
            'avg_forman_curvature': float(avg_forman_curvature)
        }

    def _compute_connectivity_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Compute connectivity metrics for robustness assessment."""
        connectivity_metrics = {}
        
        # Algebraic connectivity
        try:
            if nx.is_connected(graph):
                laplacian_matrix = nx.laplacian_matrix(graph)
                eigenvalues = np.linalg.eigvals(laplacian_matrix.toarray())
                eigenvalues = np.real(eigenvalues)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                if len(eigenvalues) > 1:
                    connectivity_metrics['algebraic_connectivity'] = float(np.sort(eigenvalues)[1])
                else:
                    connectivity_metrics['algebraic_connectivity'] = 0.0
            else:
                connectivity_metrics['algebraic_connectivity'] = 0.0
        except:
            connectivity_metrics['algebraic_connectivity'] = 0.0
            
        # Edge and node connectivity
        try:
            if nx.is_connected(graph):
                connectivity_metrics['edge_connectivity'] = float(nx.edge_connectivity(graph))
                connectivity_metrics['node_connectivity'] = float(nx.node_connectivity(graph))
            else:
                connectivity_metrics['edge_connectivity'] = 0.0
                connectivity_metrics['node_connectivity'] = 0.0
        except:
            connectivity_metrics['edge_connectivity'] = 0.0
            connectivity_metrics['node_connectivity'] = 0.0
            
        # Expansion ratio
        try:
            if graph.number_of_nodes() > 1:
                min_cut_size = min(len(list(nx.node_boundary(graph, [n]))) for n in graph.nodes())
                connectivity_metrics['expansion_ratio'] = float(min_cut_size / graph.number_of_nodes())
            else:
                connectivity_metrics['expansion_ratio'] = 0.0
        except:
            connectivity_metrics['expansion_ratio'] = 0.0
            
        return connectivity_metrics

    def _compute_efficiency_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Compute efficiency metrics for information flow assessment."""
        try:
            if nx.is_connected(graph):
                global_efficiency = nx.global_efficiency(graph)
                local_efficiency = nx.local_efficiency(graph)
            else:
                # For disconnected graphs, compute efficiency on largest component
                largest_cc = max(nx.connected_components(graph), key=len)
                if len(largest_cc) > 1:
                    subgraph = graph.subgraph(largest_cc)
                    global_efficiency = nx.global_efficiency(subgraph)
                    local_efficiency = nx.local_efficiency(subgraph)
                else:
                    global_efficiency = 0.0
                    local_efficiency = 0.0
        except:
            global_efficiency = 0.0
            local_efficiency = 0.0
            
        return {
            'global_efficiency': float(global_efficiency),
            'local_efficiency': float(local_efficiency)
        }

    def _compute_clustering_path_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Compute clustering and path length metrics."""
        try:
            # Clustering coefficient
            clustering_coeff = nx.average_clustering(graph)
            
            # Average shortest path length
            if nx.is_connected(graph):
                avg_path_length = nx.average_shortest_path_length(graph)
            else:
                # For disconnected graphs, compute on largest component
                largest_cc = max(nx.connected_components(graph), key=len)
                if len(largest_cc) > 1:
                    subgraph = graph.subgraph(largest_cc)
                    avg_path_length = nx.average_shortest_path_length(subgraph)
                else:
                    avg_path_length = 0.0
                    
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error computing clustering/path metrics: {e}")
            clustering_coeff = 0.0
            avg_path_length = 0.0
            
        return {
            'clustering_coefficient': float(clustering_coeff),
            'avg_path_length': float(avg_path_length)
        }

    def _compute_spectral_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Compute spectral properties of the graph."""
        try:
            # Adjacency matrix eigenvalues
            adj_matrix = nx.adjacency_matrix(graph)
            eigenvalues = np.linalg.eigvals(adj_matrix.toarray())
            eigenvalues = np.real(eigenvalues)
            
            spectral_radius = float(np.max(np.abs(eigenvalues)))
            spectral_gap = float(spectral_radius - np.max(eigenvalues[eigenvalues < spectral_radius])) if len(eigenvalues) > 1 else 0.0
            
            # Laplacian spectral gap
            if nx.is_connected(graph):
                laplacian_matrix = nx.laplacian_matrix(graph)
                laplacian_eigenvalues = np.linalg.eigvals(laplacian_matrix.toarray())
                laplacian_eigenvalues = np.real(laplacian_eigenvalues)
                laplacian_eigenvalues = laplacian_eigenvalues[laplacian_eigenvalues > 1e-10]
                if len(laplacian_eigenvalues) > 1:
                    laplacian_spectral_gap = float(np.sort(laplacian_eigenvalues)[1])
                else:
                    laplacian_spectral_gap = 0.0
            else:
                laplacian_spectral_gap = 0.0
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error computing spectral metrics: {e}")
            spectral_radius = 0.0
            spectral_gap = 0.0
            laplacian_spectral_gap = 0.0
            
        return {
            'spectral_radius': spectral_radius,
            'spectral_gap': spectral_gap,
            'laplacian_spectral_gap': laplacian_spectral_gap
        }


def compute_spectral_radius_directed(adj_directed: np.ndarray) -> float:
    """
    Compute spectral radius rho(A) of a directed adjacency matrix.
    For directed graphs, rho = max(|eigenvalues|). The model uses oriented (directed)
    adjacency; rho(A_dir) is a more appropriate proxy than rho(A_undir) from analyze_graph.
    """
    try:
        adj = np.asarray(adj_directed, dtype=float)
        if adj.size == 0:
            return float("nan")
        eigenvalues = np.linalg.eigvals(adj)
        eigenvalues = np.real(eigenvalues)
        return float(np.max(np.abs(eigenvalues)))
    except Exception:
        return float("nan")

    def _compute_modularity_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Compute modularity metrics for regional/functional subnetworks."""
        try:
            # Newman-Girvan modularity
            if graph.number_of_nodes() > 1:
                # Use Louvain method for community detection
                communities = nx.community.louvain_communities(graph, weight='weight' if nx.get_edge_attributes(graph, 'weight') else None)
                modularity = nx.community.modularity(graph, communities)
                
                # Number of communities
                num_communities = len(communities)
                
                # Community size distribution entropy
                community_sizes = [len(c) for c in communities]
                if len(set(community_sizes)) > 1:
                    hist_result = np.histogram(community_sizes, bins=min(20, len(set(community_sizes))))
                    if len(hist_result) >= 2 and hist_result[0].size > 0:
                        community_size_entropy = stats.entropy(hist_result[0] + 1e-10)
                    else:
                        community_size_entropy = 0.0
                else:
                    community_size_entropy = 0.0
                    
                # Intra-community density vs inter-community density
                intra_edges = 0
                inter_edges = 0
                for u, v in graph.edges():
                    u_community = next(i for i, c in enumerate(communities) if u in c)
                    v_community = next(i for i, c in enumerate(communities) if v in c)
                    if u_community == v_community:
                        intra_edges += 1
                    else:
                        inter_edges += 1
                
                total_edges = graph.number_of_edges()
                if total_edges > 0:
                    intra_density = intra_edges / total_edges
                    inter_density = inter_edges / total_edges
                    modularity_ratio = intra_density / (inter_density + 1e-10)
                else:
                    intra_density = 0.0
                    inter_density = 0.0
                    modularity_ratio = 0.0
                    
            else:
                modularity = 0.0
                num_communities = 0
                community_size_entropy = 0.0
                intra_density = 0.0
                inter_density = 0.0
                modularity_ratio = 0.0
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error computing modularity metrics: {e}")
            modularity = 0.0
            num_communities = 0
            community_size_entropy = 0.0
            intra_density = 0.0
            inter_density = 0.0
            modularity_ratio = 0.0
            
        return {
            'newman_girvan_modularity': float(modularity),
            'num_communities': int(num_communities),
            'community_size_entropy': float(community_size_entropy),
            'intra_community_density': float(intra_density),
            'inter_community_density': float(inter_density),
            'modularity_ratio': float(modularity_ratio)
        }

    def _compute_redundant_pathway_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Compute metrics for redundant-but-diverse pathways from sensory to motor."""
        try:
            # Identify sensory and motor nodes (assuming they're labeled or in specific ranges)
            # For now, we'll use a heuristic based on node degrees and positions
            nodes = list(graph.nodes())
            if not nodes:
                return self._empty_redundant_pathway_metrics()
                
            # Assume first few nodes are motor (output) and last few are sensory (input)
            # This is a simplified assumption - in practice, these would be properly labeled
            motor_nodes = nodes[:min(4, len(nodes)//4)]
            sensory_nodes = nodes[-min(8, len(nodes)//4):]
            
            if not motor_nodes or not sensory_nodes:
                return self._empty_redundant_pathway_metrics()
            
            # Find multiple paths from each sensory node to each motor node
            path_diversity = []
            path_redundancy = []
            path_lengths = []
            
            for sensory in sensory_nodes:
                for motor in motor_nodes:
                    try:
                        # Find all simple paths
                        all_paths = list(nx.all_simple_paths(graph, sensory, motor, cutoff=10))
                        if len(all_paths) > 1:
                            # Path redundancy (number of alternative paths)
                            path_redundancy.append(len(all_paths))
                            
                            # Path diversity (how different the paths are)
                            path_diversity_score = self._compute_path_diversity(all_paths)
                            path_diversity.append(path_diversity_score)
                            
                            # Path lengths
                            path_lengths.extend([len(p) for p in all_paths])
                            
                    except nx.NetworkXNoPath:
                        continue
            
            # Compute aggregate metrics
            avg_path_redundancy = np.mean(path_redundancy) if path_redundancy else 0.0
            avg_path_diversity = np.mean(path_diversity) if path_diversity else 0.0
            avg_path_length = np.mean(path_lengths) if path_lengths else 0.0
            
            # Pathway coverage (how many sensory-motor pairs have multiple paths)
            total_pairs = len(sensory_nodes) * len(motor_nodes)
            covered_pairs = len([1 for r in path_redundancy if r > 1])
            pathway_coverage = covered_pairs / total_pairs if total_pairs > 0 else 0.0
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error computing redundant pathway metrics: {e}")
            return self._empty_redundant_pathway_metrics()
            
        return {
            'avg_path_redundancy': float(avg_path_redundancy),
            'avg_path_diversity': float(avg_path_diversity),
            'avg_path_length': float(avg_path_length),
            'pathway_coverage': float(pathway_coverage),
            'total_sensory_motor_pairs': int(total_pairs),
            'covered_pairs': int(covered_pairs)
        }

    def _compute_path_diversity(self, paths: List[List]) -> float:
        """Compute diversity between multiple paths."""
        if len(paths) < 2:
            return 0.0
            
        # Compute Jaccard distance between all path pairs
        diversities = []
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                set_i = set(paths[i])
                set_j = set(paths[j])
                if set_i or set_j:  # Avoid division by zero
                    jaccard_distance = 1 - len(set_i & set_j) / len(set_i | set_j)
                    diversities.append(jaccard_distance)
                    
        return np.mean(diversities) if diversities else 0.0

    def _empty_redundant_pathway_metrics(self) -> Dict[str, float]:
        """Return empty metrics when computation fails."""
        return {
            'avg_path_redundancy': 0.0,
            'avg_path_diversity': 0.0,
            'avg_path_length': 0.0,
            'pathway_coverage': 0.0,
            'total_sensory_motor_pairs': 0,
            'covered_pairs': 0
        }

    def _compute_interpretability_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Compute metrics for interpretability and structural clarity."""
        try:
            # Structural regularity (how regular the graph structure is)
            degrees = [d for n, d in graph.degree()]
            degree_variance = np.var(degrees) if len(degrees) > 1 else 0.0
            degree_regularity = 1.0 / (1.0 + degree_variance)  # Higher is more regular
            
            # Hierarchical structure (using betweenness centrality distribution)
            betweenness_centrality = nx.betweenness_centrality(graph)
            if betweenness_centrality:
                bc_values = list(betweenness_centrality.values())
                bc_variance = np.var(bc_values) if len(bc_values) > 1 else 0.0
                hierarchical_structure = bc_variance / (np.mean(bc_values) + 1e-10)  # Higher means more hierarchical
            else:
                hierarchical_structure = 0.0
                
            # Symmetry (how symmetric the graph is)
            # Simplified: check if the graph has any automorphisms
            try:
                automorphisms = list(nx.automorphisms(graph))
                symmetry_score = len(automorphisms) / (graph.number_of_nodes() + 1e-10)
            except:
                symmetry_score = 0.0
                
            # Clustering hierarchy (how clustered the graph is at different scales)
            clustering_hierarchy = self._compute_clustering_hierarchy(graph)
            
            # Structural balance (ratio of balanced to unbalanced triangles)
            structural_balance = self._compute_structural_balance(graph)
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error computing interpretability metrics: {e}")
            degree_regularity = 0.0
            hierarchical_structure = 0.0
            symmetry_score = 0.0
            clustering_hierarchy = 0.0
            structural_balance = 0.0
            
        return {
            'degree_regularity': float(degree_regularity),
            'hierarchical_structure': float(hierarchical_structure),
            'symmetry_score': float(symmetry_score),
            'clustering_hierarchy': float(clustering_hierarchy),
            'structural_balance': float(structural_balance)
        }

    def _compute_clustering_hierarchy(self, graph: nx.Graph) -> float:
        """Compute clustering hierarchy at different scales."""
        try:
            # Compute clustering coefficient at different neighborhood sizes
            hierarchy_scores = []
            for node in list(graph.nodes())[:min(20, graph.number_of_nodes())]:
                try:
                    # Local clustering at different scales
                    neighbors = list(graph.neighbors(node))
                    if len(neighbors) > 1:
                        local_clustering = nx.clustering(graph, node)
                        hierarchy_scores.append(local_clustering)
                except:
                    continue
                    
            return np.mean(hierarchy_scores) if hierarchy_scores else 0.0
            
        except:
            return 0.0

    def _compute_structural_balance(self, graph: nx.Graph) -> float:
        """Compute structural balance based on triangle properties."""
        try:
            triangles = list(nx.triangles(graph).values())
            if not triangles:
                return 0.0
                
            # Count balanced vs unbalanced triangles
            # This is a simplified version - in practice, you'd need signed edges
            balanced_triangles = sum(1 for t in triangles if t > 0)
            total_triangles = len(triangles)
            
            return balanced_triangles / total_triangles if total_triangles > 0 else 0.0
            
        except:
            return 0.0

    def compute_robustness_score(self, metrics: Dict[str, float]) -> float:
        """Compute a composite robustness score from all metrics."""
        # Original robustness components
        weights = {
            'entropy': 0.5,
            'curvature': 0.5, 
            # 'connectivity': 0.15,
            # 'efficiency': 0.15,
            # 'modularity': 0.15,
            # 'redundancy': 0.15,
            # 'interpretability': 0.10
        }
        
        # Normalize and combine scores
        # TE is already normalized to [0,1] by definition.
        entropy_score = float(np.clip(metrics.get('te', 0.0), 0.0, 1.0))
        # ORC is signed; map to (0,1) monotonically for a bounded composite score.
        orc_val = float(metrics.get('orc', metrics.get('avg_ricci_curvature', 0.0)))
        curvature_score = float(1.0 / (1.0 + np.exp(-orc_val)))
        # connectivity_score = min(1.0, max(0.0, metrics.get('algebraic_connectivity', 0.0) / 2.0))
        # efficiency_score = min(1.0, max(0.0, metrics.get('global_efficiency', 0.0)))
        
        # # New metrics
        # modularity_score = min(1.0, max(0.0, metrics.get('newman_girvan_modularity', 0.0) + 0.5))  # Shift to positive range
        # redundancy_score = min(1.0, max(0.0, metrics.get('pathway_coverage', 0.0)))
        # interpretability_score = min(1.0, max(0.0, metrics.get('degree_regularity', 0.0)))
        
        robustness_score = (
            weights['entropy'] * entropy_score +
            weights['curvature'] * curvature_score
            # weights['connectivity'] * connectivity_score +
            # weights['efficiency'] * efficiency_score +
            # weights['modularity'] * modularity_score +
            # weights['redundancy'] * redundancy_score +
            # weights['interpretability'] * interpretability_score
        )
        
        return float(robustness_score)

    def analyze_graph_batch(self, graphs: List[nx.Graph]) -> List[Dict[str, float]]:
        """Analyze a batch of graphs efficiently."""
        metrics_list = []
        for i, graph in enumerate(graphs):
            if self.logger and i % 100 == 0:
                self.logger.info(f"Analyzing graph {i+1}/{len(graphs)}")
            try:
                metrics = self.analyze_graph(graph)
                # Ensure metrics is a dictionary
                if isinstance(metrics, dict):
                    metrics_list.append(metrics)
                else:
                    if self.logger:
                        self.logger.warning(f"Graph {i} returned non-dict metrics: {type(metrics)}")
                    metrics_list.append({})
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error analyzing graph {i}: {e}")
                # Add empty metrics for failed analysis
                metrics_list.append({})
        return metrics_list

    def get_metric_summary(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for a list of metric dictionaries."""
        if not metrics_list:
            return {}
            
        summary = {}
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
            
        for key in all_keys:
            values = [metrics.get(key, 0.0) for metrics in metrics_list if isinstance(metrics.get(key, 0.0), (int, float))]
            if values:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
                
        return summary