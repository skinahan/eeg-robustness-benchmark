#!/usr/bin/env python3
"""
Basic test script for the Architecture Refinement project.

This script tests the core functionality without running the full optimization pipeline.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import networkx as nx

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
 """Test that all modules can be imported."""
 print("Testing imports...")
 
 try:
 from architecture_refinement.config import Config
 from architecture_refinement.utils import setup_logging
 from architecture_refinement.graph_generator import ModularSmallWorldGraphGenerator
 from architecture_refinement.topology_analyzer import TopologyAnalyzer
 from architecture_refinement.optimizer import MultiObjectiveOptimizer
 from architecture_refinement.architecture_converter import WiredCfCConverter
 print(" All imports successful")
 return True
 except ImportError as e:
 print(f" Import failed: {e}")
 return False

def test_config():
 """Test configuration system."""
 print("\nTesting configuration...")
 
 try:
 from architecture_refinement.config import Config
 
 config = Config()
 
 # Test basic config access
 assert config.graph_generation.min_units == 32
 assert config.optimization.n_trials == 100
 assert config.architecture.input_size == 64
 
 print(" Configuration system working")
 return True
 except Exception as e:
 print(f" Configuration test failed: {e}")
 return False

def test_graph_generation():
 """Test graph generation."""
 print("\nTesting graph generation...")
 
 try:
 from architecture_refinement.config import Config
 from architecture_refinement.graph_generator import ModularSmallWorldGraphGenerator
 
 config = Config()
 config.graph_generation.num_candidates = 5 # Small number for testing
 
 generator = ModularSmallWorldGraphGenerator(config.graph_generation)
 
 # Generate a few test graphs
 candidate_graphs = generator.generate_candidate_graphs(num_candidates=3)
 
 assert len(candidate_graphs) == 3
 
 for graph, params in candidate_graphs:
 assert isinstance(graph, nx.Graph)
 assert graph.number_of_nodes() > 0
 assert graph.number_of_edges() > 0
 
 print(" Graph generation working")
 return True
 except Exception as e:
 print(f" Graph generation test failed: {e}")
 return False

def test_topology_analysis():
 """Test topology analysis."""
 print("\nTesting topology analysis...")
 
 try:
 from architecture_refinement.config import Config
 from architecture_refinement.topology_analyzer import TopologyAnalyzer
 
 config = Config()
 analyzer = TopologyAnalyzer(config)
 
 # Create a simple test graph
 G = nx.Graph()
 G.add_nodes_from(range(10))
 G.add_edges_from([(i, (i+1) % 10) for i in range(10)])
 
 # Analyze the graph
 metrics = analyzer.analyze_graph(G)
 
 assert 'num_nodes' in metrics
 assert 'num_edges' in metrics
 assert 'density' in metrics
 assert 'clustering_coefficient' in metrics
 
 print(" Topology analysis working")
 return True
 except Exception as e:
 print(f" Topology analysis test failed: {e}")
 return False

def test_architecture_conversion():
 """Test architecture conversion."""
 print("\nTesting architecture conversion...")
 
 try:
 from architecture_refinement.config import Config
 from architecture_refinement.architecture_converter import WiredCfCConverter
 
 config = Config()
 converter = WiredCfCConverter(config.architecture)
 
 # Create a simple test graph
 G = nx.Graph()
 G.add_nodes_from(range(20))
 G.add_edges_from([(i, (i+1) % 20) for i in range(20)])
 
 # Add some sensory nodes
 for i in range(8):
 G.add_node(f"sensory_{i}")
 G.add_edge(f"sensory_{i}", i)
 
 # Convert to WiredCfC architecture
 architecture = converter.convert_graph_to_wiredcfc(G, input_size=8, output_size=4)
 
 assert architecture.input_size == 8
 assert architecture.output_size == 4
 assert architecture.wiring_matrix.shape == (32, 32) # 8 + 20 + 4
 
 print(" Architecture conversion working")
 return True
 except Exception as e:
 print(f" Architecture conversion test failed: {e}")
 return False

def test_utils():
 """Test utility functions."""
 print("\nTesting utility functions...")
 
 try:
 from architecture_refinement.utils import setup_logging, calculate_statistics
 
 # Test logging setup
 logger = setup_logging(level="INFO")
 assert logger is not None
 
 # Test statistics calculation
 data = [1.0, 2.0, 3.0, 4.0, 5.0]
 stats = calculate_statistics(data)
 
 assert 'mean' in stats
 assert 'std' in stats
 assert stats['mean'] == 3.0
 assert stats['std'] > 0
 
 print(" Utility functions working")
 return True
 except Exception as e:
 print(f" Utility functions test failed: {e}")
 return False

def main():
 """Run all tests."""
 print("=" * 60)
 print("Architecture Refinement - Basic Functionality Test")
 print("=" * 60)
 
 tests = [
 test_imports,
 test_config,
 test_graph_generation,
 test_topology_analysis,
 test_architecture_conversion,
 test_utils
 ]
 
 passed = 0
 total = len(tests)
 
 for test in tests:
 try:
 if test():
 passed += 1
 except Exception as e:
 print(f" Test {test.__name__} crashed: {e}")
 
 print("\n" + "=" * 60)
 print(f"Test Results: {passed}/{total} tests passed")
 print("=" * 60)
 
 if passed == total:
 print(" All tests passed! The system is working correctly.")
 return True
 else:
 print(" Some tests failed. Please check the errors above.")
 return False

if __name__ == "__main__":
 success = main()
 sys.exit(0 if success else 1)
