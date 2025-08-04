"""
Fitness Evaluator for HyperNEAT CfC Evolution

This module evaluates the fitness of HyperNEAT-evolved CfC networks
on EEG classification tasks, including noise resilience testing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import time
import logging

from .cfc_phenotype import CfCPhenotype, HyperNEATCfC
from .hyperneat_genome import HyperNEATGenome

import sys

class HyperNEATFitnessEvaluator:
    """
    Evaluates fitness of HyperNEAT-evolved CfC networks.
    
    This class trains and evaluates CfC networks developed from HyperNEAT genomes
    on EEG classification tasks, measuring accuracy, noise resilience, and complexity.
    """
    
    def __init__(
        self,
        substrate,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        max_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = 'cpu',
        noise_types: List[str] = None,
        noise_intensities: List[float] = None
    ):
        self.substrate = substrate
        self.phenotype = CfCPhenotype(substrate)
        
        # Data
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        # Training parameters
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Noise testing parameters
        self.noise_types = noise_types or ['dropout', 'gaussian', 'eog']
        self.noise_intensities = noise_intensities or [0.1, 0.25, 0.5]
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def evaluate_genome(self, genome: HyperNEATGenome) -> Dict[str, float]:
        """
        Evaluate a HyperNEAT genome by developing it into a CfC network and testing it.
        
        Args:
            genome: HyperNEAT genome to evaluate
            
        Returns:
            Dictionary containing fitness scores
        """
        try:
            # Develop genome into CfC network
            model = self.phenotype.develop(genome)
            model.to(self.device)
            
            # Log architecture information
            if hasattr(model, 'use_evolved_architecture') and model.use_evolved_architecture:
                self.logger.info(f"Using evolved HyperNEAT architecture with {len(model.connections)} connections")
            else:
                self.logger.info("Using standard CfC architecture")
            
            # Train the model
            train_accuracy = self._train_model(model)
            
            # Evaluate on clean test data
            clean_accuracy = self._evaluate_model(model, self.test_data[0], self.test_data[1])
            
            # Test noise resilience
            noise_resilience = self._evaluate_noise_resilience(model)
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(model)
            
            # Log connection information if using evolved architecture
            if hasattr(model, 'connections') and model.connections:
                evolved_connections = len(model.connections)
                print(f"Evolved connections: {evolved_connections}")
            
            print(f"Complexity score: {complexity_score}")
            print(f"Noise resilience: {noise_resilience}")
            print(f"Clean accuracy: {clean_accuracy}")
            print(f"Train accuracy: {train_accuracy}")
            # Calculate overall fitness
            overall_fitness = self._calculate_overall_fitness(
                clean_accuracy, noise_resilience, complexity_score
            )
            print(f"Overall fitness: {overall_fitness}")
            
            # Update genome fitness
            genome.fitness = overall_fitness
            
            return {
                'clean_accuracy': clean_accuracy,
                'train_accuracy': train_accuracy,
                'noise_resilience': noise_resilience,
                'complexity_score': complexity_score,
                'overall_fitness': overall_fitness,
                'parameter_count': model.get_parameter_count()
            }
            
        except Exception as e:
            print(e)
            print(e.traceback)
            sys.exit(-1)
            self.logger.warning(f"Evaluation failed for genome: {e}")
            # Return minimal fitness for failed evaluations
            return {
                'clean_accuracy': 0.0,
                'train_accuracy': 0.0,
                'noise_resilience': 0.0,
                'complexity_score': 1.0,  # High complexity penalty
                'overall_fitness': 0.0,
                'parameter_count': 0
            }
    
    def _train_model(self, model: HyperNEATCfC) -> float:
        """Train the CfC model and return training accuracy"""
        model.train()
        
        # Prepare data
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs, _ = model(X_val_tensor)
                val_preds = torch.argmax(val_outputs, dim=1)
                val_acc = accuracy_score(y_val, val_preds.cpu().numpy())
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    break
        
        return best_val_acc
    
    def _evaluate_model(self, model: HyperNEATCfC, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model on given data"""
        model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        with torch.no_grad():
            outputs, _ = model(X_tensor)
            preds = torch.argmax(outputs, dim=1)
            accuracy = accuracy_score(y, preds.cpu().numpy())
        
        return accuracy
    
    def _evaluate_noise_resilience(self, model: HyperNEATCfC) -> float:
        """Evaluate model's resilience to different types of noise"""
        model.eval()
        
        X_test, y_test = self.test_data
        resilience_scores = []
        
        for noise_type in self.noise_types:
            for intensity in self.noise_intensities:
                # Apply noise to test data
                X_noisy = self._apply_noise(X_test, noise_type, intensity)
                
                # Evaluate on noisy data
                accuracy = self._evaluate_model(model, X_noisy, y_test)
                resilience_scores.append(accuracy)
        
        # Return average resilience score
        return np.mean(resilience_scores) if resilience_scores else 0.0
    
    def _apply_noise(self, X: np.ndarray, noise_type: str, intensity: float) -> np.ndarray:
        """Apply noise to the data"""
        X_noisy = X.copy()
        
        if noise_type == 'dropout':
            # Randomly zero out some features
            mask = np.random.random(X.shape) > intensity
            X_noisy = X_noisy * mask
            
        elif noise_type == 'gaussian':
            # Add Gaussian noise
            noise = np.random.normal(0, intensity, X.shape)
            X_noisy = X_noisy + noise
            
        elif noise_type == 'eog':
            # Simulate EOG artifacts (add high-amplitude, low-frequency components)
            batch_size, seq_len, n_channels = X.shape
            for i in range(batch_size):
                # Add random EOG-like artifact to random channels
                artifact_channels = np.random.choice(n_channels, size=max(1, n_channels//4), replace=False)
                for ch in artifact_channels:
                    # Create low-frequency artifact
                    artifact = np.sin(2 * np.pi * np.random.uniform(0.1, 0.5) * np.arange(seq_len))
                    artifact *= intensity * np.random.uniform(0.5, 2.0)
                    X_noisy[i, :, ch] += artifact
        
        return X_noisy
    
    def _calculate_complexity_score(self, model: HyperNEATCfC) -> float:
        """Calculate complexity score based on model parameters and structure"""
        param_count = model.get_parameter_count()
        
        # Normalize by typical ranges (0 = simple, 1 = complex)
        # Assuming typical CfC models have 10k-100k parameters
        param_score = min(param_count / 5000, 1.0)
        
        # Consider connection density
        wiring = model.wiring
        total_possible_connections = (wiring.input_dim * wiring.units + 
                                   wiring.units * wiring.units + 
                                   wiring.units * wiring.output_dim)
        # Count non-zero elements in adjacency matrix (actual connections)
        actual_connections = np.count_nonzero(wiring.adjacency_matrix)
        density_score = actual_connections / max(total_possible_connections, 1)
        
        # Combine scores (lower is better)
        complexity_score = (param_score + density_score) / 2
        return complexity_score
    
    def _calculate_overall_fitness(
        self, 
        accuracy: float, 
        noise_resilience: float, 
        complexity_score: float
    ) -> float:
        """Calculate overall fitness score"""
        # Weights for different objectives
        weights = {
            'accuracy': 1.0,
            'noise_resilience': 0.1,
            'complexity': 0.1
        }
        
        # Complexity penalty (lower complexity is better)
        complexity_penalty = 1.0 - complexity_score
        
        # Calculate weighted fitness
        fitness = (
            weights['accuracy'] * accuracy +
            weights['noise_resilience'] * noise_resilience +
            weights['complexity'] * complexity_penalty
        )
        
        return fitness
    
    def evaluate_population(self, genomes: List[HyperNEATGenome]) -> List[Dict[str, float]]:
        """Evaluate a population of genomes"""
        results = []
        
        for i, genome in enumerate(genomes):
            self.logger.info(f"Evaluating genome {i+1}/{len(genomes)}")
            result = self.evaluate_genome(genome)
            results.append(result)
        
        return results
    
    def get_best_genome(self, genomes: List[HyperNEATGenome]) -> Tuple[HyperNEATGenome, Dict[str, float]]:
        """Get the best genome from a population"""
        results = self.evaluate_population(genomes)
        
        # Find best genome
        best_idx = np.argmax([r['overall_fitness'] for r in results])
        best_genome = genomes[best_idx]
        best_result = results[best_idx]
        
        return best_genome, best_result 