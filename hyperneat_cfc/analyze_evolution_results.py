#!/usr/bin/env python3
"""
Analyze HyperNEAT Evolution Results

This script analyzes the results from HyperNEAT evolution, loads the best evolved model,
and tests it on EEG data to evaluate its performance.
"""

import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns



# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from hyperneat_cfc import (
    CfCSubstrate,
    HyperNEATGenome,
    CfCPhenotype
)

from run_hyperneat_demo import load_real_eeg_data

# Import MOABB and related modules
from moabb.datasets import BNCI2014_001
from sklearn.preprocessing import LabelEncoder
from config import get_paradigm


class HyperNEATResultsAnalyzer:
    """Analyzer for HyperNEAT evolution results."""
    
    def __init__(self, results_dir: str = "../hyperneat_real_eeg_results"):
        self.results_dir = results_dir
        self.evolution_history = None
        self.best_genome = None
        self.best_model = None
        
    def load_results(self):
        """Load evolution results from files."""
        print("Loading HyperNEAT evolution results...")
        
        # Load evolution history
        history_path = os.path.join(self.results_dir, "evolution_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.evolution_history = json.load(f)
            print(f"Loaded evolution history: {len(self.evolution_history)} generations")
        else:
            print(f"Warning: Evolution history not found at {history_path}")
            
        # Load best genome
        best_genome_path = os.path.join(self.results_dir, "best_genome_final.json")
        if os.path.exists(best_genome_path):
            self.best_genome = HyperNEATGenome.load(best_genome_path)
            print(f"Loaded best genome with fitness: {self.best_genome.fitness:.4f}")
        else:
            print(f"Warning: Best genome not found at {best_genome_path}")
            
    def analyze_evolution_progress(self):
        """Analyze and plot evolution progress."""
        if not self.evolution_history:
            print("No evolution history available for analysis")
            return
            
        print("\nAnalyzing evolution progress...")
        
        generations = [gen['generation'] for gen in self.evolution_history]
        best_fitness = [gen['best_fitness'] for gen in self.evolution_history]
        avg_fitness = [gen['avg_fitness'] for gen in self.evolution_history]
        worst_fitness = [gen['worst_fitness'] for gen in self.evolution_history]
        std_fitness = [gen['std_fitness'] for gen in self.evolution_history]
        
        # Print summary statistics
        print(f"Evolution Summary:")
        print(f"  Generations: {len(generations)}")
        print(f"  Initial best fitness: {best_fitness[0]:.4f}")
        print(f"  Final best fitness: {best_fitness[-1]:.4f}")
        print(f"  Fitness improvement: {best_fitness[-1] - best_fitness[0]:.4f}")
        print(f"  Best fitness overall: {max(best_fitness):.4f} (generation {generations[best_fitness.index(max(best_fitness))]})") 
        
        # Create evolution plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Fitness over generations
        ax1.plot(generations, best_fitness, 'g-', label='Best Fitness', linewidth=2)
        ax1.plot(generations, avg_fitness, 'b-', label='Average Fitness', linewidth=2)
        ax1.plot(generations, worst_fitness, 'r-', label='Worst Fitness', linewidth=2)
        ax1.fill_between(generations, 
                        [avg - std for avg, std in zip(avg_fitness, std_fitness)],
                        [avg + std for avg, std in zip(avg_fitness, std_fitness)],
                        alpha=0.3, color='blue', label='±1 Std Dev')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('HyperNEAT Evolution Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Fitness diversity over generations
        ax2.plot(generations, std_fitness, 'purple', linewidth=2, label='Fitness Std Dev')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness Standard Deviation')
        ax2.set_title('Population Diversity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.results_dir, "evolution_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Evolution analysis plot saved to: {output_path}")
        plt.show()
        
    def analyze_genome_structure(self):
        """Analyze the structure of the best evolved genome."""
        if not self.best_genome:
            print("No best genome available for analysis")
            return
            
        print("\nAnalyzing best genome structure...")
        
        # Count nodes by type
        node_types = {}
        activation_types = {}
        for node in self.best_genome.nodes:
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
            activation_types[node.activation] = activation_types.get(node.activation, 0) + 1
            
        print(f"Genome Structure:")
        print(f"  Total nodes: {len(self.best_genome.nodes)}")
        for node_type, count in node_types.items():
            print(f"    {node_type}: {count}")
            
        print(f"  Activation functions:")
        for activation, count in activation_types.items():
            print(f"    {activation}: {count}")
            
        # Count connections
        enabled_connections = [conn for conn in self.best_genome.connections if conn.enabled]
        print(f"  Total connections: {len(self.best_genome.connections)}")
        print(f"  Enabled connections: {len(enabled_connections)}")
        print(f"  Connection density: {len(enabled_connections) / len(self.best_genome.nodes)**2:.3f}")
        
        # Analyze connection weights
        weights = [conn.weight for conn in enabled_connections]
        if weights:
            print(f"  Connection weights:")
            print(f"    Mean: {np.mean(weights):.3f}")
            print(f"    Std: {np.std(weights):.3f}")
            print(f"    Range: [{np.min(weights):.3f}, {np.max(weights):.3f}]")
            
    def instantiate_best_model(self):
        """Instantiate the best evolved model."""
        if not self.best_genome:
            print("No best genome available")
            return None
            
        print("\nInstantiating best evolved model...")
        
        # Create substrate with same parameters as training
        substrate = CfCSubstrate(
            input_size=22,  # Number of EEG channels from BNCI2014_001
            hidden_size=6,  # Same as in run_hyperneat_demo.py
            output_size=4,  # Number of classes in BNCI2014_001
            layout_type="hierarchical"
        )

        substrate.visualize_layout()
        
        # Create phenotype
        phenotype = CfCPhenotype(substrate)
        
        # Develop the genome into a model with EEG parameters
        self.best_model = phenotype.develop(self.best_genome, n_chans=22, n_times=1000)
        
        print(f"Model instantiated:")
        print(f"  Input size: {substrate.input_size}")
        print(f"  Hidden size: {substrate.hidden_size}")
        print(f"  Output size: {substrate.output_size}")
        print(f"  Parameters: {self.best_model.get_parameter_count():,}")
        
        return self.best_model
        
    def load_test_data(self, subject_id=1, resample=250.0):
        """Load EEG test data."""
        print(f"\nLoading test EEG data for subject {subject_id}...")
        
        # Load dataset
        dataset = BNCI2014_001()
        dataset.subject_list = [subject_id]
        
        # Get paradigm
        paradigm = get_paradigm(resample=resample)
        
        # Load data
        X, y, metadata = paradigm.get_data(dataset, subjects=[subject_id])
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print(f"Test data loaded:")
        print(f"  Shape: {X.shape}")
        print(f"  Classes: {len(np.unique(y_encoded))}")
        print(f"  Samples per class: {np.bincount(y_encoded)}")
        
        return X, y_encoded, metadata, label_encoder
        
    def train_model(self, X_train, y_train, X_val, y_val, max_epochs=50, batch_size=16, learning_rate=1e-3):
        """Train the evolved model."""
        if not self.best_model:
            print("No model available for training")
            return 0.0
            
        print(f"\nTraining evolved model...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        
        # Set device
        device = torch.device('cpu')  # Use CPU for compatibility
        self.best_model.to(device)
        self.best_model.train()
        
        # Convert to tensors and transpose to (batch, time, channels)
        X_train_tensor = torch.FloatTensor(X_train).transpose(1, 2).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).transpose(1, 2).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
        
        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.best_model.parameters(), lr=learning_rate)
        
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        
        print(f"\nStarting training...")
        
        for epoch in range(max_epochs):
            try:
                # Training phase
                self.best_model.train()
                optimizer.zero_grad()
                
                # Forward pass
                outputs, _ = self.best_model(X_train_tensor)
                
                # Calculate loss
                loss = criterion(outputs, y_train_tensor)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at epoch {epoch}")
                    break
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Validation phase
                self.best_model.eval()
                with torch.no_grad():
                    val_outputs, _ = self.best_model(X_val_tensor)
                    val_preds = torch.argmax(val_outputs, dim=-1)
                    val_acc = accuracy_score(y_val, val_preds.cpu().numpy())
                    
                    train_losses.append(loss.item())
                    val_accuracies.append(val_acc)
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Print progress
                    if (epoch + 1) % 10 == 0:
                        print(f"  Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, Val Acc = {val_acc:.4f}")
                    
                    # Early stopping
                    # if patience_counter >= patience:
                    #     print(f"  Early stopping at epoch {epoch+1}")
                    #     break
                        
            except Exception as e:
                print(f"Training error at epoch {epoch}: {e}")
                break
        
        print(f"Training completed!")
        print(f"  Best validation accuracy: {best_val_acc:.4f}")
        
        # Plot training curves
        if train_losses and val_accuracies:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Training loss
            ax1.plot(train_losses)
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
            
            # Validation accuracy
            ax2.plot(val_accuracies)
            ax2.set_title('Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = os.path.join(self.results_dir, "training_curves.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {output_path}")
            plt.show()
        
        return best_val_acc

    def test_model_performance(self, X_test, y_test, batch_size=16):
        """Test the evolved model performance."""
        if not self.best_model:
            print("No model available for testing")
            return None, None
            
        print(f"\nTesting model performance on {len(X_test)} samples...")
        
        # Set model to evaluation mode
        self.best_model.eval()
        device = torch.device('cpu')
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_X = X_test[i:i+batch_size]
                
                # Convert to tensor and ensure correct shape: (batch, time, channels)
                batch_X_tensor = torch.FloatTensor(batch_X).transpose(1, 2).to(device)
                
                # Forward pass
                try:
                    output, _ = self.best_model(batch_X_tensor)
                    # Get predictions from output
                    batch_predictions = torch.argmax(output, dim=-1)
                    predictions.extend(batch_predictions.cpu().numpy())
                except Exception as e:
                    print(f"Error during forward pass: {e}")
                    print(f"Input shape: {batch_X_tensor.shape}")
                    import traceback
                    traceback.print_exc()
                    return None, None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Model Performance:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, predictions))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Evolved HyperNEAT CfC Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        output_path = os.path.join(self.results_dir, "confusion_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {output_path}")
        plt.show()
        
        return accuracy, predictions
        
    def run_full_analysis(self):
        """Run complete analysis of evolution results."""
        print("=" * 60)
        print("HyperNEAT Evolution Results Analysis")
        print("=" * 60)
        
        # Load results
        self.load_results()
        
        # Analyze evolution progress
        self.analyze_evolution_progress()
        
        # Analyze genome structure
        self.analyze_genome_structure()
        
        # Instantiate best model
        self.instantiate_best_model()
        
        # Load dataset
        dataset = BNCI2014_001()
        dataset.subject_list = [1]
        
        # Get paradigm
        paradigm = get_paradigm(resample=250.0)
        
        # Load data
        X, y, metadata = paradigm.get_data(dataset, subjects=[1])
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print(f"Full data shape: {X.shape}")
        print(f"Labels: {np.unique(y_encoded)}")
        print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        print(f"Metadata columns: {metadata.columns.tolist()}")
        
        train_ratio = 0.8
        val_ratio = 0.1
        y = y_encoded
        
        # Filter to training session only (0train)
        if 'session' in metadata.columns:
            train_mask = metadata['session'] == '0train'
            
            if train_mask.sum() > 0:
                X_train = X[train_mask]
                y_train = y_encoded[train_mask]
                train_metadata = metadata[train_mask]
            else:
                raise ValueError("No training session data found")

            test_mask = metadata['session'] == '1test'
            if test_mask.sum() > 0:
                X_test = X[test_mask]
                y_test = y_encoded[test_mask]
                test_metadata = metadata[test_mask]
                
                print(f"Test session (1test) data:")
                print(f"  Samples: {len(X_test)}")  
                print(f"  Shape: {X_test.shape}")
            else:
                raise ValueError("No test session data found")
        else:
            raise ValueError("No session column found in metadata")
        
        
        n_samples = len(X_train)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        
        # Split data
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        
        print(f"Training session (0train) data:")
        print(f"  Samples: {len(X_train)}")
        print(f"  Shape: {X_train.shape}")

        
        # Train the evolved model
        print(f"\n" + "="*60)
        print("TRAINING THE EVOLVED MODEL")
        print("="*60)
        train_accuracy = self.train_model(X_train, y_train, X_val, y_val, max_epochs=50)
        
        # Test the trained model
        print(f"\n" + "="*60)
        print("TESTING THE TRAINED MODEL")
        print("="*60)
        test_accuracy, predictions = self.test_model_performance(X_test, y_test)
        
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Evolution completed over {len(self.evolution_history)} generations")
        print(f"Best evolved genome fitness: {self.best_genome.fitness:.4f}")
        print(f"Model parameters: {self.best_model.get_parameter_count():,}")
        print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        if test_accuracy is not None:
            print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print("=" * 60)
        
        return {
            'evolution_history': self.evolution_history,
            'best_genome': self.best_genome,
            'best_model': self.best_model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'predictions': predictions
        }


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze HyperNEAT evolution results")
    parser.add_argument("--results-dir", type=str, default="../hyperneat_real_eeg_results",
                       help="Directory containing evolution results")
    parser.add_argument("--subject", type=int, default=1,
                       help="Subject ID for test data")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = HyperNEATResultsAnalyzer(args.results_dir)
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    return results


if __name__ == "__main__":
    main()