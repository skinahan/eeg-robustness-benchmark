import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation

from .robust_model_builder import RobustNeuroevolutionModelBuilder
from .architecture_genome import ArchitectureGenome
from config import get_paradigm
from augmentation.noise import EEGNoiseAugmentor, TrainOnlyNoiseClassifier
from globals import set_seeds, get_seed


class RealDataFitnessEvaluator:
    """Fitness evaluator that uses real BNCI2014_001 EEG data for testing evolved architectures"""
    
    def __init__(self, 
                 subject_list: List[int] = None,
                 resample: float = None,
                 noise_type: str = None,
                 noise_intensity: float = None,
                 max_epochs: int = 50,
                 cv_folds: int = 3):
        """
        Initialize the real data fitness evaluator
        
        Args:
            subject_list: List of subjects to use for evaluation
            resample: Resampling frequency (None for original 250Hz)
            noise_type: Type of noise to apply ('dropout', 'gaussian', 'eog', None)
            noise_intensity: Intensity of noise (0.0 to 1.0)
            max_epochs: Maximum training epochs per fold
            cv_folds: Number of cross-validation folds
        """
        self.subject_list = subject_list or list(range(1, 6))  # Use first 5 subjects for speed
        self.resample = resample
        self.noise_type = noise_type
        self.noise_intensity = noise_intensity
        self.max_epochs = max_epochs
        self.cv_folds = cv_folds
        
        # Setup dataset and paradigm
        self.dataset = BNCI2014_001()
        self.dataset.subject_list = self.subject_list
        
        self.paradigm = MotorImagery(
            events=["left_hand", "right_hand"],
            fmin=8,
            fmax=35,
            tmin=0.0,
            tmax=None,
            baseline=None,
            resample=self.resample,
            n_classes=2
        )
        
        # Determine input dimensions based on resampling
        if self.resample == 125.0:
            self.n_times = 500
        else:
            self.n_times = 1001
        self.n_chans = 22
        self.n_outputs = 2
    
    def evaluate_genome(self, genome: ArchitectureGenome) -> Dict[str, float]:
        """
        Evaluate a genome using real BNCI2014_001 data
        
        Args:
            genome: The architecture genome to evaluate
            
        Returns:
            Dictionary with fitness scores
        """
        try:
            # Create model from genome
            model = RobustNeuroevolutionModelBuilder.create_model(genome)
            classifier = RobustNeuroevolutionModelBuilder.create_classifier(genome)
            
            # Set training parameters from genome
            classifier.max_epochs = self.max_epochs
            classifier.train_split = None
            classifier.callbacks = []
            
            # Apply noise if specified
            if self.noise_type is not None and self.noise_intensity is not None:
                classifier = TrainOnlyNoiseClassifier(
                    base_pipeline=classifier,
                    noise_type=self.noise_type,
                    intensity=self.noise_intensity,
                    seed=get_seed()
                )
            
            # Get data for evaluation
            X, y, metadata = self.paradigm.get_data(self.dataset, subjects=self.subject_list)
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Simple parameter grid for hyperparameter tuning
            param_grid = {
                'optimizer__lr': [genome.learning_rate],
                'batch_size': [genome.batch_size]
            }
            
            # Cross-validation evaluation
            grid = GridSearchCV(
                classifier,
                param_grid=param_grid,
                cv=min(self.cv_folds, len(self.subject_list)),
                scoring='roc_auc',
                n_jobs=1,
                return_train_score=True
            )
            
            # Fit the model
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                grid.fit(X, y_encoded)
            
            # Extract scores
            train_score = np.mean(grid.cv_results_['mean_train_score'])
            test_score = grid.best_score_
            
            # Calculate additional metrics
            y_pred = grid.predict(X)
            accuracy = accuracy_score(y_encoded, y_pred)
            
            # Calculate complexity score
            complexity_score = genome.get_complexity_score()
            
            # Estimate noise resilience based on architecture properties
            noise_resilience = self._estimate_noise_resilience(genome, test_score)
            
            # Estimate overfitting score
            overfitting_score = max(0.0, min(1.0, train_score - test_score))
            
            # Calculate overall fitness
            overall_fitness = self._calculate_overall_fitness(
                accuracy=accuracy,
                noise_resilience=noise_resilience,
                complexity_score=complexity_score,
                overfitting_score=overfitting_score
            )
            
            return {
                'accuracy': accuracy,
                'noise_resilience': noise_resilience,
                'complexity_score': complexity_score,
                'overfitting_score': overfitting_score,
                'overall_fitness': overall_fitness,
                'train_score': train_score,
                'test_score': test_score
            }
            
        except Exception as e:
            print(f"Error evaluating genome: {e}")
            # Return default low scores for failed genomes
            return {
                'accuracy': 0.0,
                'noise_resilience': 0.0,
                'complexity_score': 0.0,
                'overfitting_score': 0.0,
                'overall_fitness': 0.0,
                'train_score': 0.0,
                'test_score': 0.0
            }
    
    def _estimate_noise_resilience(self, genome: ArchitectureGenome, test_score: float) -> float:
        """
        Estimate noise resilience based on architecture properties
        
        Args:
            genome: The architecture genome
            test_score: The test score achieved
            
        Returns:
            Estimated noise resilience score (0.0 to 1.0)
        """
        # Base resilience from test score
        base_resilience = test_score
        
        # Bonus for CfC/NCP layers (known to be noise resilient)
        cfcncp_layers = [layer for layer in genome.layers if layer.layer_type in ['cfc', 'ncp']]
        cfcncp_bonus = min(0.1, len(cfcncp_layers) * 0.05)
        
        # Bonus for appropriate regularization
        regularization_bonus = 0.0
        if genome.dropout_rate > 0.1:
            regularization_bonus += 0.05
        if genome.weight_decay > 0.001:
            regularization_bonus += 0.05
        
        # Penalty for overly complex architectures
        complexity_penalty = max(0.0, (genome.get_parameter_count() - 100000) / 1000000)
        
        # Calculate final resilience
        resilience = base_resilience + cfcncp_bonus + regularization_bonus - complexity_penalty
        return max(0.0, min(1.0, resilience))
    
    def _calculate_overall_fitness(self, 
                                 accuracy: float, 
                                 noise_resilience: float, 
                                 complexity_score: float, 
                                 overfitting_score: float) -> float:
        """
        Calculate overall fitness score
        
        Args:
            accuracy: Model accuracy
            noise_resilience: Estimated noise resilience
            complexity_score: Model complexity score
            overfitting_score: Overfitting prevention score
            
        Returns:
            Overall fitness score
        """
        weights = {
            'accuracy': 0.4,
            'noise_resilience': 0.3,
            'complexity': 0.2,
            'overfitting': 0.1
        }
        
        # Invert complexity score (lower complexity is better)
        complexity_penalty = 1.0 - complexity_score
        
        fitness = (
            weights['accuracy'] * accuracy +
            weights['noise_resilience'] * noise_resilience +
            weights['complexity'] * complexity_penalty +
            weights['overfitting'] * overfitting_score
        )
        
        return max(0.0, min(1.0, fitness))
    
    def evaluate_population(self, population: List[ArchitectureGenome]) -> List[Dict[str, float]]:
        """
        Evaluate a population of genomes
        
        Args:
            population: List of genomes to evaluate
            
        Returns:
            List of fitness dictionaries
        """
        results = []
        for i, genome in enumerate(population):
            print(f"Evaluating genome {i+1}/{len(population)}...")
            result = self.evaluate_genome(genome)
            results.append(result)
            
            # Update genome with fitness scores
            genome.accuracy = result['accuracy']
            genome.noise_resilience = result['noise_resilience']
            genome.complexity_score = result['complexity_score']
            genome.overfitting_score = result['overfitting_score']
            genome.overall_fitness = result['overall_fitness']
        
        return results


class FastRealDataEvaluator(RealDataFitnessEvaluator):
    """Fast version that uses random sampling from Subject 1 for debugging"""
    
    def __init__(self, 
                 sample_size: int = 48,
                 **kwargs):
        """
        Initialize fast evaluator with hard-coded Subject 1
        
        Args:
            sample_size: Number of EEG samples to use for evaluation (default: 48)
            **kwargs: Other parameters passed to parent
        """
        # Extract specific parameters for fast evaluation
        max_epochs = kwargs.pop('max_epochs', 10)  # Very few epochs for debugging
        cv_folds = kwargs.pop('cv_folds', 2)  # Fewer folds
        
        # Hard-code to use Subject 1 only
        self.sample_size = sample_size
        
        # Call parent constructor with remaining kwargs
        super().__init__(
            subject_list=[1],  # Hard-code to Subject 1
            max_epochs=max_epochs,
            cv_folds=cv_folds,
            **kwargs
        )
    
    def evaluate_genome(self, genome: ArchitectureGenome) -> Dict[str, float]:
        """
        Evaluate a genome using Subject 1 data with debugging
        
        Args:
            genome: The architecture genome to evaluate
            
        Returns:
            Dictionary with fitness scores
        """
        import random
        
        try:
            # Get data for Subject 1
            print(f"Getting data for Subject 1...")
            X, y, metadata = self.paradigm.get_data(self.dataset)
            print(f"Raw data shape: {X.shape}, labels: {np.unique(y)}")
            
            # Filter to only use '0train' set
            train_mask = metadata['session'] == '0train'
            X = X[train_mask]
            y = y[train_mask]
            
            metadata = metadata[train_mask]
            print(f"After train filter: {X.shape}, labels: {np.unique(y)}")
            
            # Randomly sample a subset of the data
            if len(X) > self.sample_size:
                # Randomly select indices
                indices = random.sample(range(len(X)), self.sample_size)
                X = X[indices]
                y = y[indices]
                metadata = metadata.iloc[indices]
                print(f"After sampling: {X.shape}, labels: {np.unique(y)}")
            
            # Encode labels as categorical values
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            print(f"After label encoding: {np.unique(y_encoded)}")
            
            # Create a simple train/test split for speed
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=get_seed(), stratify=y_encoded
            )
            print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            print(f"Train labels: {np.unique(y_train)}, Test labels: {np.unique(y_test)}")
            
            # Create model from genome with fixed parameters for speed
            classifier = RobustNeuroevolutionModelBuilder.create_classifier(genome)
            
            # Set simple training parameters
            classifier.max_epochs = self.max_epochs
            classifier.batch_size = 16  # Smaller batch size for debugging
            classifier.optimizer__lr = 1e-3
            classifier.optimizer__weight_decay = 1e-4
            classifier.train_split = None
            classifier.callbacks = []
            classifier.verbose = 1  # Enable verbose for debugging
            
            # Apply noise if specified
            if self.noise_type is not None and self.noise_intensity is not None:
                classifier = TrainOnlyNoiseClassifier(
                    base_pipeline=classifier,
                    noise_type=self.noise_type,
                    intensity=self.noise_intensity,
                    seed=get_seed()
                )
            
            print(f"Training classifier with {self.max_epochs} epochs...")
            # Train the classifier
            classifier.fit(X_train, y_train)
            
            print(f"Evaluating classifier...")
            # Evaluate
            y_pred = classifier.predict(X_test)
            y_pred_proba = classifier.predict_proba(X_test)
            
            print(f"Predictions: {y_pred[:10]}")
            print(f"True labels: {y_test[:10]}")
            print(f"Prediction probabilities shape: {y_pred_proba.shape}")
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            test_score = roc_auc_score(y_test, y_pred_proba[:, 1]) if len(np.unique(y_test)) > 1 else 0.5
            
            print(f"Accuracy: {accuracy}, Test Score: {test_score}")
            
            # Calculate fitness components
            complexity_score = genome.get_complexity_score()
            noise_resilience = self._estimate_noise_resilience(genome, test_score)
            overfitting_score = min(1.0, test_score / max(accuracy, 0.1))  # Simple overfitting estimate
            overall_fitness = self._calculate_overall_fitness(
                accuracy, noise_resilience, complexity_score, overfitting_score
            )
            
            return {
                'accuracy': accuracy,
                'test_score': test_score,
                'complexity_score': complexity_score,
                'noise_resilience': noise_resilience,
                'overfitting_score': overfitting_score,
                'overall_fitness': overall_fitness,
                'subject_used': 1,
                'samples_used': len(X)
            }
            
        except Exception as e:
            print(f"Error evaluating genome: {e}")
            import traceback
            traceback.print_exc()
            return {
                'accuracy': 0.0,
                'test_score': 0.0,
                'complexity_score': 0.0,
                'noise_resilience': 0.0,
                'overfitting_score': 0.0,
                'overall_fitness': 0.0,
                'subject_used': 1,
                'samples_used': 0
            } 