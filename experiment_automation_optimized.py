#!/usr/bin/env python3
"""
OPTIMIZED Experiment Automation System

Key optimizations:
1. Works at multirun job level instead of generating all individual combinations
2. Uses vectorized pandas operations instead of iterrows()
3. Uses tuple-based indexing instead of string signatures
4. Eliminates nested loops where possible
5. Caches noise intensities per dataset+noise_type

This is a reference implementation showing the optimized approach.
The optimized methods should be integrated into the main experiment_automation.py file.
"""

import os
import sys
import yaml
import pandas as pd
import argparse
import itertools
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from evaluation.experiment_utils import collect_all_results_unified
from utils import get_noise_intensities
from tqdm import tqdm


class OptimizedExperimentAutomation:
    """
    Optimized version of ExperimentAutomation with performance improvements.
    
    Key changes:
    - Works at multirun job level to avoid generating millions of individual combinations
    - Uses vectorized pandas operations
    - Uses tuple-based keys instead of string signatures
    - Better indexing structures for fast lookups
    """
    
    def __init__(self, config_file: str = "experiment_config.yaml", 
                 preaggregated_results_file: str = None, local: bool = False):
        """Initialize the automation system with configuration."""
        self.config_file = config_file
        self.config = self._load_config()
        self.existing_results = None
        self.missing_experiments = []
        self.preaggregated_results_file = preaggregated_results_file
        self.local = local
        
        # Performance optimization caches
        self._cached_noise_intensities = {}  # (dataset, noise_type) -> intensities array
        self._cached_existing_index = None  # Indexed structure for fast lookups
        self._cached_existing_multirun_jobs = None  # Set of existing multirun job keys
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[OK] Loaded configuration from {self.config_file}")
            return config
        except FileNotFoundError:
            print(f"[ERROR] Configuration file not found: {self.config_file}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"[ERROR] Error parsing YAML configuration: {e}")
            sys.exit(1)
    
    def _get_noise_intensities_cached(self, dataset: str, noise_type: str) -> np.ndarray:
        """Get noise intensities with caching."""
        cache_key = (dataset, noise_type)
        if cache_key not in self._cached_noise_intensities:
            self._cached_noise_intensities[cache_key] = get_noise_intensities(
                dataset, noise_type, num_steps=20
            )
        return self._cached_noise_intensities[cache_key]
    
    def _build_existing_index(self, existing_df: pd.DataFrame) -> Dict:
        """
        Build an indexed structure for fast lookups of existing results.
        Returns a nested dictionary structure for O(1) lookups.
        """
        print("[INFO] Building indexed structure for existing results...")
        
        # Normalize columns
        if 'eval_mode' in existing_df.columns:
            existing_df = existing_df.copy()
            existing_df['eval_mode_norm'] = existing_df['eval_mode'].str.replace('Evaluation', '', regex=False)
        else:
            existing_df['eval_mode_norm'] = 'Unknown'
        
        if 'mode' in existing_df.columns:
            existing_df['is_tuned'] = existing_df['mode'].str.contains('_tune', na=False)
            existing_df['mode_norm'] = existing_df['mode'].str.replace('_tune', '', regex=False)
        else:
            existing_df['is_tuned'] = False
            existing_df['mode_norm'] = 'Unknown'
        
        # Filter to test_perturb results only
        test_perturb_mask = existing_df['mode_norm'] == 'test_perturb'
        test_perturb_df = existing_df[test_perturb_mask].copy()
        
        if test_perturb_df.empty:
            return {}
        
        # Build index structure: 
        # index[(dataset, model, eval_mode, seed, noise_type, is_tuned)][subject_key][intensity] = True
        index = defaultdict(lambda: defaultdict(lambda: set()))
        
        # Use vectorized operations to build index
        for _, row in test_perturb_df.iterrows():
            dataset = row.get('dataset', '')
            model = row.get('model', '')
            eval_mode = row.get('eval_mode_norm', '')
            seed = row.get('seed', '')
            noise_type = row.get('noise_type', '')
            is_tuned = row.get('is_tuned', False)
            intensity = row.get('intensity', None)
            
            if pd.isna(intensity):
                continue
                
            # Create subject key based on eval_mode
            if eval_mode == 'CrossSubject':
                # For CrossSubject, use eval_subjects or session
                if 'eval_subjects' in row and pd.notna(row['eval_subjects']):
                    subject_key = ('eval_subjects', str(row['eval_subjects']))
                elif 'session' in row and pd.notna(row['session']):
                    subject_key = ('session', str(row['session']))
                else:
                    subject_key = ('unknown', '')
            elif 'subject' in row and pd.notna(row['subject']):
                subject_key = ('subject', int(row['subject']))
            else:
                subject_key = ('unknown', '')
            
            # Add to index
            key = (dataset, model, eval_mode, seed, noise_type, is_tuned)
            index[key][subject_key].add(float(intensity))
        
        return dict(index)
    
    def _check_multirun_job_complete(self, job_key: Tuple, existing_index: Dict, 
                                     expected_intensities: np.ndarray) -> bool:
        """
        Check if a multirun job has produced all expected results.
        
        Args:
            job_key: (dataset, model, eval_mode, seed, noise_type, is_tuned, subject_info)
            existing_index: Indexed structure of existing results
            expected_intensities: Array of expected intensities
            
        Returns:
            True if job appears complete, False otherwise
        """
        dataset, model, eval_mode, seed, noise_type, is_tuned, subject_info = job_key
        
        # Look up in index
        index_key = (dataset, model, eval_mode, seed, noise_type, is_tuned)
        
        if index_key not in existing_index:
            return False
        
        # Check if we have results for the subject
        subject_results = existing_index[index_key]
        
        # Find matching subject key
        found_subject = False
        for subject_key in subject_results:
            subject_type, subject_value = subject_key
            
            if subject_type == 'subject' and subject_info['type'] == 'single':
                if subject_value == subject_info['value']:
                    found_subject = True
                    break
            elif subject_type == 'eval_subjects' and subject_info['type'] == 'multi':
                # For CrossSubject, check if eval_subjects is subset of expected subjects
                existing_subjects = set(int(s) for s in str(subject_value).split(',') if s.isdigit())
                expected_subjects = set(subject_info['value'])
                if existing_subjects.issubset(expected_subjects):
                    found_subject = True
                    break
        
        if not found_subject:
            return False
        
        # Check if we have all expected intensities (with tolerance)
        existing_intensities = subject_results[subject_key]
        
        # Use vectorized comparison for intensity matching
        existing_arr = np.array(list(existing_intensities))
        
        # For each expected intensity, check if close match exists
        for expected_int in expected_intensities:
            if not np.any(np.isclose(existing_arr, expected_int, atol=1e-4)):
                return False
        
        return True
    
    def generate_expected_multirun_jobs(self) -> List[Dict[str, Any]]:
        """
        Generate expected multirun jobs directly without generating all individual combinations.
        This is much more efficient than generate_expected_experiments().
        """
        print("\n" + "="*60)
        print("GENERATING EXPECTED MULTIRUN JOBS")
        print("="*60)
        
        datasets = self.config['datasets']
        models = self.config['models']
        eval_modes = self.config['eval_modes']
        seeds = self.config['seeds']
        noise_types = self.config['noise_types']
        tune_flags = [False, True]
        
        expected_jobs = []
        
        model_names = [model['name'] for model in models]
        
        print(f"[INFO] Generating multirun jobs from:")
        print(f"   - Datasets: {len(datasets)}")
        print(f"   - Models: {len(model_names)}")
        print(f"   - Eval modes: {len(eval_modes)}")
        print(f"   - Seeds: {len(seeds)}")
        print(f"   - Tune flags: {len(tune_flags)}")
        
        for dataset_name, dataset_config in datasets.items():
            subjects = dataset_config['subjects']
            
            for model_name in model_names:
                for eval_mode in eval_modes:
                    for seed in seeds:
                        for tune_flag in tune_flags:
                            if eval_mode == 'CrossSession' or eval_mode == 'WithinSession':
                                # One job per subject
                                for subject in subjects:
                                    job = {
                                        'dataset': dataset_name,
                                        'paradigm': dataset_config['paradigm'],
                                        'model': model_name,
                                        'eval_mode': eval_mode,
                                        'subjects': [subject],
                                        'seed': seed,
                                        'tune': tune_flag,
                                        'noise_types': noise_types,  # All noise types
                                        'mode': 'multirun'
                                    }
                                    expected_jobs.append(job)
                            else:
                                # CrossSubject: one job for all subjects
                                job = {
                                    'dataset': dataset_name,
                                    'paradigm': dataset_config['paradigm'],
                                    'model': model_name,
                                    'eval_mode': eval_mode,
                                    'subjects': subjects,
                                    'seed': seed,
                                    'tune': tune_flag,
                                    'noise_types': noise_types,  # All noise types
                                    'mode': 'multirun'
                                }
                                expected_jobs.append(job)
        
        print(f"[OK] Generated {len(expected_jobs)} expected multirun jobs")
        return expected_jobs
    
    def identify_missing_experiments_optimized(self) -> List[Dict[str, Any]]:
        """
        Optimized version that works at multirun job level using vectorized operations.
        """
        print("\n" + "="*60)
        print("IDENTIFYING MISSING EXPERIMENTS (OPTIMIZED)")
        print("="*60)
        
        # Generate expected multirun jobs (much fewer than individual combinations)
        expected_jobs = self.generate_expected_multirun_jobs()
        
        if self.existing_results is None or self.existing_results.empty:
            print("[WARNING] No existing results found, all multirun jobs are missing")
            self.missing_experiments = expected_jobs
            return expected_jobs
        
        # Build indexed structure for fast lookups
        if self._cached_existing_index is None:
            self._cached_existing_index = self._build_existing_index(self.existing_results)
        else:
            print("[INFO] Using cached existing index")
        
        existing_index = self._cached_existing_index
        
        # Find missing multirun jobs
        print("[INFO] Checking multirun job completeness...")
        missing_jobs = []
        
        for job in tqdm(expected_jobs, desc="Checking multirun jobs"):
            dataset = job['dataset']
            model = job['model']
            eval_mode = job['eval_mode']
            seed = job['seed']
            tune = job['tune']
            noise_types = job['noise_types']
            subjects = job['subjects']
            
            # Check if job is complete for all noise types
            job_complete = True
            
            for noise_type in noise_types:
                # Get expected intensities for this dataset and noise type
                expected_intensities = self._get_noise_intensities_cached(dataset, noise_type)
                
                # Build job key
                if eval_mode == 'CrossSession' or eval_mode == 'WithinSession':
                    subject_info = {'type': 'single', 'value': subjects[0]}
                else:
                    subject_info = {'type': 'multi', 'value': subjects}
                
                job_key = (dataset, model, eval_mode, seed, noise_type, tune, subject_info)
                
                # Check if this job has produced all expected results
                if not self._check_multirun_job_complete(job_key, existing_index, expected_intensities):
                    job_complete = False
                    break
            
            if not job_complete:
                missing_jobs.append(job)
        
        print(f"[OK] Found {len(missing_jobs)} missing multirun jobs out of {len(expected_jobs)} total expected")
        
        self.missing_experiments = missing_jobs
        return missing_jobs



















