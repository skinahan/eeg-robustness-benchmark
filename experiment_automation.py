#!/usr/bin/env python3
"""
Comprehensive Experiment Automation System

This script provides a complete automation solution for EEG experiments:
1. Loads experimental configurations from YAML
2. Aggregates existing results from all datasets
3. Identifies missing experimental combinations
4. Generates shell scripts with missing experiment commands

Usage:
    python experiment_automation.py [--config CONFIG_FILE] [--output-dir OUTPUT_DIR]
"""

import os
import sys
import yaml
import pandas as pd
import argparse
import itertools
import pickle
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

class ExperimentAutomation:
    """Main class for experiment automation."""
    
    def __init__(self, config_file: str = "experiment_config.yaml", preaggregated_results_file: str = None, local: bool = False, use_cached: bool = False, use_optimized: bool = True, legacy: bool = False):
        """Initialize the automation system with configuration.
        
        Args:
            config_file: Path to configuration YAML file
            preaggregated_results_file: Path to pre-aggregated results CSV (skips aggregation)
            local: If True, generate Python script for local execution
            use_cached: If True, use cached processed results
            use_optimized: If True, use optimized multirun job-level approach (default: True)
            legacy: If True, use legacy experimental protocol (no subject chunking). Can also be set in config file.
        """
        self.config_file = config_file
        self.config = self._load_config()
        # Legacy mode can be set via config file or parameter (parameter takes precedence)
        config_legacy = self.config.get('experiment_settings', {}).get('legacy', False)
        self.legacy = legacy if legacy else config_legacy
        if self.legacy:
            print(f"[INFO] Legacy mode enabled: Using original experimental protocol (no subject chunking)")
        self.existing_results = None
        self.missing_experiments = []
        self.preaggregated_results_file = preaggregated_results_file
        self.local = local
        self.use_cached = use_cached
        self.use_optimized = use_optimized  # Use optimized approach by default
        # Performance optimization caches
        self._cached_noise_intensities = {}  # (dataset, noise_type) -> intensities array
        self._cached_existing_signatures = None
        self._cached_existing_df_normalized = None  # Normalized DataFrame for fast lookups
        self._cached_intensity_tolerance_map = None  # Intensity tolerance mapping
        self._cached_metadata = None  # Metadata for quick filtering (unique values, etc.)
        # Optimized approach caches
        self._cached_existing_index = None  # Indexed structure for fast lookups (optimized approach)
        # Cache file path
        self.cache_file = os.path.join(current_dir, ".experiment_cache.pkl")
        print(f"Expecting: {self.cache_file}")
        
    def _invalidate_caches(self):
        """Invalidate all performance caches."""
        self._cached_existing_signatures = None
        self._cached_existing_df_normalized = None
        self._cached_intensity_tolerance_map = None
        self._cached_metadata = None
        self._cached_existing_index = None  # Optimized approach cache
        if hasattr(self, 'expected_test_perturb_results'):
            delattr(self, 'expected_test_perturb_results')
    
    def _save_cache(self, df_normalized: pd.DataFrame, signatures: set, metadata: Dict[str, Any], existing_index: Dict = None):
        """Save processed results to cache file.
        
        Args:
            df_normalized: Normalized DataFrame for original approach
            signatures: Set of signatures for original approach
            metadata: Metadata dictionary
            existing_index: Indexed structure for optimized approach (optional)
        """
        try:
            cache_data = {
                'df_normalized': df_normalized,
                'signatures': signatures,
                'metadata': metadata,
                'existing_index': existing_index,  # Add optimized index to cache
                'timestamp': datetime.now().isoformat()
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"[OK] Saved processed results cache to {self.cache_file}")
        except Exception as e:
            print(f"[WARNING] Failed to save cache: {e}")
    
    def _load_cache(self) -> Optional[Dict[str, Any]]:
        """Load processed results from cache file."""
        if not os.path.exists(self.cache_file):
            return None
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"[OK] Loaded processed results cache from {self.cache_file}")
            print(f"[INFO] Cache timestamp: {cache_data.get('timestamp', 'unknown')}")
            return cache_data
        except Exception as e:
            print(f"[WARNING] Failed to load cache: {e}")
            return None
    
    def _apply_cache_data(self, cache_data: Dict[str, Any]) -> None:
        """Apply cached data to instance variables.
        
        Args:
            cache_data: Dictionary containing cached data
        """
        # Load optimized index if available
        if 'existing_index' in cache_data and cache_data['existing_index'] is not None:
            self._cached_existing_index = cache_data['existing_index']
            print("[INFO] Loaded optimized index from cache")
        
        # Load normalized DataFrame if available
        if 'df_normalized' in cache_data and cache_data['df_normalized'] is not None:
            self._cached_existing_df_normalized = cache_data['df_normalized']
            print("[INFO] Loaded normalized DataFrame from cache")
        
        # Load signatures if available
        if 'signatures' in cache_data and cache_data['signatures'] is not None:
            self._cached_existing_signatures = cache_data['signatures']
            print("[INFO] Loaded signatures from cache")
        
        # Load metadata if available
        if 'metadata' in cache_data and cache_data['metadata'] is not None:
            self._cached_metadata = cache_data['metadata']
            print("[INFO] Loaded metadata from cache")
        
        # Populate existing_results from cached data for compatibility
        # Prefer df_normalized if available, otherwise use empty DataFrame
        if self._cached_existing_df_normalized is not None and not self._cached_existing_df_normalized.empty:
            self.existing_results = self._cached_existing_df_normalized.copy()
            print(f"[INFO] Populated existing_results from cache ({len(self.existing_results)} rows)")
        elif self.existing_results is None or self.existing_results.empty:
            # If no cached DataFrame, create empty DataFrame for compatibility
            self.existing_results = pd.DataFrame()
            print("[INFO] No cached DataFrame available, using empty DataFrame")
    
    def _extract_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract metadata for quick filtering from DataFrame."""
        if df.empty:
            return {
                'datasets': set(),
                'models': set(),
                'eval_modes': set(),
                'seeds': set(),
                'noise_types': set(),
                'has_test_perturb': False
            }
        
        metadata = {
            'datasets': set(df['dataset'].unique()) if 'dataset' in df.columns else set(),
            'models': set(df['model'].unique()) if 'model' in df.columns else set(),
            'eval_modes': set(df['eval_mode_normalized'].unique()) if 'eval_mode_normalized' in df.columns else set(),
            'seeds': set(str(s) for s in df['seed'].unique()) if 'seed' in df.columns else set(),
            'noise_types': set(df['noise_type'].unique()) if 'noise_type' in df.columns else set(),
            'has_test_perturb': False
        }
        
        # Check if we have test_perturb results
        if 'mode_normalized' in df.columns:
            metadata['has_test_perturb'] = 'test_perturb' in df['mode_normalized'].values
        
        return metadata
    
    def _get_noise_intensities_cached(self, dataset: str, noise_type: str) -> np.ndarray:
        """Get noise intensities with caching to avoid repeated calls."""
        cache_key = (dataset, noise_type)
        if cache_key not in self._cached_noise_intensities:
            self._cached_noise_intensities[cache_key] = get_noise_intensities(
                dataset, noise_type, num_steps=20
            )
        return self._cached_noise_intensities[cache_key]
        
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
    
    def load_preaggregated_results(self) -> pd.DataFrame:
        """Load pre-aggregated results from CSV file."""
        print("\n" + "="*60)
        print("LOADING PRE-AGGREGATED RESULTS")
        print("="*60)
        
        try:
            self.existing_results = pd.read_csv(self.preaggregated_results_file)
            
            # Invalidate caches when new results are loaded
            self._invalidate_caches()
            
            print(f"[OK] Loaded {len(self.existing_results)} result rows from {self.preaggregated_results_file}")
            print(f"[INFO] Results summary:")
            print(f"   - Datasets: {self.existing_results['dataset'].unique()}")
            print(f"   - Models: {self.existing_results['model'].unique()}")
            print(f"   - Eval modes: {self.existing_results['eval_mode'].unique()}")
            print(f"   - Modes: {self.existing_results['mode'].unique()}")
            if 'seed' in self.existing_results.columns:
                print(f"   - Seeds: {sorted(self.existing_results['seed'].unique())}")
            return self.existing_results
        except FileNotFoundError:
            print(f"[ERROR] Pre-aggregated results file not found: {self.preaggregated_results_file}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Error loading pre-aggregated results: {e}")
            sys.exit(1)
    
    def aggregate_existing_results(self) -> pd.DataFrame:
        """Aggregate all existing results from the results directory."""
        print("\n" + "="*60)
        print("AGGREGATING EXISTING RESULTS")
        print("="*60)
        
        # Use the updated aggregation function
        self.existing_results = collect_all_results_unified()
        
        # Invalidate caches when new results are loaded
        self._invalidate_caches()
        
        if self.existing_results is not None:
            print(f"[OK] Aggregated {len(self.existing_results)} result rows")
            print(f"[INFO] Results summary:")
            print(f"   - Datasets: {self.existing_results['dataset'].unique()}")
            print(f"   - Models: {self.existing_results['model'].unique()}")
            print(f"   - Eval modes: {self.existing_results['eval_mode'].unique()}")
            print(f"   - Modes: {self.existing_results['mode'].unique()}")
            if 'seed' in self.existing_results.columns:
                print(f"   - Seeds: {sorted(self.existing_results['seed'].unique())}")
        else:
            print("[WARNING] No existing results found")
            self.existing_results = pd.DataFrame()
        
        return self.existing_results
    
    def generate_expected_experiments(self) -> List[Dict[str, Any]]:
        """
        Generate all expected test_perturb results based on configuration, 
        then identify which multirun jobs are needed to generate missing results.
        Optimized version using itertools.product for better performance.
        """
        print("\n" + "="*60)
        print("GENERATING EXPECTED TEST_PERTURB EXPERIMENTS")
        print("="*60)
        
        # Get base configurations
        datasets = self.config['datasets']
        models = self.config['models']
        eval_modes = self.config['eval_modes']
        seeds = self.config['seeds']
        noise_types = self.config['noise_types']
        
        # Note: We'll compute noise intensities dynamically per dataset and noise type
        # This ensures consistency with unified_experiment_runner.py
        
        # Use itertools.product for efficient cartesian product generation
        expected_test_perturb_results = []
        
        # Pre-extract values for faster iteration
        dataset_items = list(datasets.items())
        model_names = [model['name'] for model in models]
        tune_flags = [False, True]
        
        print(f"[INFO] Generating combinations from:")
        print(f"   - Datasets: {len(dataset_items)}")
        print(f"   - Models: {len(model_names)}")
        print(f"   - Eval modes: {len(eval_modes)}")
        print(f"   - Seeds: {len(seeds)}")
        print(f"   - Noise types: {len(noise_types)}")
        print(f"   - Intensities: Dynamic (based on saturation points)")
        print(f"   - Tune flags: {len(tune_flags)}")
        
        # Generate combinations using nested loops since intensities are now dynamic
        # We can't use itertools.product with dynamic intensities
        
        print("[INFO] Processing combinations...")
        total_combinations = 0
        
        # Calculate total combinations for progress tracking
        # OPTIMIZED: Pre-compute and cache all intensity arrays to avoid repeated calls
        intensity_cache = {}
        for dataset_name, dataset_config in dataset_items:
            for noise_type in noise_types:
                cache_key = (dataset_name, noise_type)
                intensities = get_noise_intensities(dataset_name, noise_type, num_steps=20)
                intensity_cache[cache_key] = intensities
                total_combinations += len(model_names) * len(eval_modes) * len(seeds) * len(intensities) * len(tune_flags)
        
        print(f"[INFO] Total combinations to process: {total_combinations}")
        
        # Process combinations with dynamic intensities
        processed = 0
        for dataset_name, dataset_config in dataset_items:
            for model_name in model_names:
                for eval_mode in eval_modes:
                    for seed in seeds:
                        for noise_type in noise_types:
                            # Get dynamic intensities from cache (avoid repeated calls)
                            cache_key = (dataset_name, noise_type)
                            intensities = intensity_cache[cache_key]
                            
                            for intensity in intensities:
                                for tune_flag in tune_flags:
                                    processed += 1
                                    if processed % 1000 == 0:
                                        print(f"[INFO] Processed {processed}/{total_combinations} combinations...")
                                    
                                    subjects = dataset_config['subjects']
                                    
                                    if eval_mode == 'CrossSession' or eval_mode == 'WithinSession':
                                        # For CrossSession and WithinSession, create separate entries for each subject
                                        for subject in subjects:
                                            experiment = {
                                                'dataset': dataset_name,
                                                'paradigm': dataset_config['paradigm'],
                                                'subject': subject,
                                                'model': model_name,
                                                'eval_mode': eval_mode,
                                                'mode': 'test_perturb',
                                                'seed': seed,
                                                'noise_type': noise_type,
                                                'intensity': intensity,
                                                'tune': tune_flag
                                            }
                                            expected_test_perturb_results.append(experiment)
                                    else:
                                        # For Cross-Subject, results are aggregated across subjects
                                        experiment = {
                                            'dataset': dataset_name,
                                            'paradigm': dataset_config['paradigm'],
                                            'subjects': subjects,  # All subjects together
                                            'model': model_name,
                                            'eval_mode': eval_mode,
                                            'mode': 'test_perturb',
                                            'seed': seed,
                                            'noise_type': noise_type,
                                            'intensity': intensity,
                                            'tune': tune_flag
                                        }
                                        expected_test_perturb_results.append(experiment)
        
        print(f"[OK] Generated {len(expected_test_perturb_results)} expected test_perturb results")
        
        # Store for use in identify_missing_experiments
        self.expected_test_perturb_results = expected_test_perturb_results
        
        # For now, return empty list - multirun jobs will be determined in identify_missing_experiments
        return []
    
    def generate_expected_multirun_jobs(self) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Generate expected multirun jobs directly without generating all individual combinations.
        This is much more efficient than generate_expected_experiments().
        
        Returns:
            List of multirun job dictionaries
        """
        print("\n" + "="*60)
        print("GENERATING EXPECTED MULTIRUN JOBS (OPTIMIZED)")
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
    
    def _build_existing_index(self, existing_df: pd.DataFrame) -> Dict:
        """
        OPTIMIZED: Build an indexed structure for fast lookups of existing results.
        Returns a nested dictionary structure for O(1) lookups.
        
        Structure: index[(dataset, model, eval_mode, seed, noise_type, is_tuned)][subject_key][intensity] = True
        
        Args:
            existing_df: DataFrame with existing results
            
        Returns:
            Nested dictionary index structure
        """
        print("[INFO] Building indexed structure for existing results (optimized)...")
        
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
        
        # Check if 'tune' column exists (preferred), otherwise use inferred value
        if 'tune' in existing_df.columns:
            existing_df['is_tuned'] = existing_df['tune'].astype(bool)
        
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
            elif eval_mode == 'WithinSession':
                # For WithinSession, combine subject and session since results are per (subject, session)
                # However, multirun jobs produce results for all sessions, so we aggregate by subject only
                # Store session-specific intensities but match by subject (aggregating across sessions)
                if 'subject' in row and pd.notna(row['subject']):
                    # Store subject only - we'll check that all sessions are present when matching
                    subject_key = ('subject', int(row['subject']))
                else:
                    subject_key = ('unknown', '')
            elif 'subject' in row and pd.notna(row['subject']):
                # For CrossSession, also use subject only (one session per job)
                subject_key = ('subject', int(row['subject']))
            else:
                subject_key = ('unknown', '')
            
            # Add to index
            key = (dataset, model, eval_mode, seed, noise_type, is_tuned)
            index[key][subject_key].add(float(intensity))
        
        # Convert nested defaultdicts to regular dicts for pickling compatibility
        # This ensures the structure can be pickled and cached
        result = {}
        for key, subject_dict in index.items():
            result[key] = {}
            for subject_key, intensity_set in subject_dict.items():
                result[key][subject_key] = intensity_set
        
        return result
    
    def _check_multirun_job_complete(self, job_key: Tuple, existing_index: Dict, 
                                     expected_intensities: np.ndarray) -> bool:
        """
        OPTIMIZED: Check if a multirun job has produced all expected results.
        
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
        matching_subject_key = None
        for subject_key in subject_results:
            subject_type, subject_value = subject_key
            
            if subject_type == 'subject' and subject_info['type'] == 'single':
                if subject_value == subject_info['value']:
                    found_subject = True
                    matching_subject_key = subject_key
                    break
            elif subject_type == 'eval_subjects' and subject_info['type'] == 'multi':
                # For CrossSubject, check if eval_subjects is subset of expected subjects
                existing_subjects = set(int(s) for s in str(subject_value).split(',') if s.isdigit())
                expected_subjects = set(subject_info['value'])
                if existing_subjects.issubset(expected_subjects):
                    found_subject = True
                    matching_subject_key = subject_key
                    break
        
        if not found_subject:
            return False
        
        # Check if we have all expected intensities (with tolerance)
        existing_intensities = subject_results[matching_subject_key]
        
        # Use vectorized comparison for intensity matching
        existing_arr = np.array(list(existing_intensities))
        
        # For each expected intensity, check if close match exists
        for expected_int in expected_intensities:
            if not np.any(np.isclose(existing_arr, expected_int, atol=1e-4)):
                return False
        
        return True
    
    def identify_missing_experiments_optimized(self) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Identify missing multirun jobs directly without generating all individual combinations.
        Works at multirun job level using vectorized operations.
        
        Returns:
            List of missing multirun job dictionaries
        """
        print("\n" + "="*60)
        print("IDENTIFYING MISSING EXPERIMENTS (OPTIMIZED)")
        print("="*60)
        
        # Generate expected multirun jobs (much fewer than individual combinations)
        expected_jobs = self.generate_expected_multirun_jobs()
        
        # Build indexed structure for fast lookups
        # Check if we have cached index first (even if existing_results is empty)
        if self._cached_existing_index is None:
            # Try to load from cache if use_cached flag is set
            if self.use_cached:
                cache_data = self._load_cache()
                if cache_data:
                    self._apply_cache_data(cache_data)
            
            # Build index if still not available
            if self._cached_existing_index is None:
                if self.existing_results is None or self.existing_results.empty:
                    print("[WARNING] No existing results found and no cached index, all multirun jobs are missing")
                    self.missing_experiments = expected_jobs
                    return expected_jobs
                # Build index from existing results
                self._cached_existing_index = self._build_existing_index(self.existing_results)
        else:
            print("[INFO] Using cached existing index")
        
        # Verify we have a valid index (should not be None at this point, but check for empty dict)
        if self._cached_existing_index is None or len(self._cached_existing_index) == 0:
            print("[WARNING] No existing index available, all multirun jobs are missing")
            self.missing_experiments = expected_jobs
            return expected_jobs
        
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
        
        # Save cache for future use (if not loading from cache)
        if not self.use_cached and self._cached_existing_index is not None:
            # Also save the normalized DataFrame and signatures for backward compatibility
            # Build minimal normalized DataFrame from existing results
            if self._cached_existing_df_normalized is None:
                # Create a minimal normalized DataFrame for cache compatibility
                if 'mode' in self.existing_results.columns:
                    test_perturb_df = self.existing_results[
                        self.existing_results['mode'].str.replace('_tune', '', regex=False) == 'test_perturb'
                    ].copy()
                else:
                    test_perturb_df = pd.DataFrame()
                self._cached_existing_df_normalized = test_perturb_df
            # Save cache with optimized index
            metadata = {}
            if not self._cached_existing_df_normalized.empty:
                metadata = self._extract_metadata(self._cached_existing_df_normalized)
            self._save_cache(
                self._cached_existing_df_normalized,
                set(),  # Signatures not needed for optimized approach
                metadata,
                self._cached_existing_index
            )
        
        self.missing_experiments = missing_jobs
        return missing_jobs
    
    def identify_missing_experiments(self) -> List[Dict[str, Any]]:
        """
        Identify missing test_perturb results and map them to required multirun jobs.
        
        Uses optimized approach by default (use_optimized=True) which works at multirun job level.
        Set use_optimized=False to use the original approach that generates all individual combinations.
        """
        # Use optimized approach by default
        if self.use_optimized:
            return self.identify_missing_experiments_optimized()
        
        # Original approach (for backward compatibility)
        print("\n" + "="*60)
        print("IDENTIFYING MISSING TEST_PERTURB RESULTS (ORIGINAL APPROACH)")
        print("="*60)
        
        # Generate expected test_perturb results (cached if already generated)
        if not hasattr(self, 'expected_test_perturb_results'):
            self.generate_expected_experiments()  # This populates self.expected_test_perturb_results
        else:
            print("[INFO] Using cached expected test_perturb results")
        
        # Initialize missing_combinations for diagnostics
        missing_combinations = {}
        
        # Try to load from cache if use_cached flag is set
        if self.use_cached:
            cache_data = self._load_cache()
            if cache_data:
                self._cached_existing_df_normalized = cache_data['df_normalized']
                self._cached_existing_signatures = cache_data['signatures']
                self._cached_metadata = cache_data.get('metadata', {})
                print("[INFO] Using cached processed results (skipping aggregation)")
                # Still need existing_results for some operations, but can be minimal
                if self.existing_results is None or self.existing_results.empty:
                    # Create minimal existing_results from cached data
                    if not self._cached_existing_df_normalized.empty:
                        self.existing_results = self._cached_existing_df_normalized.copy()
                    else:
                        self.existing_results = pd.DataFrame()
        
        if self.existing_results is None or self.existing_results.empty:
            print("[WARNING] No existing results found, all test_perturb results are missing")
            missing_test_perturb_results = self.expected_test_perturb_results
            
            # Populate missing_combinations for diagnostics when no existing results
            for expected_result in self.expected_test_perturb_results:
                combo_key = (expected_result['dataset'], expected_result['model'], 
                           expected_result['eval_mode'], expected_result['seed'],
                           expected_result['noise_type'])
                if combo_key not in missing_combinations:
                    missing_combinations[combo_key] = []
                # For CrossSubject, track subjects tuple; for others, track subject
                if 'subjects' in expected_result:
                    missing_combinations[combo_key].append(f"subjects_{expected_result['subjects']}")
                else:
                    missing_combinations[combo_key].append(expected_result.get('subject', 'no_subject'))
        else:
            # Convert expected results to DataFrame for vectorized operations
            print("[INFO] Converting expected results to DataFrame for efficient comparison...")
            expected_df = pd.DataFrame(self.expected_test_perturb_results)
            
            # Prepare existing results for comparison
            existing_df = self.existing_results.copy()
            
            # Normalize eval_mode column in existing results for matching
            if 'eval_mode' in existing_df.columns:
                existing_df['eval_mode_normalized'] = existing_df['eval_mode'].str.replace('Evaluation', '', regex=False)
            else:
                existing_df['eval_mode_normalized'] = 'Unknown'
                
            # Normalize mode column to handle tuning suffixes
            if 'mode' in existing_df.columns:
                existing_df['mode_normalized'] = existing_df['mode'].str.replace('_tune', '', regex=False)
            else:
                existing_df['mode_normalized'] = 'Unknown'
            
            # Create a set of existing result signatures for fast lookup (with caching)
            # OPTIMIZED: Use vectorized operations instead of iterrows()
            if self._cached_existing_signatures is None or self._cached_existing_df_normalized is None:
                print("[INFO] Creating lookup set for existing results (vectorized)...")
                
                # Use vectorized operations to normalize columns
                df_work = existing_df.copy()
                # Check if 'tune' column exists (preferred), otherwise infer from mode
                if 'tune' in df_work.columns:
                    # Use the 'tune' column directly if it exists
                    df_work['is_tuned'] = df_work['tune'].astype(bool)
                else:
                    # Fall back to inferring from mode string
                    df_work['is_tuned'] = df_work['mode'].astype(str).str.contains('_tune', na=False)
                
                # Filter to test_perturb results only
                test_perturb_mask = df_work['mode_normalized'] == 'test_perturb'
                df_test_perturb = df_work[test_perturb_mask].copy()
                
                if not df_test_perturb.empty:
                    # Create signatures using vectorized string operations
                    # Base signature parts
                    df_test_perturb['sig_dataset'] = df_test_perturb['dataset'].astype(str)
                    df_test_perturb['sig_model'] = df_test_perturb['model'].astype(str)
                    df_test_perturb['sig_eval_mode'] = df_test_perturb['eval_mode_normalized'].astype(str)
                    df_test_perturb['sig_seed'] = df_test_perturb['seed'].astype(str)
                    df_test_perturb['sig_noise_type'] = df_test_perturb['noise_type'].astype(str)
                    df_test_perturb['sig_intensity'] = df_test_perturb['intensity'].astype(str)
                    df_test_perturb['sig_tune'] = df_test_perturb['is_tuned'].astype(str)
                    
                    # Create subject key using vectorized operations
                    def create_subject_key(row):
                        eval_mode_norm = str(row.get('eval_mode_normalized', ''))
                        if eval_mode_norm == 'CrossSubject':
                            if pd.notna(row.get('eval_subjects')):
                                return f"eval_subjects_{row['eval_subjects']}"
                            elif pd.notna(row.get('session')):
                                return f"session_{row['session']}"
                            else:
                                return 'no_subject'
                        elif pd.notna(row.get('subject')):
                            return str(int(row['subject']))
                        else:
                            return 'no_subject'
                    
                    df_test_perturb['sig_subject'] = df_test_perturb.apply(create_subject_key, axis=1)
                    
                    # Combine into signature using vectorized concatenation
                    df_test_perturb['signature'] = (
                        df_test_perturb['sig_dataset'] + '|' +
                        df_test_perturb['sig_model'] + '|' +
                        df_test_perturb['sig_eval_mode'] + '|' +
                        df_test_perturb['sig_seed'] + '|' +
                        df_test_perturb['sig_noise_type'] + '|' +
                        df_test_perturb['sig_intensity'] + '|' +
                        'test_perturb|' +
                        df_test_perturb['sig_tune'] + '|' +
                        df_test_perturb['sig_subject']
                    )
                    
                    existing_signatures = set(df_test_perturb['signature'].values)
                    
                    # Cache normalized DataFrame for future use
                    self._cached_existing_df_normalized = df_test_perturb.copy()
                else:
                    existing_signatures = set()
                    self._cached_existing_df_normalized = pd.DataFrame()
                
                # Cache the signatures for future use
                self._cached_existing_signatures = existing_signatures
                
                # Extract metadata for quick filtering
                self._cached_metadata = self._extract_metadata(df_test_perturb)
                
                # Save cache for future use
                if not self.use_cached:  # Only save if we're not loading from cache
                    self._save_cache(
                        self._cached_existing_df_normalized,
                        self._cached_existing_signatures,
                        self._cached_metadata
                    )
            else:
                print("[INFO] Using cached existing result signatures")
                existing_signatures = self._cached_existing_signatures
                if self._cached_metadata is None:
                    self._cached_metadata = self._extract_metadata(self._cached_existing_df_normalized)
            
            # EARLY FILTERING: Quickly eliminate impossible matches using DataFrame operations
            print("[INFO] Applying early filtering to eliminate impossible matches...")
            
            # Track which expected results can be quickly determined as missing
            quickly_missing_indices = set()
            
            if self._cached_metadata and not self._cached_existing_df_normalized.empty:
                metadata = self._cached_metadata
                
                # Filter expected results based on metadata
                for idx, expected_result in enumerate(self.expected_test_perturb_results):
                    # Quick checks: if any key field doesn't exist in existing results, mark as missing immediately
                    dataset_match = expected_result['dataset'] in metadata['datasets']
                    model_match = expected_result['model'] in metadata['models']
                    eval_mode_match = expected_result['eval_mode'] in metadata['eval_modes']
                    seed_match = str(expected_result['seed']) in metadata['seeds']
                    noise_type_match = expected_result['noise_type'] in metadata['noise_types']
                    
                    # If we don't have test_perturb results at all, all are missing
                    if not metadata['has_test_perturb']:
                        quickly_missing_indices.add(idx)
                        continue
                    
                    # If any critical field doesn't match, it's definitely missing
                    if not dataset_match or not model_match or not noise_type_match or not seed_match:
                        quickly_missing_indices.add(idx)
                        continue
                    
                    # For non-CrossSubject modes, if eval_mode doesn't match, it's missing
                    if expected_result['eval_mode'] != 'CrossSubject' and not eval_mode_match:
                        quickly_missing_indices.add(idx)
                        continue
                    
                    # For CrossSubject with no matching eval_mode, also mark as missing
                    if expected_result['eval_mode'] == 'CrossSubject' and not eval_mode_match:
                        quickly_missing_indices.add(idx)
                        continue
            else:
                # No existing results or metadata, all are missing
                quickly_missing_indices = set(range(len(self.expected_test_perturb_results)))
            
            # Results that need detailed checking (not in quickly_missing_indices)
            needs_detailed_check = [idx for idx in range(len(self.expected_test_perturb_results)) 
                                   if idx not in quickly_missing_indices]
            
            print(f"[INFO] Early filtering: {len(quickly_missing_indices)} results immediately marked as missing, {len(needs_detailed_check)} need detailed checking")
            
            # OPTIMIZATION: If all results were marked as quickly missing, skip detailed checking entirely
            if len(needs_detailed_check) == 0:
                print("[INFO] All expected results were marked as missing by early filtering - skipping detailed checking")
                missing_test_perturb_results = [self.expected_test_perturb_results[idx] for idx in quickly_missing_indices]
                found_expected_indices = set()
            else:
                # Find missing results using set operations
                print("[INFO] Identifying missing experiments using vectorized comparison...")
                missing_test_perturb_results = []
                
                # Pre-compute intensity mapping for better performance
                # OPTIMIZED: Build mapping once for all unique intensities
                print("[INFO] Pre-computing intensity mappings...")
                # Get intensities from cached normalized DataFrame if available, otherwise from existing_df
                if self._cached_existing_df_normalized is not None and not self._cached_existing_df_normalized.empty:
                    existing_intensities = self._cached_existing_df_normalized['intensity'].dropna().unique()
                else:
                    existing_intensities = existing_df['intensity'].dropna().unique()
                intensity_mapping = {}
                
                if len(existing_intensities) > 0:
                    existing_intensities_arr = np.array(existing_intensities)
                    
                    # Get all unique expected intensities from results that need checking
                    expected_intensities = set()
                    for idx in needs_detailed_check:
                        expected_result = self.expected_test_perturb_results[idx]
                        expected_intensities.add(expected_result['intensity'])
                    
                    # Build mapping for all expected intensities at once
                    for expected_intensity in expected_intensities:
                        # Vectorized: Find if any existing intensity is close enough
                        matches = np.isclose(existing_intensities_arr, expected_intensity, atol=1e-4)
                        if np.any(matches):
                            # Find first matching intensity
                            matching_intensity = existing_intensities_arr[matches][0]
                            intensity_mapping[expected_intensity] = float(matching_intensity)
                        else:
                            intensity_mapping[expected_intensity] = None
                
                print("[INFO] Checking missing experiments...")
                
                # OPTIMIZED: Instead of iterating through expected results, iterate through existing results
                # and mark matching expected results as found. This is more efficient when 
                # len(existing_results) < len(expected_results), which is the typical case.
                
                # Helper function to build signature from expected result
                def build_expected_signature(expected_result, use_matching_intensity=True):
                    """Build signature for an expected result, using matching intensity if available."""
                    intensity = expected_result['intensity']
                    matching_intensity = intensity_mapping.get(intensity) if use_matching_intensity else None
                    eval_mode = expected_result['eval_mode']
                    
                    if matching_intensity is None:
                        intensity_to_use = intensity
                    else:
                        intensity_to_use = matching_intensity
                    
                    signature_parts = [
                        expected_result['dataset'],
                        expected_result['model'],
                        eval_mode,
                        str(expected_result['seed']),
                        expected_result['noise_type'],
                        str(intensity_to_use),
                        'test_perturb',
                        str(expected_result['tune'])
                    ]
                    
                    # Add subject/eval_subjects for signature
                    if eval_mode == 'CrossSubject':
                        if 'subjects' in expected_result:
                            subjects_tuple = tuple(sorted(expected_result['subjects']))
                            signature_parts.append(f"subjects_{subjects_tuple}")
                        else:
                            signature_parts.append('no_subject')
                    elif 'subject' in expected_result:
                        # Ensure subject is converted to int then string for consistency with existing results
                        signature_parts.append(str(int(expected_result['subject'])))
                    else:
                        signature_parts.append('no_subject')
                    
                    return '|'.join(str(part) for part in signature_parts)
                
                # Create a set to track which expected results have been found
                # Use a set of indices to track found expected results
                found_expected_indices = set()
                
                # For non-CrossSubject: build a mapping from signature to expected result indices
                # For CrossSubject: we'll handle separately with more complex matching
                expected_signature_to_indices = {}  # signature -> set of indices
                crosssubject_expected_results = []  # Store CrossSubject expected results separately
                
                # Only process results that need detailed checking
                for idx in needs_detailed_check:
                    expected_result = self.expected_test_perturb_results[idx]
                    eval_mode = expected_result['eval_mode']
                    if eval_mode == 'CrossSubject':
                        crosssubject_expected_results.append((idx, expected_result))
                    else:
                        # Build signature with matching intensity
                        signature = build_expected_signature(expected_result, use_matching_intensity=True)
                        if signature not in expected_signature_to_indices:
                            expected_signature_to_indices[signature] = set()
                        expected_signature_to_indices[signature].add(idx)
                
                # Now iterate through existing results and mark matching expected results as found
                # OPTIMIZATION: Only iterate if we have results that need checking
                if self._cached_existing_df_normalized is not None and not self._cached_existing_df_normalized.empty:
                    df_norm = self._cached_existing_df_normalized
                    
                    for _, existing_row in tqdm(df_norm.iterrows(), total=len(df_norm), desc="Checking existing results"):
                        existing_signature = existing_row['signature']
                        existing_eval_mode = existing_row['sig_eval_mode']
                        
                        # For non-CrossSubject modes, use exact signature matching
                        if existing_eval_mode != 'CrossSubject':
                            if existing_signature in expected_signature_to_indices:
                                # Mark all expected results with this signature as found
                                found_expected_indices.update(expected_signature_to_indices[existing_signature])
                        else:
                            # For CrossSubject, need to check subset matching
                            # Extract existing result details
                            existing_dataset = existing_row['sig_dataset']
                            existing_model = existing_row['sig_model']
                            existing_seed = existing_row['sig_seed']
                            existing_noise_type = existing_row['sig_noise_type']
                            existing_intensity = float(existing_row['sig_intensity'])
                            existing_tune = existing_row['sig_tune']
                            existing_subject_sig = str(existing_row['sig_subject'])
                            
                            # Extract eval_subjects from existing result
                            if existing_subject_sig.startswith('eval_subjects_'):
                                eval_subjects_str = existing_subject_sig.replace('eval_subjects_', '')
                                existing_eval_subjects = set(int(s) for s in eval_subjects_str.split(',') if s.isdigit())
                            else:
                                existing_eval_subjects = set()
                            
                            # Check against all CrossSubject expected results
                            for idx, expected_result in crosssubject_expected_results:
                                if idx in found_expected_indices:
                                    continue  # Already found
                                
                                # Check base criteria match
                                if (expected_result['dataset'] == existing_dataset and
                                    expected_result['model'] == existing_model and
                                    str(expected_result['seed']) == existing_seed and
                                    expected_result['noise_type'] == existing_noise_type and
                                    str(expected_result['tune']) == existing_tune):
                                    
                                    # Check intensity match (with tolerance)
                                    intensity = expected_result['intensity']
                                    matching_intensity = intensity_mapping.get(intensity)
                                    intensity_to_check = matching_intensity if matching_intensity is not None else intensity
                                    
                                    if np.isclose(existing_intensity, intensity_to_check, atol=1e-4):
                                        # Check if existing eval_subjects are subset of expected subjects
                                        if 'subjects' in expected_result:
                                            expected_subjects = set(expected_result['subjects'])
                                            if existing_eval_subjects.issubset(expected_subjects):
                                                found_expected_indices.add(idx)
                
                # Collect missing results (those not found + those quickly determined as missing)
                # Note: missing_test_perturb_results was already initialized above for early-exit case
                if len(needs_detailed_check) > 0:  # Only do this if we did detailed checking
                    missing_test_perturb_results = []
                    
                    # Add quickly determined missing results
                    for idx in quickly_missing_indices:
                        missing_test_perturb_results.append(self.expected_test_perturb_results[idx])
                    
                    # Add results that were checked but not found
                    for idx in needs_detailed_check:
                        if idx not in found_expected_indices:
                            missing_test_perturb_results.append(self.expected_test_perturb_results[idx])
            
            # Build missing_combinations for diagnostics
            # Collect all missing indices (quickly missing + not found in detailed check)
            missing_combinations = {}
            all_missing_indices = quickly_missing_indices.copy()
            if len(needs_detailed_check) > 0:
                all_missing_indices.update(set(needs_detailed_check) - found_expected_indices)
            
            for idx in all_missing_indices:
                expected_result = self.expected_test_perturb_results[idx]
                
                # Track missing combinations for diagnostics
                combo_key = (expected_result['dataset'], expected_result['model'], 
                           expected_result['eval_mode'], expected_result['seed'],
                           expected_result['noise_type'])
                if combo_key not in missing_combinations:
                    missing_combinations[combo_key] = []
                # For CrossSubject, track subjects tuple; for others, track subject
                if 'subjects' in expected_result:
                    missing_combinations[combo_key].append(f"subjects_{expected_result['subjects']}")
                else:
                    missing_combinations[combo_key].append(expected_result.get('subject', 'no_subject'))
        
        print(f"[OK] Found {len(missing_test_perturb_results)} missing test_perturb results out of {len(self.expected_test_perturb_results)} total expected")
        
        # Print detailed diagnostics of missing combinations
        if missing_combinations:
            print(f"\n[INFO] Missing combinations breakdown:")
            print(f"   Total missing combinations: {len(missing_combinations)}")
            
            # Group by eval_mode and noise_type for better understanding
            missing_by_eval_noise = {}
            for combo_key, subjects in missing_combinations.items():
                dataset, model, eval_mode, seed, noise_type = combo_key
                key = (eval_mode, noise_type)
                if key not in missing_by_eval_noise:
                    missing_by_eval_noise[key] = 0
                missing_by_eval_noise[key] += len(subjects) * 20  # 20 intensities per combination
            
            print(f"   Missing by eval_mode and noise_type:")
            for (eval_mode, noise_type), count in sorted(missing_by_eval_noise.items()):
                print(f"     - {eval_mode} + {noise_type}: {count} missing results")
            
            # Show first few missing combinations as examples
            print(f"\n[INFO] Example missing combinations (first 10):")
            for i, (combo_key, subjects) in enumerate(list(missing_combinations.items())[:10]):
                dataset, model, eval_mode, seed, noise_type = combo_key
                print(f"     {i+1}. {dataset} | {model} | {eval_mode} | seed={seed} | {noise_type} | subjects={subjects[:3]}{'...' if len(subjects) > 3 else ''}")
        else:
            print("[INFO] No missing combinations found - all expected results are present!")
        
        # Now map missing results to required multirun jobs
        print("\n" + "="*60)
        print("MAPPING TO REQUIRED MULTIRUN JOBS")
        print("="*60)
        
        required_multirun_jobs = set()
        
        for missing_result in missing_test_perturb_results:
            # Only create multirun job for the specific model that has missing results
            model = missing_result['model']
            seed = missing_result['seed']
            
            if missing_result['eval_mode'] == 'CrossSession' or missing_result['eval_mode'] == 'WithinSession':
                # For CrossSession, each subject needs its own multirun job for specific model and seed
                job_key = (
                    missing_result['dataset'],
                    missing_result['eval_mode'], 
                    missing_result['subject'],
                    missing_result['tune'],
                    model,
                    seed
                )
                required_multirun_jobs.add(job_key)
            else:
                # For Cross-Subject, all subjects are processed together for specific model and seed
                subjects_tuple = tuple(missing_result['subjects'])
                job_key = (
                    missing_result['dataset'],
                    missing_result['eval_mode'],
                    subjects_tuple,
                    missing_result['tune'],
                    model,
                    seed
                )
                required_multirun_jobs.add(job_key)
        
        # Convert to list of multirun job dictionaries
        missing_experiments = []
        for job_key in required_multirun_jobs:
            dataset, eval_mode, subjects_or_subject, tune, model, seed = job_key
            
            if eval_mode == 'CrossSession' or eval_mode == 'WithinSession':
                # Single subject for CrossSession
                subjects = [subjects_or_subject]
            else:
                # Multiple subjects for Cross-Subject
                subjects = list(subjects_or_subject)
            
            multirun_job = {
                'dataset': dataset,
                'eval_mode': eval_mode,
                'subjects': subjects,
                'tune': tune,
                'model': model,
                'seed': seed,
                'mode': 'multirun',  # This will generate test_perturb results
                'paradigm': next(config['paradigm'] for name, config in self.config['datasets'].items() if name == dataset)
            }
            missing_experiments.append(multirun_job)
        
        self.missing_experiments = missing_experiments
        print(f"[OK] Mapped to {len(missing_experiments)} required multirun jobs")
        
        # Print summary of missing multirun jobs
        if missing_experiments:
            print("\n[INFO] Missing multirun jobs summary:")
            missing_df = pd.DataFrame(missing_experiments)
            
            print("   By dataset:")
            for dataset, count in missing_df['dataset'].value_counts().items():
                print(f"     - {dataset}: {count} multirun jobs")
            
            print("   By model:")
            for model, count in missing_df['model'].value_counts().items():
                print(f"     - {model}: {count} multirun jobs")
            
            print("   By eval_mode:")
            for eval_mode, count in missing_df['eval_mode'].value_counts().items():
                print(f"     - {eval_mode}: {count} multirun jobs")
            
            print("   By tune flag:")
            for tune, count in missing_df['tune'].value_counts().items():
                print(f"     - {'tuned' if tune else 'not tuned'}: {count} multirun jobs")
            
            print("   By seed:")
            for seed, count in missing_df['seed'].value_counts().items():
                print(f"     - seed {seed}: {count} multirun jobs")
        
        return missing_experiments
    
    def generate_python_script(self, output_dir: str = None) -> str:
        """Generate Python script for local execution of missing experiments with parallel processing."""
        print("\n" + "="*60)
        print("GENERATING PYTHON EXECUTION SCRIPT")
        print("="*60)
        
        if output_dir is None:
            output_dir = self.config['output']['script_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        script_file = os.path.join(output_dir, "run_missing_multirun_jobs.py")
        
        # Separate experiments into tuned and non-tuned groups
        non_tuned_experiments = [exp for exp in self.missing_experiments if not exp['tune']]
        tuned_experiments = [exp for exp in self.missing_experiments if exp['tune']]
        
        print(f"[INFO] Separated experiments: {len(non_tuned_experiments)} non-tuned, {len(tuned_experiments)} tuned")
        
        with open(script_file, 'w') as f:
            f.write("#!/usr/bin/env python3\n")
            f.write('"""\n')
            f.write("Generated Python automation script for local experiment execution\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total missing multirun jobs: {len(self.missing_experiments)}\n")
            f.write(f"Non-tuned jobs: {len(non_tuned_experiments)}\n")
            f.write(f"Tuned jobs: {len(tuned_experiments)}\n")
            f.write("OPTIMIZED: Runs non-tuned jobs first, aggregates, then runs tuned jobs\n")
            f.write('"""\n\n')
            
            f.write("import os\n")
            f.write("import sys\n")
            f.write("import gc\n")
            f.write("import torch\n")
            f.write("import time\n")
            f.write("from datetime import datetime\n")
            f.write("from tqdm import tqdm\n")
            f.write("from concurrent.futures import ProcessPoolExecutor, as_completed\n")
            f.write("from typing import Dict, Any, Tuple, Optional\n\n")
            
            f.write("# Add evaluation directory to path\n")
            f.write("current_dir = os.path.dirname(os.path.abspath(__file__))\n")
            f.write("project_root = os.path.dirname(current_dir)\n")
            f.write("sys.path.insert(0, project_root)\n")
            f.write("sys.path.insert(0, os.path.join(project_root, 'evaluation'))\n\n")
            
            f.write("from evaluation.unified_experiment_runner import UnifiedExperimentRunner\n")
            f.write("from evaluation.experiment_utils import collect_all_results_unified\n")
            f.write("from globals import set_seeds\n\n")
            
            # Add legacy mode support
            f.write("# Legacy mode flag: Set to True to use original experimental protocol (no subject chunking)\n")
            f.write("# Can also be set via environment variable: USE_LEGACY_MODE=1\n")
            f.write("USE_LEGACY_MODE = os.environ.get('USE_LEGACY_MODE', '0').lower() in ('1', 'true', 'yes')\n")
            if self.legacy:
                f.write("# Legacy mode enabled via experiment_automation.py --legacy flag or config file\n")
                f.write("USE_LEGACY_MODE = True\n")
            f.write("\n")
            
            f.write("def cleanup_memory():\n")
            f.write("    \"\"\"Perform aggressive garbage collection and clear CUDA cache.\"\"\"\n")
            f.write("    gc.collect()\n")
            f.write("    if torch.cuda.is_available():\n")
            f.write("        torch.cuda.empty_cache()\n")
            f.write("        torch.cuda.synchronize()\n\n")
            
            # Add the run_single_experiment function for parallel execution
            f.write("def run_single_experiment(exp_config: Dict[str, Any], job_num: int, total_jobs: int) -> Tuple[int, bool, Optional[str], float]:\n")
            f.write("    \"\"\"\n")
            f.write("    Run a single experiment in a separate process.\n")
            f.write("    \n")
            f.write("    Args:\n")
            f.write("        exp_config: Experiment configuration dictionary\n")
            f.write("        job_num: Job number for logging\n")
            f.write("        total_jobs: Total number of jobs\n")
            f.write("        \n")
            f.write("    Returns:\n")
            f.write("        Tuple of (job_num, success, error_message, elapsed_time)\n")
            f.write("    \"\"\"\n")
            f.write("    job_start_time = time.time()\n")
            f.write("    \n")
            f.write("    # Re-import modules in the subprocess (necessary for multiprocessing)\n")
            f.write("    sys.path.insert(0, project_root)\n")
            f.write("    sys.path.insert(0, os.path.join(project_root, 'evaluation'))\n")
            f.write("    from evaluation.unified_experiment_runner import UnifiedExperimentRunner\n")
            f.write("    from globals import set_seeds\n")
            f.write("    \n")
            f.write("    print(f'\\n{\"-\"*60}')\n")
            f.write("    print(f'Job {job_num}/{total_jobs}')\n")
            f.write("    print(f'Dataset: {exp_config[\"dataset\"]} | Model: {exp_config[\"model\"]} | '\n")
            f.write("          f'Eval: {exp_config[\"eval_mode\"]} | Subjects: {exp_config[\"subjects\"]} | '\n")
            f.write("          f'Seed: {exp_config[\"seed\"]} | Tune: {exp_config[\"tune\"]}')\n")
            f.write("    print(f'{\"-\"*60}')\n\n")
            
            f.write("    # Set seed for reproducibility\n")
            f.write("    set_seeds(exp_config['seed'])\n\n")
            
            f.write("    try:\n")
            f.write("        # Create and run experiment\n")
            f.write("        # For CrossSubject mode, use chunked training to reduce memory usage\n")
            f.write("        # subject_chunk_size=3 loads 3 subjects at a time (default)\n")
            f.write("        # Legacy mode disables chunked training to follow original protocol\n")
            f.write("        subject_chunk_size = None if USE_LEGACY_MODE else (3 if exp_config['eval_mode'] == 'CrossSubject' else None)\n")
            f.write("        \n")
            f.write("        runner = UnifiedExperimentRunner(\n")
            f.write("            model=exp_config['model'],\n")
            f.write("            dataset=exp_config['dataset'],\n")
            f.write("            subjects=exp_config['subjects'],\n")
            f.write("            mode='test_perturb',\n")
            f.write("            eval_mode=exp_config['eval_mode'],\n")
            f.write("            seed=exp_config['seed'],\n")
            f.write("            noise_type='gaussian',  # multirun handles all noise types\n")
            f.write("            intensity=10.0,  # multirun handles all intensities\n")
            f.write("            tune=exp_config['tune'],\n")
            f.write("            overwrite=False,\n")
            f.write("            subject_chunk_size=subject_chunk_size,  # Enable chunked training for CrossSubject (unless legacy mode)\n")
            f.write("            legacy=USE_LEGACY_MODE  # Use legacy experimental protocol if enabled\n")
            f.write("        )\n\n")
            
            f.write("        results = runner.run_experiment()\n")
            f.write("        job_time = time.time() - job_start_time\n")
            f.write("        print(f'[SUCCESS] Job {job_num} completed in {job_time/60:.2f} minutes.')\n")
            f.write("        if results is not None:\n")
            f.write("            print(f'Results shape: {results.shape}')\n\n")
            
            f.write("        # Clean up\n")
            f.write("        del runner\n")
            f.write("        if results is not None:\n")
            f.write("            del results\n")
            f.write("        cleanup_memory()\n")
            f.write("        \n")
            f.write("        return (job_num, True, None, job_time)\n\n")
            
            f.write("    except Exception as e:\n")
            f.write("        job_time = time.time() - job_start_time\n")
            f.write("        print(f'[ERROR] Job {job_num} failed after {job_time/60:.2f} minutes: {e}')\n")
            f.write("        import traceback\n")
            f.write("        error_msg = traceback.format_exc()\n")
            f.write("        print(error_msg)\n")
            f.write("        cleanup_memory()\n")
            f.write("        return (job_num, False, str(e), job_time)\n\n")
            
            f.write("def run_experiments():\n")
            f.write("    \"\"\"Run all missing experiments in two phases: non-tuned first, then tuned.\"\"\"\n")
            f.write(f"    total_jobs = {len(self.missing_experiments)}\n")
            f.write(f"    non_tuned_jobs = {len(non_tuned_experiments)}\n")
            f.write(f"    tuned_jobs = {len(tuned_experiments)}\n")
            f.write("    print(f'Starting local experiment execution...')\n")
            f.write("    print(f'Total multirun jobs to execute: {total_jobs}')\n")
            f.write("    print(f'Non-tuned jobs: {non_tuned_jobs}')\n")
            f.write("    print(f'Tuned jobs: {tuned_jobs}')\n")
            f.write("    print(f'Started at: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')\n\n")
            
            # Write non-tuned experiments
            f.write("    non_tuned_experiments = [\n")
            for exp in non_tuned_experiments:
                f.write("        {\n")
                f.write(f"            'dataset': '{exp['dataset']}',\n")
                f.write(f"            'eval_mode': '{exp['eval_mode']}',\n")
                f.write(f"            'subjects': {exp['subjects']},\n")
                f.write(f"            'tune': {exp['tune']},\n")
                f.write(f"            'model': '{exp['model']}',\n")
                f.write(f"            'seed': {exp['seed']},\n")
                f.write(f"            'paradigm': '{exp['paradigm']}'\n")
                f.write("        },\n")
            f.write("    ]\n\n")
            
            # Write tuned experiments
            f.write("    tuned_experiments = [\n")
            for exp in tuned_experiments:
                f.write("        {\n")
                f.write(f"            'dataset': '{exp['dataset']}',\n")
                f.write(f"            'eval_mode': '{exp['eval_mode']}',\n")
                f.write(f"            'subjects': {exp['subjects']},\n")
                f.write(f"            'tune': {exp['tune']},\n")
                f.write(f"            'model': '{exp['model']}',\n")
                f.write(f"            'seed': {exp['seed']},\n")
                f.write(f"            'paradigm': '{exp['paradigm']}'\n")
                f.write("        },\n")
            f.write("    ]\n\n")
            
            f.write("    failed_jobs = []\n")
            f.write("    successful_jobs = 0\n")
            f.write("    start_time = time.time()\n")
            f.write("    \n")
            f.write("    # Initialize phase counters for interruption handling\n")
            f.write("    phase1_successful = 0\n")
            f.write("    phase2_successful = 0\n")
            f.write("    \n")
            f.write("    # Run experiments in parallel with configurable workers\n")
            f.write("    max_workers = 1\n")
            f.write("    \n")
            f.write("    def run_experiment_batch(experiments, phase_name, job_offset=0):\n")
            f.write("        \"\"\"Run a batch of experiments and return results.\"\"\"\n")
            f.write("        if not experiments:\n")
            f.write("            print(f'No {phase_name} experiments to run.')\n")
            f.write("            return [], 0\n")
            f.write("        \n")
            f.write("        print(f'\\n{\"=\"*60}')\n")
            f.write("        print(f'PHASE: {phase_name.upper()}')\n")
            f.write("        print(f'{\"=\"*60}')\n")
            f.write("        print(f'Running {len(experiments)} {phase_name} experiments...')\n")
            f.write("        print(f'Started at: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')\n")
            f.write("        \n")
            f.write("        batch_failed_jobs = []\n")
            f.write("        batch_successful_jobs = 0\n")
            f.write("        \n")
            f.write("        # Sort experiments by model to potentially improve cache efficiency\n")
            f.write("        experiments.sort(key=lambda x: x['model'])\n")
            f.write("        \n")
            f.write("        try:\n")
            f.write("            with ProcessPoolExecutor(max_workers=max_workers) as executor:\n")
            f.write("                # Submit all experiments to the executor\n")
            f.write("                future_to_job = {}\n")
            f.write("                for i, exp in enumerate(experiments):\n")
            f.write("                    job_num = i + 1 + job_offset\n")
            f.write("                    future = executor.submit(run_single_experiment, exp, job_num, total_jobs)\n")
            f.write("                    future_to_job[future] = (job_num, exp)\n")
            f.write("                \n")
            f.write("                # Process completed experiments as they finish\n")
            f.write("                completed = 0\n")
            f.write("                pending_futures = set(future_to_job.keys())\n")
            f.write("                \n")
            f.write("                try:\n")
            f.write("                    with tqdm(total=len(experiments), desc=f'{phase_name} progress') as pbar:\n")
            f.write("                        for future in as_completed(future_to_job):\n")
            f.write("                            job_num, exp = future_to_job[future]\n")
            f.write("                            completed += 1\n")
            f.write("                            pending_futures.discard(future)\n")
            f.write("                            \n")
            f.write("                            try:\n")
            f.write("                                result_job_num, success, error_msg, elapsed_time = future.result()\n")
            f.write("                                \n")
            f.write("                                if success:\n")
            f.write("                                    batch_successful_jobs += 1\n")
            f.write("                                else:\n")
            f.write("                                    # Job failed - stop immediately\n")
            f.write("                                    batch_failed_jobs.append((result_job_num, exp, error_msg))\n")
            f.write("                                    print(f'\\n[FAILURE DETECTED] Job {result_job_num} failed. Stopping all remaining jobs...')\n")
            f.write("                                    \n")
            f.write("                                    # Cancel all pending futures\n")
            f.write("                                    cancelled_count = 0\n")
            f.write("                                    for pending_future in pending_futures:\n")
            f.write("                                        if pending_future.cancel():\n")
            f.write("                                            cancelled_count += 1\n")
            f.write("                                    \n")
            f.write("                                    print(f'Cancelled {cancelled_count} pending jobs')\n")
            f.write("                                    print(f'Waiting for running jobs to complete before exiting...')\n")
            f.write("                                    \n")
            f.write("                                    # Wait for currently running jobs to complete (with timeout)\n")
            f.write("                                    running_futures = [f for f in pending_futures if not f.cancelled()]\n")
            f.write("                                    if running_futures:\n")
            f.write("                                        try:\n")
            f.write("                                            for running_future in as_completed(running_futures, timeout=300):\n")
            f.write("                                                run_job_num, run_exp = future_to_job[running_future]\n")
            f.write("                                                try:\n")
            f.write("                                                    run_result_job_num, run_success, run_error_msg, run_elapsed_time = running_future.result()\n")
            f.write("                                                    if run_success:\n")
            f.write("                                                        batch_successful_jobs += 1\n")
            f.write("                                                    else:\n")
            f.write("                                                        batch_failed_jobs.append((run_result_job_num, run_exp, run_error_msg))\n")
            f.write("                                                except Exception as run_e:\n")
            f.write("                                                    batch_failed_jobs.append((run_job_num, run_exp, f'Error during shutdown: {run_e}'))\n")
            f.write("                                        except TimeoutError:\n")
            f.write("                                            print(f'[WARNING] Timeout waiting for {len(running_futures)} running jobs to complete')\n")
            f.write("                                            print(f'These jobs may still be running in background processes')\n")
            f.write("                                    \n")
            f.write("                                    # Break out of the as_completed loop\n")
            f.write("                                    break\n")
            f.write("                                \n")
            f.write("                            except Exception as e:\n")
            f.write("                                # Critical error - stop immediately\n")
            f.write("                                print(f'\\n[CRITICAL ERROR] Job {job_num} crashed: {e}')\n")
            f.write("                                import traceback\n")
            f.write("                                traceback.print_exc()\n")
            f.write("                                batch_failed_jobs.append((job_num, exp, str(e)))\n")
            f.write("                                print(f'[FAILURE DETECTED] Job {job_num} crashed. Stopping all remaining jobs...')\n")
            f.write("                                \n")
            f.write("                                # Cancel all pending futures\n")
            f.write("                                cancelled_count = 0\n")
            f.write("                                for pending_future in pending_futures:\n")
            f.write("                                    if pending_future.cancel():\n")
            f.write("                                        cancelled_count += 1\n")
            f.write("                                \n")
            f.write("                                print(f'Cancelled {cancelled_count} pending jobs')\n")
            f.write("                                print(f'Waiting for running jobs to complete before exiting...')\n")
            f.write("                                \n")
            f.write("                                # Wait for currently running jobs to complete (with timeout)\n")
            f.write("                                running_futures = [f for f in pending_futures if not f.cancelled()]\n")
            f.write("                                if running_futures:\n")
            f.write("                                    try:\n")
            f.write("                                        for running_future in as_completed(running_futures, timeout=300):\n")
            f.write("                                            run_job_num, run_exp = future_to_job[running_future]\n")
            f.write("                                            try:\n")
            f.write("                                                run_result_job_num, run_success, run_error_msg, run_elapsed_time = running_future.result()\n")
            f.write("                                                if run_success:\n")
            f.write("                                                    batch_successful_jobs += 1\n")
            f.write("                                                else:\n")
            f.write("                                                    batch_failed_jobs.append((run_result_job_num, run_exp, run_error_msg))\n")
            f.write("                                            except Exception as run_e:\n")
            f.write("                                                batch_failed_jobs.append((run_job_num, run_exp, f'Error during shutdown: {run_e}'))\n")
            f.write("                                    except TimeoutError:\n")
            f.write("                                        print(f'[WARNING] Timeout waiting for {len(running_futures)} running jobs to complete')\n")
            f.write("                                        print(f'These jobs may still be running in background processes')\n")
            f.write("                                \n")
            f.write("                                # Break out of the as_completed loop\n")
            f.write("                                break\n")
            f.write("                            \n")
            f.write("                            pbar.update(1)\n")
            f.write("                \n")
            f.write("                except KeyboardInterrupt:\n")
            f.write("                    print(f'\\n\\n[INTERRUPTED] KeyboardInterrupt received during {phase_name} phase')\n")
            f.write("                    print(f'Completed {completed}/{len(experiments)} jobs before interruption')\n")
            f.write("                    print(f'Cancelling {len(pending_futures)} pending jobs...')\n")
            f.write("                    \n")
            f.write("                    # Cancel all pending futures\n")
            f.write("                    cancelled_count = 0\n")
            f.write("                    for future in pending_futures:\n")
            f.write("                        if future.cancel():\n")
            f.write("                            cancelled_count += 1\n")
            f.write("                    \n")
            f.write("                    print(f'Cancelled {cancelled_count} pending jobs')\n")
            f.write("                    print(f'Waiting for {len(pending_futures) - cancelled_count} running jobs to complete...')\n")
            f.write("                    \n")
            f.write("                    # Wait for running jobs to complete (with timeout)\n")
            f.write("                    running_futures = [f for f in pending_futures if not f.cancelled()]\n")
            f.write("                    if running_futures:\n")
            f.write("                        try:\n")
            f.write("                            for future in as_completed(running_futures, timeout=300):  # 5 minute timeout\n")
            f.write("                                job_num, exp = future_to_job[future]\n")
            f.write("                                try:\n")
            f.write("                                    result_job_num, success, error_msg, elapsed_time = future.result()\n")
            f.write("                                    if success:\n")
            f.write("                                        batch_successful_jobs += 1\n")
            f.write("                                    else:\n")
            f.write("                                        batch_failed_jobs.append((result_job_num, exp, error_msg))\n")
            f.write("                                except Exception as e:\n")
            f.write("                                    print(f'[WARNING] Job {job_num} result unavailable: {e}')\n")
            f.write("                                    batch_failed_jobs.append((job_num, exp, f'Interrupted: {e}'))\n")
            f.write("                        except TimeoutError:\n")
            f.write("                            print(f'[WARNING] Timeout waiting for {len(running_futures)} running jobs to complete')\n")
            f.write("                            print(f'These jobs may still be running in background processes')\n")
            f.write("                    \n")
            f.write("                    print(f'Gracefully shut down {phase_name} phase')\n")
            f.write("                    raise  # Re-raise to propagate to outer handler\n")
            f.write("                \n")
            f.write("                # Cleanup after batch\n")
            f.write("                cleanup_memory()\n")
            f.write("        \n")
            f.write("        except KeyboardInterrupt:\n")
            f.write("            # Cleanup on interruption\n")
            f.write("            cleanup_memory()\n")
            f.write("            raise  # Re-raise to propagate to outer handler\n")
            f.write("        \n")
            f.write("        print(f'\\n{phase_name.upper()} PHASE COMPLETE')\n")
            f.write("        print(f'Successful: {batch_successful_jobs}')\n")
            f.write("        print(f'Failed: {len(batch_failed_jobs)}')\n")
            f.write("        \n")
            f.write("        return batch_failed_jobs, batch_successful_jobs\n")
            f.write("    \n")
            f.write("    try:\n")
            f.write("        # PHASE 1: Run non-tuned experiments\n")
            f.write("        phase1_failed, phase1_successful = run_experiment_batch(non_tuned_experiments, 'non-tuned', 0)\n")
            f.write("        failed_jobs.extend(phase1_failed)\n")
            f.write("        successful_jobs += phase1_successful\n")
            f.write("        \n")
            f.write("        # Stop execution if Phase 1 had failures\n")
            f.write("        if phase1_failed:\n")
            f.write("            total_time = time.time() - start_time\n")
            f.write("            print(f'\\n{\"=\"*60}')\n")
            f.write("            print('EXECUTION STOPPED DUE TO FAILURES IN PHASE 1')\n")
            f.write("            print(f'{\"=\"*60}')\n")
            f.write("            print(f'Stopped at: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')\n")
            f.write("            print(f'Total runtime: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)')\n")
            f.write("            print(f'Phase 1 successful: {phase1_successful}')\n")
            f.write("            print(f'Phase 1 failed: {len(phase1_failed)}')\n")
            f.write("            print(f'\\nFailed jobs:')\n")
            f.write("            for job_num, exp, error in phase1_failed:\n")
            f.write("                print(f'  Job {job_num}: {exp[\"dataset\"]} | {exp[\"model\"]} | '\n")
            f.write("                      f'{exp[\"eval_mode\"]} | seed={exp[\"seed\"]} - Error: {error}')\n")
            f.write("            print(f'\\nExecution stopped. Phase 2 will not run.')\n")
            f.write("            sys.exit(1)\n")
            f.write("        \n")
            f.write("        # Aggregate results after non-tuned phase\n")
            f.write("        if phase1_successful > 0:\n")
            f.write("            print(f'\\n{\"=\"*60}')\n")
            f.write("            print('AGGREGATING RESULTS AFTER NON-TUNED PHASE')\n")
            f.write("            print(f'{\"=\"*60}')\n")
            f.write("            try:\n")
            f.write("                print('Calling collect_all_results_unified()...')\n")
            f.write("                aggregated_results = collect_all_results_unified()\n")
            f.write("                if aggregated_results is not None:\n")
            f.write("                    print(f'Aggregated {len(aggregated_results)} result rows')\n")
            f.write("                else:\n")
            f.write("                    print('No results found to aggregate')\n")
            f.write("            except Exception as e:\n")
            f.write("                print(f'Error during aggregation: {e}')\n")
            f.write("                import traceback\n")
            f.write("                traceback.print_exc()\n")
            f.write("        \n")
            f.write("        # PHASE 2: Run tuned experiments\n")
            f.write("        phase2_failed, phase2_successful = run_experiment_batch(tuned_experiments, 'tuned', len(non_tuned_experiments))\n")
            f.write("        failed_jobs.extend(phase2_failed)\n")
            f.write("        successful_jobs += phase2_successful\n")
            f.write("        \n")
            f.write("        # Final aggregation after tuned phase\n")
            f.write("        if phase2_successful > 0:\n")
            f.write("            print(f'\\n{\"=\"*60}')\n")
            f.write("            print('FINAL AGGREGATION AFTER TUNED PHASE')\n")
            f.write("            print(f'{\"=\"*60}')\n")
            f.write("            try:\n")
            f.write("                print('Calling collect_all_results_unified()...')\n")
            f.write("                aggregated_results = collect_all_results_unified()\n")
            f.write("                if aggregated_results is not None:\n")
            f.write("                    print(f'Final aggregated {len(aggregated_results)} result rows')\n")
            f.write("                else:\n")
            f.write("                    print('No results found to aggregate')\n")
            f.write("            except Exception as e:\n")
            f.write("                print(f'Error during final aggregation: {e}')\n")
            f.write("                import traceback\n")
            f.write("                traceback.print_exc()\n")
            f.write("        \n")
            f.write("        # Final summary\n")
            f.write("        total_time = time.time() - start_time\n")
            f.write("        print(f'\\n{\"=\"*60}')\n")
            f.write("        print('EXPERIMENT EXECUTION COMPLETE')\n")
            f.write("        print(f'{\"=\"*60}')\n")
            f.write("        print(f'Completed at: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')\n")
            f.write("        print(f'Total runtime: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)')\n")
            f.write("        print(f'Total jobs: {total_jobs}')\n")
            f.write("        print(f'Successful: {successful_jobs}')\n")
            f.write("        print(f'Failed: {len(failed_jobs)}')\n")
            f.write("        print(f'Non-tuned successful: {phase1_successful}')\n")
            f.write("        print(f'Tuned successful: {phase2_successful}')\n")
            f.write("        if successful_jobs > 0:\n")
            f.write("            print(f'Average time per job: {total_time/successful_jobs/60:.2f} minutes')\n\n")
            
            f.write("        if failed_jobs:\n")
            f.write("            print(f'\\nFailed jobs:')\n")
            f.write("            for job_num, exp, error in failed_jobs:\n")
            f.write("                print(f'  Job {job_num}: {exp[\"dataset\"]} | {exp[\"model\"]} | '\n")
            f.write("                      f'{exp[\"eval_mode\"]} | seed={exp[\"seed\"]} - Error: {error}')\n")
            f.write("            sys.exit(1)\n")
            f.write("        else:\n")
            f.write("            print('\\nAll jobs completed successfully!')\n")
            f.write("            sys.exit(0)\n")
            f.write("    \n")
            f.write("    except KeyboardInterrupt:\n")
            f.write("        # Handle graceful shutdown on Ctrl+C\n")
            f.write("        total_time = time.time() - start_time\n")
            f.write("        print(f'\\n\\n{\"=\"*60}')\n")
            f.write("        print('EXPERIMENT EXECUTION INTERRUPTED')\n")
            f.write("        print(f'{\"=\"*60}')\n")
            f.write("        print(f'Interrupted at: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')\n")
            f.write("        print(f'Total runtime before interruption: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)')\n")
            f.write("        print(f'\\nProgress Summary:')\n")
            f.write("        print(f'  Total jobs: {total_jobs}')\n")
            f.write("        print(f'  Successful: {successful_jobs}')\n")
            f.write("        print(f'  Failed: {len(failed_jobs)}')\n")
            f.write("        print(f'  Remaining: {total_jobs - successful_jobs - len(failed_jobs)}')\n")
            f.write("        \n")
            f.write("        if failed_jobs:\n")
            f.write("            print(f'\\nFailed jobs before interruption:')\n")
            f.write("            for job_num, exp, error in failed_jobs[:10]:  # Show first 10\n")
            f.write("                print(f'  Job {job_num}: {exp[\"dataset\"]} | {exp[\"model\"]} | '\n")
            f.write("                      f'{exp[\"eval_mode\"]} | seed={exp[\"seed\"]} - Error: {error}')\n")
            f.write("            if len(failed_jobs) > 10:\n")
            f.write("                print(f'  ... and {len(failed_jobs) - 10} more failed jobs')\n")
            f.write("        \n")
            f.write("        print(f'\\nNote: Completed experiments have been saved. You can re-run this script')\n")
            f.write("        print(f'      to continue with remaining experiments.')\n")
            f.write("        \n")
            f.write("        # Cleanup\n")
            f.write("        cleanup_memory()\n")
            f.write("        \n")
            f.write("        # Exit with code 130 (standard for SIGINT/KeyboardInterrupt)\n")
            f.write("        sys.exit(130)\n\n")
            
            f.write("if __name__ == '__main__':\n")
            f.write("    run_experiments()\n")
        
        # Make script executable
        os.chmod(script_file, 0o755)
        
        print(f"[OK] Generated Python execution script: {script_file}")
        print(f"[INFO] Script contains {len(self.missing_experiments)} multirun experiments")
        print(f"[INFO] Non-tuned experiments: {len(non_tuned_experiments)}")
        print(f"[INFO] Tuned experiments: {len(tuned_experiments)}")
        print(f"[INFO] Experiments will be run in two phases with aggregation between phases")
        print(f"[INFO] Each experiment uses UnifiedExperimentRunner in test_perturb mode")
        print(f"[INFO] Uses existing collect_all_results_unified() for aggregation")
        
        return script_file
    
    def generate_shell_script(self, output_dir: str = None) -> str:
        """Generate shell script with missing multirun sbatch commands."""
        print("\n" + "="*60)
        print("GENERATING SBATCH SHELL SCRIPT")
        print("="*60)
        
        if output_dir is None:
            output_dir = self.config['output']['script_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        script_file = os.path.join(output_dir, self.config['output']['missing_script_file'])
        
        with open(script_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Generated multirun sbatch automation script\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total missing multirun jobs: {len(self.missing_experiments)}\n\n")
            
            f.write("set -e  # Exit on any error\n\n")
            
            f.write("echo \"Starting multirun experiment automation...\"\n")
            f.write("echo \"Total multirun jobs to submit: {}\"\n\n".format(len(self.missing_experiments)))
            
            for i, exp in enumerate(self.missing_experiments, 1):
                # Build sbatch command arguments
                if exp['eval_mode'] == 'CrossSession' or exp['eval_mode'] == 'WithinSession':
                    # For CrossSession, single subject
                    subjects_str = str(exp['subjects'][0])
                else:
                    # For CrossSubject, space-separated subjects
                    subjects_str = " ".join(map(str, exp['subjects']))
                
                # Handle tuning flag
                tune_flag = "true" if exp['tune'] else "false"
                
                # Get model name and seed
                model = exp['model']
                seed = exp['seed']
                
                # Generate sbatch command with appropriate time limits
                # UPDATED: Since multirun now processes one seed at a time, reduce time limits by factor of 5
                # Base estimates: CrossSession without tuning ~3 hours / 5 = ~36 minutes
                # WithinSession takes ~5x longer than CrossSession
                # Tuning adds significant overhead (~20x for CrossSession)
                # SSVEP (Lee2019) is ~3x slower than Motor Imagery due to:
                #   - 4 classes vs 2 classes (~2x overhead)
                #   - Longer time windows (~1.5x overhead)
                # CLUSTER LIMIT: Maximum time is 7 days
                
                if exp['dataset'] == 'Lee2019_SSVEP':
                    # SSVEP-specific timeouts (reduced by factor of 5)
                    if exp['eval_mode'] == 'CrossSession':
                        if exp['tune']:
                            # CrossSession with tuning: ~2.5 days / 5 = ~12 hours
                            slurm_args = "--time=3-12:00:00 --mem=12G"
                        else:
                            # CrossSession without tuning: ~3 hours * 3 / 5 = ~1.8 hours
                            slurm_args = "--time=0-02:00:00 --mem=12G"
                            
                    elif exp['eval_mode'] == 'WithinSession':
                        if exp['tune']:
                            # WithinSession with tuning: ~12.5 days / 5 = ~2.5 days
                            slurm_args = "--time=5-12:00:00 --mem=12G"
                        else:
                            # WithinSession without tuning: ~15 hours * 3 / 5 = ~9 hours
                            slurm_args = "--time=0-10:00:00 --mem=12G"
                            
                    else:
                        # CrossSubject with fold-by-fold: Using 96G memory budget
                        # Time increased slightly to account for running 3 folds + aggregation
                        slurm_args = "--time=3-00:00:00 --mem=96G"
                        
                else:
                    # Motor Imagery timeouts (reduced by factor of 5)
                    # UPDATED: Increased timeouts for TUNED jobs based on observed timeout failures
                    if exp['eval_mode'] == 'CrossSession':
                        if exp['tune']:
                            # CrossSession with tuning: Increased from 36h to 3 days due to timeout failures
                            # Original estimate was ~2.5 days / 5 = ~12 hours, but actual runtime is longer
                            slurm_args = "--time=3-00:00:00 --mem=12G"
                        else:
                            # CrossSession without tuning: ~3 hours / 5 = ~36 minutes
                            slurm_args = "--time=0-08:00:00 --mem=12G"
                            
                    elif exp['eval_mode'] == 'WithinSession':
                        if exp['tune']:
                            # WithinSession with tuning: Increased from 3 days to 5 days due to timeout failures
                            # Original estimate was ~12.5 days / 5 = ~2.5 days, but actual runtime is longer
                            slurm_args = "--time=5-00:00:00 --mem=12G"
                        else:
                            # WithinSession without tuning: ~15 hours / 5 = ~3 hours
                            slurm_args = "--time=0-04:00:00 --mem=12G"
                            
                    else:
                        # CrossSubject with fold-by-fold: Using 64G memory budget
                        # Time increased slightly to account for running 3 folds + aggregation
                        slurm_args = "--time=1-12:00:00 --mem=64G"
                
                # Format: sbatch {slurm_args} unified_eval_script.sh {subject} {dataset} {eval_mode} {tune_flag} {model} {seed} [legacy_flag]
                # Use CrossSubject-specific script for CrossSubject eval mode
                # UPDATED: Use fold-by-fold script for CrossSubject to reduce memory usage (unless legacy mode)
                # Legacy mode uses the original script without fold-by-fold processing
                # Chunked training (subject_chunk_size=3) is automatically enabled via run_crosssubject_folds.py (unless legacy mode)
                if exp['eval_mode'] == 'CrossSubject':
                    if self.legacy:
                        # Legacy mode: use original script without fold-by-fold processing
                        script_name = "unified_eval_script_crosssubject.sh"
                    else:
                        # Normal mode: use fold-by-fold script for memory optimization
                        script_name = "unified_eval_script_crosssubject_foldbyfold.sh"
                else:
                    script_name = "unified_eval_script.sh"
                # Pass legacy flag if enabled (only for fold-by-fold script)
                legacy_flag = "true" if self.legacy else "false"
                if exp['eval_mode'] == 'CrossSubject' and not self.legacy:
                    # Only pass legacy flag to fold-by-fold script
                    command = f"sbatch {slurm_args} {script_name} {subjects_str} {exp['dataset']} {exp['eval_mode']} {tune_flag} {model} {seed} {legacy_flag}"
                else:
                    # Other scripts don't need legacy flag (legacy mode uses different script)
                    command = f"sbatch {slurm_args} {script_name} {subjects_str} {exp['dataset']} {exp['eval_mode']} {tune_flag} {model} {seed}"
                
                # Write sbatch command
                f.write(f"# Multirun Job {i}/{len(self.missing_experiments)}\n")
                f.write(f"# Dataset: {exp['dataset']} | Model: {model} | Eval: {exp['eval_mode']} | Subjects: {exp['subjects']} | Seed: {seed}")
                if exp['tune']:
                    f.write(" | TUNED")
                f.write("\n")
                f.write(f"# Timeout: {slurm_args}\n")
                f.write(f"# This multirun will generate test_perturb results for model: {model}\n")
                f.write(f"# This multirun will generate test_perturb results for seed: {seed}\n")
                f.write(f"# This multirun will generate test_perturb results for all noise types and intensities\n")
                f.write(f"echo \"Submitting multirun job {i}/{len(self.missing_experiments)}...\"\n")
                f.write(f"{command}\n")
                f.write("if [ $? -eq 0 ]; then\n")
                f.write(f"    echo \"[SUCCESS] Multirun job {i} submitted successfully\"\n")
                f.write("else\n")
                f.write(f"    echo \"[ERROR] Multirun job {i} submission failed\"\n")
                f.write("    exit 1\n")
                f.write("fi\n")
                f.write("sleep 1  # Brief pause between submissions\n\n")
        
        # Make script executable
        os.chmod(script_file, 0o755)
        
        print(f"[OK] Generated sbatch shell script: {script_file}")
        print(f"[INFO] Script contains {len(self.missing_experiments)} multirun sbatch commands")
        print(f"[INFO] Each sbatch command format: sbatch unified_eval_script.sh <subjects> <dataset> <eval_mode> <tune_flag>")
        if self.legacy:
            print(f"[INFO] Legacy mode enabled: CrossSubject experiments use original script (unified_eval_script_crosssubject.sh)")
            print(f"[INFO] Legacy mode: No subject chunking or fold-by-fold processing (matches original behavior)")
        else:
            print(f"[INFO] CrossSubject experiments use fold-by-fold mode (unified_eval_script_crosssubject_foldbyfold.sh)")
            print(f"[INFO] Memory optimization: Chunked training enabled (subject_chunk_size=3) for CrossSubject mode")
            print(f"[INFO] Memory budget: 64G per job (chunked training + fold-by-fold significantly reduces actual peak memory usage)")
        
        return script_file
    
    def generate_summary_report(self, output_dir: str = None) -> str:
        """Generate a summary report of missing experiments."""
        if output_dir is None:
            output_dir = self.config['output']['script_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        report_file = os.path.join(output_dir, "missing_experiments_report.csv")
        
        if self.missing_experiments:
            missing_df = pd.DataFrame(self.missing_experiments)
            missing_df.to_csv(report_file, index=False)
            print(f"[OK] Generated summary report: {report_file}")
        else:
            print("[OK] No missing experiments to report")
        
        return report_file
    
    def generate_completion_report(self) -> None:
        """
        Generate and print a completion report showing which combinations of
        (dataset, model, tune, eval_mode) are complete.
        """
        print("\n" + "="*60)
        print("COMPLETION STATUS REPORT")
        print("="*60)
        
        # Get all unique combinations from config
        datasets = self.config['datasets']
        models = [model['name'] for model in self.config['models']]
        eval_modes = self.config['eval_modes']
        seeds = self.config['seeds']
        tune_flags = [False, True]
        
        # Generate expected multirun jobs to check against
        expected_jobs = self.generate_expected_multirun_jobs()
        
        # Build a set of complete job keys for fast lookup
        # A job is complete if it's NOT in missing_experiments
        missing_job_keys = set()
        for job in self.missing_experiments:
            # Create a unique key for the job
            subjects_key = tuple(sorted(job['subjects']))
            
            job_key = (
                job['dataset'],
                job['model'],
                job['eval_mode'],
                job['seed'],
                job['tune'],
                subjects_key
            )
            missing_job_keys.add(job_key)
        
        # For each (dataset, model, tune, eval_mode) combination, check if all required jobs are complete
        completion_data = []
        
        for dataset_name in datasets.keys():
            for model_name in models:
                for tune_flag in tune_flags:
                    for eval_mode in eval_modes:
                        # Get all expected jobs for this combination
                        expected_jobs_for_combo = [
                            job for job in expected_jobs
                            if (job['dataset'] == dataset_name and
                                job['model'] == model_name and
                                job['eval_mode'] == eval_mode and
                                job['tune'] == tune_flag)
                        ]
                        
                        if not expected_jobs_for_combo:
                            continue  # Skip if no expected jobs for this combination
                        
                        # Check how many are missing
                        missing_count = 0
                        for job in expected_jobs_for_combo:
                            subjects_key = tuple(sorted(job['subjects']))
                            
                            job_key = (
                                job['dataset'],
                                job['model'],
                                job['eval_mode'],
                                job['seed'],
                                job['tune'],
                                subjects_key
                            )
                            
                            if job_key in missing_job_keys:
                                missing_count += 1
                        
                        total_jobs = len(expected_jobs_for_combo)
                        complete_jobs = total_jobs - missing_count
                        is_complete = (missing_count == 0)
                        
                        completion_data.append({
                            'dataset': dataset_name,
                            'model': model_name,
                            'tune': 'Tuned' if tune_flag else 'Baseline',
                            'eval_mode': eval_mode,
                            'complete': is_complete,
                            'complete_jobs': complete_jobs,
                            'total_jobs': total_jobs,
                            'missing_jobs': missing_count
                        })
        
        # Sort by dataset, then model, then eval_mode, then tune
        completion_data.sort(key=lambda x: (x['dataset'], x['model'], x['eval_mode'], x['tune']))
        
        # Print formatted table
        print("\nCompletion Status by (Dataset, Model, Tune, Eval Mode):")
        print("-" * 100)
        
        # Header
        header = f"{'Dataset':<20} {'Model':<20} {'Tune':<10} {'Eval Mode':<15} {'Status':<10} {'Progress':<20}"
        print(header)
        print("-" * 100)
        
        # Data rows
        for row in completion_data:
            status = "✓ Complete" if row['complete'] else "✗ Incomplete"
            progress = f"{row['complete_jobs']}/{row['total_jobs']} jobs"
            
            print(f"{row['dataset']:<20} {row['model']:<20} {row['tune']:<10} {row['eval_mode']:<15} "
                  f"{status:<10} {progress:<20}")
        
        print("-" * 100)
        
        # Summary statistics
        total_combinations = len(completion_data)
        complete_combinations = sum(1 for row in completion_data if row['complete'])
        incomplete_combinations = total_combinations - complete_combinations
        
        print(f"\nSummary:")
        print(f"  Total combinations: {total_combinations}")
        print(f"  Complete: {complete_combinations} ({100*complete_combinations/total_combinations:.1f}%)")
        print(f"  Incomplete: {incomplete_combinations} ({100*incomplete_combinations/total_combinations:.1f}%)")
        
        # Group by dataset and model for easier reading
        print("\n" + "="*60)
        print("COMPLETION STATUS BY DATASET AND MODEL")
        print("="*60)
        
        # Group data by dataset and model
        by_dataset_model = defaultdict(lambda: defaultdict(list))
        for row in completion_data:
            by_dataset_model[row['dataset']][row['model']].append(row)
        
        for dataset_name in sorted(by_dataset_model.keys()):
            print(f"\n{dataset_name}:")
            for model_name in sorted(by_dataset_model[dataset_name].keys()):
                model_rows = by_dataset_model[dataset_name][model_name]
                print(f"  {model_name}:")
                for row in model_rows:
                    status_symbol = "✓" if row['complete'] else "✗"
                    print(f"    {status_symbol} {row['tune']:<10} {row['eval_mode']:<15} "
                          f"({row['complete_jobs']}/{row['total_jobs']} jobs)")
        
        print("\n" + "="*60)
    
    def run_full_automation(self, output_dir: str = None) -> Tuple[str, str]:
        """Run the complete automation process."""
        print("[START] Starting Experiment Automation")
        print("="*60)
        
        # Step 1: Load existing results (either aggregate or load pre-aggregated)
        if self.use_cached:
            print("[INFO] Using cached results (skipping aggregation)")
            # Load cache data and apply it to instance variables
            cache_data = self._load_cache()
            if cache_data:
                self._apply_cache_data(cache_data)
            else:
                print("[WARNING] Cache file not found or failed to load, using empty results")
                self.existing_results = pd.DataFrame()
        elif self.preaggregated_results_file:
            self.load_preaggregated_results()
        else:
            self.aggregate_existing_results()
        
        # Step 2: Identify missing experiments
        self.identify_missing_experiments()
        
        # Step 3: Generate completion report
        self.generate_completion_report()
        
        # Step 4: Generate script (Python or shell based on local flag)
        if self.local:
            script_file = self.generate_python_script(output_dir)
        else:
            script_file = self.generate_shell_script(output_dir)
        
        # Step 5: Generate summary report
        report_file = self.generate_summary_report(output_dir)
        
        print("\n" + "="*60)
        print("AUTOMATION COMPLETE")
        print("="*60)
        print(f"[INFO] Output directory: {output_dir or self.config['output']['script_dir']}")
        print(f"[INFO] Generated script: {script_file}")
        print(f"[INFO] Summary report: {report_file}")
        print(f"[INFO] Missing experiments: {len(self.missing_experiments)}")
        
        if len(self.missing_experiments) > 0:
            print("\n[INFO] Next steps:")
            if self.local:
                print("1. Review the generated Python script")
                print("2. Run: python <script_file>")
                print("3. Monitor progress (tqdm will show a progress bar)")
                print("4. Check for any failed experiments in the output")
            else:
                print("1. Review the generated shell script")
                print("2. Run: chmod +x <script_file>")
                print("3. Execute: ./<script_file>")
                print("4. Monitor progress and handle any failures")
        
        return script_file, report_file


def main():
    """Main entry point for the automation script."""
    parser = argparse.ArgumentParser(description="EEG Experiment Automation System")
    parser.add_argument("--config", type=str, default="experiment_config.yaml",
                       help="Path to configuration YAML file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for generated scripts and reports")
    parser.add_argument("--preaggregated-results", type=str, default=None,
                       help="Path to pre-aggregated results CSV file (skips aggregation step)")
    parser.add_argument("--aggregate-only", action="store_true",
                       help="Only aggregate existing results, don't generate missing experiments")
    parser.add_argument("--missing-only", action="store_true",
                       help="Only identify missing experiments, don't aggregate")
    parser.add_argument("--local", action="store_true",
                       help="Generate Python script for local execution instead of sbatch script")
    parser.add_argument("--use-cached", action="store_true",
                       help="Use cached processed results (skips aggregation and signature generation)")
    parser.add_argument("--use-original", action="store_true",
                       help="Use original approach (generates all individual combinations). Default is optimized approach.")
    parser.add_argument("--legacy", action="store_true",
                       help="Use legacy experimental protocol (disables subject chunking and other memory optimizations to match original behavior)")
    
    args = parser.parse_args()
    
    # Initialize automation system
    use_optimized = not args.use_original  # Default to optimized unless --use-original is specified
    automation = ExperimentAutomation(args.config, args.preaggregated_results, args.local, args.use_cached, use_optimized, legacy=args.legacy)
    
    if args.aggregate_only:
        # Only aggregate results (ignore pre-aggregated file for this mode)
        if args.preaggregated_results:
            print("[WARNING] --preaggregated-results ignored in --aggregate-only mode")
        automation.aggregate_existing_results()
    elif args.missing_only:
        # Only identify missing experiments
        if args.use_cached:
            print("[INFO] Using cached results (skipping aggregation)")
            # Load cache data and apply it to instance variables
            cache_data = automation._load_cache()
            if cache_data:
                automation._apply_cache_data(cache_data)
            else:
                print("[WARNING] Cache file not found or failed to load, using empty results")
                automation.existing_results = pd.DataFrame()
        elif args.preaggregated_results:
            automation.load_preaggregated_results()
        else:
            automation.aggregate_existing_results()
        automation.identify_missing_experiments()
        
        # Generate completion report
        automation.generate_completion_report()
        
        # Generate script based on local flag
        if args.local:
            automation.generate_python_script(args.output_dir)
        else:
            automation.generate_shell_script(args.output_dir)
        
        automation.generate_summary_report(args.output_dir)
    else:
        # Full automation
        automation.run_full_automation(args.output_dir)


if __name__ == "__main__":
    main()
