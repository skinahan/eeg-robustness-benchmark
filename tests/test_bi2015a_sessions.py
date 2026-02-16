#!/usr/bin/env python3
"""
Test script to verify dataset session structure for BI2015a and Lee2019_SSVEP.
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from moabb.datasets import BI2015a, Lee2019_SSVEP
from config import get_paradigm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut

def test_dataset_sessions(dataset_name, dataset_class, subject_id=1):
    """Test that a dataset loads correctly and report session structure."""
    print(f"\n{'=' * 60}")
    print(f"Testing {dataset_name} dataset session structure...")
    print("=" * 60)
    
    try:
        # Load dataset
        dataset = dataset_class()
        dataset.subject_list = [subject_id]
        
        # Get paradigm
        paradigm = get_paradigm(resample=None, dataset=dataset_name)
        
        # Load data
        print(f"Loading data for subject {subject_id}...")
        X, y, metadata = paradigm.get_data(dataset, subjects=[subject_id])
        
        print(f"\nData shape: {X.shape}")
        print(f"Number of samples: {len(X)}")
        
        # Check sessions
        if 'session' in metadata.columns:
            sessions = metadata['session'].unique()
            print(f"\nSessions found: {sorted(sessions)}")
            print(f"Number of sessions: {len(sessions)}")
            
            # Show session distribution
            print("\nSession distribution:")
            for session in sorted(sessions):
                session_mask = metadata['session'] == session
                session_count = session_mask.sum()
                print(f"  {session}: {session_count} samples")
            
            # Check that LeaveOneGroupOut would work correctly
            print("\n" + "-" * 60)
            print("Testing LeaveOneGroupOut compatibility...")
            
            groups = metadata['session'].values
            logo = LeaveOneGroupOut()
            n_splits = logo.get_n_splits(X, y, groups)
            print(f"LeaveOneGroupOut would create {n_splits} folds")
            
            if n_splits == len(sessions):
                print("[OK] SUCCESS: LeaveOneGroupOut correctly identifies all sessions!")
            else:
                print(f"[ERROR] WARNING: Expected {len(sessions)} folds, but got {n_splits}")
            
            # Test iteration
            print("\nFold structure:")
            for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
                test_session = groups[test_idx][0]
                train_sessions = sorted(set(groups[train_idx]))
                print(f"  Fold {i+1}: Test session = {test_session}, Train sessions = {train_sessions}")
            
            return True, len(sessions)
        else:
            print("\n[ERROR] ERROR: No 'session' column found in metadata")
            print(f"Metadata columns: {metadata.columns.tolist()}")
            return False, 0
    except Exception as e:
        print(f"\n[ERROR] ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

if __name__ == "__main__":
    print("=" * 60)
    print("DATASET SESSION STRUCTURE TEST")
    print("=" * 60)
    
    results = {}
    
    # Test BI2015a
    success_bi2015a, n_sessions_bi2015a = test_dataset_sessions("BI2015a", BI2015a, subject_id=1)
    results["BI2015a"] = (success_bi2015a, n_sessions_bi2015a)
    
    # Test Lee2019_SSVEP
    success_lee2019, n_sessions_lee2019 = test_dataset_sessions("Lee2019_SSVEP", Lee2019_SSVEP, subject_id=1)
    results["Lee2019_SSVEP"] = (success_lee2019, n_sessions_lee2019)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for dataset_name, (success, n_sessions) in results.items():
        status = "[OK] PASS" if success else "[ERROR] FAIL"
        print(f"{dataset_name}: {status} - {n_sessions} sessions per subject")
    
    # Verify expectations
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    if results["BI2015a"][0] and results["BI2015a"][1] == 3:
        print("[OK] BI2015a: Has 3 sessions per subject as expected")
    elif results["BI2015a"][0]:
        print(f"⚠ BI2015a: Has {results['BI2015a'][1]} sessions (expected 3)")
    else:
        print("[ERROR] BI2015a: Failed to load")
    
    if results["Lee2019_SSVEP"][0] and results["Lee2019_SSVEP"][1] == 2:
        print("[OK] Lee2019_SSVEP: Has 2 sessions per subject as expected")
    elif results["Lee2019_SSVEP"][0]:
        print(f"⚠ Lee2019_SSVEP: Has {results['Lee2019_SSVEP'][1]} sessions (expected 2)")
    else:
        print("[ERROR] Lee2019_SSVEP: Failed to load")
    
    # Overall result
    all_passed = all(success for success, _ in results.values())
    if all_passed:
        print("\n[OK] All datasets loaded successfully!")
        sys.exit(0)
    else:
        print("\n[ERROR] Some datasets failed to load. Please review the output above.")
        sys.exit(1)



