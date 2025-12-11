#!/usr/bin/env python3
"""
Test script for the new 3-fold CrossSubject evaluation mode.
"""

import sys
import os
import numpy as np

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from evaluation.unified_experiment_runner import ThreeFoldSubjectSplit


def test_three_fold_split():
    """Test the ThreeFoldSubjectSplit with different numbers of subjects."""
    
    print("=" * 80)
    print("Testing ThreeFoldSubjectSplit")
    print("=" * 80)
    
    # Test case 1: 9 subjects (evenly divisible by 3)
    print("\n--- Test Case 1: 9 subjects (evenly divisible by 3) ---")
    n_subjects = 9
    n_samples_per_subject = 100
    
    # Create mock data
    X = np.random.randn(n_subjects * n_samples_per_subject, 10, 100)
    y = np.random.randint(0, 2, n_subjects * n_samples_per_subject)
    groups = np.repeat(np.arange(1, n_subjects + 1), n_samples_per_subject)
    
    splitter = ThreeFoldSubjectSplit()
    
    print(f"Total subjects: {n_subjects}")
    print(f"Total samples: {len(X)}")
    print(f"Expected folds: {splitter.get_n_splits()}")
    print(f"Expected eval subjects per fold: {n_subjects // 3}")
    print(f"Expected train subjects per fold: {2 * n_subjects // 3}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        train_subjects = np.unique(groups[train_idx])
        test_subjects = np.unique(groups[test_idx])
        
        print(f"\nFold {fold_idx}:")
        print(f"  Eval subjects: {sorted(test_subjects)} (count: {len(test_subjects)})")
        print(f"  Train subjects: {sorted(train_subjects)} (count: {len(train_subjects)})")
        print(f"  Eval samples: {len(test_idx)}")
        print(f"  Train samples: {len(train_idx)}")
        
        # Verify no overlap
        overlap = set(train_subjects) & set(test_subjects)
        assert len(overlap) == 0, f"Found overlap between train and test: {overlap}"
        
        # Verify all subjects are accounted for
        all_subjects = set(train_subjects) | set(test_subjects)
        expected_subjects = set(range(1, n_subjects + 1))
        assert all_subjects == expected_subjects, f"Missing subjects: {expected_subjects - all_subjects}"
    
    print("\n✓ Test Case 1 passed!")
    
    # Test case 2: 10 subjects (not evenly divisible by 3)
    print("\n--- Test Case 2: 10 subjects (not evenly divisible by 3) ---")
    n_subjects = 10
    n_samples_per_subject = 100
    
    X = np.random.randn(n_subjects * n_samples_per_subject, 10, 100)
    y = np.random.randint(0, 2, n_subjects * n_samples_per_subject)
    groups = np.repeat(np.arange(1, n_subjects + 1), n_samples_per_subject)
    
    splitter = ThreeFoldSubjectSplit()
    
    print(f"Total subjects: {n_subjects}")
    print(f"Total samples: {len(X)}")
    print(f"Expected folds: {splitter.get_n_splits()}")
    print(f"Expected eval subjects per fold: {n_subjects // 3}")
    print(f"Remainder subjects (added to training): {n_subjects % 3}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        train_subjects = np.unique(groups[train_idx])
        test_subjects = np.unique(groups[test_idx])
        
        print(f"\nFold {fold_idx}:")
        print(f"  Eval subjects: {sorted(test_subjects)} (count: {len(test_subjects)})")
        print(f"  Train subjects: {sorted(train_subjects)} (count: {len(train_subjects)})")
        print(f"  Eval samples: {len(test_idx)}")
        print(f"  Train samples: {len(train_idx)}")
        
        # Verify no overlap
        overlap = set(train_subjects) & set(test_subjects)
        assert len(overlap) == 0, f"Found overlap between train and test: {overlap}"
        
        # Verify eval set has exactly 3 subjects
        assert len(test_subjects) == 3, f"Expected 3 eval subjects, got {len(test_subjects)}"
    
    print("\n✓ Test Case 2 passed!")
    
    # Test case 3: BNCI2014_001 (9 subjects)
    print("\n--- Test Case 3: BNCI2014_001 (9 subjects) ---")
    n_subjects = 9
    n_samples_per_subject = 288  # Typical BNCI2014_001
    
    X = np.random.randn(n_subjects * n_samples_per_subject, 22, 1000)
    y = np.random.randint(0, 2, n_subjects * n_samples_per_subject)
    groups = np.repeat(np.arange(1, n_subjects + 1), n_samples_per_subject)
    
    splitter = ThreeFoldSubjectSplit()
    
    print(f"Total subjects: {n_subjects}")
    print(f"Total samples: {len(X)}")
    print(f"Samples per subject: {n_samples_per_subject}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        train_subjects = np.unique(groups[train_idx])
        test_subjects = np.unique(groups[test_idx])
        
        print(f"\nFold {fold_idx}:")
        print(f"  Eval subjects: {sorted(test_subjects)}")
        print(f"  Train subjects: {sorted(train_subjects)}")
        
        # Verify expected sample counts
        expected_test_samples = len(test_subjects) * n_samples_per_subject
        expected_train_samples = len(train_subjects) * n_samples_per_subject
        
        assert len(test_idx) == expected_test_samples, \
            f"Expected {expected_test_samples} test samples, got {len(test_idx)}"
        assert len(train_idx) == expected_train_samples, \
            f"Expected {expected_train_samples} train samples, got {len(train_idx)}"
    
    print("\n✓ Test Case 3 passed!")
    
    # Test edge case: exactly 3 subjects
    print("\n--- Test Case 4: 3 subjects (minimum) ---")
    n_subjects = 3
    n_samples_per_subject = 100
    
    X = np.random.randn(n_subjects * n_samples_per_subject, 10, 100)
    y = np.random.randint(0, 2, n_subjects * n_samples_per_subject)
    groups = np.repeat(np.arange(1, n_subjects + 1), n_samples_per_subject)
    
    splitter = ThreeFoldSubjectSplit()
    
    print(f"Total subjects: {n_subjects}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        train_subjects = np.unique(groups[train_idx])
        test_subjects = np.unique(groups[test_idx])
        
        print(f"\nFold {fold_idx}:")
        print(f"  Eval subjects: {sorted(test_subjects)} (count: {len(test_subjects)})")
        print(f"  Train subjects: {sorted(train_subjects)} (count: {len(train_subjects)})")
        
        # With 3 subjects, each fold should have 1 eval subject and 2 train subjects
        assert len(test_subjects) == 1, f"Expected 1 eval subject, got {len(test_subjects)}"
        assert len(train_subjects) == 2, f"Expected 2 train subjects, got {len(train_subjects)}"
    
    print("\n✓ Test Case 4 passed!")
    
    # Test error case: less than 3 subjects
    print("\n--- Test Case 5: 2 subjects (should raise error) ---")
    n_subjects = 2
    n_samples_per_subject = 100
    
    X = np.random.randn(n_subjects * n_samples_per_subject, 10, 100)
    y = np.random.randint(0, 2, n_subjects * n_samples_per_subject)
    groups = np.repeat(np.arange(1, n_subjects + 1), n_samples_per_subject)
    
    splitter = ThreeFoldSubjectSplit()
    
    try:
        for train_idx, test_idx in splitter.split(X, y, groups):
            pass
        print("✗ Expected ValueError but none was raised!")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    print("\n" + "=" * 80)
    print("All tests passed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_three_fold_split()


