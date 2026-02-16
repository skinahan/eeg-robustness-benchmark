"""
Diagnostic script to identify where fold_idx is being lost in the data pipeline.

The issue: fold_idx is set in _evaluate_cv_fold but gets dropped during aggregation.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("DIAGNOSING fold_idx LOSS IN DATA PIPELINE")
print("=" * 80)

print("\n[1] Checking where fold_idx is set:")
print("-" * 80)
print("Location: evaluation/unified_experiment_runner.py")
print("  - Line 655: fold_idx is added to results in _evaluate_cv_fold()")
print("    all_results[i].update({'fold_idx': fold_idx, ...})")
print("  - Line 900: fold_idx is added in _evaluate_without_tuning()")
print("  - Line 942: fold_idx is added in _evaluate_perturb()")

print("\n[2] Checking where fold_idx is dropped:")
print("-" * 80)
print("Location: evaluation/unified_experiment_runner.py, _aggregate_fold_results()")
print("\n  A. WithinSession mode (lines 1274-1449):")
print("     - Line 1276: Checks if 'fold_idx' in results_df.columns")
print("     - Lines 1297-1340: For test_perturb mode, aggregates by noise_type/intensity/session")
print("       PROBLEM: fold_idx is NOT included in agg_row dictionary")
print("     - Lines 1401-1445: For regular modes, aggregates by session")
print("       PROBLEM: fold_idx is NOT included in agg_row dictionary")
print("     - Line 1447: Returns pd.DataFrame(agg_results) - fold_idx is lost!")

print("\n  B. CrossSession mode (lines 1451-1454):")
print("     - Line 1453: EXPLICITLY DROPS fold_idx:")
print("       results_df = results_df.drop(columns=['fold_idx'])")
print("     - This is intentional (CrossSession uses LeaveOneGroupOut, not folds)")

print("\n  C. CrossSubject mode (lines 1455-1459):")
print("     - Line 1459: Returns results_df as-is")
print("     - fold_idx SHOULD be preserved here")

print("\n[3] ROOT CAUSE:")
print("-" * 80)
print("For WithinSession mode:")
print("  - fold_idx is correctly set in individual fold results")
print("  - But when aggregating folds (taking mean across folds per session),")
print("    fold_idx is NOT preserved in the aggregated output")
print("  - This causes multiple fold results to be collapsed into a single row")
print("  - Result: Clean scores from different folds get averaged together,")
print("    but if there are multiple rows with same (model, dataset, seed, subject,")
print("    tune, mode, session, eval_mode) but different fold_idx, they appear as")
print("    separate rows with different clean scores")

print("\n[4] IMPACT:")
print("-" * 80)
print("  - Sanity check expects clean scores to be identical for same experimental setup")
print("  - But if fold_idx is missing, rows from different folds appear as separate")
print("    experimental setups, causing false violations")
print("  - Pattern: 2-3 unique clean scores per noise type suggests unaggregated folds")

print("\n[5] SOLUTION OPTIONS:")
print("-" * 80)
print("  Option A: Preserve fold_idx in aggregated results")
print("    - Add 'fold_idx' to agg_row dictionaries in _aggregate_fold_results")
print("    - But this might break existing analysis that expects aggregated results")
print("")
print("  Option B: Include fold_idx in grouping for sanity check")
print("    - Already attempted, but fold_idx doesn't exist in final data")
print("")
print("  Option C: Re-aggregate data properly")
print("    - Re-run experiments with fold_idx preserved, OR")
print("    - Post-process existing data to add fold_idx back (if possible)")
print("")
print("  Option D: Accept that folds are aggregated")
print("    - Modify sanity check to allow slight variation in clean scores")
print("    - Or aggregate/deduplicate clean scores within each group")

print("\n[6] RECOMMENDED FIX:")
print("-" * 80)
print("  For WithinSession mode in test_perturb:")
print("    - The aggregation is taking mean across folds, which is correct")
print("    - But if there are multiple rows with same grouping but different fold_idx,")
print("      they should be aggregated together")
print("    - The issue might be that aggregation is happening at the wrong level")
print("    - OR: fold_idx is being lost before aggregation, causing rows to not be grouped")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)

