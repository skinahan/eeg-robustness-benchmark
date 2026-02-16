#!/usr/bin/env python3
"""
Detailed analysis of matching issue between missing experiments and existing results.
"""

import pandas as pd
import ast
import numpy as np

# Read files
missing_file = r"generated_scripts_core_models_bnci2014_cross_session\missing_experiments_report.csv"
all_results_file = r"sol_results\MotorImagery\BNCI2014_001\all_results.csv"

print("="*80)
print("ANALYZING MATCHING ISSUE")
print("="*80)

# Read missing experiments
print("\n1. Reading missing experiments report...")
missing_df = pd.read_csv(missing_file)
print(f"   Total missing: {len(missing_df)}")
print(f"   Columns: {missing_df.columns.tolist()}")
print(f"   Sample:")
print(missing_df.head(3).to_string())

# Read a sample of all_results to understand structure
print("\n2. Reading all_results.csv structure...")
# Read first 100 rows to understand structure
sample_df = pd.read_csv(all_results_file, nrows=100)
print(f"   Sample rows: {len(sample_df)}")
print(f"   Columns: {sample_df.columns.tolist()}")
print(f"\n   Sample data:")
print(sample_df.head(3).to_string())

# Check eval_mode values
if 'eval_mode' in sample_df.columns:
    print(f"\n   Unique eval_mode values in sample: {sample_df['eval_mode'].unique()}")
    
# Check mode values
if 'mode' in sample_df.columns:
    print(f"   Unique mode values in sample: {sample_df['mode'].unique()}")

# Now let's simulate the matching logic from experiment_automation.py
print("\n" + "="*80)
print("3. SIMULATING MATCHING LOGIC")
print("="*80)

# Read full all_results (this might take a while, but we need it)
print("   Reading full all_results.csv...")
try:
    full_df = pd.read_csv(all_results_file)
    print(f"   Total rows: {len(full_df)}")
    
    # Filter for BNCI2014_001
    if 'dataset' in full_df.columns:
        bnci_df = full_df[full_df['dataset'] == 'BNCI2014_001']
        print(f"   BNCI2014_001 rows: {len(bnci_df)}")
        
        # Normalize eval_mode (as done in experiment_automation.py line 371)
        if 'eval_mode' in bnci_df.columns:
            bnci_df = bnci_df.copy()
            bnci_df['eval_mode_normalized'] = bnci_df['eval_mode'].str.replace('Evaluation', '', regex=False)
            print(f"   Unique eval_mode_normalized: {bnci_df['eval_mode_normalized'].unique()}")
            
            # Filter for CrossSession
            crosssession_df = bnci_df[bnci_df['eval_mode_normalized'] == 'CrossSession']
            print(f"   CrossSession rows: {len(crosssession_df)}")
            
            # Normalize mode (as done in line 377)
            if 'mode' in crosssession_df.columns:
                crosssession_df = crosssession_df.copy()
                crosssession_df['mode_normalized'] = crosssession_df['mode'].str.replace('_tune', '', regex=False)
                
                # Filter for test_perturb results
                test_perturb_df = crosssession_df[crosssession_df['mode_normalized'] == 'test_perturb']
                print(f"   test_perturb rows: {len(test_perturb_df)}")
                
                if len(test_perturb_df) > 0:
                    print(f"\n   Sample test_perturb results:")
                    cols_to_show = ['dataset', 'model', 'eval_mode', 'eval_mode_normalized', 'seed', 'mode', 'mode_normalized', 'subject', 'tune'] if 'subject' in test_perturb_df.columns else ['dataset', 'model', 'eval_mode', 'eval_mode_normalized', 'seed', 'mode', 'mode_normalized']
                    available_cols = [c for c in cols_to_show if c in test_perturb_df.columns]
                    print(test_perturb_df[available_cols].head(10).to_string())
                    
                    # Now check a specific missing experiment
                    print("\n" + "="*80)
                    print("4. CHECKING SPECIFIC MISSING EXPERIMENT")
                    print("="*80)
                    
                    # Take first missing experiment
                    missing_row = missing_df.iloc[0]
                    print(f"\n   Missing experiment:")
                    print(f"     dataset: {missing_row['dataset']}")
                    print(f"     eval_mode: {missing_row['eval_mode']}")
                    print(f"     subjects: {missing_row['subjects']}")
                    print(f"     tune: {missing_row['tune']}")
                    print(f"     model: {missing_row['model']}")
                    print(f"     seed: {missing_row['seed']}")
                    print(f"     mode: {missing_row['mode']}")
                    
                    # Parse subjects
                    try:
                        subjects_list = ast.literal_eval(missing_row['subjects'])
                        subject = subjects_list[0] if isinstance(subjects_list, list) and len(subjects_list) > 0 else None
                        print(f"     parsed subject: {subject}")
                    except:
                        subject = None
                        print(f"     failed to parse subject")
                    
                    # Search for matching test_perturb results
                    # Build search criteria matching the signature logic
                    search = test_perturb_df[
                        (test_perturb_df['model'] == missing_row['model']) &
                        (test_perturb_df['seed'] == missing_row['seed'])
                    ]
                    
                    # Check tune flag
                    if 'tune' in test_perturb_df.columns:
                        search = search[search['tune'] == missing_row['tune']]
                        print(f"\n   After filtering by model, seed, tune: {len(search)} rows")
                    else:
                        # Check mode for _tune suffix
                        if missing_row['tune']:
                            search = search[search['mode'].str.contains('_tune', na=False)]
                            print(f"\n   After filtering by model, seed, tune (via mode): {len(search)} rows")
                        else:
                            search = search[~search['mode'].str.contains('_tune', na=False)]
                            print(f"\n   After filtering by model, seed, tune=False (via mode): {len(search)} rows")
                    
                    # Check subject
                    if subject is not None and 'subject' in test_perturb_df.columns:
                        subject_search = search[search['subject'] == subject]
                        print(f"   After filtering by subject={subject}: {len(subject_search)} rows")
                        
                        if len(subject_search) > 0:
                            print(f"\n   [OK] FOUND MATCHING RESULTS!")
                            print(f"   Sample matching rows:")
                            print(subject_search.head(5).to_string())
                        else:
                            print(f"\n   [ERROR] NO MATCHING RESULTS for subject={subject}")
                            print(f"   Available subjects in filtered results: {sorted(search['subject'].unique()) if 'subject' in search.columns else 'N/A'}")
                            
                            # Check if there are results for this model/seed but different subject
                            if len(search) > 0:
                                print(f"\n   But there ARE results for this model/seed combination!")
                                print(f"   Total matching rows (any subject): {len(search)}")
                                print(f"   Subjects in matching rows: {sorted(search['subject'].unique()) if 'subject' in search.columns else 'N/A'}")
                    else:
                        print(f"\n   Cannot check subject (subject={subject}, 'subject' in columns: {'subject' in test_perturb_df.columns})")
                        if len(search) > 0:
                            print(f"   But found {len(search)} rows matching model/seed/tune")
                            print(search.head(5).to_string())
                
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
