#!/usr/bin/env python3
"""
Test signature matching to identify the exact issue.
"""

import pandas as pd
import ast

# Simulate the signature building from experiment_automation.py
def build_existing_signature(row):
    """Build signature as done in experiment_automation.py for existing results."""
    sig_parts = [
        str(row.get('dataset', '')),
        str(row.get('model', '')),
        str(row.get('eval_mode_normalized', '')),
        str(row.get('seed', '')),
        str(row.get('noise_type', '')),
        str(row.get('intensity', '')),
        'test_perturb',
        str(row.get('is_tuned', False)),
    ]
    
    # Subject handling
    eval_mode_norm = str(row.get('eval_mode_normalized', ''))
    if eval_mode_norm == 'CrossSubject':
        if pd.notna(row.get('eval_subjects')):
            sig_parts.append(f"eval_subjects_{row['eval_subjects']}")
        elif pd.notna(row.get('session')):
            sig_parts.append(f"session_{row['session']}")
        else:
            sig_parts.append('no_subject')
    elif pd.notna(row.get('subject')):
        sig_parts.append(str(int(row['subject'])))
    else:
        sig_parts.append('no_subject')
    
    return '|'.join(sig_parts)

def build_expected_signature(expected_result, intensity_to_use):
    """Build signature as done in experiment_automation.py for expected results."""
    signature_parts = [
        expected_result['dataset'],
        expected_result['model'],
        expected_result['eval_mode'],
        str(expected_result['seed']),
        expected_result['noise_type'],
        str(intensity_to_use),
        'test_perturb',
        str(expected_result['tune'])
    ]
    
    # Add subject/eval_subjects for signature
    if expected_result['eval_mode'] == 'CrossSubject':
        if 'subjects' in expected_result:
            subjects_tuple = tuple(sorted(expected_result['subjects']))
            signature_parts.append(f"subjects_{subjects_tuple}")
        else:
            signature_parts.append('no_subject')
    elif 'subject' in expected_result:
        signature_parts.append(str(expected_result['subject']))
    else:
        signature_parts.append('no_subject')
    
    return '|'.join(str(part) for part in signature_parts)

# Read files
print("Reading files...")
missing_df = pd.read_csv(r"generated_scripts_core_models_bnci2014_cross_session\missing_experiments_report.csv")
print(f"Missing experiments: {len(missing_df)}")

# Read sample of all_results
print("Reading all_results.csv (sample)...")
all_results = pd.read_csv(r"sol_results\MotorImagery\BNCI2014_001\all_results.csv", nrows=5000)
print(f"Sample rows: {len(all_results)}")

# Filter and normalize as done in experiment_automation.py
if 'dataset' in all_results.columns and 'eval_mode' in all_results.columns:
    # Filter for BNCI2014_001
    bnci = all_results[all_results['dataset'] == 'BNCI2014_001'].copy()
    
    # Normalize eval_mode
    bnci['eval_mode_normalized'] = bnci['eval_mode'].str.replace('Evaluation', '', regex=False)
    
    # Filter for CrossSession
    crosssession = bnci[bnci['eval_mode_normalized'] == 'CrossSession'].copy()
    
    # Normalize mode
    if 'mode' in crosssession.columns:
        crosssession['mode_normalized'] = crosssession['mode'].str.replace('_tune', '', regex=False)
        crosssession['is_tuned'] = crosssession['mode'].astype(str).str.contains('_tune', na=False)
        
        # Filter for test_perturb
        test_perturb = crosssession[crosssession['mode_normalized'] == 'test_perturb'].copy()
        
        print(f"\nTest_perturb results: {len(test_perturb)}")
        
        if len(test_perturb) > 0:
            # Build signatures for existing results
            print("\nBuilding signatures for existing results...")
            test_perturb['signature'] = test_perturb.apply(build_existing_signature, axis=1)
            existing_signatures = set(test_perturb['signature'].values)
            print(f"Unique existing signatures: {len(existing_signatures)}")
            
            # Now check a missing experiment
            print("\n" + "="*80)
            print("Testing missing experiment matching...")
            print("="*80)
            
            missing_row = missing_df.iloc[0]
            print(f"\nMissing experiment:")
            for col in missing_row.index:
                print(f"  {col}: {missing_row[col]}")
            
            # Parse subjects
            try:
                subjects_list = ast.literal_eval(missing_row['subjects'])
                subject = subjects_list[0] if isinstance(subjects_list, list) and len(subjects_list) > 0 else None
            except:
                subject = None
            
            # Build expected signature (we need to create a test_perturb expected result)
            # The missing experiment is a multirun job, but we need to check if test_perturb results exist
            # Let's check for a specific intensity (e.g., 0.0 or first intensity)
            
            # First, let's see what intensities exist for this model/seed/subject
            if subject is not None:
                matching_existing = test_perturb[
                    (test_perturb['model'] == missing_row['model']) &
                    (test_perturb['seed'] == missing_row['seed']) &
                    (test_perturb['subject'] == subject) &
                    (test_perturb['is_tuned'] == missing_row['tune'])
                ]
                
                print(f"\nExisting results matching model={missing_row['model']}, seed={missing_row['seed']}, subject={subject}, tune={missing_row['tune']}:")
                print(f"  Count: {len(matching_existing)}")
                
                if len(matching_existing) > 0:
                    print(f"  Sample signatures:")
                    for sig in list(matching_existing['signature'].head(3)):
                        print(f"    {sig}")
                    
                    # Now build expected signature for comparison
                    # We need to create an expected test_perturb result
                    # Let's use the first intensity from matching results
                    if 'intensity' in matching_existing.columns and len(matching_existing) > 0:
                        sample_intensity = matching_existing['intensity'].iloc[0]
                        sample_noise_type = matching_existing['noise_type'].iloc[0] if 'noise_type' in matching_existing.columns else 'gaussian'
                        
                        expected_result = {
                            'dataset': missing_row['dataset'],
                            'model': missing_row['model'],
                            'eval_mode': 'CrossSession',  # This is what's in missing_row
                            'seed': missing_row['seed'],
                            'noise_type': sample_noise_type,
                            'intensity': sample_intensity,
                            'tune': missing_row['tune'],
                            'subject': subject
                        }
                        
                        expected_sig = build_expected_signature(expected_result, sample_intensity)
                        print(f"\n  Expected signature (for intensity={sample_intensity}):")
                        print(f"    {expected_sig}")
                        
                        if expected_sig in existing_signatures:
                            print(f"\n  [OK] SIGNATURE MATCHES!")
                        else:
                            print(f"\n  [ERROR] SIGNATURE DOES NOT MATCH!")
                            print(f"  Let's check what's different...")
                            
                            # Find closest match
                            parts_expected = expected_sig.split('|')
                            print(f"\n  Expected signature parts:")
                            for i, part in enumerate(parts_expected):
                                print(f"    [{i}] {part}")
                            
                            # Check if any existing signature has same prefix
                            matching_prefixes = [sig for sig in existing_signatures if sig.startswith('|'.join(parts_expected[:7]))]
                            if matching_prefixes:
                                print(f"\n  Found {len(matching_prefixes)} signatures with matching prefix:")
                                for sig in matching_prefixes[:3]:
                                    parts = sig.split('|')
                                    print(f"    {sig}")
                                    print(f"    Parts:")
                                    for i, part in enumerate(parts):
                                        marker = "[OK]" if i < len(parts_expected) and part == parts_expected[i] else "[ERROR]"
                                        print(f"      [{i}] {marker} {part}")
                else:
                    print(f"\n  [ERROR] NO MATCHING RESULTS FOUND")
                    print(f"  This suggests the matching logic is working correctly - the experiment is truly missing")
                    
                    # But let's check if there are results for this model/seed but different subject
                    any_subject = test_perturb[
                        (test_perturb['model'] == missing_row['model']) &
                        (test_perturb['seed'] == missing_row['seed']) &
                        (test_perturb['is_tuned'] == missing_row['tune'])
                    ]
                    if len(any_subject) > 0:
                        print(f"\n  BUT: Found {len(any_subject)} results for this model/seed/tune (any subject)")
                        print(f"  Subjects: {sorted(any_subject['subject'].unique())}")
                        print(f"  This suggests results exist but for different subjects")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
