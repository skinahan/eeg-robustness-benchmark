import pandas as pd
import os

# Check unified CSV
csv_path = '../evaluation/results/unified_all_results.csv'
if os.path.exists(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False, nrows=1000)
    print(f"Total rows in sample: {len(df)}")
    
    # Filter to test_perturb
    if 'mode' in df.columns:
        df_test = df[df['mode'] == 'test_perturb'].copy()
        print(f"Test perturb rows: {len(df_test)}")
    else:
        df_test = df.copy()
        print("No 'mode' column found")
    
    if len(df_test) > 0:
        print("\n=== Columns related to fold/eval/session ===")
        relevant_cols = [c for c in df_test.columns if any(x in c.lower() for x in ['fold', 'eval', 'session'])]
        for col in relevant_cols:
            unique_vals = df_test[col].dropna().unique()[:5]
            print(f"  {col}: {len(df_test[col].dropna().unique())} unique values, sample: {list(unique_vals)}")
        
        print("\n=== All columns ===")
        print(list(df_test.columns))
        
        print("\n=== Sample row (first test_perturb row) ===")
        sample = df_test.iloc[0]
        for col in df_test.columns:
            if col in ['model', 'dataset', 'seed', 'subject', 'tune', 'session', 'eval_mode', 'fold_idx', 'noise_type', 'intensity', 'clean_score']:
                print(f"  {col}: {sample[col]} (type: {type(sample[col]).__name__})")
else:
    print(f"CSV not found: {csv_path}")

