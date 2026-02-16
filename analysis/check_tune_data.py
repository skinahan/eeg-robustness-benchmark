import pandas as pd
import numpy as np

df = pd.read_csv('../evaluation/results/unified_all_results.csv', low_memory=False)
print('Total rows:', len(df))
print('Rows with test_perturb:', len(df[df['mode'] == 'test_perturb']))
print('Rows with BNCI2014_001:', len(df[df['dataset'] == 'BNCI2014_001']))

df_filtered = df[
    (df['mode'] == 'test_perturb') & 
    (df['dataset'] == 'BNCI2014_001') & 
    (df['clean_score'].notna())
].copy()

print('After filtering:', len(df_filtered))
print('Tune values:', sorted(df_filtered['tune'].dropna().unique()))
print('Tune value counts:', df_filtered['tune'].value_counts().to_dict())
print('Tune dtype:', df_filtered['tune'].dtype)

# Check for tune=True with different comparisons
print('\nChecking tune=True matches:')
tune_true_bool = df_filtered[df_filtered['tune'] == True]
tune_true_np = df_filtered[df_filtered['tune'] == np.True_]
tune_true_1 = df_filtered[df_filtered['tune'] == 1]
tune_true_str = df_filtered[df_filtered['tune'].astype(str) == 'True']

print(f'  tune == True: {len(tune_true_bool)} rows')
print(f'  tune == np.True_: {len(tune_true_np)} rows')
print(f'  tune == 1: {len(tune_true_1)} rows')
print(f'  tune.astype(str) == "True": {len(tune_true_str)} rows')

# Check what values tune actually has
print('\nUnique tune values (with types):')
for val in df_filtered['tune'].dropna().unique():
    print(f'  {val} (type: {type(val).__name__})')

# Check if there are any rows that should match a violation
print('\nChecking for a specific violation match:')
violation_example = {
    'model': 'cnn_ncp',
    'dataset': 'BNCI2014_001',
    'seed': 300,
    'subject': 7,
    'tune': True,
    'session': '0train',
    'eval_mode': 'WithinSession'
}

# Check each filter individually
print('Individual filter matches:')
for key, val in violation_example.items():
    if key == 'tune':
        continue
    matches = (df_filtered[key] == val).sum()
    print(f'  {key} == {val}: {matches} rows')

# Build mask step by step to see where we lose rows
print('\nBuilding mask step by step:')
mask = pd.Series(True, index=df_filtered.index)
print(f'  Initial: {mask.sum()} rows')

mask = mask & (df_filtered['model'] == violation_example['model'])
print(f'  After model filter: {mask.sum()} rows')

mask = mask & (df_filtered['dataset'] == violation_example['dataset'])
print(f'  After dataset filter: {mask.sum()} rows')

mask = mask & (df_filtered['seed'] == violation_example['seed'])
print(f'  After seed filter: {mask.sum()} rows')

mask = mask & (df_filtered['subject'] == violation_example['subject'])
print(f'  After subject filter: {mask.sum()} rows')

mask = mask & (df_filtered['session'] == violation_example['session'])
print(f'  After session filter: {mask.sum()} rows')

mask = mask & (df_filtered['eval_mode'] == violation_example['eval_mode'])
print(f'  After eval_mode filter: {mask.sum()} rows')

# Show what values exist in the data for this combination
if mask.sum() == 0:
    print('\nNo rows match. Checking what values exist for this combination:')
    
    # Check what sessions exist for cnn_ncp, seed 300, subject 7, WithinSession
    temp_mask = (
        (df_filtered['model'] == 'cnn_ncp') &
        (df_filtered['seed'] == 300) &
        (df_filtered['subject'] == 7) &
        (df_filtered['eval_mode'] == 'WithinSession')
    )
    if temp_mask.sum() > 0:
        sessions = df_filtered[temp_mask]['session'].unique()
        print(f'  Sessions for cnn_ncp, seed 300, subject 7, WithinSession ({temp_mask.sum()} rows):')
        for sess in sorted(sessions):
            count = (df_filtered[temp_mask]['session'] == sess).sum()
            print(f'    "{sess}": {count} rows')
    else:
        print('  No rows for cnn_ncp, seed 300, subject 7, WithinSession')
    
    # Check what eval_modes exist for cnn_ncp, seed 300, subject 7, session 0train
    temp_mask = (
        (df_filtered['model'] == 'cnn_ncp') &
        (df_filtered['seed'] == 300) &
        (df_filtered['subject'] == 7) &
        (df_filtered['session'] == '0train')
    )
    if temp_mask.sum() > 0:
        eval_modes = df_filtered[temp_mask]['eval_mode'].unique()
        print(f'  Eval modes for cnn_ncp, seed 300, subject 7, session 0train ({temp_mask.sum()} rows):')
        for em in sorted(eval_modes):
            count = (df_filtered[temp_mask]['eval_mode'] == em).sum()
            print(f'    "{em}": {count} rows')
    else:
        print('  No rows for cnn_ncp, seed 300, subject 7, session 0train')
    
    # Check what subjects exist for cnn_ncp, seed 300, session 0train, WithinSession
    temp_mask = (
        (df_filtered['model'] == 'cnn_ncp') &
        (df_filtered['seed'] == 300) &
        (df_filtered['session'] == '0train') &
        (df_filtered['eval_mode'] == 'WithinSession')
    )
    if temp_mask.sum() > 0:
        subjects = df_filtered[temp_mask]['subject'].unique()
        print(f'  Subjects for cnn_ncp, seed 300, session 0train, WithinSession ({temp_mask.sum()} rows):')
        print(f'    {sorted(subjects)}')
    else:
        print('  No rows for cnn_ncp, seed 300, session 0train, WithinSession')
    
    # Show the actual 80 rows that match model, dataset, seed, subject
    print('\n  The 80 rows that match model=cnn_ncp, dataset=BNCI2014_001, seed=300, subject=7:')
    temp_mask = (
        (df_filtered['model'] == 'cnn_ncp') &
        (df_filtered['dataset'] == 'BNCI2014_001') &
        (df_filtered['seed'] == 300) &
        (df_filtered['subject'] == 7)
    )
    if temp_mask.sum() > 0:
        print(f'    Session values: {sorted(df_filtered[temp_mask]["session"].unique())}')
        print(f'    Eval_mode values: {sorted(df_filtered[temp_mask]["eval_mode"].unique())}')
        print(f'    Tune values: {sorted(df_filtered[temp_mask]["tune"].unique())}')
        print(f'\n    Sample rows:')
        sample = df_filtered[temp_mask][['model', 'seed', 'subject', 'session', 'eval_mode', 'tune']].head(5)
        print(sample.to_string())

