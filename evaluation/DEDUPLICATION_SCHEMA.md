# Results Deduplication Schema

## Overview

This document defines the schema and strategy for deduplicating experimental results when collecting from both `sol_results` and `results` directories. The goal is to ensure that each unique experiment appears exactly once in the final aggregated dataset, with `sol_results` serving as the primary source of truth.

**Important**: This process is **completely non-destructive**. All original CSV files in both `sol_results` and `results` directories remain untouched. We only read from these files, perform deduplication in memory, and write the final deduplicated results to a new unified CSV file (typically `evaluation/results/unified_all_results.csv`). No original data files are modified, overwritten, or deleted.

## Core Principles

1. **Non-Destructive Processing**: All original CSV files are read-only. We never modify, overwrite, or delete any files in `sol_results` or `results` directories. All deduplication happens in memory, and only the final aggregated result is written to a new output file.
2. **Primary Source Priority**: `sol_results` is the primary source. Results from `results` directory are only used to fill gaps or when explicitly needed.
3. **Experiment Identity**: Two rows represent the same experiment if they share the same experiment signature (defined below).
4. **Careful Deduplication**: We deduplicate at the experiment level, not just exact row matches, to handle cases where the same experiment may have been run multiple times or stored in different formats.
5. **Metadata Preservation**: When deduplicating, preserve the most complete and reliable metadata available.

---

## Experiment Signature (Unique Identifier)

An experiment is uniquely identified by the following combination of fields:

### Core Identifying Fields

1. **`dataset`** (str): Dataset identifier (e.g., "BNCI2014_001", "Lee2019_SSVEP", "BI2015a")
2. **`model`** (str): Model name (e.g., "cnn_ncp", "eegnet", "reegnet")
3. **`eval_mode`** (str): Evaluation mode (e.g., "WithinSessionEvaluation", "CrossSessionEvaluation", "CrossSubjectEvaluation")
4. **`seed`** (int): Random seed used for the experiment
5. **`mode`** (str): Experiment mode (e.g., "test_perturb", "test_perturb_tune", "multirun")
6. **`tune`** (bool): Whether hyperparameter tuning was used
7. **`noise_type`** (str): Type of noise perturbation (e.g., "gaussian", "eog", "dropout", "spike", "clean")
8. **`intensity`** (float): Noise intensity level

### Subject/Session Identification (Context-Dependent)

The subject/session identification depends on `eval_mode`:

- **WithinSessionEvaluation**: Use `(subject, session)`
- **CrossSessionEvaluation**: Use `(subject, session)` (session identifies which session was used for evaluation)
- **CrossSubjectEvaluation**: Use either:
  - `eval_subjects` (if available): Comma-separated list of evaluation subjects
  - `session` (fallback): Session identifier that encodes the fold/evaluation group

**Note**: For CrossSubject mode, the `session` field may contain fold information like `"fold_0_eval_subjects_1,2,3"`. This should be normalized to a consistent format.

### Complete Signature Tuple

```python
signature = (
    dataset,           # str
    model,             # str
    eval_mode,         # str
    seed,              # int
    mode,              # str
    tune,              # bool
    noise_type,        # str
    intensity,         # float
    subject_key        # str or int (see below)
)
```

Where `subject_key` is determined as:
- For `eval_mode == "CrossSubjectEvaluation"`:
  - If `eval_subjects` exists and is not null: `f"eval_subjects_{eval_subjects}"`
  - Else if `session` exists and is not null: `f"session_{session}"`
  - Else: `"no_subject"`
- For other eval_modes:
  - If `subject` exists and is not null: `int(subject)` (normalized to int)
  - Else: `"no_subject"`
- Additionally, for WithinSession/CrossSession: include `session` in the key if it's meaningful

**Refined Signature for WithinSession/CrossSession**:
```python
signature = (
    dataset,
    model,
    eval_mode,
    seed,
    mode,
    tune,
    noise_type,
    intensity,
    subject,      # int
    session       # str (for WithinSession/CrossSession)
)
```

**Signature for CrossSubject**:
```python
signature = (
    dataset,
    model,
    eval_mode,
    seed,
    mode,
    tune,
    noise_type,
    intensity,
    subject_key   # str: "eval_subjects_{...}" or "session_{...}"
)
```

---

## Data Source Priority

### Priority Order

1. **`sol_results`** (Primary Source)
   - These are the curated, verified results
   - Always prefer results from `sol_results` when duplicates exist
   - If a result exists in both directories, keep the one from `sol_results`

2. **`results`** (Secondary Source)
   - Used to fill gaps in `sol_results`
   - Only included if the experiment doesn't exist in `sol_results`
   - May contain older or less reliable results

### Collection Strategy

**All operations are read-only on source files. No files are modified.**

1. **Phase 1: Collect from `sol_results` (Read-Only)**
   - Walk through `sol_results/{paradigm}/{dataset}/` directory structure
   - **Read** all CSV files (excluding aggregate files like `all_results.csv` if present)
   - Load data into memory as pandas DataFrames
   - Build a dictionary/index keyed by experiment signature
   - Handle inter-dataset duplicates here - keep only one unique experiment that matches the signature
   - Track source: mark all as coming from `sol_results`
   - **Original files remain untouched**

2. **Phase 2: Collect from `results` (Read-Only, with deduplication)**
   - Walk through `results/{paradigm}/{dataset}/` directory structure
   - For each CSV file:
     - **Read** the file into memory
     - Parse and extract experiment signature
     - Check if this signature already exists in the `sol_results` collection
     - If **not** in `sol_results`, add it to the in-memory collection (mark source as `results`)
     - If **already** in `sol_results`, skip it (don't add duplicate)
   - **Original files remain untouched**

3. **Phase 3: Final Aggregation (Write to New File Only)**
   - Combine all collected results into a single DataFrame in memory
   - Perform final validation and cleanup (all in memory)
   - **Write** the deduplicated results to a **new** output file (e.g., `evaluation/results/unified_all_results.csv`)
   - **No original source files are modified or overwritten**

---

## Deduplication Rules

### Rule 1: Signature-Based Deduplication

Two rows are considered duplicates if they have the same experiment signature (as defined above).

**Implementation**:
```python
# Create signature column
df['experiment_signature'] = df.apply(create_signature, axis=1)

# Deduplicate: keep first occurrence (which will be from sol_results due to collection order)
df_deduped = df.drop_duplicates(subset=['experiment_signature'], keep='first')
```

### Rule 2: Source Priority

When the same signature appears multiple times:
- **Keep**: The row from `sol_results` (if available)
- **Discard**: Rows from `results` directory

**Implementation**:
- Add a `source` column during collection: `'sol_results'` or `'results'`
- Sort by source priority (sol_results first)
- Use `drop_duplicates(keep='first')` which will keep the first (highest priority) occurrence

### Rule 3: Metadata Normalization

Before deduplication, normalize key fields:

1. **`eval_mode`**: Normalize to canonical forms:
   - `"WithinSessionEvaluation"`, `"CrossSessionEvaluation"`, `"CrossSubjectEvaluation"`
   - Handle variations: `"CrossSession"` → `"CrossSessionEvaluation"`

2. **`tune`**: Convert to boolean:
   - `True`, `False`, `1`, `0`, `"True"`, `"False"` → boolean

3. **`intensity`**: Normalize to float:
   - Handle string representations: `"10.0"` → `10.0`
   - Handle slight floating-point differences (use tolerance for comparison if needed)

4. **`subject`**: Normalize to int:
   - Convert string `"7"` → `7`
   - Handle NaN/null values consistently

5. **`mode`**: Normalize to canonical forms:
   - `"test_perturb"`, `"test_perturb_tune"`, `"multirun"`, etc.

### Rule 4: Handling Missing/Incomplete Data

- If a row from `sol_results` has missing values but a duplicate from `results` has complete data:
  - **Still prefer `sol_results`** (primary source)
  - Log warnings about missing data for manual review
- If a row from `sol_results` is completely empty or invalid:
  - Consider falling back to `results` version (with logging)

---

## Implementation Steps

### Step 1: Data Collection with Source Tracking

**Note**: All file operations are read-only. Original files are never modified.

```python
def collect_results_with_source(paradigm: str, dataset: str):
    """
    Collect results from both directories, tracking source.
    
    This function only READS from source files. No files are modified.
    
    Returns: DataFrame with 'source' column indicating origin.
    """
    all_dfs = []
    
    # Phase 1: Collect from sol_results (primary) - READ ONLY
    sol_results = collect_from_directory(
        os.path.join("sol_results", paradigm, dataset),
        source="sol_results"
    )
    # Note: collect_from_directory only reads files, never modifies them
    
    # Build index of existing signatures from sol_results
    sol_signatures = set()
    if sol_results is not None and not sol_results.empty:
        sol_results['experiment_signature'] = sol_results.apply(create_signature, axis=1)
        sol_signatures = set(sol_results['experiment_signature'].unique())
        all_dfs.append(sol_results)
    
    # Phase 2: Collect from results (secondary, with deduplication) - READ ONLY
    results_dir = os.path.join("results", paradigm, dataset)
    if os.path.exists(results_dir):
        results_df = collect_from_directory(results_dir, source="results")
        # Note: collect_from_directory only reads files, never modifies them
        if results_df is not None and not results_df.empty:
            results_df['experiment_signature'] = results_df.apply(create_signature, axis=1)
            # Filter out duplicates that already exist in sol_results (in memory only)
            results_df = results_df[~results_df['experiment_signature'].isin(sol_signatures)]
            all_dfs.append(results_df)
    
    # Combine (all operations in memory)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined
    return None
```

### Step 2: Signature Creation Function

```python
def create_experiment_signature(row: pd.Series) -> tuple:
    """
    Create a unique signature tuple for an experiment row.
    """
    # Normalize core fields
    dataset = str(row.get('dataset', '')).strip()
    model = str(row.get('model', '')).strip()
    eval_mode = normalize_eval_mode(row.get('eval_mode', ''))
    seed = int(row.get('seed', 0))
    mode = normalize_mode(row.get('mode', ''))
    tune = normalize_bool(row.get('tune', False))
    noise_type = str(row.get('noise_type', '')).strip()
    intensity = normalize_float(row.get('intensity', 0.0))
    
    # Determine subject/session key based on eval_mode
    if eval_mode == "CrossSubjectEvaluation":
        if pd.notna(row.get('eval_subjects')):
            subject_key = f"eval_subjects_{row['eval_subjects']}"
        elif pd.notna(row.get('session')):
            subject_key = f"session_{row['session']}"
        else:
            subject_key = "no_subject"
        return (dataset, model, eval_mode, seed, mode, tune, noise_type, intensity, subject_key)
    else:
        # WithinSession or CrossSession
        subject = int(row.get('subject', 0)) if pd.notna(row.get('subject')) else 0
        session = str(row.get('session', '')).strip() if pd.notna(row.get('session')) else ''
        return (dataset, model, eval_mode, seed, mode, tune, noise_type, intensity, subject, session)
```

### Step 3: Normalization Functions

```python
def normalize_eval_mode(eval_mode: str) -> str:
    """Normalize eval_mode to canonical form."""
    if pd.isna(eval_mode) or eval_mode == '':
        return 'UnknownEvaluation'
    eval_mode = str(eval_mode).strip()
    if not eval_mode.endswith('Evaluation'):
        eval_mode = eval_mode + 'Evaluation'
    return eval_mode

def normalize_mode(mode: str) -> str:
    """Normalize mode to canonical form."""
    if pd.isna(mode) or mode == '':
        return 'unknown'
    return str(mode).strip().lower()

def normalize_bool(value) -> bool:
    """Normalize to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 't')
    return False

def normalize_float(value) -> float:
    """Normalize to float."""
    if pd.isna(value):
        return 0.0
    return float(value)
```

### Step 4: Final Deduplication and Output

**Note**: All operations are in-memory. The output is written to a NEW file, never overwriting source files.

```python
def final_deduplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform final deduplication with source priority.
    
    All operations are in-memory. No source files are modified.
    """
    # Ensure source column exists and set priority
    if 'source' not in df.columns:
        df['source'] = 'unknown'
    
    # Create priority order (sol_results = 0, results = 1, unknown = 2)
    source_priority = {'sol_results': 0, 'results': 1, 'unknown': 2}
    df['source_priority'] = df['source'].map(source_priority).fillna(2)
    
    # Sort by priority (sol_results first)
    df = df.sort_values('source_priority', ascending=True)
    
    # Drop duplicates, keeping first (highest priority) - all in memory
    df_deduped = df.drop_duplicates(
        subset=['experiment_signature'],
        keep='first'
    )
    
    # Drop temporary columns
    df_deduped = df_deduped.drop(columns=['source_priority'], errors='ignore')
    
    return df_deduped

def save_unified_results(df: pd.DataFrame, output_path: str):
    """
    Save the deduplicated results to a NEW output file.
    
    This function writes to a new file (typically evaluation/results/unified_all_results.csv).
    It never modifies or overwrites any files in sol_results/ or results/ directories.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Deduplicated results saved to: {output_path}")
    print(f"[INFO] Original source files in sol_results/ and results/ remain unchanged")
```

---

## Validation and Logging

### Validation Checks

1. **Signature Completeness**: Ensure all required fields for signature creation are present
2. **Data Type Consistency**: Verify normalized fields have correct types
3. **Duplicate Detection**: Log statistics on duplicates found and removed
4. **Source Statistics**: Report counts of results from each source

### Logging Output

```
[INFO] Collecting results for MotorImagery - BNCI2014_001
[INFO] Found 50000 rows in sol_results
[INFO] Found 75000 rows in results
[INFO] Filtered 25000 duplicate rows from results (already in sol_results)
[INFO] Final aggregated: 50000 rows (50000 from sol_results, 0 from results)
[INFO] Deduplication removed 0 additional exact duplicates
```

---

## Edge Cases and Special Handling

### Case 1: Same Experiment, Different Metrics

If the same experiment signature appears with different metric values:
- This indicates a data integrity issue
- **Action**: Log a warning and keep the `sol_results` version
- **Investigation**: Flag for manual review

### Case 2: Missing Signature Components

If required signature fields are missing:
- **Action**: Skip the row and log a warning
- **Reason**: Cannot reliably identify the experiment

### Case 3: CrossSubject Session Format Variations

CrossSubject mode may have session strings in different formats:
- `"fold_0_eval_subjects_1,2,3"`
- `"fold_0"`
- `"eval_subjects_1,2,3"`
- **Action**: Normalize to a consistent format before signature creation

### Case 4: Intensity Floating-Point Precision

Intensities may be stored with slight precision differences:
- `10.0` vs `10.0000001`
- **Action**: Round to reasonable precision (e.g., 3 decimal places) before signature creation

---

## Testing Strategy

1. **Unit Tests**: Test signature creation with various input formats
2. **Integration Tests**: Test full collection and deduplication pipeline
3. **Validation Tests**: Verify no duplicates remain after deduplication
4. **Source Priority Tests**: Verify sol_results takes precedence

---

## Migration Notes

When implementing this schema:

1. **Non-Destructive Guarantee**: The implementation must guarantee that no original CSV files are ever modified. All file operations should be read-only, and only the final unified output file should be written. This is a critical requirement.
2. **Backward Compatibility**: Ensure existing code that uses `collect_all_results_unified()` continues to work
3. **Gradual Rollout**: Consider adding a flag to enable/disable new deduplication logic
4. **Data Validation**: Run validation checks on existing aggregated results to identify issues
5. **Documentation**: Update function docstrings to reflect new behavior and emphasize read-only operations

---

## Summary

This schema ensures:
- ✅ **Non-destructive processing**: All original CSV files remain untouched; only a new unified output file is created
- ✅ `sol_results` is the primary source of truth
- ✅ Proper deduplication based on experiment identity (not just exact row matches)
- ✅ No loss of unique experiments from either directory
- ✅ Clear logging and validation of the deduplication process
- ✅ Handling of edge cases and data quality issues
- ✅ All operations are read-only on source files; deduplication happens entirely in memory

