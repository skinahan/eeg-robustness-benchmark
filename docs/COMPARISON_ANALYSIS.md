# Comparison: experiment_automation.py vs experiment_automation_optimized.py

## Executive Summary

**experiment_automation.py**: Complete, production-ready implementation with full feature set
**experiment_automation_optimized.py**: Incomplete reference implementation showing optimized approach (ends at line 358)

---

## 1. Completeness & Functionality

### experiment_automation.py ✅ COMPLETE
- **Full class implementation**: `ExperimentAutomation` with all methods
- **Result loading**: `load_preaggregated_results()`, `aggregate_existing_results()`
- **Missing experiment identification**: `generate_expected_experiments()`, `identify_missing_experiments()`
- **Script generation**: 
  - `generate_python_script()` - For local execution with parallel processing
  - `generate_shell_script()` - For SLURM/sbatch execution
- **Reporting**: `generate_summary_report()` - CSV report generation
- **Full automation**: `run_full_automation()` - Complete workflow
- **CLI interface**: `main()` function with argparse
- **Cache management**: File-based caching with pickle

### experiment_automation_optimized.py ❌ INCOMPLETE
- **Partial class implementation**: `OptimizedExperimentAutomation` 
- **Missing methods**:
  - No result loading methods
  - No script generation methods
  - No reporting methods
  - No full automation workflow
  - No CLI interface
  - No cache persistence (only in-memory)
- **Only implements**:
  - `generate_expected_multirun_jobs()` 
  - `identify_missing_experiments_optimized()`
  - Helper methods for indexing and checking

---

## 2. Core Algorithm Differences

### Approach to Missing Experiment Identification

#### experiment_automation.py (Original)
```
1. Generate ALL individual test_perturb combinations
   - For each: dataset × model × eval_mode × seed × noise_type × intensity × tune × subject
   - Result: Millions of individual combinations
   
2. Compare each expected result against existing results
   - Build string signatures for fast lookup
   - Use set operations for comparison
   - Early filtering with metadata
   
3. Map missing results → multirun jobs
   - Group missing results by (dataset, model, eval_mode, subject, tune, seed)
   - Create multirun job for each unique group
```

**Complexity**: O(expected_results × existing_results) in worst case
**Memory**: High (stores all individual combinations)

#### experiment_automation_optimized.py (Optimized)
```
1. Generate multirun jobs directly
   - For each: dataset × model × eval_mode × seed × tune × subject
   - Result: Thousands of multirun jobs (much fewer)
   
2. Check if each multirun job is complete
   - Build nested dictionary index: index[key][subject][intensities]
   - Check if job produced all expected intensities for all noise types
   
3. Return missing multirun jobs directly
```

**Complexity**: O(multirun_jobs × noise_types × intensities) - typically much better
**Memory**: Lower (only stores multirun jobs, not individual combinations)

---

## 3. Data Structures & Indexing

### experiment_automation.py
- **String signatures**: `"dataset|model|eval_mode|seed|noise_type|intensity|test_perturb|tune|subject"`
- **Set-based lookup**: `existing_signatures = set(df['signature'].values)`
- **Metadata filtering**: Quick elimination using unique value sets
- **Intensity mapping**: Pre-computed tolerance mapping for fuzzy matching

### experiment_automation_optimized.py
- **Tuple-based keys**: `(dataset, model, eval_mode, seed, noise_type, is_tuned)`
- **Nested dictionary index**: 
  ```python
  index[(dataset, model, eval_mode, seed, noise_type, is_tuned)][subject_key][intensity] = True
  ```
- **Subject key tuples**: `('subject', int)` or `('eval_subjects', str)`
- **Direct intensity sets**: Stores intensities as sets for fast membership testing

---

## 4. Performance Optimizations

### experiment_automation.py
- ✅ Caching of noise intensities per (dataset, noise_type)
- ✅ Caching of processed results (normalized DataFrame, signatures, metadata)
- ✅ File-based cache persistence (pickle)
- ✅ Early filtering using metadata (eliminates impossible matches quickly)
- ✅ Vectorized pandas operations where possible
- ✅ Pre-computed intensity tolerance mapping
- ⚠️ Still uses `iterrows()` in some places
- ⚠️ Generates all individual combinations first (memory intensive)

### experiment_automization_optimized.py
- ✅ Caching of noise intensities per (dataset, noise_type)
- ✅ Works at multirun job level (avoids generating millions of combinations)
- ✅ Tuple-based indexing (faster than string operations)
- ✅ Nested dictionary structure for O(1) lookups
- ✅ Vectorized intensity comparison with numpy
- ⚠️ Still uses `iterrows()` in `_build_existing_index()` (line 119)
- ❌ No cache persistence (only in-memory)
- ❌ No early filtering optimizations

---

## 5. Feature Comparison Matrix

| Feature | experiment_automation.py | experiment_automation_optimized.py |
|---------|-------------------------|-----------------------------------|
| **Result Loading** | ✅ Yes | ❌ No |
| **Result Aggregation** | ✅ Yes | ❌ No |
| **Expected Experiments Generation** | ✅ Individual combinations | ✅ Multirun jobs only |
| **Missing Identification** | ✅ Complete | ✅ Partial (only multirun level) |
| **Python Script Generation** | ✅ Yes (with parallel processing) | ❌ No |
| **Shell Script Generation** | ✅ Yes (sbatch) | ❌ No |
| **CSV Report Generation** | ✅ Yes | ❌ No |
| **Full Automation Workflow** | ✅ Yes | ❌ No |
| **CLI Interface** | ✅ Yes (argparse) | ❌ No |
| **Cache Persistence** | ✅ Yes (pickle file) | ❌ No |
| **Early Filtering** | ✅ Yes (metadata-based) | ❌ No |
| **Intensity Tolerance** | ✅ Yes (pre-computed mapping) | ✅ Yes (numpy.isclose) |
| **Progress Tracking** | ✅ Yes (tqdm) | ✅ Yes (tqdm) |

---

## 6. Code Quality & Maintainability

### experiment_automation.py
- **Lines of code**: ~1480 lines
- **Documentation**: Comprehensive docstrings
- **Error handling**: Robust (try/except blocks)
- **Logging**: Detailed progress messages
- **Modularity**: Well-separated methods
- **Production-ready**: Yes

### experiment_automation_optimized.py
- **Lines of code**: ~358 lines (incomplete)
- **Documentation**: Good docstrings for implemented methods
- **Error handling**: Basic
- **Logging**: Basic progress messages
- **Modularity**: Good structure for implemented parts
- **Production-ready**: No (incomplete)

---

## 7. Use Cases

### Use experiment_automation.py when:
- ✅ You need a complete, working solution
- ✅ You need script generation (Python or shell)
- ✅ You need reporting and full automation
- ✅ You're running in production
- ✅ You need cache persistence across runs

### Use experiment_automation_optimized.py when:
- ✅ You want to understand the optimized approach
- ✅ You're integrating optimizations into the main file
- ✅ You only need to identify missing multirun jobs (no scripts)
- ⚠️ You're willing to implement missing features yourself

---

## 8. Key Insights

### What the Optimized Version Does Better:
1. **Memory efficiency**: Works at multirun job level, avoiding millions of individual combinations
2. **Algorithmic efficiency**: Direct multirun job checking vs. generate-all-then-map approach
3. **Indexing structure**: Nested dictionaries with tuple keys are more efficient than string signatures

### What the Original Version Does Better:
1. **Completeness**: Full feature set, production-ready
2. **Cache persistence**: File-based caching survives restarts
3. **Early filtering**: Metadata-based quick elimination
4. **Flexibility**: Can work with individual results or multirun jobs

### Best Approach (Hybrid):
The optimized approach should be **integrated into** the original file:
- Replace `generate_expected_experiments()` + `identify_missing_experiments()` 
- With `generate_expected_multirun_jobs()` + `identify_missing_experiments_optimized()`
- Keep all other features (script generation, reporting, etc.)
- Add cache persistence to optimized methods
- Add early filtering optimizations

---

## 9. Performance Estimates

### For a typical experiment setup:
- **Datasets**: 2
- **Models**: 10
- **Eval modes**: 3
- **Seeds**: 5
- **Noise types**: 2
- **Intensities**: 20 per noise type
- **Subjects**: 10 per dataset
- **Tune flags**: 2

**Individual combinations**: ~2 × 10 × 3 × 5 × 2 × 20 × 10 × 2 = **2,400,000 combinations**

**Multirun jobs**: ~2 × 10 × 3 × 5 × 2 × (10 + 1) = **6,600 jobs** (CrossSession: 10 per subject, CrossSubject: 1 for all)

**Memory savings**: ~364x fewer items to track
**Time savings**: Proportional to the reduction in items to check

---

## 10. Recommendations

1. **For immediate use**: Use `experiment_automation.py` - it's complete and works
2. **For optimization**: Integrate the optimized methods from `experiment_automation_optimized.py` into the main file
3. **Integration strategy**:
   - Add `generate_expected_multirun_jobs()` as an alternative method
   - Add `identify_missing_experiments_optimized()` as an alternative method
   - Add a flag to choose between approaches
   - Keep all existing features (script generation, etc.)
   - Add cache persistence to optimized methods
   - Add early filtering to optimized methods











