# Orientation Duplication Diagnosis

## Problem
`orient_ro` (random_oriented) and `orient_sym` (symmetric) produce **identical** scores in the final report, which is impossible given different wirings.

## Data Flow Traced

### 1. Path Generation
- `create_output_path(model_name, ...)` uses `short_run_id(model_name)` (12-char hash) as directory segment
- `orient_ro` → `83003f17dbdb`
- `orient_sym` → `ec1823c6dbdc`
- **Paths are distinct** ✓

### 2. Result File Lookup (_find_result_files)
- Walks `repo_root/results`, finds CSVs where:
  - `short_id in path_parts` (as directory component) AND
  - `"test_perturb"` in path
- With distinct short_ids, each model should find **different files** ✓

### 3. Model Registration (nas_pilot_registry)
- Loads ALL `*.json` from `pilot_dir/selected_architectures/`
- Creates one factory per arch; each captures `arch_snapshot` (dict) with correct `hidden_edge_orientation`
- **Registration looks correct** ✓

### 4. Runner Invocation
- Each model runs in **separate subprocess**: `subprocess.run(cmd, ..., "--model", model_name)`
- Fresh registry load each run
- **No cross-process state** ✓

## Suspected Root Causes

### A) Model Cache Cross-Load
When `orient_sym` runs, if it loads a cached model from `orient_ro` (wrong path), it would produce identical results. Cache path includes `model_name`, so this should not happen unless there is a path normalization bug on Windows.

### B) Shared Pilot Dir → Factory Closure
When both models are registered from the same pilot_dir, **both JSON files are in the same directory**. The registry iterates `for p in sorted(arch_dir.glob("*.json"))`. If there is a closure bug where the *last* arch_snapshot is used for all factories (e.g. late binding), then both orient_ro and orient_sym would use orient_sym's wiring (last in sorted order).

**This is a plausible bug**: In Python, default args are bound at def time. The loop:
```python
for p in arch_files:
    arch = json.load(...)
    arch_snapshot = dict(arch)
    def _factory(..., _arch=arch_snapshot, ...):
        ...
```
Each iteration creates a new `_factory` with `_arch=arch_snapshot`. The key: `arch_snapshot` is rebound each iteration. When the def statement executes, it captures the *current* value of `arch_snapshot`. So factory_ro should get orient_ro's arch, factory_sym gets orient_sym's arch. **This should be correct** unless sorted(arch_files) puts them in an unexpected order.

### C) Result Path Collision
If `create_output_path` returns the **same path** for both models (e.g. due to "//" normalization on Windows creating an ambiguous path), both would write to the same directory and overwrite.

### D) Analyzer Finds Same Files for Both
If `_find_result_files` has a bug where both model names match the same set of files (e.g. fallback substring match is too broad), we'd load identical data.

## Implemented Fix: Separate Pilot Dirs

The run script now creates **model-specific pilot directories**:
- `pilot_orient_ro/selected_architectures/` contains only `orient_ro.json`
- `pilot_orient_sym/selected_architectures/` contains only `orient_sym.json`

Each model run uses `--nas_pilot_dir pilot_<model>/` so the registry loads **only** that model's JSON. No shared state, no closure confusion.

The main `out_dir/selected_architectures/` still has both JSONs for the analyzer, which matches results by `model_name` via `_find_result_files(repo_root, model_name)`.
