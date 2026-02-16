# Plot 2 File Collision Diagnostic

## Purpose

When SW and NSW models show **identical max_drop** across all regimes, the cause is often misattributed to "file collision" (analysis reading the same CSV for both). This diagnostic isolates the real cause.

## What the diagnostic does

1. **short_run_id check** — Confirms SW and NSW have different hashes (no hash collision).
2. **File discovery** — Lists paths found for each model; verifies no overlap.
3. **Path overlap** — Fails if any file is shared between models.
4. **Raw CSV comparison** — Compares first file per model: row counts, columns, max_drop.
5. **Adjacency check** (optional `--check_adj`) — Compares `hidden_adj_undirected` from pilot JSONs.

## How to run

```bash
# Basic (uses default model pair)
python architecture_refinement/run_plot2_file_collision_diagnostic.py

# With specific models
python architecture_refinement/run_plot2_file_collision_diagnostic.py \
  --model_a plot2_intermediate_sparse_sw_2 \
  --model_b plot2_intermediate_sparse_nsw_3

# With adjacency check (requires --output_dir from intermediate diagnostic)
python architecture_refinement/run_plot2_file_collision_diagnostic.py \
  --output_dir path/to/intermediate_output \
  --check_adj
```

## Root cause when files are correct but max_drop is identical

If the diagnostic shows:
- **No path overlap** (different files for SW vs NSW)
- **Correct model names** in each CSV
- **Identical corrupted_roc_auc** at every intensity

Then the issue is **not** file collision. It is a **runner/model bug**:

1. **Checkpoint reuse** — Cache path should include `model_name`; verify `model_cache_manager._get_cache_path` uses it.
2. **NAS pilot closure** — Each model's factory must capture its own `_arch` snapshot. A Python closure bug could cause all models to use the same wiring.
3. **Identical adjacencies** — Run with `--check_adj` to verify the JSON files have different `hidden_adj_undirected` matrices.

## Changes made to analysis pipeline

- **`_find_result_files`** — Uses exact path-segment matching: `short_run_id` must appear as a directory component, not a substring.
- **`_load_model_results`** — If CSV has a `model` column, verifies it matches the requested `model_name`; skips file and warns on mismatch.
