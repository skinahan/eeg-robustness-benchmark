"""
CfC Branching Inference Speed Benchmark

Quantifies the relative inference speed of the CfC recurrent compartment when using
branched vs unbranched temporal processing. Uses BNCI2014_001 subject 1 data,
3 fixed seeds, and follows ML conference benchmarking standards (warm-up, median
latency, CUDA synchronization).

See experiments/CFC_BRANCHING_BENCHMARK_SPEC.md for full specification.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root for imports
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from config import get_paradigm
from moabb.datasets import BNCI2014_001
from globals import set_seeds
from models.branched_wiredcfc import create_branched_wiredcfc_classifier
from architecture_refinement.arbitrary_wiring import load_architecture_from_file

# Benchmark constants (ML conference standards)
N_WARMUP = 50
N_TRIALS = 200
BATCH_SIZE = 32
SEEDS = [42, 43, 44]
DEFAULT_ARCHITECTURE_PATH = "outputs/architectures/best_architecture_4_trial_178.json"

# Typical unified experiment runner parameters (for extrapolation)
# Sources: experiment_config.yaml (seeds), globals.py (max_epochs), evaluation patterns
UNIFIED_RUNNER_N_MODELS = 1
UNIFIED_RUNNER_N_SEEDS = 5  # experiment_config.yaml
UNIFIED_RUNNER_N_SUBJECTS = 9  # BNCI2014_001
UNIFIED_RUNNER_MAX_EPOCHS = 200  # globals.get_max_epochs_for_dataset
UNIFIED_RUNNER_BATCHES_PER_EPOCH = 5  # train + val batches (estimate)
UNIFIED_RUNNER_TEST_PERTURB_BATCHES = 100  # 20 steps * 4 noise types * ~1.25 batches


def load_bnci2014_001_subject1():
    """Load BNCI2014_001 data for subject 1.

    Returns:
        tuple: (X, y, n_chans, n_times, n_outputs)
        X: np.ndarray shape (n_epochs, n_chans, n_times)
        y: labels
    """
    dataset = BNCI2014_001()
    paradigm = get_paradigm(dataset="BNCI2014_001")
    X, y, metadata = paradigm.get_data(dataset, subjects=[1], return_epochs=False)

    # Handle different return types from MOABB
    if hasattr(X, "get_data"):
        X = X.get_data()
    X = np.asarray(X, dtype=np.float32)

    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    n_chans = X.shape[1]
    n_times = X.shape[2]
    n_outputs = len(np.unique(y))

    return X, y, n_chans, n_times, n_outputs


def benchmark_model(model, X_batch, device, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    """Benchmark inference speed for a model.

    Follows MLPerf/TorchBench standards:
    - Warm-up runs to amortize CUDA kernel launch
    - Median latency (robust to outliers)
    - torch.cuda.synchronize() for GPU

    Args:
        model: nn.Module in eval mode
        X_batch: torch.Tensor (B, n_chans, n_times)
        device: torch device
        n_warmup: number of warm-up forward passes
        n_trials: number of timed forward passes

    Returns:
        dict: median_latency_ms, mean_latency_ms, std_latency_ms,
              per_sample_ms, throughput_samples_per_sec
    """
    model.eval()
    model.to(device)
    X_batch = X_batch.to(device)

    with torch.no_grad():
        # Warm-up
        for _ in range(n_warmup):
            _ = model(X_batch)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        latencies_ms = []
        for _ in range(n_trials):
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model(X_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    median_ms = float(np.median(latencies_ms))
    mean_ms = float(np.mean(latencies_ms))
    std_ms = float(np.std(latencies_ms))

    batch_size = X_batch.shape[0]
    per_sample_ms = median_ms / batch_size
    throughput = batch_size / (median_ms / 1000.0) if median_ms > 0 else 0.0

    return {
        "median_latency_ms": median_ms,
        "mean_latency_ms": mean_ms,
        "std_latency_ms": std_ms,
        "per_sample_ms": per_sample_ms,
        "throughput_samples_per_sec": throughput,
        "batch_size": batch_size,
    }


def estimate_unified_runner_savings(
    branched_median_ms: float,
    unbranched_median_ms: float,
    n_models: int = UNIFIED_RUNNER_N_MODELS,
    n_seeds: int = UNIFIED_RUNNER_N_SEEDS,
    n_subjects: int = UNIFIED_RUNNER_N_SUBJECTS,
    max_epochs: int = UNIFIED_RUNNER_MAX_EPOCHS,
    batches_per_epoch: int = UNIFIED_RUNNER_BATCHES_PER_EPOCH,
    test_perturb_batches: int = UNIFIED_RUNNER_TEST_PERTURB_BATCHES,
) -> dict:
    """Estimate inference time savings for a typical unified experiment runner run.

    Extrapolates from per-batch latency to total inference time across:
    - Training epochs (validation forward passes)
    - Test/perturb evaluation (inference-only)

    Note: This is an *estimate*. Training is dominated by backward passes;
    inference speedup mainly affects validation and test phases. The estimate
    uses conservative parameters matching experiment_config.yaml and globals.py.

    Returns:
        dict with: total_batches, unbranched_time_sec, branched_time_sec,
                   time_saved_sec, time_saved_pct
    """
    batches_per_run = max_epochs * batches_per_epoch + test_perturb_batches
    total_batches = n_models * n_seeds * n_subjects * batches_per_run

    unbranched_time_sec = total_batches * (unbranched_median_ms / 1000.0)
    branched_time_sec = total_batches * (branched_median_ms / 1000.0)
    time_saved_sec = unbranched_time_sec - branched_time_sec
    time_saved_pct = (
        (time_saved_sec / unbranched_time_sec * 100.0) if unbranched_time_sec > 0 else 0.0
    )

    return {
        "total_inference_batches": total_batches,
        "unbranched_time_sec": unbranched_time_sec,
        "branched_time_sec": branched_time_sec,
        "time_saved_sec": time_saved_sec,
        "time_saved_pct": time_saved_pct,
        "params": {
            "n_models": n_models,
            "n_seeds": n_seeds,
            "n_subjects": n_subjects,
            "max_epochs": max_epochs,
            "batches_per_epoch": batches_per_epoch,
            "test_perturb_batches": test_perturb_batches,
        },
    }


def run_benchmark(
    architecture_path=None,
    batch_size=BATCH_SIZE,
    n_warmup=N_WARMUP,
    n_trials=N_TRIALS,
    seeds=SEEDS,
):
    """Run full benchmark: branched vs unbranched across seeds."""
    architecture_path = Path(architecture_path or DEFAULT_ARCHITECTURE_PATH)
    if not architecture_path.is_absolute():
        architecture_path = _repo_root / architecture_path

    if not architecture_path.exists():
        raise FileNotFoundError(
            f"Architecture file not found: {architecture_path}\n"
            "Ensure outputs/architectures/ contains the architecture JSON."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {architecture_path}")
    print(f"Batch size: {batch_size}, Warm-up: {n_warmup}, Trials: {n_trials}")
    print(f"Seeds: {seeds}")
    print()

    # Load data
    print("Loading BNCI2014_001 subject 1...")
    X, y, n_chans, n_times, n_outputs = load_bnci2014_001_subject1()
    n_samples = len(X)
    print(f"  n_samples={n_samples}, n_chans={n_chans}, n_times={n_times}")

    # Create batches (pad if necessary)
    n_batches = (n_samples + batch_size - 1) // batch_size
    n_pad = n_batches * batch_size - n_samples
    if n_pad > 0:
        X_padded = np.concatenate([X, X[:n_pad]], axis=0)
    else:
        X_padded = X

    # Use first batch for benchmarking (representative)
    X_batch_np = X_padded[:batch_size]
    X_batch = torch.from_numpy(X_batch_np).float()

    results = {"branched": [], "unbranched": []}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print("=" * 60)
        set_seeds(seed)

        # Load wiring (BranchedWiredCfC uses projection layers when dims don't match)
        wiring_branched = load_architecture_from_file(str(architecture_path))
        wiring_unbranched = load_architecture_from_file(str(architecture_path))

        # Create models - both use BranchedWiredCfC for fair comparison (same wiring handling).
        # Branched: default bin_len=48, bin_stride=44 -> multiple bins, seq len L=48
        # Unbranched: bin_len=bin_stride=512 -> single bin, seq len T2 (~125)
        # Post-downsample T2 is ~125 for BNCI2014_001; 512 ensures one bin.
        classifier_branched = create_branched_wiredcfc_classifier(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            wiring=wiring_branched,
        )
        model_branched = classifier_branched.module_

        classifier_unbranched = create_branched_wiredcfc_classifier(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            wiring=wiring_unbranched,
            bin_len=512,
            bin_stride=512,
        )
        model_unbranched = classifier_unbranched.module_

        # Benchmark branched
        print("  Benchmarking branched...")
        r_branched = benchmark_model(
            model_branched, X_batch, device, n_warmup=n_warmup, n_trials=n_trials
        )
        results["branched"].append(r_branched)
        print(
            f"    median={r_branched['median_latency_ms']:.3f} ms, "
            f"throughput={r_branched['throughput_samples_per_sec']:.1f} samples/s"
        )

        # Benchmark unbranched
        print("  Benchmarking unbranched...")
        r_unbranched = benchmark_model(
            model_unbranched, X_batch, device, n_warmup=n_warmup, n_trials=n_trials
        )
        results["unbranched"].append(r_unbranched)
        print(
            f"    median={r_unbranched['median_latency_ms']:.3f} ms, "
            f"throughput={r_unbranched['throughput_samples_per_sec']:.1f} samples/s"
        )

        speedup = r_unbranched["median_latency_ms"] / r_branched["median_latency_ms"]
        print(f"  Speedup (branched vs unbranched): {speedup:.2f}x")

    # Aggregate
    def agg(lst, key):
        vals = [r[key] for r in lst]
        return np.mean(vals), np.std(vals)

    branched_median_mean, branched_median_std = agg(results["branched"], "median_latency_ms")
    unbranched_median_mean, unbranched_median_std = agg(
        results["unbranched"], "median_latency_ms"
    )
    branched_throughput_mean, branched_throughput_std = agg(
        results["branched"], "throughput_samples_per_sec"
    )
    unbranched_throughput_mean, unbranched_throughput_std = agg(
        results["unbranched"], "throughput_samples_per_sec"
    )

    speedups = [
        results["unbranched"][i]["median_latency_ms"]
        / results["branched"][i]["median_latency_ms"]
        for i in range(len(seeds))
    ]
    speedup_mean = np.mean(speedups)
    speedup_std = np.std(speedups)

    # Report
    print("\n" + "=" * 60)
    print("SUMMARY (mean +/- std across seeds)")
    print("=" * 60)
    print(f"Branched   median latency: {branched_median_mean:.3f} +/- {branched_median_std:.3f} ms")
    print(
        f"           throughput:      {branched_throughput_mean:.1f} +/- "
        f"{branched_throughput_std:.1f} samples/s"
    )
    print(
        f"Unbranched median latency: {unbranched_median_mean:.3f} +/- "
        f"{unbranched_median_std:.3f} ms"
    )
    print(
        f"           throughput:      {unbranched_throughput_mean:.1f} +/- "
        f"{unbranched_throughput_std:.1f} samples/s"
    )
    print(f"Speedup (branched faster): {speedup_mean:.2f} +/- {speedup_std:.2f}x")
    print("=" * 60)

    # Extrapolate to typical unified experiment runner run
    est = estimate_unified_runner_savings(
        branched_median_ms=branched_median_mean,
        unbranched_median_ms=unbranched_median_mean,
    )
    print("\n" + "=" * 60)
    print("UNIFIED EXPERIMENT RUNNER EXTRAPOLATION (estimate)")
    print("=" * 60)
    print(
        f"  Assumptions: {est['params']['n_models']} model(s), "
        f"{est['params']['n_seeds']} seeds, {est['params']['n_subjects']} subjects, "
        f"{est['params']['max_epochs']} max_epochs"
    )
    print(
        f"  Batches per run: {est['params']['max_epochs']}*{est['params']['batches_per_epoch']} "
        f"+ {est['params']['test_perturb_batches']} = "
        f"{est['params']['max_epochs']*est['params']['batches_per_epoch'] + est['params']['test_perturb_batches']}"
    )
    print(f"  Total inference batches: {est['total_inference_batches']:,}")
    print(
        f"  Unbranched inference time: {est['unbranched_time_sec']/3600:.2f} h "
        f"({est['unbranched_time_sec']:.0f} s)"
    )
    print(
        f"  Branched inference time:   {est['branched_time_sec']/3600:.2f} h "
        f"({est['branched_time_sec']:.0f} s)"
    )
    print(
        f"  Time saved (branched):     {est['time_saved_sec']/3600:.2f} h "
        f"({est['time_saved_sec']:.0f} s, {est['time_saved_pct']:.1f}%)"
    )
    print("=" * 60)

    return {
        "branched": results["branched"],
        "unbranched": results["unbranched"],
        "summary": {
            "branched_median_ms": (branched_median_mean, branched_median_std),
            "unbranched_median_ms": (unbranched_median_mean, unbranched_median_std),
            "speedup": (speedup_mean, speedup_std),
        },
        "unified_runner_estimate": est,
        "device": str(device),
        "batch_size": batch_size,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CfC branching inference speed benchmark"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=DEFAULT_ARCHITECTURE_PATH,
        help="Path to architecture JSON file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=N_WARMUP,
        help=f"Warm-up passes (default: {N_WARMUP})",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help=f"Timed trials per run (default: {N_TRIALS})",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(map(str, SEEDS)),
        help=f"Comma-separated seeds (default: {','.join(map(str, SEEDS))})",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    run_benchmark(
        architecture_path=args.architecture,
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        n_trials=args.n_trials,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
