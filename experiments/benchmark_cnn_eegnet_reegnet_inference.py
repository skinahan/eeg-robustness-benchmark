"""
CNN-NCP, EEGNet, REEGNet Inference Speed Benchmark

Compares inference speed of three EEG classification models: CNN-NCP, EEGNet,
and REEGNet. Uses the exact same setup as the CfC branching benchmark:
BNCI2014_001 subject 1, 3 fixed seeds, ML conference standards (warm-up,
median latency, CUDA synchronization), and unified experiment runner extrapolation.

See experiments/CNN_EEGNET_REEGNET_BENCHMARK_SPEC.md for specification.
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
from models.cnnncp import create_cnnncp_classifier
from models.eegnet import create_eegnet_classifier
from models.reegnet import create_reegnet_classifier

# Benchmark constants (same as CfC branching benchmark)
N_WARMUP = 50
N_TRIALS = 200
BATCH_SIZE = 32
SEEDS = [42, 43, 44]

# Typical unified experiment runner parameters (for extrapolation)
UNIFIED_RUNNER_N_MODELS = 1
UNIFIED_RUNNER_N_SEEDS = 5
UNIFIED_RUNNER_N_SUBJECTS = 9
UNIFIED_RUNNER_MAX_EPOCHS = 200
UNIFIED_RUNNER_BATCHES_PER_EPOCH = 5
UNIFIED_RUNNER_TEST_PERTURB_BATCHES = 100

MODELS = {
    "cnn_ncp": create_cnnncp_classifier,
    "eegnet": create_eegnet_classifier,
    "reegnet": create_reegnet_classifier,
}


def load_bnci2014_001_subject1():
    """Load BNCI2014_001 data for subject 1."""
    dataset = BNCI2014_001()
    paradigm = get_paradigm(dataset="BNCI2014_001")
    X, y, metadata = paradigm.get_data(dataset, subjects=[1], return_epochs=False)

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
    """Benchmark inference speed for a model (MLPerf/TorchBench standards)."""
    model.eval()
    model.to(device)
    X_batch = X_batch.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(X_batch)

        if device.type == "cuda":
            torch.cuda.synchronize()

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


def estimate_unified_runner_time(
    median_ms: float,
    n_models: int = UNIFIED_RUNNER_N_MODELS,
    n_seeds: int = UNIFIED_RUNNER_N_SEEDS,
    n_subjects: int = UNIFIED_RUNNER_N_SUBJECTS,
    max_epochs: int = UNIFIED_RUNNER_MAX_EPOCHS,
    batches_per_epoch: int = UNIFIED_RUNNER_BATCHES_PER_EPOCH,
    test_perturb_batches: int = UNIFIED_RUNNER_TEST_PERTURB_BATCHES,
) -> dict:
    """Estimate inference time for a typical unified experiment runner run."""
    batches_per_run = max_epochs * batches_per_epoch + test_perturb_batches
    total_batches = n_models * n_seeds * n_subjects * batches_per_run
    time_sec = total_batches * (median_ms / 1000.0)
    return {
        "total_inference_batches": total_batches,
        "inference_time_sec": time_sec,
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
    batch_size=BATCH_SIZE,
    n_warmup=N_WARMUP,
    n_trials=N_TRIALS,
    seeds=SEEDS,
    models_to_run=None,
):
    """Run full benchmark for CNN-NCP, EEGNet, and REEGNet."""
    models_to_run = models_to_run or list(MODELS.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Warm-up: {n_warmup}, Trials: {n_trials}")
    print(f"Seeds: {seeds}")
    print(f"Models: {models_to_run}")
    print()

    # Load data
    print("Loading BNCI2014_001 subject 1...")
    X, y, n_chans, n_times, n_outputs = load_bnci2014_001_subject1()
    n_samples = len(X)
    print(f"  n_samples={n_samples}, n_chans={n_chans}, n_times={n_times}")

    # Create batch
    n_batches = (n_samples + batch_size - 1) // batch_size
    n_pad = n_batches * batch_size - n_samples
    X_padded = np.concatenate([X, X[:n_pad]], axis=0) if n_pad > 0 else X
    X_batch_np = X_padded[:batch_size]
    X_batch = torch.from_numpy(X_batch_np).float()

    results = {name: [] for name in models_to_run}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print("=" * 60)
        set_seeds(seed)

        for model_name in models_to_run:
            factory = MODELS.get(model_name)
            if factory is None:
                print(f"  Skipping unknown model: {model_name}")
                continue

            print(f"  Benchmarking {model_name}...")
            classifier = factory(
                n_chans=n_chans,
                n_times=n_times,
                n_outputs=n_outputs,
            )
            # Ensure module is built (EEGNet/REEGNet don't initialize in factory)
            classifier.initialize()
            model = classifier.module_

            r = benchmark_model(
                model, X_batch, device, n_warmup=n_warmup, n_trials=n_trials
            )
            results[model_name].append(r)
            print(
                f"    median={r['median_latency_ms']:.3f} ms, "
                f"throughput={r['throughput_samples_per_sec']:.1f} samples/s"
            )

    # Aggregate
    def agg(lst, key):
        vals = [r[key] for r in lst]
        return np.mean(vals), np.std(vals)

    print("\n" + "=" * 60)
    print("SUMMARY (mean +/- std across seeds)")
    print("=" * 60)

    summaries = {}
    for model_name in models_to_run:
        lst = results[model_name]
        if not lst:
            continue
        median_mean, median_std = agg(lst, "median_latency_ms")
        throughput_mean, throughput_std = agg(lst, "throughput_samples_per_sec")
        summaries[model_name] = {
            "median_ms": (median_mean, median_std),
            "throughput": (throughput_mean, throughput_std),
        }
        print(
            f"{model_name:10} median latency: {median_mean:.3f} +/- {median_std:.3f} ms  "
            f"throughput: {throughput_mean:.1f} +/- {throughput_std:.1f} samples/s"
        )
    print("=" * 60)

    # Extrapolation
    print("\n" + "=" * 60)
    print("UNIFIED EXPERIMENT RUNNER EXTRAPOLATION (estimate)")
    print("=" * 60)
    est = estimate_unified_runner_time(median_ms=0)  # params only
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
    print()
    for model_name in models_to_run:
        if model_name not in summaries:
            continue
        median_mean, _ = summaries[model_name]["median_ms"]
        model_est = estimate_unified_runner_time(median_ms=median_mean)
        print(
            f"  {model_name:10} inference time: {model_est['inference_time_sec']/3600:.2f} h "
            f"({model_est['inference_time_sec']:.0f} s)"
        )

    # Relative speed (fastest = 1.0)
    medians = {n: summaries[n]["median_ms"][0] for n in models_to_run if n in summaries}
    if medians:
        fastest = min(medians.values())
        print()
        print("  Relative to fastest:")
        for name, m in sorted(medians.items(), key=lambda x: x[1]):
            print(f"    {name:10} {m/fastest:.2f}x (slower)")

    print("=" * 60)

    return {
        "results": results,
        "summaries": summaries,
        "device": str(device),
        "batch_size": batch_size,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CNN-NCP, EEGNet, REEGNet inference speed benchmark"
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
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS.keys()),
        help=f"Comma-separated models (default: {','.join(MODELS.keys())})",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]

    run_benchmark(
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        n_trials=args.n_trials,
        seeds=seeds,
        models_to_run=models_to_run,
    )


if __name__ == "__main__":
    main()
