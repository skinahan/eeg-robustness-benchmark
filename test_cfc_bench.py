# bench_cfc.py
import time
import torch
from ncps.torch import CfC as BaselineCfC
from models.fastcfc_improved import CfCOptimized, CfCCellFused, compiled

def measure_latency(model, x, timespans=None, warmup=20, iters=100):
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model(x, timespans=timespans)
        if x.is_cuda:
            torch.cuda.synchronize()
        # timed
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x, timespans=timespans)
        if x.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters  # ms per inference

def main():
    torch.set_float32_matmul_precision("high")  # enable TF32 where available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Match typical shapes *after* your CNN/SPP front-end
    B, L, C = 16, 256, 64
    H = 128
    P = 64

    x = torch.randn(B, L, C, device=device)
    timespans = None#torch.ones(B, L, device=device)  # comment out to test implicit 1.0

    # 1) Baseline (ncps)
    base = BaselineCfC(
        input_size=C, units=H, proj_size=P, return_sequences=True,
        batch_first=True, mixed_memory=False, mode="default",
    ).to(device)

    from ncps.torch import CfCCell as CfCCell_no_fuse
    # 2) Optimized wrapper, original cell path (no fusion)
    opt_no_fuse = CfCOptimized(
        input_size=C, units=H, proj_size=P, return_sequences=True,
        batch_first=True, mixed_memory=False, mode="default",
        activation="lecun_tanh", use_amp=False,
        cell_cls=CfCCell_no_fuse  # defaults to fused; we override to force non-fused by passing a stub below if you wish
    ).to(device)

    # To force "non-fused" inside the wrapper, you can pass your original CfCCell class
    # from your codebase, e.g.:
    # from your_module import CfCCell as OrigCell
    # opt_no_fuse = CfCOptimized(..., cell_cls=OrigCell, use_amp=False).to(device)

    # 3) Optimized wrapper + FUSED cell
    opt_fused = CfCOptimized(
        input_size=C, units=H, proj_size=P, return_sequences=True,
        batch_first=True, mixed_memory=False, mode="default",
        activation="lecun_tanh", use_amp=False,
        cell_cls=CfCCellFused
    ).to(device)

    # # 4) FUSED + AMP
    opt_fused_amp = CfCOptimized(
        input_size=C, units=H, proj_size=P, return_sequences=True,
        batch_first=True, mixed_memory=False, mode="default",
        activation="lecun_tanh", use_amp=False,
        cell_cls=CfCCellFused
    ).to(device)

    # 5) FUSED + AMP + torch.compile
    opt_fused_amp_comp = compiled(opt_fused_amp, fullgraph=True)
    # prime the compiled graph
    _ = opt_fused_amp_comp(x, timespans=timespans)

    ms_base   = measure_latency(base, x, timespans=timespans)
    ms_opt    = measure_latency(opt_no_fuse, x, timespans=timespans)
    ms_fused  = measure_latency(opt_fused, x, timespans=timespans)
    ms_amp    = measure_latency(opt_fused_amp, x, timespans=timespans)
    ms_comp   = measure_latency(opt_fused_amp_comp, x, timespans=timespans)

    print(f"Device: {device}")
    print(f"Shapes: x=({B},{L},{C}), H={H}, P={P}")
    print(f"Baseline CfC (ncps)             : {ms_base:8.3f} ms / inf")
    print(f"Optimized wrapper (no fusion)   : {ms_opt:8.3f} ms / inf  (x{ms_base/max(ms_opt,1e-6):.2f})")
    print(f"Optimized + FUSED cell          : {ms_fused:8.3f} ms / inf  (x{ms_base/max(ms_fused,1e-6):.2f})")
    print(f"+ AMP                            : {ms_amp:8.3f} ms / inf  (x{ms_base/max(ms_amp,1e-6):.2f})")
    print(f"+ AMP + torch.compile            : {ms_comp:8.3f} ms / inf  (x{ms_base/max(ms_comp,1e-6):.2f})")

if __name__ == "__main__":
    main()
