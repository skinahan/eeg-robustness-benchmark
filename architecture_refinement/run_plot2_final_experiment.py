"""
Plot 2 Final Experiment — End-to-End Orchestrator

Runs the complete Plot 2 "final attempt" pipeline per the 2026-02-17 spec:
  0) Pre-flight: no resume; fresh run_id or --force_search
  1) Proxy viability (V1–V5 gates)
  2) Mini-scale topology study
  3) Phase-6 integrity checks
  4) Analysis + report generation
  5) Optional: scale-up (user-triggered after GO)

Errors or NO-GO results terminate early with informative messages.
Detailed diagnostic logging to stdout and optional log file.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging to stdout and optionally to file."""
    log = logging.getLogger("plot2_final")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    h_stdout = logging.StreamHandler(sys.stdout)
    h_stdout.setLevel(logging.INFO)
    h_stdout.setFormatter(fmt)
    log.addHandler(h_stdout)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        h_file = logging.FileHandler(log_file, encoding="utf-8")
        h_file.setLevel(logging.DEBUG)
        h_file.setFormatter(fmt)
        log.addHandler(h_file)
    return log


def _run_cmd(
    cmd: List[str],
    cwd: Path,
    log: logging.Logger,
    check: bool = True,
    timeout: Optional[int] = None,
) -> Tuple[int, str, str]:
    """Run command; return (returncode, stdout, stderr)."""
    log.info("Running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        if out:
            for line in out.strip().splitlines():
                log.debug("  stdout: %s", line)
        if err:
            for line in err.strip().splitlines():
                log.warning("  stderr: %s", line)
        if proc.returncode != 0:
            log.error("Command exited with code %d", proc.returncode)
            if out.strip():
                for line in out.strip().splitlines():
                    log.error("  stdout: %s", line)
        if check and proc.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {proc.returncode}")
        return proc.returncode, out, err
    except subprocess.TimeoutExpired as e:
        log.error("Command timed out after %s seconds", timeout)
        raise RuntimeError(f"Command timed out: {e}") from e


def _fail(msg: str, log: logging.Logger, exit_code: int = 1) -> None:
    """Log error and exit."""
    log.error("=" * 60)
    log.error("PLOT 2 FINAL EXPERIMENT: EARLY TERMINATION")
    log.error("%s", msg)
    log.error("=" * 60)
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Step 1: Proxy Viability
# ---------------------------------------------------------------------------


def _run_proxy_viability(
    repo_root: Path,
    output_dir: Path,
    M_ref: int,
    H: int,
    log: logging.Logger,
    relaxed_v2: bool = False,
    relaxed_v4: bool = False,
) -> bool:
    """
    Run proxy viability. Returns True if GO (all gates pass), False if NO-GO.
    Raises on script failure.
    """
    script = repo_root / "architecture_refinement" / "run_plot2_proxy_viability.py"
    if not script.exists():
        _fail(f"Proxy viability script not found: {script}", log)

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--output_dir", str(output_dir),
        "--M_ref", str(M_ref),
        "--H", str(H),
        "--relaxed_v2",
        "--relaxed_v4",
    ]

    rc, out, err = _run_cmd(cmd, repo_root, log, check=False)
    if rc != 0:
        _fail(
            f"Proxy viability script exited with code {rc}. "
            "Fix generator/bounds/coverage and re-run Step 1.",
            log,
        )

    # GO/NO-GO #1 checks
    report_path = output_dir / "proxy_viability_report.json"
    if not report_path.exists():
        _fail(f"Proxy viability report not found: {report_path}", log)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    all_pass = report.get("all_gates_pass", False)
    gates = report.get("gates", {})

    for gname, gdata in gates.items():
        ok = gdata.get("pass", False)
        reasons = gdata.get("reasons", [])
        status = "PASS" if ok else "FAIL"
        log.info("  %s: %s", gname, status)
        if not ok and reasons:
            for r in reasons:
                log.warning("    %s", r)

    if not all_pass:
        log.error("Proxy viability NO-GO: one or more gates failed.")
        return False

    # Artifact checks
    frozen_path = output_dir / "frozen_bin_edges.json"
    if not frozen_path.exists():
        _fail(f"frozen_bin_edges.json not found: {frozen_path}", log)

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    te_res_lo = frozen.get("te_res_lo", float("nan"))
    te_res_hi = frozen.get("te_res_hi", float("nan"))
    sigma_lo = frozen.get("sigma_lo", float("nan"))
    sigma_hi = frozen.get("sigma_hi", float("nan"))

    if not (te_res_lo < te_res_hi and sigma_lo < sigma_hi):
        _fail(
            f"Degenerate bounds: TE_res [{te_res_lo}, {te_res_hi}], sigma [{sigma_lo}, {sigma_hi}]. "
            "Bounds must satisfy lo < hi.",
            log,
        )

    for name in ["mu_te_by_k.json", "mu_orc_by_k.json"]:
        p = output_dir / name
        if not p.exists():
            _fail(f"Required artifact missing: {p}", log)

    log.info("Proxy viability GO: all gates passed, artifacts valid.")
    return True


# ---------------------------------------------------------------------------
# Step 2: Mini-Scale Topology Study
# ---------------------------------------------------------------------------


def _run_topology_study(
    repo_root: Path,
    proxy_viability_dir: Path,
    scale: str,
    subjects: List[int],
    B: int,
    S: int,
    M_max: int,
    log: logging.Logger,
    force_search: bool = True,
    smoke: bool = False,
    run_id: Optional[str] = None,
    target_snr_dbs: str = "-12",
) -> Path:
    """
    Run topology study. Returns plot2_dir path.
    Raises on failure.
    """
    script = repo_root / "architecture_refinement" / "run_plot2_topology_study.py"
    if not script.exists():
        _fail(f"Topology study script not found: {script}", log)

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    plot2_dir = (repo_root / "architecture_refinement" / "outputs" / "plot2_topology_study" / run_id).resolve()

    cmd = [
        sys.executable,
        str(script),
        "--run_id", run_id,
        "--scale", scale,
        "--subjects", *[str(s) for s in subjects],
        "--generator_mode", "ws_flex",
        "--perturbation_types", "ar1_drift",
        f"--target_snr_dbs={target_snr_dbs}",
        "--alpha_grid", "0,0.25,0.5,0.75,1.0",
        "--selection_coverage_level", "regime_cl_bins_fixed",
        "--proxy_viability_dir", str(proxy_viability_dir),
        "--B", str(B),
        "--S", str(S),
        "--M_max", str(M_max),
        "--force_search",
    ]
    if smoke:
        cmd.extend(["--dry_run"])
        log.info("Smoke mode: using dry_run (no training).")

    rc, out, err = _run_cmd(cmd, repo_root, log, check=False, timeout=None)
    if rc != 0:
        _fail(
            f"Topology study exited with code {rc}. "
            "Do NOT interpret results; run integrity checks to localize failure.",
            log,
        )

    # Parse plot2_dir from output
    match = re.search(r"\[PLOT2\] Plot2 directory:\s*(.+)", out)
    if match:
        parsed = match.group(1).strip()
        if Path(parsed).is_dir():
            plot2_dir = Path(parsed).resolve()

    if not plot2_dir.exists():
        _fail(f"Plot2 directory not found: {plot2_dir}", log)

    manifest_path = plot2_dir / "plot2_manifest.json"
    if not manifest_path.exists():
        manifest_path = plot2_dir / "manifest.json"
    if not manifest_path.exists():
        _fail(f"Manifest not found in {plot2_dir}", log)

    # Schema validation (basic)
    try:
        from architecture_refinement.plot2_schema import validate_manifest
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest, strict=True)
    except Exception as e:
        _fail(f"Manifest schema validation failed: {e}", log)

    log.info("Topology study complete. Plot2 directory: %s", plot2_dir)
    return plot2_dir


# ---------------------------------------------------------------------------
# Step 3: Integrity Checks
# ---------------------------------------------------------------------------


def _run_integrity_checks(
    repo_root: Path,
    plot2_dir: Path,
    log: logging.Logger,
    strict_spec: bool = True,
) -> bool:
    """Run Phase-6 integrity checks. Returns True if all pass."""
    script = repo_root / "architecture_refinement" / "run_plot2_integrity_checks.py"
    if not script.exists():
        _fail(f"Integrity checks script not found: {script}", log)

    cmd = [
        sys.executable,
        str(script),
        "--plot2_dir", str(plot2_dir),
        "--strict_spec",
    ]
    rc, out, err = _run_cmd(cmd, repo_root, log, check=False)
    if rc != 0:
        _fail(
            f"Integrity checks FAILED (exit {rc}). "
            "Fix the failing phase, then re-run Step 2 with --force_search.",
            log,
        )
    return True


# ---------------------------------------------------------------------------
# Step 4: Analysis
# ---------------------------------------------------------------------------


def _run_analysis(
    repo_root: Path,
    plot2_dir: Path,
    log: logging.Logger,
) -> Path:
    """Run analysis. Returns analysis output directory."""
    script = repo_root / "architecture_refinement" / "analyze_plot2_results.py"
    if not script.exists():
        _fail(f"Analysis script not found: {script}", log)

    cmd = [
        sys.executable,
        str(script),
        "--plot2_dir", str(plot2_dir),
    ]
    rc, out, err = _run_cmd(cmd, repo_root, log, check=False)
    if rc != 0:
        _fail(f"Analysis script exited with code {rc}.", log)

    report_path = plot2_dir / "analysis" / "report.txt"
    if report_path.exists():
        log.info("Report written: %s", report_path)
    return plot2_dir / "analysis"


# ---------------------------------------------------------------------------
# Step 4 GO/NO-GO (scale-up decision)
# ---------------------------------------------------------------------------


def _evaluate_scale_up_decision(analysis_dir: Path, log: logging.Logger) -> bool:
    """
    Evaluate whether to proceed to scale-up per spec §4.
    Returns True if GO (proceed to scale-up), False if NO-GO (pivot).
    """
    report_path = analysis_dir / "report.txt"
    if not report_path.exists():
        log.warning("Report not found; cannot evaluate scale-up decision.")
        return False

    text = report_path.read_text(encoding="utf-8")
    # Parse M4 verdict from report (format: "M4 verdict: GO=True, NO-GO=False")
    if "M4 verdict: GO=True" in text and "NO-GO=False" in text:
        log.info("M4 verdict: GO — primary claim signal detected. Proceed to scale-up.")
        return True
    if "NO-GO=True" in text:
        log.info("M4 verdict: NO-GO — pivot justified by diagnostics.")
        return False
    # Diagnostic signal: even if TPE doesn't win, actionable direction
    if "actionable" in text.lower() or "diagnostic" in text.lower():
        log.info("Diagnostic signal present; consider scale-up for confirmation.")
        return True
    log.info("Inconclusive; recommend pivot unless diagnostics indicate pipeline fix.")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot 2 Final Experiment — end-to-end orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--proxy_output_dir",
        type=str,
        default="architecture_refinement/outputs/proxy_viability_plot2_final_mini",
        help="Output directory for proxy viability (Step 1).",
    )
    parser.add_argument("--M_ref", type=int, default=5000, help="Reference set size for proxy viability.")
    parser.add_argument("--H", type=int, default=32, help="Hidden size.")
    parser.add_argument(
        "--scale",
        type=str,
        default="mini",
        choices=["mini", "full"],
        help="Topology study scale.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="*",
        default=[1, 3, 4],
        help="Subject IDs for mini-scale.",
    )
    parser.add_argument("--B", type=int, default=8, help="Topologies per method.")
    parser.add_argument("--S", type=int, default=1, help="Training seeds per topology.")
    parser.add_argument(
        "--M_max",
        type=int,
        default=200,
        help="Max search budget for topology study (reduces saturation; default 200).",
    )
    parser.add_argument(
        "--target_snr_dbs",
        type=str,
        default="-12",
        help="Comma-separated target SNRs (e.g. -12 or -12,-6). Mini: default -12 per audit (locked config).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: dry_run topology study (B=2, no training).",
    )
    parser.add_argument(
        "--relaxed_v2",
        action="store_true",
        help="Use relaxed V2 gate in proxy viability.",
    )
    parser.add_argument(
        "--relaxed_v4",
        action="store_true",
        help="Use relaxed V4 gate in proxy viability.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path to write detailed log.",
    )
    parser.add_argument(
        "--stop_after",
        type=str,
        default=None,
        choices=["proxy", "topology", "integrity", "analysis"],
        help="Stop after this step (for debugging).",
    )
    args = parser.parse_args()

    repo_root = _REPO_ROOT
    log_file = Path(args.log_file).resolve() if args.log_file else None
    log = _setup_logging(log_file)

    log.info("=" * 60)
    log.info("PLOT 2 FINAL EXPERIMENT — START")
    log.info("=" * 60)
    log.info("Pre-flight: no resume; using --force_search for topology study.")

    proxy_dir = (repo_root / args.proxy_output_dir).resolve()

    # Step 1: Proxy viability
    log.info("")
    log.info("--- Step 1: Proxy Viability ---")
    if not _run_proxy_viability(
        repo_root, proxy_dir, args.M_ref, args.H, log,
        relaxed_v2=args.relaxed_v2, relaxed_v4=args.relaxed_v4,
    ):
        _fail(
            "Proxy viability NO-GO: one or more V-gates failed. "
            "Fix generator/bounds/coverage. Do NOT proceed to training.",
            log,
        )
    if args.stop_after == "proxy":
        log.info("Stopping after proxy (--stop_after=proxy).")
        return 0

    # Step 2: Topology study
    log.info("")
    log.info("--- Step 2: Mini-Scale Topology Study ---")
    plot2_dir = _run_topology_study(
        repo_root,
        proxy_dir,
        scale=args.scale,
        subjects=args.subjects,
        B=args.B,
        S=args.S,
        M_max=args.M_max,
        log=log,
        force_search=True,
        smoke=args.smoke,
        target_snr_dbs=args.target_snr_dbs,
    )
    if args.stop_after == "topology":
        log.info("Stopping after topology (--stop_after=topology).")
        log.info("Plot2 directory: %s", plot2_dir)
        return 0

    # Step 3: Integrity checks
    log.info("")
    log.info("--- Step 3: Phase-6 Integrity Checks ---")
    _run_integrity_checks(repo_root, plot2_dir, log, strict_spec=True)
    if args.stop_after == "integrity":
        log.info("Stopping after integrity (--stop_after=integrity).")
        return 0

    # Step 4: Analysis
    log.info("")
    log.info("--- Step 4: Analysis + Report ---")
    analysis_dir = _run_analysis(repo_root, plot2_dir, log)
    go_scale_up = _evaluate_scale_up_decision(analysis_dir, log)

    log.info("")
    log.info("=" * 60)
    log.info("PLOT 2 FINAL EXPERIMENT — COMPLETE")
    log.info("=" * 60)
    log.info("Plot2 directory: %s", plot2_dir)
    log.info("Report: %s", analysis_dir / "report.txt")
    log.info("Scale-up decision: %s", "GO" if go_scale_up else "NO-GO (pivot)")
    if go_scale_up:
        log.info("To scale up: run topology study with --scale full --B 12 --S 3, then repeat Steps 3–4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
